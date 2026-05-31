import jax
import jax.numpy as jnp
import numpy as np

from rl_x.algorithms.tr_vbd_moe.flax_full_jit.default_config import get_config
from rl_x.algorithms.tr_vbd_moe.flax_full_jit.policy import TRVBDMoEPolicy, tanh_log_det_from_raw
from rl_x.algorithms.tr_vbd_moe.flax_full_jit.tr_vbd_moe import compute_tr_vbd_moe_actor_loss


def assert_allclose(actual, expected, *, rtol=1e-6, atol=1e-6):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol)


def make_policy(nr_experts=3, action_dim=2, observation_dim=4):
    return TRVBDMoEPolicy(
        action_dim=action_dim,
        action_scale=jnp.ones((action_dim,), dtype=jnp.float32),
        policy_observation_indices=jnp.arange(observation_dim),
        nr_experts=nr_experts,
        hidden_dim=16,
        layers=2,
        log_std_min=-4.0,
        log_std_max=1.0,
        min_std=0.0,
        ent_start=0.01,
        kl_start=0.02,
        use_norm=False,
        use_skip=False,
    )


def initialize_policy_params(policy, batch_size=5, observation_dim=4):
    observations = jnp.linspace(-0.5, 0.5, batch_size * observation_dim, dtype=jnp.float32).reshape(batch_size, observation_dim)
    return policy.init(jax.random.PRNGKey(0), observations)["params"], observations


def test_default_config_has_tr_vbd_moe_actor_and_distributional_critic_defaults():
    cfg = get_config("tr_vbd_moe.flax_full_jit")

    assert cfg.nr_experts == 4
    assert cfg.nr_actor_samples_per_expert == 1
    assert cfg.min_log_responsibility == -20.0
    assert cfg.hl_gauss is True
    assert cfg.lmbda == 0.95
    assert cfg.aux_loss_mult == 1.0
    assert cfg.kl_bound == 0.1


def test_mixture_log_prob_matches_manual_logsumexp_for_raw_samples():
    policy = make_policy()
    params, observations = initialize_policy_params(policy)
    gate_logits, means, log_stds = policy.distribution(params, observations)
    raw_actions = jnp.asarray(
        [
            [0.2, -0.1],
            [-0.3, 0.5],
            [0.0, 0.4],
            [0.9, -0.7],
            [-0.2, -0.6],
        ],
        dtype=jnp.float32,
    )

    actual = policy.mixture_log_prob_from_raw_dist(raw_actions, gate_logits, means, log_stds)
    component_log_probs = policy.component_raw_log_probs(raw_actions, means, log_stds)
    manual = jax.nn.logsumexp(jax.nn.log_softmax(gate_logits, axis=-1) + component_log_probs, axis=-1)
    manual = manual - tanh_log_det_from_raw(raw_actions, policy.action_scale)

    assert_allclose(actual, manual)


def test_old_responsibilities_are_selected_old_moe_posteriors():
    policy = make_policy(nr_experts=4)
    params, observations = initialize_policy_params(policy)
    sample_info = policy.sample_expert_actions(params, observations, jax.random.PRNGKey(1), nr_samples=3)

    actual = policy.log_responsibilities_for_expert_samples(
        params,
        observations,
        sample_info["raw_actions"],
        min_log_responsibility=-100.0,
    )

    gate_logits, means, log_stds = policy.distribution(params, observations)
    component_log_probs = policy.component_raw_log_probs(sample_info["raw_actions"], means, log_stds)
    log_joint = jax.nn.log_softmax(gate_logits, axis=-1)[:, None, None, :] + component_log_probs
    log_responsibilities = log_joint - jax.nn.logsumexp(log_joint, axis=-1, keepdims=True)
    expert_indices = jnp.broadcast_to(jnp.arange(policy.nr_experts)[None, :, None, None], sample_info["raw_actions"].shape[:-1] + (1,))
    expected = jnp.take_along_axis(log_responsibilities, expert_indices, axis=-1).squeeze(axis=-1)

    assert actual.shape == (observations.shape[0], policy.nr_experts, 3)
    assert_allclose(actual, expected)


def test_old_responsibilities_keep_action_path_gradients():
    policy = make_policy(nr_experts=4)
    params, observations = initialize_policy_params(policy)
    sample_info = policy.sample_expert_actions(params, observations, jax.random.PRNGKey(2), nr_samples=3)

    def responsibility_sum(raw_actions):
        return jnp.sum(
            policy.log_responsibilities_for_expert_samples(
                jax.lax.stop_gradient(params),
                observations,
                raw_actions,
                min_log_responsibility=-100.0,
            )
        )

    action_grad = jax.grad(responsibility_sum)(sample_info["raw_actions"])

    assert jnp.linalg.norm(action_grad) > 1e-6


def test_joint_kl_is_zero_for_identical_actor_params():
    policy = make_policy()
    params, observations = initialize_policy_params(policy)

    gate_kl, expert_kl, joint_kl = policy.joint_kl_components(params, params, observations)

    assert_allclose(gate_kl, jnp.zeros_like(gate_kl))
    assert_allclose(expert_kl, jnp.zeros_like(expert_kl))
    assert_allclose(joint_kl, jnp.zeros_like(joint_kl))


def test_tr_vbd_moe_actor_loss_matches_lagrangian_formula():
    vbd_bound = jnp.asarray(1.3, dtype=jnp.float32)
    joint_kl = jnp.asarray(0.25, dtype=jnp.float32)
    entropy = jnp.asarray(0.7, dtype=jnp.float32)
    temperature = jnp.asarray(0.4, dtype=jnp.float32)
    lagrangian = jnp.asarray(2.0, dtype=jnp.float32)
    action_size_target = jnp.asarray(3.0, dtype=jnp.float32)
    kl_bound = 0.1

    loss, metrics = compute_tr_vbd_moe_actor_loss(
        vbd_bound,
        joint_kl,
        entropy,
        temperature,
        lagrangian,
        action_size_target,
        kl_bound,
        reduce_kl=True,
        update_entropy_lagrangian=True,
        update_kl_lagrangian=True,
    )

    expected_actor_loss = vbd_bound + lagrangian * joint_kl
    expected_entropy_loss = temperature * (entropy - action_size_target)
    expected_kl_loss = -lagrangian * (joint_kl - kl_bound)
    expected_loss = expected_actor_loss + expected_entropy_loss + expected_kl_loss

    assert_allclose(loss, expected_loss)
    assert_allclose(metrics["actor_loss"], expected_actor_loss)
    assert_allclose(metrics["vbd_bound"], vbd_bound)
    assert_allclose(metrics["trust_region_loss"], lagrangian * joint_kl)
    assert_allclose(metrics["entropy_lagrangian_loss"], expected_entropy_loss)
    assert_allclose(metrics["kl_lagrangian_loss"], expected_kl_loss)
