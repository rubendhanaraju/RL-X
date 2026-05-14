from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl_x.algorithms.reppo_dime.flax_full_jit.default_config import get_config
from rl_x.algorithms.reppo_dime.flax_full_jit.policy import DIMEPolicy, LOG_2_PI
from rl_x.algorithms.reppo_dime.flax_full_jit.reppo_dime import (
    compute_reppo_dime_actor_loss,
    compute_reppo_dime_critic_loss,
    compute_reppo_dime_lambda_targets,
)
from rl_x.algorithms.reppo_dime.flax_full_jit.utils import get_action_scale, hl_gauss


def assert_allclose(actual, expected, *, rtol=1e-6, atol=1e-6):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol)


def legacy_hl_gauss_reference(inp, nr_bins, v_min, v_max, epsilon=0.0):
    # Mirrors reppo_code/src/jaxrl/utils.py::hl_gauss.
    x = jnp.clip(inp, v_min, v_max).squeeze(-1) / (1.0 - epsilon)
    bin_width = (v_max - v_min) / (nr_bins - 1)
    sigma = bin_width * 0.75
    support = jnp.linspace(
        v_min - bin_width / 2.0,
        v_max + bin_width / 2.0,
        nr_bins + 1,
        dtype=jnp.float32,
    )
    cdf_evals = jax.scipy.special.erf((support - x[..., None]) / (jnp.sqrt(2.0) * sigma))
    z = cdf_evals[..., -1:] - cdf_evals[..., :1]
    target_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
    target_probs = target_probs / z
    uniform = jnp.ones_like(target_probs) / nr_bins
    return (1.0 - epsilon) * target_probs + epsilon * uniform


def legacy_lambda_targets_reference(rewards, values, terminations, truncations, gamma, lmbda):
    # Mirrors reppo_code/src/jaxrl/reppo_dime.py::learn_step with zero importance weights.
    lambda_return = values[-1]
    next_truncated = jnp.ones_like(truncations[0])
    targets = []
    for timestep in reversed(range(rewards.shape[0])):
        lambda_sum = lmbda * lambda_return + (1.0 - lmbda) * values[timestep]
        delta = gamma * jnp.where(next_truncated, values[timestep], (1.0 - terminations[timestep]) * lambda_sum)
        lambda_return = rewards[timestep] + delta
        targets.insert(0, lambda_return)
        next_truncated = truncations[timestep]
    return jnp.stack(targets)


def legacy_cosine_schedule(step, diffusion_steps, schedule_min, schedule_s, schedule_power):
    # Mirrors reppo_code/src/networks/diffusion/schedulers.py::get_cosine_schedule.
    t = (diffusion_steps - step) / diffusion_steps
    offset = 1.0 + schedule_s
    return (1.0 - schedule_min) * jnp.cos(0.5 * jnp.pi * (offset - t) / offset) ** schedule_power + schedule_min


def make_zero_score_policy(action_dim=2, observation_dim=3, action_scale=None):
    if action_scale is None:
        action_scale = jnp.ones((action_dim,), dtype=jnp.float32)

    return DIMEPolicy(
        action_dim=action_dim,
        action_scale=jnp.asarray(action_scale, dtype=jnp.float32),
        policy_observation_indices=jnp.arange(observation_dim),
        diffusion_steps=4,
        diffusion_init_std=2.5,
        diffusion_friction=1.0,
        learn_forward=True,
        learn_backward=False,
        learn_prior=False,
        learn_betas=False,
        learn_dt=False,
        per_step_dt=False,
        per_dim_friction=True,
        learn_friction=True,
        learn_mass_matrix=False,
        dt=0.125,
        dt_schedule_min=0.001,
        dt_schedule_s=0.008,
        dt_schedule_power=2.0,
        eval_ode_coef=1.0,
        ent_start=0.01,
        kl_start=0.01,
        score_model_use_path_gradient=False,
        score_model_use_target_score=False,
        score_model_layer_norm=False,
        score_model_layer_norm_type="LayerNorm",
        score_model_nr_layers=2,
        score_model_nr_hidden_units=8,
        score_model_nr_time_hidden_units=4,
        score_model_time_coder_out=3,
        score_model_outer_clip=1e4,
        score_model_inner_clip=1e2,
        score_model_weight_init=0.0,
        score_model_bias_init=0.0,
    )


def initialize_policy_params(policy, action_dim=2, observation_dim=3):
    variables = policy.init(
        jax.random.PRNGKey(0),
        jnp.zeros((action_dim,), dtype=jnp.float32),
        jnp.zeros((observation_dim,), dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.zeros((action_dim,), dtype=jnp.float32),
    )
    return variables["params"]


def test_default_config_matches_legacy_reference_hyperparameters():
    cfg = get_config("reppo_dime.flax_full_jit")

    assert cfg.nr_steps == 128
    assert cfg.nr_minibatches == 128
    assert cfg.nr_epochs == 4
    assert cfg.gamma == 0.99
    assert cfg.lmbda == 0.95
    assert cfg.kl_bound == 0.1
    assert cfg.kl_action_rep == 4
    assert cfg.actor_kl_clip_mode == "clipped"
    assert cfg.ent_target_mult == 3.0
    assert cfg.diffusion_steps == 8
    assert cfg.diffusion_init_std == 2.5
    assert cfg.diffusion_friction == 1.0
    assert cfg.dt == 0.125
    assert cfg.per_dim_friction is True
    assert cfg.use_env_action_scale is False
    assert cfg.action_clipping is True
    assert cfg.action_clip_value == pytest.approx(0.999)


def test_action_scale_is_unit_by_default_like_legacy_tanh_actor():
    class Space:
        shape = (2,)
        low = jnp.asarray([-0.01, -2.0], dtype=jnp.float32)
        high = jnp.asarray([0.01, 4.0], dtype=jnp.float32)
        center = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
        scale = jnp.asarray([0.01, 2.0], dtype=jnp.float32)

    env = SimpleNamespace(single_action_space=Space())
    config = SimpleNamespace(algorithm=SimpleNamespace(use_env_action_scale=False))

    assert_allclose(get_action_scale(config, env), jnp.ones((2,), dtype=jnp.float32))

    config.algorithm.use_env_action_scale = True
    assert_allclose(get_action_scale(config, env), jnp.asarray([1.0, 1.5], dtype=jnp.float32))


def test_hl_gauss_matches_legacy_jax_reference():
    values = jnp.asarray([[-120.0], [-10.0], [0.0], [123.0], [520.0]], dtype=jnp.float32)
    actual = hl_gauss(values, nr_bins=151, v_min=-100.0, v_max=500.0)
    expected = legacy_hl_gauss_reference(values, nr_bins=151, v_min=-100.0, v_max=500.0)

    assert actual.shape == (5, 151)
    assert_allclose(jnp.sum(actual, axis=-1), jnp.ones((5,), dtype=jnp.float32), atol=2e-6)
    assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_lambda_targets_match_legacy_reference_with_zero_importance_weights():
    rewards = jnp.asarray(
        [
            [0.4, -0.2, 1.0],
            [1.2, 0.1, -0.4],
            [-0.5, 0.3, 0.2],
            [0.7, -1.0, 0.5],
        ],
        dtype=jnp.float32,
    )
    values = jnp.asarray(
        [
            [0.2, 0.5, -0.1],
            [1.1, -0.3, 0.4],
            [-0.7, 0.8, 0.6],
            [0.9, -1.2, 0.0],
        ],
        dtype=jnp.float32,
    )
    terminations = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    truncations = jnp.asarray(
        [
            [False, False, False],
            [False, True, False],
            [False, False, False],
            [True, False, False],
        ]
    )
    importance_weights = jnp.zeros_like(rewards)

    actual = compute_reppo_dime_lambda_targets(
        rewards,
        values,
        terminations,
        truncations,
        importance_weights,
        gamma=0.99,
        lmbda=0.95,
    )
    expected = legacy_lambda_targets_reference(
        rewards,
        values,
        terminations,
        truncations,
        gamma=0.99,
        lmbda=0.95,
    )

    assert_allclose(actual, expected)


def test_critic_loss_matches_legacy_reference_formula():
    critic_update_loss = jnp.asarray([0.2, 0.8, 0.4], dtype=jnp.float32)
    pred_emb = jnp.asarray([[0.1, 0.2], [0.3, -0.1], [1.0, -0.4]], dtype=jnp.float32)
    target_next_embs = jnp.asarray([[0.0, 0.4], [0.3, 0.2], [0.5, -0.6]], dtype=jnp.float32)
    pred_rew = jnp.asarray([[0.5], [-0.2], [0.8]], dtype=jnp.float32)
    rewards = jnp.asarray([0.1, -0.2, 1.2], dtype=jnp.float32)
    value = jnp.asarray([1.0, 2.0, -0.5], dtype=jnp.float32)
    target_values = jnp.asarray([0.8, 1.5, -0.25], dtype=jnp.float32)
    terminations = jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32)
    truncations = jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float32)

    actual_loss, actual_metrics = compute_reppo_dime_critic_loss(
        critic_update_loss,
        pred_emb,
        pred_rew,
        value,
        target_next_embs,
        rewards,
        target_values,
        terminations,
        truncations,
        aux_loss_mult=1.7,
    )

    aux_emb_loss = jnp.square(pred_emb - target_next_embs)
    aux_rew_loss = jnp.square(pred_rew - rewards[:, None])
    aux_loss = jnp.mean((1.0 - terminations[:, None]) * jnp.concatenate([aux_emb_loss, aux_rew_loss], axis=-1), axis=-1)
    expected_loss = jnp.mean((1.0 - truncations) * (critic_update_loss + 1.7 * aux_loss))
    expected_value_loss = jnp.mean(jnp.square(value - target_values))

    assert_allclose(actual_loss, expected_loss)
    assert_allclose(actual_metrics["value_loss"], expected_value_loss)
    assert_allclose(actual_metrics["critic_update_loss"], jnp.mean(critic_update_loss))
    assert_allclose(actual_metrics["aux_loss"], jnp.mean(aux_loss))
    assert_allclose(actual_metrics["reward_aux_loss"], jnp.mean(aux_rew_loss))


@pytest.mark.parametrize("mode", ["full", "clipped", "value"])
def test_actor_loss_matches_legacy_reference_formula(mode):
    policy_log_prob = jnp.asarray([0.2, -0.4, 0.1], dtype=jnp.float32)
    value = jnp.asarray([1.0, 0.5, -0.25], dtype=jnp.float32)
    entropy = jnp.asarray([-0.2, 0.0, 0.3], dtype=jnp.float32)
    kl = jnp.asarray([0.05, 0.2, 0.1], dtype=jnp.float32)
    temperature = jnp.asarray(0.7, dtype=jnp.float32)
    lagrangian = jnp.asarray(2.0, dtype=jnp.float32)
    action_size_target = jnp.asarray(6.0, dtype=jnp.float32)
    kl_bound = 0.1
    reduce_kl = True

    actual_loss, actual_metrics = compute_reppo_dime_actor_loss(
        policy_log_prob,
        value,
        entropy,
        kl,
        temperature,
        lagrangian,
        action_size_target,
        kl_bound,
        reduce_kl,
        mode,
        update_entropy_lagrangian=True,
        update_kl_lagrangian=True,
    )

    sac_loss = policy_log_prob * temperature - value
    if mode == "full":
        expected_actor_loss = sac_loss + kl * lagrangian * reduce_kl
    elif mode == "clipped":
        expected_actor_loss = jnp.where(kl < kl_bound, sac_loss, kl * lagrangian * reduce_kl)
    else:
        expected_actor_loss = sac_loss

    expected_entropy_loss = temperature * (action_size_target + entropy)
    expected_lagrangian_loss = -lagrangian * (kl - kl_bound)
    expected_loss = jnp.mean(expected_actor_loss) + jnp.mean(expected_entropy_loss) + jnp.mean(expected_lagrangian_loss)

    assert_allclose(actual_loss, expected_loss)
    assert_allclose(actual_metrics["actor_loss"], jnp.mean(expected_actor_loss))
    assert_allclose(actual_metrics["entropy_lagrangian_loss"], jnp.mean(expected_entropy_loss))
    assert_allclose(actual_metrics["kl_lagrangian_loss"], jnp.mean(expected_lagrangian_loss))


def test_diffusion_core_terms_match_legacy_jax_equations():
    policy = make_zero_score_policy()
    params = initialize_policy_params(policy)
    x = jnp.asarray([0.4, -1.2], dtype=jnp.float32)

    for step in (0, 1, 3):
        expected_dt = 0.125 * legacy_cosine_schedule(
            jnp.asarray(step, dtype=jnp.float32),
            diffusion_steps=4,
            schedule_min=0.001,
            schedule_s=0.008,
            schedule_power=2.0,
        )
        assert_allclose(policy.delta_t(jnp.asarray(step, dtype=jnp.float32), params), expected_dt)

    assert_allclose(policy.friction(params), jnp.ones((2,), dtype=jnp.float32))
    assert_allclose(policy.prior_score(x, params), -x / (2.5**2))

    expected_log_prob = -0.5 * jnp.sum(jnp.square(x / 2.5) + 2.0 * jnp.log(2.5) + LOG_2_PI)
    assert_allclose(policy.prior_log_prob(x, params), expected_log_prob)


def test_dime_kl_is_zero_for_identical_actor_params():
    policy = make_zero_score_policy()
    params = initialize_policy_params(policy)
    observations = jnp.asarray(
        [
            [0.0, 0.5, -0.2],
            [1.0, -0.5, 0.4],
            [-0.2, 0.1, 0.8],
        ],
        dtype=jnp.float32,
    )

    kl = policy.kl_divergence(
        params,
        params,
        observations,
        jax.random.PRNGKey(3),
        nr_action_samples=4,
        reverse_kl=False,
    )

    assert kl.shape == (3,)
    assert_allclose(kl, jnp.zeros((3,), dtype=jnp.float32), atol=1e-5)


def test_reverse_kl_stays_explicitly_unsupported_like_reference_path():
    policy = make_zero_score_policy()
    params = initialize_policy_params(policy)
    observations = jnp.zeros((1, 3), dtype=jnp.float32)

    with pytest.raises(NotImplementedError, match="Reverse KL"):
        policy.kl_divergence(
            params,
            params,
            observations,
            jax.random.PRNGKey(4),
            nr_action_samples=1,
            reverse_kl=True,
        )
