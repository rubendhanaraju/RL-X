from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rl_x.algorithms.reppo.flax_full_jit.base import (
    RePPOBase,
    compute_reppo_actor_loss,
    compute_reppo_critic_loss,
    compute_reppo_lambda_targets,
)
from rl_x.algorithms.reppo.flax_full_jit.default_config import get_config
from rl_x.algorithms.reppo.flax_full_jit.policy import Policy
from rl_x.algorithms.reppo.flax_full_jit.utils import (
    get_action_scale,
    hl_gauss,
    tanh_normal_log_prob_from_raw,
)


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


def legacy_lambda_targets_reference(rewards, values, terminations, truncations, importance_weights, gamma, lmbda):
    # Mirrors reppo_code/src/jaxrl/reppo.py::learn_step.
    lambda_return = values[-1]
    next_truncated = jnp.ones_like(truncations[0])
    next_importance_weight = jnp.zeros_like(importance_weights[0])
    targets = []
    for timestep in reversed(range(rewards.shape[0])):
        importance_lambda = jnp.exp(next_importance_weight) * lmbda
        lambda_sum = importance_lambda * lambda_return + (1.0 - importance_lambda) * values[timestep]
        delta = gamma * jnp.where(next_truncated, values[timestep], (1.0 - terminations[timestep]) * lambda_sum)
        lambda_return = rewards[timestep] + delta
        targets.insert(0, lambda_return)
        next_truncated = truncations[timestep]
        next_importance_weight = importance_weights[timestep]
    return jnp.stack(targets)


def legacy_update_stats(mean, var, count, observation):
    # Mirrors reppo_code/src/env_utils/jax_wrappers.py::NormalizeVec._compute_stats.
    batch_mean = jnp.mean(observation, axis=0)
    batch_var = jnp.var(observation, axis=0)
    batch_count = observation.shape[0]
    total_count = count + batch_count
    delta = batch_mean - mean
    new_mean = mean + delta * batch_count / total_count
    m_a = var * count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + jnp.square(delta) * count * batch_count / total_count
    return new_mean, m2 / total_count


def make_policy(action_dim=2, observation_dim=3, action_scale=None):
    if action_scale is None:
        action_scale = jnp.ones((action_dim,), dtype=jnp.float32)

    return Policy(
        action_dim=action_dim,
        action_scale=jnp.asarray(action_scale, dtype=jnp.float32),
        policy_observation_indices=jnp.arange(observation_dim),
        hidden_dim=8,
        layers=2,
        min_std=0.0,
        ent_start=0.01,
        kl_start=0.01,
        use_norm=True,
        use_skip=False,
    )


def initialize_policy_params(policy, batch_size=3, observation_dim=3):
    variables = policy.init(
        jax.random.PRNGKey(0),
        jnp.zeros((batch_size, observation_dim), dtype=jnp.float32),
    )
    return variables["params"]


def make_normalizer_model(policy_indices=(0, 1), critic_indices=(1, 2)):
    model = object.__new__(RePPOBase)
    model.enable_observation_normalization = True
    model.normalizer_epsilon = 1e-2
    model.policy_observation_indices = jnp.asarray(policy_indices)
    model.critic_observation_indices = jnp.asarray(critic_indices)
    model.os_shape = (3,)
    return model


def test_default_config_matches_legacy_reference_hyperparameters():
    cfg = get_config("reppo.flax_full_jit")

    assert cfg.nr_steps == 128
    assert cfg.nr_minibatches == 128
    assert cfg.nr_epochs == 4
    assert cfg.gamma == 0.99
    assert cfg.lmbda == 0.95
    assert cfg.lmbda_min == 0.5
    assert cfg.kl_bound == 0.1
    assert cfg.kl_action_rep == 16
    assert cfg.actor_kl_clip_mode == "clipped"
    assert cfg.ent_target_mult == 0.5
    assert cfg.actor_min_std == 0.0
    assert cfg.use_env_action_scale is False
    assert cfg.normalizer_epsilon == pytest.approx(1e-2)
    assert cfg.randomize_initial_episode_steps is True


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
    values = jnp.asarray([[-120.0], [-10.0], [0.0], [75.0], [120.0]], dtype=jnp.float32)
    actual = hl_gauss(values, nr_bins=151, v_min=-100.0, v_max=100.0)
    expected = legacy_hl_gauss_reference(values, nr_bins=151, v_min=-100.0, v_max=100.0)

    assert actual.shape == (5, 151)
    assert_allclose(jnp.sum(actual, axis=-1), jnp.ones((5,), dtype=jnp.float32), atol=2e-6)
    assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_online_observation_normalizer_matches_legacy_normalize_vec_formula():
    model = make_normalizer_model()
    reset_observation = jnp.asarray(
        [
            [1.0, 2.0, -1.0],
            [3.0, 4.0, 1.0],
            [5.0, 6.0, 3.0],
        ],
        dtype=jnp.float32,
    )
    next_observation = jnp.asarray(
        [
            [2.0, 1.0, 0.0],
            [4.0, 3.0, 2.0],
            [6.0, 5.0, 4.0],
        ],
        dtype=jnp.float32,
    )

    state = model.initialize_observation_normalizer(reset_observation)
    expected_policy_mean = jnp.mean(reset_observation[:, [0, 1]], axis=0)
    expected_policy_var = jnp.var(reset_observation[:, [0, 1]], axis=0)
    expected_critic_mean = jnp.mean(reset_observation[:, [1, 2]], axis=0)
    expected_critic_var = jnp.var(reset_observation[:, [1, 2]], axis=0)

    assert_allclose(state["policy_mean"], expected_policy_mean)
    assert_allclose(state["policy_var"], expected_policy_var)
    assert_allclose(state["critic_mean"], expected_critic_mean)
    assert_allclose(state["critic_var"], expected_critic_var)
    assert_allclose(state["count"], jnp.asarray(3.0, dtype=jnp.float32))

    state = model.update_observation_normalizer(state, next_observation)
    expected_policy_mean, expected_policy_var = legacy_update_stats(
        expected_policy_mean,
        expected_policy_var,
        jnp.asarray(3.0, dtype=jnp.float32),
        next_observation[:, [0, 1]],
    )
    expected_critic_mean, expected_critic_var = legacy_update_stats(
        expected_critic_mean,
        expected_critic_var,
        jnp.asarray(3.0, dtype=jnp.float32),
        next_observation[:, [1, 2]],
    )

    assert_allclose(state["policy_mean"], expected_policy_mean)
    assert_allclose(state["policy_var"], expected_policy_var)
    assert_allclose(state["critic_mean"], expected_critic_mean)
    assert_allclose(state["critic_var"], expected_critic_var)
    assert_allclose(state["count"], jnp.asarray(6.0, dtype=jnp.float32))

    normalized_policy = model.normalize_observation(next_observation, state, "policy")
    normalized_selected = (next_observation[:, [0, 1]] - expected_policy_mean) / jnp.sqrt(expected_policy_var + 1e-2)
    assert_allclose(normalized_policy[:, [0, 1]], normalized_selected)
    assert_allclose(normalized_policy[:, 2], next_observation[:, 2])


def test_lambda_targets_match_legacy_reference_with_importance_weights():
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
    importance_weights = jnp.log(
        jnp.asarray(
            [
                [1.0, 0.8, 0.5],
                [0.7, 1.0, 0.6],
                [0.9, 0.5, 1.0],
                [1.0, 0.7, 0.8],
            ],
            dtype=jnp.float32,
        )
    )

    actual = compute_reppo_lambda_targets(
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
        importance_weights,
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

    actual_loss, actual_metrics = compute_reppo_critic_loss(
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
    action_size_target = jnp.asarray(1.0, dtype=jnp.float32)
    kl_bound = 0.1
    reduce_kl = True

    actual_loss, actual_metrics = compute_reppo_actor_loss(
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


def test_policy_sample_action_matches_legacy_tanh_normal_formula():
    policy = make_policy()
    params = initialize_policy_params(policy)
    observations = jnp.asarray(
        [
            [0.0, 0.5, -0.2],
            [1.0, -0.5, 0.4],
            [-0.2, 0.1, 0.8],
        ],
        dtype=jnp.float32,
    )
    key = jax.random.PRNGKey(7)
    exploration_scale = jnp.asarray([[1.0], [0.75], [0.5]], dtype=jnp.float32)

    action, log_prob, entropy, sample_info = policy.sample_action(params, observations, key, exploration_scale)
    mean, std = policy.distribution(params, observations, exploration_scale)
    raw_action = mean + std * jax.random.normal(key, shape=mean.shape)
    expected_action = jnp.tanh(raw_action)
    expected_log_prob = tanh_normal_log_prob_from_raw(raw_action, mean, std, jnp.ones((2,), dtype=jnp.float32))

    assert_allclose(action, expected_action)
    assert_allclose(log_prob, expected_log_prob)
    assert_allclose(entropy, -expected_log_prob)
    assert_allclose(sample_info["raw_action"], raw_action)


def test_behavior_importance_weight_matches_legacy_density_ratio_formula():
    policy = make_policy()
    params = initialize_policy_params(policy)
    observations = jnp.asarray(
        [
            [0.0, 0.5, -0.2],
            [1.0, -0.5, 0.4],
            [-0.2, 0.1, 0.8],
        ],
        dtype=jnp.float32,
    )
    exploration_scale = jnp.asarray([[1.0], [0.75], [0.5]], dtype=jnp.float32)
    _, _, _, sample_info = policy.sample_action(params, observations, jax.random.PRNGKey(8), exploration_scale)

    actual = policy.behavior_importance_weight(
        params,
        observations,
        sample_info,
        exploration_scale,
        lmbda_min=0.5,
    )
    mean, base_std = policy.distribution(params, observations, 1.0)
    _, behavior_std = policy.distribution(params, observations, exploration_scale)
    raw_action = sample_info["raw_action"]
    raw_importance = tanh_normal_log_prob_from_raw(
        raw_action,
        mean,
        base_std,
        jnp.ones((2,), dtype=jnp.float32),
    ) - tanh_normal_log_prob_from_raw(
        raw_action,
        mean,
        behavior_std,
        jnp.ones((2,), dtype=jnp.float32),
    )
    expected = jnp.clip(jnp.nan_to_num(raw_importance, nan=jnp.log(0.5)), min=jnp.log(0.5), max=jnp.log(1.0))

    assert_allclose(actual, expected)


@pytest.mark.parametrize("reverse_kl", [False, True])
def test_policy_kl_matches_legacy_monte_carlo_formula(reverse_kl):
    policy = make_policy()
    params = initialize_policy_params(policy)
    target_params = jax.tree_util.tree_map(
        lambda x: x + jnp.asarray(0.03, dtype=x.dtype) if jnp.issubdtype(x.dtype, jnp.floating) and x.ndim > 0 else x,
        params,
    )
    observations = jnp.asarray(
        [
            [0.0, 0.5, -0.2],
            [1.0, -0.5, 0.4],
            [-0.2, 0.1, 0.8],
        ],
        dtype=jnp.float32,
    )
    key = jax.random.PRNGKey(9)
    nr_action_samples = 16

    actual = policy.kl_divergence(params, target_params, observations, key, nr_action_samples, reverse_kl)

    mean, std = policy.distribution(params, observations, 1.0)
    target_mean, target_std = policy.distribution(target_params, observations, 1.0)
    noise = jax.random.normal(key, shape=(nr_action_samples,) + mean.shape)
    if reverse_kl:
        raw_action = mean[None] + std[None] * noise
        current_log_prob = tanh_normal_log_prob_from_raw(raw_action, mean[None], std[None], jnp.ones((2,), dtype=jnp.float32))
        target_log_prob = tanh_normal_log_prob_from_raw(raw_action, target_mean[None], target_std[None], jnp.ones((2,), dtype=jnp.float32))
        expected = jnp.mean(current_log_prob - target_log_prob, axis=0)
    else:
        raw_action = target_mean[None] + target_std[None] * noise
        target_log_prob = tanh_normal_log_prob_from_raw(raw_action, target_mean[None], target_std[None], jnp.ones((2,), dtype=jnp.float32))
        current_log_prob = tanh_normal_log_prob_from_raw(raw_action, mean[None], std[None], jnp.ones((2,), dtype=jnp.float32))
        expected = jnp.mean(target_log_prob - current_log_prob, axis=0)

    assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
