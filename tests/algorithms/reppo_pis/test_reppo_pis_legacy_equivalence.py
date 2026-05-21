import sys
import types
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from rl_x.algorithms.reppo_pis.flax_full_jit.default_config import get_config
from rl_x.algorithms.reppo_pis.flax_full_jit.policy import PISPolicy, LOG_2_PI
from rl_x.algorithms.reppo_pis.flax_full_jit.reppo_pis import (
    RePPO_PIS,
    compute_entropy_via_importance_sampling,
    compute_reppo_pis_adjoint_actor_loss,
    compute_reppo_pis_critic_loss,
    compute_reppo_pis_lambda_targets,
    find_optimum_kl_lagrangian,
)


def assert_allclose(actual, expected, *, rtol=1e-6, atol=1e-6):
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol)


def legacy_geometric_scheduler_terms(sigma_max, sigma_min, t):
    ratio = sigma_max / sigma_min
    sigma_t = sigma_min * ratio ** (1.0 - t) * jnp.sqrt(2.0 * jnp.log(ratio))
    sigma_t_0 = jnp.sqrt(sigma_max**2 * (1.0 - ratio ** (-2.0 * t)))
    sigma_t_0_at_1 = jnp.sqrt(sigma_max**2 * (1.0 - ratio ** (-2.0)))
    sigma_ratio = sigma_t_0 / sigma_t_0_at_1
    sigma_t_0T = sigma_t_0 * jnp.sqrt(1.0 - sigma_ratio**2)
    mu_t_0T = sigma_t_0**2 / sigma_t_0_at_1**2
    return sigma_t, sigma_t_0_at_1, sigma_t_0T, mu_t_0T


def legacy_lambda_targets_reference(rewards, values, terminations, truncations, importance_weights, gamma, lmbda):
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


def legacy_parallel_nary_search_reference(f, low, high, n_points=64, rtol=1e-4, atol=1e-6, max_iter=50):
    f_batched = jax.vmap(f)

    def cond_fun(state):
        low, high, iteration = state
        return (iteration < max_iter) & ((high / low) > (1.0 + rtol)) & (jnp.abs(high - low) > atol)

    def body_fun(state):
        low, high, iteration = state
        grid = jnp.geomspace(low, high, n_points)
        values = f_batched(grid)
        idx = jnp.argmin(values)
        ratio = high / low

        is_at_lower = idx == 0
        high_lower = grid[1]
        low_lower = high_lower / ratio

        is_at_upper = idx == n_points - 1
        low_upper = grid[n_points - 2]
        high_upper = low_upper * ratio

        low_bracketing = grid[jnp.maximum(0, idx - 1)]
        high_bracketing = grid[jnp.minimum(n_points - 1, idx + 1)]

        next_low = jnp.where(is_at_lower, low_lower, jnp.where(is_at_upper, low_upper, low_bracketing))
        next_high = jnp.where(is_at_lower, high_lower, jnp.where(is_at_upper, high_upper, high_bracketing))
        return next_low, next_high, iteration + 1

    low, high, _ = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (jnp.float32(low), jnp.float32(high), 0),
    )
    return (low + high) / 2


def legacy_find_optimum_kl_lagrangian_reference(
    w_t,
    old_ctrl,
    ctrl_target,
    eps,
    min_value=1e-3,
    max_value=1e3,
    norm_weights=True,
):
    if norm_weights:
        w_t_factor = w_t.mean()
    else:
        w_t_factor = 1.0
    w_t_norm = w_t / w_t_factor
    sse = 0.5 * jnp.sum(jnp.square(old_ctrl - ctrl_target), axis=-1)
    w2 = w_t_norm**2

    def eval_single_lambda(lam):
        w_lam_2 = (w_t_norm + lam) ** 2
        dual_grad = jnp.mean(w2 / w_lam_2 * sse) - eps
        return dual_grad**2

    min_norm_lambda = legacy_parallel_nary_search_reference(eval_single_lambda, low=min_value, high=max_value)
    return min_norm_lambda * w_t_factor


def make_zero_control_policy(action_dim=2, observation_dim=3):
    return PISPolicy(
        action_dim=action_dim,
        action_scale=jnp.ones((action_dim,), dtype=jnp.float32),
        policy_observation_indices=jnp.arange(observation_dim),
        diffusion_steps=4,
        noise_schedule_sigma_max=2.0,
        noise_schedule_sigma_min=0.1,
        ent_start=1.0,
        kl_start=10.0,
        score_model_nr_layers=2,
        score_model_nr_hidden_units=8,
        score_model_time_mode="mlp",
        score_model_time_mlp_input="both",
        score_model_nr_time_fourier=4,
        score_model_time_fourier_range_min=0.1,
        score_model_time_fourier_range_max=100.0,
        score_model_nr_time_hidden_units=4,
        score_model_time_coder_out=3,
        score_model_action_mode="linear",
        score_model_action_mlp_input="both",
        score_model_nr_action_fourier=4,
        score_model_action_fourier_range_min=0.1,
        score_model_action_fourier_range_max=100.0,
        score_model_nr_action_hidden_units=4,
        score_model_action_coder_out=3,
        score_model_outer_clip=1e4,
        score_model_inner_clip=1e2,
        score_model_weight_init=0.0,
        score_model_bias_init=0.0,
        score_model_layer_norm=False,
        score_model_layer_norm_type="LayerNorm",
    )


def initialize_policy_params(policy, observation_dim=3):
    variables = policy.init(
        jax.random.PRNGKey(0),
        jnp.zeros((policy.action_dim,), dtype=jnp.float32),
        jnp.zeros((observation_dim,), dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    return variables["params"]


def load_local_bms_pis(monkeypatch):
    reference_root = Path(__file__).resolve().parents[3] / "reppo-bms_rl"
    if not reference_root.exists():
        pytest.skip("local reppo-bms_rl reference checkout is not available")

    distrax = types.ModuleType("distrax")

    class Bijector:
        def __init__(self, event_ndims_in=0):
            self.event_ndims_in = event_ndims_in

    class MultivariateNormalDiag:
        def __init__(self, loc, scale_diag):
            self.loc = loc
            self.scale_diag = scale_diag

        def log_prob(self, x):
            return -0.5 * jnp.sum(
                jnp.square((x - self.loc) / self.scale_diag)
                + 2.0 * jnp.log(self.scale_diag)
                + LOG_2_PI,
                axis=-1,
            )

    distrax.Bijector = Bijector
    distrax.MultivariateNormalDiag = MultivariateNormalDiag
    monkeypatch.setitem(sys.modules, "distrax", distrax)

    fake_common_utils = types.ModuleType("src.networks.reppo_dime.common.utils")
    fake_common_utils.log_prob_kernel = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "src.networks.reppo_dime.common.utils", fake_common_utils)
    monkeypatch.syspath_prepend(str(reference_root))

    from flax import nnx
    from src.networks.reppo_dime.common.jax_scheduler_pis import get_geometric_scheduler
    from src.networks.reppo_dime.jax_dime_integrators_pis import (
        logratio,
        ode_integrator,
        sde_integrator,
        sde_integrator_with_kl,
    )
    from src.networks.reppo_dime.jax_dime_models_pis import DIMEActor, PIS

    return types.SimpleNamespace(
        nnx=nnx,
        get_geometric_scheduler=get_geometric_scheduler,
        logratio=logratio,
        ode_integrator=ode_integrator,
        sde_integrator=sde_integrator,
        sde_integrator_with_kl=sde_integrator_with_kl,
        DIMEActor=DIMEActor,
        PIS=PIS,
    )


def make_local_bms_actor(bms, fwd_model):
    legacy_diffusion = bms.PIS(
        action_dim=2,
        observation_dim=3,
        fwd_model=fwd_model,
        diff_steps=4,
        scheduler=bms.get_geometric_scheduler(2.0, 0.1),
        rngs=bms.nnx.Rngs(0),
    )
    return bms.DIMEActor(
        action_dim=2,
        observation_dim=3,
        diffusion_model=legacy_diffusion,
        sde_integrator=bms.sde_integrator,
        sde_integrator_with_kl=bms.sde_integrator_with_kl,
        ode_integrator=bms.ode_integrator,
        logratio=bms.logratio,
        kl_start=10.0,
        ent_start=1.0,
        kl_bound=0.1,
    )


def test_default_config_matches_bms_reppo_pis_yaml_defaults():
    cfg = get_config("reppo_pis.flax_full_jit")

    assert cfg.total_timesteps == 80_000_000
    assert cfg.nr_steps == 128
    assert cfg.nr_minibatches == 128
    assert cfg.nr_epochs == 4
    assert cfg.nr_actor_epochs == 4
    assert cfg.nr_critic_epochs == 4
    assert cfg.batch_repetitions == 1
    assert cfg.gamma == pytest.approx(0.99)
    assert cfg.lmbda == pytest.approx(0.95)
    assert cfg.polyak == pytest.approx(1.0)
    assert cfg.v_min == pytest.approx(0.0)
    assert cfg.v_max == pytest.approx(150.0)
    assert cfg.nr_bins == 151
    assert cfg.kl_start == pytest.approx(10.0)
    assert cfg.kl_action_rep == 1
    assert cfg.actor_kl_clip_mode == "full"
    assert cfg.ent_start == pytest.approx(1.0)
    assert cfg.ent_target_mult == pytest.approx(3.5)
    assert cfg.diffusion_steps == 16
    assert cfg.noise_schedule_sigma_max == pytest.approx(2.0)
    assert cfg.noise_schedule_sigma_min == pytest.approx(0.01)
    assert cfg.loss_scaling_sigma_power == -1
    assert cfg.scale_loss_with_temperature is True
    assert cfg.onpol_entropy is True


def test_actor_optimizer_matches_bms_unclipped_actor_update():
    actor_tx = RePPO_PIS.create_actor_optimizer(None, 3e-4)
    critic_tx = RePPO_PIS.create_critic_optimizer(type("Dummy", (), {"max_grad_norm": 0.5})(), 3e-4)

    params = {"x": jnp.zeros((2,), dtype=jnp.float32)}

    assert actor_tx.init(params).__class__.__name__ == "InjectStatefulHyperparamsState"
    assert len(critic_tx.init(params)) == 3


def test_geometric_scheduler_matches_bms_reference_formula():
    policy = make_zero_control_policy()
    params = initialize_policy_params(policy)
    del params

    timesteps = jnp.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=jnp.float32)
    sigma_t, sigma_t_0_at_1, sigma_t_0T, mu_t_0T = legacy_geometric_scheduler_terms(2.0, 0.1, timesteps)

    assert_allclose(policy.sigma_t(timesteps), sigma_t)
    assert_allclose(policy.sigma_T_0(), sigma_t_0_at_1)
    assert_allclose(policy.sigma_t_0T(timesteps), sigma_t_0T)
    assert_allclose(policy.mu_t_0T_scale(timesteps), mu_t_0T)


def test_erf_squash_log_det_and_reference_density_match_bms_formula():
    policy = make_zero_control_policy()
    x = jnp.asarray([0.4, -1.2], dtype=jnp.float32)
    k = jnp.sqrt(policy.sigma_T_0() ** -2 / 2.0)

    expected_action = jax.scipy.special.erf(k * x)
    expected_log_det = jnp.log(2.0 * k) - 0.5 * jnp.log(jnp.pi) - jnp.square(k * x)
    expected_grad = -2.0 * jnp.square(k) * x
    expected_ref_log_prob = -0.5 * jnp.sum(jnp.square(x / policy.sigma_T_0()) + 2.0 * jnp.log(policy.sigma_T_0()) + LOG_2_PI)

    assert_allclose(policy.erf_forward(x), expected_action)
    assert_allclose(policy.erf_forward_log_det_jacobian(x), expected_log_det)
    assert_allclose(policy.erf_log_det_grad(x), expected_grad)
    assert_allclose(policy.ref_log_prob(x), expected_ref_log_prob)


def test_lambda_targets_match_bms_reference_with_importance_weights():
    rewards = jnp.asarray([[0.4, -0.2], [1.2, 0.1], [-0.5, 0.3]], dtype=jnp.float32)
    values = jnp.asarray([[0.2, 0.5], [1.1, -0.3], [-0.7, 0.8]], dtype=jnp.float32)
    terminations = jnp.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    truncations = jnp.asarray([[False, False], [False, True], [True, False]])
    importance_weights = jnp.log(jnp.asarray([[1.0, 0.8], [0.7, 1.0], [1.0, 0.7]], dtype=jnp.float32))

    actual = compute_reppo_pis_lambda_targets(rewards, values, terminations, truncations, importance_weights, 0.99, 0.95)
    expected = legacy_lambda_targets_reference(rewards, values, terminations, truncations, importance_weights, 0.99, 0.95)

    assert_allclose(actual, expected)


def test_critic_loss_matches_bms_embedding_only_auxiliary_loss():
    critic_update_loss = jnp.asarray([0.2, 0.8, 0.4], dtype=jnp.float32)
    pred_emb = jnp.asarray([[0.1, 0.2], [0.3, -0.1], [1.0, -0.4]], dtype=jnp.float32)
    target_next_embs = jnp.asarray([[0.0, 0.4], [0.3, 0.2], [0.5, -0.6]], dtype=jnp.float32)
    value = jnp.asarray([1.0, 2.0, -0.5], dtype=jnp.float32)
    target_values = jnp.asarray([0.8, 1.5, -0.25], dtype=jnp.float32)
    terminations = jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32)
    truncations = jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float32)

    actual_loss, actual_metrics = compute_reppo_pis_critic_loss(
        critic_update_loss,
        pred_emb,
        value,
        target_next_embs,
        target_values,
        terminations,
        truncations,
        aux_loss_mult=1.7,
    )

    aux_loss = jnp.mean((1.0 - terminations[:, None]) * jnp.square(pred_emb - target_next_embs), axis=-1)
    expected_loss = jnp.mean((1.0 - truncations) * (critic_update_loss + 1.7 * aux_loss))

    assert_allclose(actual_loss, expected_loss)
    assert_allclose(actual_metrics["value_loss"], jnp.mean(jnp.square(value - target_values)))
    assert_allclose(actual_metrics["critic_update_loss"], jnp.mean(critic_update_loss))
    assert_allclose(actual_metrics["aux_loss"], jnp.mean(aux_loss))


def test_adjoint_actor_loss_matches_bms_dual_descent_formula():
    adjoint_loss = jnp.asarray([0.2, 0.8, 0.4], dtype=jnp.float32)
    kl_loss = jnp.asarray(0.12, dtype=jnp.float32)
    entropy = jnp.asarray(-1.5, dtype=jnp.float32)
    temperature = jnp.asarray(0.7, dtype=jnp.float32)
    lagrangian = jnp.asarray(2.0, dtype=jnp.float32)
    action_size_target = jnp.asarray(3.5, dtype=jnp.float32)
    kl_bound = 0.1

    actual_loss, actual_metrics = compute_reppo_pis_adjoint_actor_loss(
        adjoint_loss,
        kl_loss,
        entropy,
        temperature,
        lagrangian,
        action_size_target,
        kl_bound,
        reduce_kl=True,
        update_entropy_lagrangian=True,
        update_kl_lagrangian=True,
    )

    expected_actor_loss = adjoint_loss + lagrangian * kl_loss
    expected_entropy_loss = temperature * (action_size_target + entropy)
    expected_lagrangian_loss = -lagrangian * (kl_loss - kl_bound)
    expected_loss = jnp.mean(expected_actor_loss) + expected_entropy_loss + expected_lagrangian_loss

    assert_allclose(actual_loss, expected_loss)
    assert_allclose(actual_metrics["actor_loss"], jnp.mean(expected_actor_loss))
    assert_allclose(actual_metrics["entropy_lagrangian_loss"], expected_entropy_loss)
    assert_allclose(actual_metrics["kl_lagrangian_loss"], expected_lagrangian_loss)


def test_entropy_importance_sampling_matches_bms_formula():
    log_weights = jnp.asarray([-0.2, 0.1, 0.4], dtype=jnp.float32)
    q_value = jnp.asarray([1.0, -0.5, 0.2], dtype=jnp.float32)
    cov_weight = jnp.asarray([0.3, 0.1, -0.2], dtype=jnp.float32)
    temperature = jnp.asarray(0.7, dtype=jnp.float32)

    actual = compute_entropy_via_importance_sampling(log_weights, q_value, temperature, cov_weight)

    log_q_tilde = q_value / temperature
    log_importance_weights = log_q_tilde + log_weights
    log_z = jax.nn.logsumexp(log_importance_weights) - jnp.log(log_weights.shape[-1])
    norm_weights = jax.nn.softmax(log_importance_weights)
    del cov_weight
    expected = -jnp.sum(norm_weights * log_q_tilde) + log_z

    assert_allclose(actual, expected)


def test_zero_control_sde_sample_matches_bms_path_weight_equations():
    policy = make_zero_control_policy()
    params = initialize_policy_params(policy)
    observation = jnp.asarray([[0.1, -0.2, 0.3]], dtype=jnp.float32)
    key = jax.random.PRNGKey(123)

    (
        action,
        raw_action,
        prior_action,
        tanh_correction_grad,
        log_weight,
        log_path_weight_deterministic,
        log_path_weight_stochastic,
        log_p_T_ref,
        cov_weight,
        tanh_correction_val,
    ) = policy.sde_sample(params, key, observation, stop_grad=True)

    sample_key = jax.random.split(key, observation.shape[0])[0]
    _, _, scan_key = jax.random.split(sample_key, 3)
    expected_raw = jnp.zeros((policy.action_dim,), dtype=jnp.float32)
    dt = 1.0 / policy.diffusion_steps
    for step in range(policy.diffusion_steps):
        noise_key, scan_key = jax.random.split(scan_key)
        noise = jax.random.normal(noise_key, expected_raw.shape)
        expected_raw = expected_raw + policy.sigma_t(jnp.asarray(step * dt, dtype=jnp.float32)) * noise * jnp.sqrt(dt)

    expected_log_p_T_ref = policy.ref_log_prob(expected_raw)
    expected_cov_weight = policy.erf_log_det_sum(expected_raw)
    expected_log_weight = -expected_log_p_T_ref + expected_cov_weight

    assert_allclose(prior_action[0], jnp.zeros((policy.action_dim,), dtype=jnp.float32))
    assert_allclose(raw_action[0], expected_raw)
    assert_allclose(action[0], policy.erf_forward(expected_raw))
    assert_allclose(tanh_correction_grad[0], policy.erf_log_det_grad(expected_raw))
    assert_allclose(log_path_weight_deterministic[0, 0], 0.0)
    assert_allclose(log_path_weight_stochastic[0, 0], 0.0)
    assert_allclose(log_p_T_ref[0, 0], expected_log_p_T_ref)
    assert_allclose(cov_weight[0], expected_cov_weight)
    assert_allclose(tanh_correction_val[0], expected_cov_weight)
    assert_allclose(log_weight[0, 0], expected_log_weight)


def test_zero_control_sde_sample_matches_local_bms_implementation(monkeypatch):
    bms = load_local_bms_pis(monkeypatch)

    class ZeroModel(bms.nnx.Module):
        def __call__(self, x, obs, step):
            del obs, step
            return jnp.zeros_like(x)

    legacy_actor = make_local_bms_actor(bms, ZeroModel())

    policy = make_zero_control_policy()
    params = initialize_policy_params(policy)
    observation = jnp.asarray([[0.1, -0.2, 0.3]], dtype=jnp.float32)
    key = jax.random.PRNGKey(123)

    actual = policy.sde_sample(params, key, observation, stop_grad=True)
    expected = legacy_actor.sde_sample(key, observation, stop_grad=True)

    assert len(actual) == len(expected)
    for actual_part, expected_part in zip(actual, expected):
        assert_allclose(actual_part, expected_part, atol=2e-6)


def test_nonzero_control_sde_path_weights_match_local_bms_implementation(monkeypatch):
    bms = load_local_bms_pis(monkeypatch)

    class LinearModel(bms.nnx.Module):
        def __call__(self, x, obs, step):
            step = jnp.atleast_1d(step)[0]
            return 0.15 * x + 0.07 * obs[: x.shape[0]] - 0.03 * step

    def linear_forward_control(self, params, raw_action, observation, timestep):
        del params
        timestep = jnp.atleast_1d(timestep)[0]
        return 0.15 * raw_action + 0.07 * observation[: self.action_dim] - 0.03 * timestep

    monkeypatch.setattr(PISPolicy, "forward_control", linear_forward_control)

    legacy_actor = make_local_bms_actor(bms, LinearModel())
    policy = make_zero_control_policy()
    params = initialize_policy_params(policy)
    observation = jnp.asarray([[0.1, -0.2, 0.3]], dtype=jnp.float32)
    key = jax.random.PRNGKey(321)

    actual = policy.sde_sample(params, key, observation, stop_grad=True)
    expected = legacy_actor.sde_sample(key, observation, stop_grad=True)

    for actual_part, expected_part in zip(actual, expected):
        assert_allclose(actual_part, expected_part, atol=2e-6)


def test_policy_kl_is_zero_for_identical_params():
    policy = make_zero_control_policy()
    params = initialize_policy_params(policy)
    observations = jnp.asarray([[0.1, -0.2, 0.3], [0.0, 0.4, -0.1]], dtype=jnp.float32)

    kl = policy.kl_divergence(params, params, observations, jax.random.PRNGKey(7), nr_action_samples=3, reverse_kl=False)

    assert kl.shape == (2,)
    assert_allclose(kl, jnp.zeros((2,), dtype=jnp.float32), atol=1e-5)


def test_optimum_geometric_kl_lagrangian_matches_reference_search_objective():
    w_t = jnp.asarray([0.7, 1.2, 0.9], dtype=jnp.float32)
    old_ctrl = jnp.asarray([[0.1, -0.2], [0.3, 0.2], [-0.4, 0.5]], dtype=jnp.float32)
    ctrl_target = jnp.asarray([[0.0, -0.1], [0.4, 0.0], [-0.2, 0.1]], dtype=jnp.float32)
    kl_bound = 0.05

    actual = find_optimum_kl_lagrangian(w_t, old_ctrl, ctrl_target, kl_bound)
    expected = legacy_find_optimum_kl_lagrangian_reference(w_t, old_ctrl, ctrl_target, kl_bound)

    assert_allclose(actual, expected)
