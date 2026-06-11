# gmm_bms_global_vs_free_energy.py
#
# JAX/Flax implementation of:
#
#   1. Regular neural BMS explorer
#   2. Global GMM-BMS surrogate
#   3. Free-energy / local-target GMM-BMS comparison method
#
# Final deployable policies are explicit GMMs:
#
#   z ~ Cat(w), x ~ N(mu_z, diag(scale_z^2)).
#
# Outputs are saved to ./tmp:
#
#   tmp/target_density.png
#   tmp/regular_bms_explorer_samples.png
#   tmp/final_global_gmm_density.png
#   tmp/final_free_energy_gmm_density.png
#   tmp/comparison_target_bms_global_free_energy.png
#   tmp/global_gmm_params.npz
#   tmp/free_energy_gmm_params.npz
#
# Install:
#
#   pip install jax flax optax matplotlib
#
# Run:
#
#   python gmm_bms_global_vs_free_energy.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax import struct
from flax.training import train_state
import optax

# ============================================================
# Config
# ============================================================


@dataclass(frozen=True)
class Config:
    dim: int = 2

    # Target GMM/Boltzmann modes
    num_modes_target: int = 8
    target_radius: float = 5.0
    target_std: float = 0.55

    # Learned GMM components
    num_components_gmm: int = 12

    # Brownian reference
    T: float = 1.0
    sigma_ref: float = 2.0
    t_eps: float = 1e-3

    # Prior p0 = N(0, prior_std^2 I)
    prior_std: float = 3.0

    # Training
    batch_size: int = 2048
    sde_steps: int = 64
    num_train_steps: int = 20000
    print_every: int = 200

    # Optimizers
    explorer_lr: float = 2e-4
    global_gmm_lr: float = 5e-4
    free_energy_lr: float = 5e-4
    grad_clip: float = 10.0

    # Damping
    explorer_eta: float = 1.0
    global_gmm_eta: float = 10.0
    free_energy_eta: float = 10.0

    # Global GMM endpoint source:
    # X_T ~ lambda * q_phi + (1-lambda) * BMS explorer terminal distribution
    lambda_global_gmm_endpoint: float = 0.5

    # Regular, non-mode-biased initialization.
    # 0.4 starts all components near origin.
    # Try prior_std for broader unbiased initialization.
    init_mean_std: float = 0.4
    init_log_scale: float = 0.0
    min_scale: float = 0.08

    # Free-energy/local method
    fe_batch_per_component: int = 512
    responsibility_floor: float = 1e-2
    fe_grid_size: int = 140

    # Output
    output_dir: str = "tmp"

    # Plotting
    plot_xlim: Tuple[float, float] = (-8.0, 8.0)
    plot_ylim: Tuple[float, float] = (-8.0, 8.0)
    plot_grid_size: int = 250
    plot_num_samples: int = 12000
    plot_num_bms_samples: int = 12000


CFG = Config()

# ============================================================
# Fixed multimodal Boltzmann target
# ============================================================


def make_ring_centers(num_modes: int, radius: float, dim: int) -> jnp.ndarray:
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, num_modes, endpoint=False)
    xy = radius * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)

    if dim == 2:
        return xy

    pad = jnp.zeros((num_modes, dim - 2))
    return jnp.concatenate([xy, pad], axis=-1)


TARGET_CENTERS = make_ring_centers(
    CFG.num_modes_target,
    CFG.target_radius,
    CFG.dim,
)

TARGET_LOG_WEIGHTS = jnp.zeros((CFG.num_modes_target,)) - jnp.log(CFG.num_modes_target)


def log_rho_single(x: jnp.ndarray) -> jnp.ndarray:
    """
    Unnormalized log density.

    rho(x) = sum_m w_m exp(-||x-c_m||^2 / (2 target_std^2))

    Normalization constants are intentionally omitted.
    """
    diff = x[None, :] - TARGET_CENTERS
    quad = jnp.sum(diff * diff, axis=-1) / (CFG.target_std**2)
    comp_log = TARGET_LOG_WEIGHTS - 0.5 * quad
    return jax.nn.logsumexp(comp_log)


target_score_single = jax.grad(log_rho_single)


@jax.vmap
def log_rho(x: jnp.ndarray) -> jnp.ndarray:
    return log_rho_single(x)


@jax.vmap
def target_score(x: jnp.ndarray) -> jnp.ndarray:
    return target_score_single(x)


# ============================================================
# Brownian reference / bridge
# ============================================================


def kappa(t: jnp.ndarray) -> jnp.ndarray:
    return (CFG.sigma_ref**2) * t


def gamma(t: jnp.ndarray) -> jnp.ndarray:
    return t / CFG.T


def sample_prior(key: jax.Array, n: int) -> jnp.ndarray:
    return CFG.prior_std * random.normal(key, (n, CFG.dim))


def prior_score(x: jnp.ndarray) -> jnp.ndarray:
    return -x / (CFG.prior_std**2)


def sample_times(key: jax.Array, n: int) -> jnp.ndarray:
    lo = CFG.t_eps * CFG.T
    hi = (1.0 - CFG.t_eps) * CFG.T
    return random.uniform(key, (n,), minval=lo, maxval=hi)


def sample_bridge(
    key: jax.Array,
    x0: jnp.ndarray,
    xT: jnp.ndarray,
    t: jnp.ndarray,
) -> jnp.ndarray:
    """
    Brownian bridge marginal:

        X_t | X_0, X_T ~ N(
            (1-gamma) X_0 + gamma X_T,
            kappa(t)(1-gamma) I
        )
    """
    g = gamma(t)[:, None]
    kap = kappa(t)[:, None]

    mean = (1.0 - g) * x0 + g * xT
    std = jnp.sqrt(jnp.maximum(kap * (1.0 - g), 1e-12))

    eps = random.normal(key, x0.shape)
    return mean + std * eps


def bms_target_drift(
    x0: jnp.ndarray,
    xt: jnp.ndarray,
    xT: jnp.ndarray,
    t: jnp.ndarray,
) -> jnp.ndarray:
    """
    Independent-coupling BMS target drift with c(t)=gamma(t):

        xi*(X,t)
        =
        sigma [
            score_p0(X0)
            + score_target(XT)
            + (Xt-X0)/kappa(t)
        ]

    score_target = ∇ log rho, since p* = rho/Z.
    """
    kap = kappa(t)[:, None]

    return CFG.sigma_ref * (prior_score(x0) + target_score(xT) + (xt - x0) / jnp.maximum(kap, 1e-12))


# ============================================================
# Regular neural BMS explorer
# ============================================================


class DriftMLP(nn.Module):
    hidden: Tuple[int, ...] = (128, 128, 128)

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        if t.ndim == 1:
            t = t[:, None]

        h = jnp.concatenate([x, t], axis=-1)

        for width in self.hidden:
            h = nn.swish(nn.Dense(width)(h))

        return nn.Dense(
            x.shape[-1],
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=nn.initializers.zeros,
        )(h)


def create_explorer_state(key: jax.Array) -> train_state.TrainState:
    model = DriftMLP()
    variables = model.init(key, jnp.zeros((1, CFG.dim)), jnp.zeros((1,)))

    tx = optax.chain(
        optax.clip_by_global_norm(CFG.grad_clip),
        optax.adam(CFG.explorer_lr),
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )


def simulate_explorer_terminal(
    params: Any,
    apply_fn: Any,
    key: jax.Array,
    n: int,
) -> jnp.ndarray:
    """
    Euler-Maruyama simulation:

        dX_t = sigma u_theta(X_t,t) dt + sigma dB_t
    """
    key_x0, key_scan = random.split(key)
    x = sample_prior(key_x0, n)

    dt = CFG.T / CFG.sde_steps
    sqrt_dt = jnp.sqrt(dt)

    keys = random.split(key_scan, CFG.sde_steps)

    def step(x_curr, inp):
        i, k = inp

        t_scalar = (i.astype(jnp.float32) / CFG.sde_steps) * CFG.T
        t = jnp.full((n,), t_scalar)

        u = apply_fn({"params": params}, x_curr, t)
        noise = random.normal(k, x_curr.shape)

        x_next = x_curr + CFG.sigma_ref * u * dt + CFG.sigma_ref * sqrt_dt * noise

        return x_next, None

    idx = jnp.arange(CFG.sde_steps)
    xT, _ = jax.lax.scan(step, x, (idx, keys))

    return xT


# ============================================================
# Explicit diagonal GMM endpoint law
# ============================================================


@struct.dataclass
class GMMState:
    params: Any
    opt_state: Any


def make_initial_gmm_params(key: jax.Array) -> dict:
    """
    Regular, non-mode-biased initialization.
    """
    k1, _ = random.split(key)

    means = CFG.init_mean_std * random.normal(
        k1,
        (CFG.num_components_gmm, CFG.dim),
    )

    logits = jnp.zeros((CFG.num_components_gmm,))

    log_scales = jnp.ones((CFG.num_components_gmm, CFG.dim)) * CFG.init_log_scale

    return {
        "logits": logits,
        "means": means,
        "log_scales": log_scales,
    }


def gmm_scales(params: dict) -> jnp.ndarray:
    return CFG.min_scale + jax.nn.softplus(params["log_scales"])


def diag_gaussian_log_prob(
    x: jnp.ndarray,
    mean: jnp.ndarray,
    scale: jnp.ndarray,
) -> jnp.ndarray:
    """
    Broadcasted log N(x; mean, diag(scale^2)).
    """
    d = x.shape[-1]
    z = (x - mean) / scale

    return -0.5 * (jnp.sum(z * z, axis=-1) + 2.0 * jnp.sum(jnp.log(scale), axis=-1) + d * jnp.log(2.0 * jnp.pi))


def component_log_probs(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """
    log q_z(x) for all components.

    x: [N, D]
    output: [N, K]
    """
    means = params["means"]
    scales = gmm_scales(params)

    return diag_gaussian_log_prob(
        x[:, None, :],
        means[None, :, :],
        scales[None, :, :],
    )


def component_scores(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """
    ∇_x log q_z(x) for all components.

    x: [N, D]
    output: [N, K, D]
    """
    means = params["means"]
    scales = gmm_scales(params)
    vars_ = scales**2

    return -(x[:, None, :] - means[None, :, :]) / vars_[None, :, :]


def gmm_log_prob(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    logits = params["logits"]
    log_w = jax.nn.log_softmax(logits)

    log_qz = component_log_probs(params, x)

    return jax.nn.logsumexp(log_w[None, :] + log_qz, axis=-1)


def sample_gmm(params: dict, key: jax.Array, n: int) -> jnp.ndarray:
    k1, k2 = random.split(key)

    comp = random.categorical(k1, params["logits"], shape=(n,))

    means = params["means"][comp]
    scales = gmm_scales(params)[comp]

    eps = random.normal(k2, (n, CFG.dim))

    return means + scales * eps


def posterior_responsibilities(
    params: dict,
    x: jnp.ndarray,
    floor: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Endpoint posterior responsibilities:

        r_z(x) = w_z q_z(x) / q_phi(x)

    Floored version:

        rtilde_z(x) = (1-floor) r_z(x) + floor/K
    """
    log_w = jax.nn.log_softmax(params["logits"])
    log_qz = component_log_probs(params, x)

    raw_logits = log_w[None, :] + log_qz
    r = jax.nn.softmax(raw_logits, axis=-1)

    K = r.shape[-1]
    r_tilde = (1.0 - floor) * r + floor / K

    return r, r_tilde


def create_global_gmm_state(key: jax.Array) -> GMMState:
    params = make_initial_gmm_params(key)

    tx = optax.chain(
        optax.clip_by_global_norm(CFG.grad_clip),
        optax.adam(CFG.global_gmm_lr),
    )

    return GMMState(params=params, opt_state=tx.init(params))


def create_free_energy_gmm_state(key: jax.Array) -> GMMState:
    params = make_initial_gmm_params(key)

    tx = optax.chain(
        optax.clip_by_global_norm(CFG.grad_clip),
        optax.adam(CFG.free_energy_lr),
    )

    return GMMState(params=params, opt_state=tx.init(params))


GLOBAL_GMM_TX = optax.chain(
    optax.clip_by_global_norm(CFG.grad_clip),
    optax.adam(CFG.global_gmm_lr),
)

FREE_ENERGY_TX = optax.chain(
    optax.clip_by_global_norm(CFG.grad_clip),
    optax.adam(CFG.free_energy_lr),
)

# ============================================================
# Global GMM-induced Markovian bridge drift
# ============================================================


def global_gmm_induced_drift(
    params: dict,
    x: jnp.ndarray,
    t: jnp.ndarray,
) -> jnp.ndarray:
    """
    Closed-form Markovian bridge drift induced by endpoint law q_phi.

        u_phi(x,t)
        =
        sigma / (kappa(T)-kappa(t))
        [
            E_phi[X_T | X_t=x] - x
        ]

    q_phi is a full GMM.
    """
    logits = params["logits"]
    log_w = jax.nn.log_softmax(logits)

    means = params["means"]
    scales = gmm_scales(params)
    variances = scales**2

    g = gamma(t)[:, None, None]
    kap = kappa(t)[:, None, None]

    m0 = jnp.zeros((CFG.dim,))
    S0_diag = jnp.ones((CFG.dim,)) * (CFG.prior_std**2)

    mzt = (1.0 - g) * m0[None, None, :] + g * means[None, :, :]

    C_diag = ((1.0 - g)**2 * S0_diag[None, None, :] + g**2 * variances[None, :, :] + kap * (1.0 - g))

    C_scale = jnp.sqrt(jnp.maximum(C_diag, 1e-12))

    log_comp_t = diag_gaussian_log_prob(
        x[:, None, :],
        mzt,
        C_scale,
    )

    beta = jax.nn.softmax(log_w[None, :] + log_comp_t, axis=-1)

    cond_mean_z = means[None, :, :] + (g * variances[None, :, :] / jnp.maximum(C_diag, 1e-12)) * (x[:, None, :] - mzt)

    xT_hat = jnp.sum(beta[:, :, None] * cond_mean_z, axis=1)

    denom = (CFG.sigma_ref**2) * (CFG.T - t)

    return CFG.sigma_ref * (xT_hat - x) / jnp.maximum(denom[:, None], 1e-12)


# ============================================================
# Local single-Gaussian induced drift for free-energy method
# ============================================================


def sample_all_old_expert_endpoints(
    params: dict,
    key: jax.Array,
    batch_per_component: int,
) -> jnp.ndarray:
    """
    Samples X_T,z ~ N(mu_z, Sigma_z) for every expert z.

    output: [K, B, D]
    """
    means = params["means"]
    scales = gmm_scales(params)

    eps = random.normal(
        key,
        (CFG.num_components_gmm, batch_per_component, CFG.dim),
    )

    return means[:, None, :] + scales[:, None, :] * eps


def sample_prior_for_all_experts(
    key: jax.Array,
    batch_per_component: int,
) -> jnp.ndarray:
    """
    output: [K, B, D]
    """
    return CFG.prior_std * random.normal(
        key,
        (CFG.num_components_gmm, batch_per_component, CFG.dim),
    )


def sample_times_for_all_experts(
    key: jax.Array,
    batch_per_component: int,
) -> jnp.ndarray:
    """
    output: [K, B]
    """
    lo = CFG.t_eps * CFG.T
    hi = (1.0 - CFG.t_eps) * CFG.T

    return random.uniform(
        key,
        (CFG.num_components_gmm, batch_per_component),
        minval=lo,
        maxval=hi,
    )


def sample_bridge_all_experts(
    key: jax.Array,
    x0: jnp.ndarray,
    xT: jnp.ndarray,
    t: jnp.ndarray,
) -> jnp.ndarray:
    """
    x0: [K, B, D]
    xT: [K, B, D]
    t:  [K, B]
    """
    g = gamma(t)[..., None]
    kap = kappa(t)[..., None]

    mean = (1.0 - g) * x0 + g * xT
    std = jnp.sqrt(jnp.maximum(kap * (1.0 - g), 1e-12))

    eps = random.normal(key, x0.shape)

    return mean + std * eps


def single_gaussian_induced_drift_all_experts(
    params: dict,
    x: jnp.ndarray,
    t: jnp.ndarray,
) -> jnp.ndarray:
    """
    Per-expert single-Gaussian induced bridge drift:

        u_phi_z(x,t)
        =
        sigma / (kappa(T)-kappa(t))
        [
            mu_z + gamma Sigma_z C_z,t^{-1}(x-m_z,t) - x
        ]

    x: [K, B, D]
    t: [K, B]

    output: [K, B, D]
    """
    means = params["means"]
    scales = gmm_scales(params)
    vars_ = scales**2

    g = gamma(t)[..., None]
    kap = kappa(t)[..., None]

    S0_diag = jnp.ones((CFG.dim,)) * (CFG.prior_std**2)

    mu = means[:, None, :]
    var = vars_[:, None, :]

    # m0 = 0
    m_t = g * mu

    C_diag = ((1.0 - g)**2 * S0_diag[None, None, :] + g**2 * var + kap * (1.0 - g))

    xT_hat = mu + (g * var / jnp.maximum(C_diag, 1e-12)) * (x - m_t)

    denom = (CFG.sigma_ref**2) * (CFG.T - t)

    return CFG.sigma_ref * (xT_hat - x) / jnp.maximum(denom[..., None], 1e-12)


# ============================================================
# Free-energy local target scores
# ============================================================


def local_target_scores_with_old_responsibilities(
    old_params: dict,
    x_z: jnp.ndarray,
) -> jnp.ndarray:
    """
    Computes

        s_z^*(x) = ∇ log rho(x) + ∇ log rtilde_z(x)

    where rtilde_z is the floored old posterior responsibility.

    x_z: [K, B, D]
    output: [K, B, D]
    """
    K, B, D = x_z.shape

    x_flat = x_z.reshape((K * B, D))

    r, r_tilde = posterior_responsibilities(
        old_params,
        x_flat,
        floor=CFG.responsibility_floor,
    )

    sq_z_all = component_scores(old_params, x_flat)
    sq_mix = jnp.sum(r[:, :, None] * sq_z_all, axis=1)

    z_ids = jnp.repeat(jnp.arange(K), B)

    sq_selected = sq_z_all[jnp.arange(K * B), z_ids, :]

    r_selected = r[jnp.arange(K * B), z_ids]
    rtilde_selected = r_tilde[jnp.arange(K * B), z_ids]

    grad_log_r = sq_selected - sq_mix

    floor = CFG.responsibility_floor

    correction = ((1.0 - floor) * r_selected) / jnp.maximum(
        rtilde_selected,
        1e-12,
    )

    grad_log_rtilde = correction[:, None] * grad_log_r

    s_rho = target_score(x_flat)

    s_local = s_rho + grad_log_rtilde

    return s_local.reshape((K, B, D))


def local_bms_target_drift_all_experts(
    old_params: dict,
    x0: jnp.ndarray,
    xt: jnp.ndarray,
    xT: jnp.ndarray,
    t: jnp.ndarray,
) -> jnp.ndarray:
    """
    Local target drift:

        xi_z^*(X,t)
        =
        sigma [
            score_p0(X0)
            + s_z^*(XT)
            + (Xt-X0)/kappa(t)
        ]
    """
    kap = kappa(t)[..., None]

    s_local = local_target_scores_with_old_responsibilities(
        old_params,
        xT,
    )

    return CFG.sigma_ref * (prior_score(x0) + s_local + (xt - x0) / jnp.maximum(kap, 1e-12))


# ============================================================
# Free-energy gate update
# ============================================================


def make_fe_grid() -> tuple[jnp.ndarray, float]:
    xs = jnp.linspace(CFG.plot_xlim[0], CFG.plot_xlim[1], CFG.fe_grid_size)
    ys = jnp.linspace(CFG.plot_ylim[0], CFG.plot_ylim[1], CFG.fe_grid_size)

    xx, yy = jnp.meshgrid(xs, ys)

    grid = jnp.stack(
        [xx.reshape(-1), yy.reshape(-1)],
        axis=-1,
    )

    dx = (CFG.plot_xlim[1] - CFG.plot_xlim[0]) / (CFG.fe_grid_size - 1)
    dy = (CFG.plot_ylim[1] - CFG.plot_ylim[0]) / (CFG.fe_grid_size - 1)

    cell_area = dx * dy

    return grid, cell_area


def estimate_free_energy_logits_grid(old_params: dict) -> jnp.ndarray:
    """
    Grid estimator:

        Z_z = ∫ rho(x) rtilde_z(x) dx

    Returns logits proportional to log Z_z.
    """
    grid, cell_area = make_fe_grid()

    _, r_tilde = posterior_responsibilities(
        old_params,
        grid,
        floor=CFG.responsibility_floor,
    )

    log_terms = log_rho(grid)[:, None] + jnp.log(jnp.maximum(r_tilde, 1e-30))

    log_Z = jax.nn.logsumexp(log_terms, axis=0) + jnp.log(cell_area)

    return log_Z


# ============================================================
# Jitted train steps
# ============================================================


@jax.jit
def train_explorer_and_global_gmm_step(
    key: jax.Array,
    explorer_state: train_state.TrainState,
    global_gmm_state: GMMState,
):
    n = CFG.batch_size

    (
        key,
        k_exp_terminal,
        k_exp_x0,
        k_exp_t,
        k_exp_bridge,
        k_gmm_endpoint,
        k_mix,
        k_gmm_x0,
        k_gmm_t,
        k_gmm_bridge,
    ) = random.split(key, 10)

    # ------------------------------------------------------------
    # 1. Regular neural BMS explorer update
    # ------------------------------------------------------------

    xT_exp = simulate_explorer_terminal(
        explorer_state.params,
        explorer_state.apply_fn,
        k_exp_terminal,
        n,
    )
    xT_exp = jax.lax.stop_gradient(xT_exp)

    x0_exp = sample_prior(k_exp_x0, n)
    t_exp = sample_times(k_exp_t, n)
    xt_exp = sample_bridge(k_exp_bridge, x0_exp, xT_exp, t_exp)

    xi_exp = bms_target_drift(x0_exp, xt_exp, xT_exp, t_exp)

    old_explorer_params = explorer_state.params

    def explorer_loss_fn(params):
        u = explorer_state.apply_fn({"params": params}, xt_exp, t_exp)
        u_old = explorer_state.apply_fn(
            {"params": old_explorer_params},
            xt_exp,
            t_exp,
        )

        fit = 0.5 * jnp.mean(jnp.sum((u - xi_exp)**2, axis=-1))

        damp = 0.5 * CFG.explorer_eta * jnp.mean(jnp.sum(
            (u - jax.lax.stop_gradient(u_old))**2,
            axis=-1,
        ))

        return fit + damp, {
            "explorer_fit": fit,
            "explorer_damp": damp,
        }

    (explorer_loss, explorer_aux), explorer_grads = jax.value_and_grad(
        explorer_loss_fn,
        has_aux=True,
    )(explorer_state.params)

    explorer_state = explorer_state.apply_gradients(grads=explorer_grads)

    # ------------------------------------------------------------
    # 2. Global GMM-BMS surrogate update
    # ------------------------------------------------------------

    xT_gmm = sample_gmm(global_gmm_state.params, k_gmm_endpoint, n)

    choose_gmm = random.bernoulli(
        k_mix,
        CFG.lambda_global_gmm_endpoint,
        shape=(n, 1),
    )

    xT_mix = jnp.where(choose_gmm, xT_gmm, xT_exp)
    xT_mix = jax.lax.stop_gradient(xT_mix)

    x0_gmm = sample_prior(k_gmm_x0, n)
    t_gmm = sample_times(k_gmm_t, n)
    xt_gmm = sample_bridge(k_gmm_bridge, x0_gmm, xT_mix, t_gmm)

    xi_gmm = bms_target_drift(x0_gmm, xt_gmm, xT_mix, t_gmm)

    old_gmm_params = global_gmm_state.params

    def global_gmm_loss_fn(params):
        u = global_gmm_induced_drift(params, xt_gmm, t_gmm)
        u_old = global_gmm_induced_drift(old_gmm_params, xt_gmm, t_gmm)

        fit = 0.5 * jnp.mean(jnp.sum((u - xi_gmm)**2, axis=-1))

        damp = 0.5 * CFG.global_gmm_eta * jnp.mean(jnp.sum(
            (u - jax.lax.stop_gradient(u_old))**2,
            axis=-1,
        ))

        return fit + damp, {
            "global_fit": fit,
            "global_damp": damp,
        }

    (global_loss, global_aux), global_grads = jax.value_and_grad(
        global_gmm_loss_fn,
        has_aux=True,
    )(global_gmm_state.params)

    updates, new_opt_state = GLOBAL_GMM_TX.update(
        global_grads,
        global_gmm_state.opt_state,
        global_gmm_state.params,
    )

    new_global_params = optax.apply_updates(
        global_gmm_state.params,
        updates,
    )

    global_gmm_state = GMMState(
        params=new_global_params,
        opt_state=new_opt_state,
    )

    metrics = {
        "explorer_loss": explorer_loss,
        "explorer_fit": explorer_aux["explorer_fit"],
        "explorer_damp": explorer_aux["explorer_damp"],
        "global_gmm_loss": global_loss,
        "global_fit": global_aux["global_fit"],
        "global_damp": global_aux["global_damp"],
        "mean_log_rho_global_gmm": jnp.mean(log_rho(xT_gmm)),
        "mean_log_rho_explorer": jnp.mean(log_rho(xT_exp)),
    }

    return key, explorer_state, global_gmm_state, metrics


@jax.jit
def free_energy_train_step(
    key: jax.Array,
    fe_state: GMMState,
):
    """
    Free-energy / local-target GMM-BMS update.

    1. Use old posterior responsibilities.
    2. Define local scores s_z^* = ∇log rho + ∇log r_z.
    3. Train each expert with local BMS drift matching.
    4. Set gate logits to log Z_z estimated on a grid.
    """
    B = CFG.fe_batch_per_component

    key, k_xT, k_x0, k_t, k_bridge, k_metric = random.split(key, 6)

    old_params = fe_state.params

    # Old local expert bridges
    xT = sample_all_old_expert_endpoints(old_params, k_xT, B)
    x0 = sample_prior_for_all_experts(k_x0, B)
    t = sample_times_for_all_experts(k_t, B)
    xt = sample_bridge_all_experts(k_bridge, x0, xT, t)

    xi_local = local_bms_target_drift_all_experts(
        old_params,
        x0,
        xt,
        xT,
        t,
    )

    def fe_loss_fn(params):
        u = single_gaussian_induced_drift_all_experts(params, xt, t)
        u_old = single_gaussian_induced_drift_all_experts(old_params, xt, t)

        fit = 0.5 * jnp.mean(jnp.sum((u - xi_local)**2, axis=-1))

        damp = 0.5 * CFG.free_energy_eta * jnp.mean(jnp.sum(
            (u - jax.lax.stop_gradient(u_old))**2,
            axis=-1,
        ))

        return fit + damp, {
            "fe_fit": fit,
            "fe_damp": damp,
        }

    (fe_loss, fe_aux), fe_grads = jax.value_and_grad(
        fe_loss_fn,
        has_aux=True,
    )(fe_state.params)

    updates, new_opt_state = FREE_ENERGY_TX.update(
        fe_grads,
        fe_state.opt_state,
        fe_state.params,
    )

    new_params = optax.apply_updates(fe_state.params, updates)

    # Free-energy gate update:
    # w_z ∝ Z_z = ∫ rho(x) r_z(x) dx.
    # Uses old responsibilities for this iteration.
    fe_logits = estimate_free_energy_logits_grid(old_params)

    new_params = {
        "logits": fe_logits,
        "means": new_params["means"],
        "log_scales": new_params["log_scales"],
    }

    fe_state = GMMState(
        params=new_params,
        opt_state=new_opt_state,
    )

    metric_samples = sample_gmm(new_params, k_metric, CFG.batch_size)

    metrics = {
        "fe_loss": fe_loss,
        "fe_fit": fe_aux["fe_fit"],
        "fe_damp": fe_aux["fe_damp"],
        "mean_log_rho_fe_gmm": jnp.mean(log_rho(metric_samples)),
    }

    return key, fe_state, metrics


# ============================================================
# Plotting
# ============================================================


def evaluate_density_on_grid(params: dict | None = None):
    xs = jnp.linspace(CFG.plot_xlim[0], CFG.plot_xlim[1], CFG.plot_grid_size)
    ys = jnp.linspace(CFG.plot_ylim[0], CFG.plot_ylim[1], CFG.plot_grid_size)

    xx, yy = jnp.meshgrid(xs, ys)

    grid = jnp.stack(
        [xx.reshape(-1), yy.reshape(-1)],
        axis=-1,
    )

    if params is None:
        vals = jnp.exp(log_rho(grid))
    else:
        vals = jnp.exp(gmm_log_prob(params, grid))

    vals = vals.reshape(CFG.plot_grid_size, CFG.plot_grid_size)

    return xx, yy, vals


def save_all_plots(
    global_params: dict,
    fe_params: dict,
    explorer_params: Any,
    apply_fn: Any,
    key: jax.Array,
    out_dir: Path,
):
    if CFG.dim != 2:
        print("Skipping plots because CFG.dim != 2.")
        return

    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)

    key_global, key_fe, key_bms = random.split(key, 3)

    xx, yy, target_vals = evaluate_density_on_grid(None)
    _, _, global_vals = evaluate_density_on_grid(global_params)
    _, _, fe_vals = evaluate_density_on_grid(fe_params)

    xx = jax.device_get(xx)
    yy = jax.device_get(yy)

    target_vals = jax.device_get(target_vals)
    global_vals = jax.device_get(global_vals)
    fe_vals = jax.device_get(fe_vals)

    target_norm = target_vals / (target_vals.sum() + 1e-12)
    global_norm = global_vals / (global_vals.sum() + 1e-12)
    fe_norm = fe_vals / (fe_vals.sum() + 1e-12)

    centers = jax.device_get(TARGET_CENTERS)

    global_samples = jax.device_get(sample_gmm(global_params, key_global, CFG.plot_num_samples))

    fe_samples = jax.device_get(sample_gmm(fe_params, key_fe, CFG.plot_num_samples))

    bms_samples = jax.device_get(
        simulate_explorer_terminal(
            explorer_params,
            apply_fn,
            key_bms,
            CFG.plot_num_bms_samples,
        ))

    global_means = jax.device_get(global_params["means"])
    fe_means = jax.device_get(fe_params["means"])

    global_weights = jax.device_get(jax.nn.softmax(global_params["logits"]))
    fe_weights = jax.device_get(jax.nn.softmax(fe_params["logits"]))

    global_scales = jax.device_get(gmm_scales(global_params))
    fe_scales = jax.device_get(gmm_scales(fe_params))

    # ------------------------------------------------------------
    # Target only
    # ------------------------------------------------------------

    plt.figure(figsize=(6.5, 5.8))
    im = plt.contourf(xx, yy, target_norm, levels=80)
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="x",
        s=90,
        linewidths=2,
        label="target modes",
    )
    plt.title(f"Boltzmann target, modes={CFG.num_modes_target}")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.axis("equal")
    plt.xlim(*CFG.plot_xlim)
    plt.ylim(*CFG.plot_ylim)
    plt.colorbar(im, label="normalized grid density")
    plt.legend(loc="upper right")
    plt.tight_layout()

    target_path = out_dir / "target_density.png"
    plt.savefig(target_path, dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # Regular BMS samples
    # ------------------------------------------------------------

    plt.figure(figsize=(6.5, 5.8))
    im = plt.contourf(xx, yy, target_norm, levels=80)
    plt.scatter(
        bms_samples[:, 0],
        bms_samples[:, 1],
        s=1,
        alpha=0.18,
        label="regular BMS samples",
    )
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="x",
        s=90,
        linewidths=2,
        label="target modes",
    )
    plt.title("Regular neural BMS explorer samples")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.axis("equal")
    plt.xlim(*CFG.plot_xlim)
    plt.ylim(*CFG.plot_ylim)
    plt.colorbar(im, label="target density")
    plt.legend(loc="upper right")
    plt.tight_layout()

    bms_path = out_dir / "regular_bms_explorer_samples.png"
    plt.savefig(bms_path, dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # Global GMM
    # ------------------------------------------------------------

    plt.figure(figsize=(6.5, 5.8))
    im = plt.contourf(xx, yy, global_norm, levels=80)
    plt.scatter(
        global_samples[:, 0],
        global_samples[:, 1],
        s=1,
        alpha=0.12,
        label="global GMM samples",
    )
    plt.scatter(
        global_means[:, 0],
        global_means[:, 1],
        marker="o",
        s=40,
        edgecolors="white",
        linewidths=0.8,
        label="global GMM means",
    )
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="x",
        s=90,
        linewidths=2,
        label="target modes",
    )
    plt.title("Final global GMM-BMS")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.axis("equal")
    plt.xlim(*CFG.plot_xlim)
    plt.ylim(*CFG.plot_ylim)
    plt.colorbar(im, label="normalized grid density")
    plt.legend(loc="upper right")
    plt.tight_layout()

    global_path = out_dir / "final_global_gmm_density.png"
    plt.savefig(global_path, dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # Free-energy GMM
    # ------------------------------------------------------------

    plt.figure(figsize=(6.5, 5.8))
    im = plt.contourf(xx, yy, fe_norm, levels=80)
    plt.scatter(
        fe_samples[:, 0],
        fe_samples[:, 1],
        s=1,
        alpha=0.12,
        label="free-energy GMM samples",
    )
    plt.scatter(
        fe_means[:, 0],
        fe_means[:, 1],
        marker="o",
        s=40,
        edgecolors="white",
        linewidths=0.8,
        label="free-energy GMM means",
    )
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="x",
        s=90,
        linewidths=2,
        label="target modes",
    )
    plt.title("Final free-energy local GMM-BMS")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.axis("equal")
    plt.xlim(*CFG.plot_xlim)
    plt.ylim(*CFG.plot_ylim)
    plt.colorbar(im, label="normalized grid density")
    plt.legend(loc="upper right")
    plt.tight_layout()

    fe_path = out_dir / "final_free_energy_gmm_density.png"
    plt.savefig(fe_path, dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # Full comparison
    # ------------------------------------------------------------

    fig, axes = plt.subplots(1, 5, figsize=(26, 4.8))

    im0 = axes[0].contourf(xx, yy, target_norm, levels=80)
    axes[0].scatter(centers[:, 0], centers[:, 1], marker="x", s=80, linewidths=2)
    axes[0].set_title("Target")
    axes[0].axis("equal")
    axes[0].set_xlim(*CFG.plot_xlim)
    axes[0].set_ylim(*CFG.plot_ylim)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].contourf(xx, yy, target_norm, levels=80)
    axes[1].scatter(bms_samples[:, 0], bms_samples[:, 1], s=1, alpha=0.16)
    axes[1].scatter(centers[:, 0], centers[:, 1], marker="x", s=80, linewidths=2)
    axes[1].set_title("Regular BMS")
    axes[1].axis("equal")
    axes[1].set_xlim(*CFG.plot_xlim)
    axes[1].set_ylim(*CFG.plot_ylim)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].contourf(xx, yy, global_norm, levels=80)
    axes[2].scatter(global_samples[:, 0], global_samples[:, 1], s=1, alpha=0.12)
    axes[2].scatter(
        global_means[:, 0],
        global_means[:, 1],
        marker="o",
        s=40,
        edgecolors="white",
        linewidths=0.8,
    )
    axes[2].scatter(centers[:, 0], centers[:, 1], marker="x", s=80, linewidths=2)
    axes[2].set_title("Global GMM-BMS")
    axes[2].axis("equal")
    axes[2].set_xlim(*CFG.plot_xlim)
    axes[2].set_ylim(*CFG.plot_ylim)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].contourf(xx, yy, fe_norm, levels=80)
    axes[3].scatter(fe_samples[:, 0], fe_samples[:, 1], s=1, alpha=0.12)
    axes[3].scatter(
        fe_means[:, 0],
        fe_means[:, 1],
        marker="o",
        s=40,
        edgecolors="white",
        linewidths=0.8,
    )
    axes[3].scatter(centers[:, 0], centers[:, 1], marker="x", s=80, linewidths=2)
    axes[3].set_title("Free-energy local GMM-BMS")
    axes[3].axis("equal")
    axes[3].set_xlim(*CFG.plot_xlim)
    axes[3].set_ylim(*CFG.plot_ylim)
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    diff = fe_norm - target_norm
    vmax = float(np.max(np.abs(diff)))

    im4 = axes[4].imshow(
        diff,
        origin="lower",
        extent=(
            CFG.plot_xlim[0],
            CFG.plot_xlim[1],
            CFG.plot_ylim[0],
            CFG.plot_ylim[1],
        ),
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
    )
    axes[4].set_title("Free-energy GMM - target")
    axes[4].set_xlim(*CFG.plot_xlim)
    axes[4].set_ylim(*CFG.plot_ylim)
    fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")

    fig.tight_layout()

    comparison_path = out_dir / "comparison_target_bms_global_free_energy.png"
    fig.savefig(comparison_path, dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------
    # Save params
    # ------------------------------------------------------------

    np.savez(
        out_dir / "global_gmm_params.npz",
        weights=global_weights,
        means=global_means,
        scales=global_scales,
        target_centers=centers,
    )

    np.savez(
        out_dir / "free_energy_gmm_params.npz",
        weights=fe_weights,
        means=fe_means,
        scales=fe_scales,
        target_centers=centers,
    )

    print(f"Saved {target_path}")
    print(f"Saved {bms_path}")
    print(f"Saved {global_path}")
    print(f"Saved {fe_path}")
    print(f"Saved {comparison_path}")
    print(f"Saved {out_dir / 'global_gmm_params.npz'}")
    print(f"Saved {out_dir / 'free_energy_gmm_params.npz'}")


# ============================================================
# Main
# ============================================================


def main():
    key = random.PRNGKey(0)

    key, k_explorer, k_global, k_fe = random.split(key, 4)

    explorer_state = create_explorer_state(k_explorer)
    global_gmm_state = create_global_gmm_state(k_global)
    fe_gmm_state = create_free_energy_gmm_state(k_fe)

    print("Target centers are fixed, but initialization is not biased toward them.")
    print("Initial GLOBAL GMM means:")
    print(jax.device_get(global_gmm_state.params["means"]))

    print("Initial FREE-ENERGY GMM means:")
    print(jax.device_get(fe_gmm_state.params["means"]))

    for step in range(1, CFG.num_train_steps + 1):
        key, explorer_state, global_gmm_state, global_metrics = (train_explorer_and_global_gmm_step(
            key,
            explorer_state,
            global_gmm_state,
        ))

        key, fe_gmm_state, fe_metrics = free_energy_train_step(
            key,
            fe_gmm_state,
        )

        if step % CFG.print_every == 0 or step == 1:
            gm = jax.device_get(global_metrics)
            fm = jax.device_get(fe_metrics)

            print(f"step {step:05d} | "
                  f"explorer_loss={gm['explorer_loss']:.4f} "
                  f"global_loss={gm['global_gmm_loss']:.4f} "
                  f"fe_loss={fm['fe_loss']:.4f} "
                  f"logrho_bms={gm['mean_log_rho_explorer']:.3f} "
                  f"logrho_global={gm['mean_log_rho_global_gmm']:.3f} "
                  f"logrho_fe={fm['mean_log_rho_fe_gmm']:.3f}")

    print("\nFinal GLOBAL GMM weights:")
    print(jax.device_get(jax.nn.softmax(global_gmm_state.params["logits"])))

    print("\nFinal GLOBAL GMM means:")
    print(jax.device_get(global_gmm_state.params["means"]))

    print("\nFinal GLOBAL GMM scales:")
    print(jax.device_get(gmm_scales(global_gmm_state.params)))

    print("\nFinal FREE-ENERGY GMM weights:")
    print(jax.device_get(jax.nn.softmax(fe_gmm_state.params["logits"])))

    print("\nFinal FREE-ENERGY GMM means:")
    print(jax.device_get(fe_gmm_state.params["means"]))

    print("\nFinal FREE-ENERGY GMM scales:")
    print(jax.device_get(gmm_scales(fe_gmm_state.params)))

    out_dir = Path(CFG.output_dir)

    key, k_plot = random.split(key)

    save_all_plots(
        global_gmm_state.params,
        fe_gmm_state.params,
        explorer_state.params,
        explorer_state.apply_fn,
        k_plot,
        out_dir,
    )


if __name__ == "__main__":
    main()
