#!/usr/bin/env python3
"""
Neural-parameterized GMM-constrained BMS-style sampler in JAX/Flax.

This is the clean neural-GMM version requested:
  - The terminal distribution is still exactly a Gaussian mixture model.
  - The GMM terminal parameters are NOT direct arrays.
  - A neural network maps learned component codes z_k to
        logits a_k, means mu_k, and diagonal log-variances ell_k.
  - The diffusion/control is still derived from the linear GMM marginal path.
  - No residual terminal parameters and no arbitrary x-dependent gates.

Output:
  ./tmp/target_vs_mixture_samples.png  # target, initial neural-GMM, trained neural-GMM
  ./tmp/learned_gmm_params.npz

Install:
  pip install "jax[cpu]" flax optax matplotlib

For GPU, install the appropriate JAX CUDA wheel from the JAX docs.
"""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax


# -----------------------------
# Fixed wide-range 2D multimodal Boltzmann target
# p(x) ∝ exp(-U(x)).
# This script keeps the toy target/plot 2D, but the neural GMM/control code
# uses TARGET_DIM and is not hard-coded to 2D except for plotting.
# -----------------------------

TARGET_CENTERS = jnp.array(
    [
        [-14.0, -12.0],
        [-13.5,   7.5],
        [ -6.5,  14.0],
        [  0.0,   0.0],
        [  7.0,  12.5],
        [ 14.0,  -8.0],
        [  4.5, -15.0],
        [ 15.0,   4.5],
    ],
    dtype=jnp.float32,
)
TARGET_DIM = int(TARGET_CENTERS.shape[-1])

_target_weights = jnp.array(
    [0.10, 0.13, 0.09, 0.18, 0.12, 0.16, 0.08, 0.14],
    dtype=jnp.float32,
)
TARGET_LOGITS = jnp.log(_target_weights)

TARGET_VARS = jnp.array(
    [
        [1.20, 1.80],
        [1.75, 1.10],
        [1.35, 1.55],
        [2.30, 1.90],
        [1.40, 1.70],
        [1.15, 1.35],
        [1.60, 1.10],
        [1.50, 1.85],
    ],
    dtype=jnp.float32,
)


def target_log_rho(x: jnp.ndarray) -> jnp.ndarray:
    """
    Unnormalized log-density log rho(x) = -U(x).
    x: (..., d)
    returns: (...)
    """
    diff = x[..., None, :] - TARGET_CENTERS
    quad = jnp.sum(diff**2 / TARGET_VARS, axis=-1)
    log_det = jnp.sum(jnp.log(TARGET_VARS), axis=-1)
    log_comp = TARGET_LOGITS - 0.5 * (quad + log_det)
    return jax.nn.logsumexp(log_comp, axis=-1)


def target_score(x: jnp.ndarray) -> jnp.ndarray:
    """
    score = grad_x log rho(x).
    x: (batch, d)
    returns: (batch, d)
    """
    diff = x[:, None, :] - TARGET_CENTERS[None, :, :]
    quad = jnp.sum(diff**2 / TARGET_VARS[None, :, :], axis=-1)
    log_det = jnp.sum(jnp.log(TARGET_VARS), axis=-1)

    log_comp = TARGET_LOGITS[None, :] - 0.5 * (quad + log_det[None, :])
    resp = jax.nn.softmax(log_comp, axis=-1)

    comp_scores = -diff / TARGET_VARS[None, :, :]
    return jnp.sum(resp[:, :, None] * comp_scores, axis=1)


# -----------------------------
# Neural terminal GMM generator
# -----------------------------


def positive_terminal_vars(raw_logvars: jnp.ndarray, var_floor: float) -> jnp.ndarray:
    return var_floor + jnp.exp(jnp.clip(raw_logvars, -9.0, 6.0))


class NeuralGMMPath(nn.Module):
    """
    Neural parameter generator for a terminal GMM.

    Learned codes z_k are mapped by a shared MLP to terminal GMM parameters:
        z_k -> (a_k, mu_k,T, raw_logvar_k,T)

    The endpoint distribution is exactly:
        q_T(x) = sum_k softmax(a)_k N(x; mu_k,T, diag(v_k,T)).

    The path remains the simple linear GMM path:
        mu_k(t) = t mu_k,T
        v_k(t) = (1-t) prior_var + t v_k,T
    """

    k: int
    dim: int
    var_floor: float = 1.0e-3
    prior_var: float = 1.0

    # Neural generator hyperparameters.
    code_dim: int = 16
    hidden: int = 128
    depth: int = 3
    code_init_std: float = 1.0

    # Output scalings for initialization.
    init_mean_scale: float = 12.0
    init_terminal_std: float = 2.5
    mean_head_init_std: float = 0.7
    logvar_output_scale: float = 0.25
    logit_output_scale: float = 0.1

    def _mlp(self, z: jnp.ndarray) -> jnp.ndarray:
        h = z
        for i in range(self.depth):
            h = nn.Dense(
                self.hidden,
                kernel_init=nn.initializers.lecun_normal(),
                bias_init=nn.initializers.zeros,
                name=f"gen_dense_{i}",
            )(h)
            h = nn.tanh(h)
        return h

    @nn.compact
    def terminal_params(self):
        # Learned component codes. These are not terminal means/variances;
        # the GMM parameters are generated through the shared network below.
        codes = self.param(
            "component_codes",
            nn.initializers.normal(self.code_init_std),
            (self.k, self.code_dim),
        )
        h = self._mlp(codes)

        logits_raw = nn.Dense(
            1,
            kernel_init=nn.initializers.normal(0.02),
            bias_init=nn.initializers.zeros,
            name="logit_head",
        )(h).squeeze(-1)
        logits = self.logit_output_scale * logits_raw

        # We scale the generated means so initialization is broad without using
        # target mode locations or plot/domain limits.
        means_raw = nn.Dense(
            self.dim,
            kernel_init=nn.initializers.normal(self.mean_head_init_std),
            bias_init=nn.initializers.zeros,
            name="mean_head",
        )(h)
        means_T = self.init_mean_scale * means_raw

        # Generate logvars around init_terminal_std, with a small learned spread.
        init_var_minus_floor = jnp.maximum(
            jnp.asarray(self.init_terminal_std**2 - self.var_floor, dtype=h.dtype),
            jnp.asarray(1.0e-6, dtype=h.dtype),
        )
        base_raw_logvar = jnp.log(init_var_minus_floor)
        raw_logvar_delta = nn.Dense(
            self.dim,
            kernel_init=nn.initializers.normal(0.02),
            bias_init=nn.initializers.zeros,
            name="logvar_head",
        )(h)
        raw_logvars_T = base_raw_logvar + self.logvar_output_scale * raw_logvar_delta
        vars_T = positive_terminal_vars(raw_logvars_T, self.var_floor)
        return logits, means_T, vars_T, raw_logvars_T

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, sigma: float) -> jnp.ndarray:
        logits, means_T, vars_T, _raw = self.terminal_params()
        return gmm_path_control_from_terminal(
            logits=logits,
            means_T=means_T,
            vars_T=vars_T,
            x=x,
            t=t,
            sigma=sigma,
            prior_var=self.prior_var,
        )


# -----------------------------
# GMM path and control
# -----------------------------


def gmm_path_control_from_terminal(
    logits: jnp.ndarray,
    means_T: jnp.ndarray,
    vars_T: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray,
    sigma: float,
    prior_var: float,
) -> jnp.ndarray:
    """
    Returns BMS control u_phi(x,t), where actual SDE drift is sigma * u_phi.

    This is the same GMM-preserving linear path as the pure GMM version, except
    terminal parameters come from a neural generator.

    x: (batch, dim)
    t: (batch,)
    """
    t = jnp.clip(t, 1.0e-4, 1.0 - 1.0e-4)
    t3 = t[:, None, None]

    vars_0 = prior_var * jnp.ones_like(vars_T)

    mu_t = t3 * means_T[None, :, :]
    dmu_dt = means_T[None, :, :]

    var_t = (1.0 - t3) * vars_0[None, :, :] + t3 * vars_T[None, :, :]
    dvar_dt = vars_T[None, :, :] - vars_0[None, :, :]

    # Component Gaussian-preserving affine drift.
    A_diag = 0.5 * (dvar_dt - sigma**2) / var_t

    x_centered = x[:, None, :] - mu_t
    component_drift = dmu_dt + A_diag * x_centered

    log_weights = jax.nn.log_softmax(logits)
    log_comp = (
        log_weights[None, :]
        - 0.5
        * jnp.sum(
            jnp.log(2.0 * jnp.pi * var_t) + x_centered**2 / var_t,
            axis=-1,
        )
    )
    resp = jax.nn.softmax(log_comp, axis=-1)

    actual_drift = jnp.sum(resp[:, :, None] * component_drift, axis=1)
    control = actual_drift / sigma
    return control


# -----------------------------
# Sampling / density utilities
# -----------------------------


def get_terminal_params(apply_fn, params):
    logits, means_T, vars_T, raw_logvars_T = apply_fn(
        {"params": params}, method=NeuralGMMPath.terminal_params
    )
    return logits, means_T, vars_T, raw_logvars_T


def sample_terminal_gmm(apply_fn, params, key, n: int) -> jnp.ndarray:
    logits, means_T, vars_T, _raw = get_terminal_params(apply_fn, params)

    key_comp, key_noise = jax.random.split(key)
    comp = jax.random.categorical(key_comp, logits, shape=(n,))
    eps = jax.random.normal(key_noise, (n, means_T.shape[-1]))

    return means_T[comp] + jnp.sqrt(vars_T[comp]) * eps


def terminal_gmm_log_prob(apply_fn, params, x: jnp.ndarray) -> jnp.ndarray:
    """
    log q_{params,T}(x) for the terminal diagonal GMM.
    x: (batch, dim)
    returns: (batch,)
    """
    logits, means_T, vars_T, _raw = get_terminal_params(apply_fn, params)
    diff = x[:, None, :] - means_T[None, :, :]
    log_weights = jax.nn.log_softmax(logits)
    log_comp = (
        log_weights[None, :]
        - 0.5
        * jnp.sum(
            jnp.log(2.0 * jnp.pi * vars_T)[None, :, :]
            + diff**2 / vars_T[None, :, :],
            axis=-1,
        )
    )
    return jax.nn.logsumexp(log_comp, axis=-1)


def sample_brownian_bridge(
    x0: jnp.ndarray,
    xT: jnp.ndarray,
    t: jnp.ndarray,
    key,
    sigma: float,
) -> jnp.ndarray:
    """
    Brownian bridge marginal:
      X_t | X_0, X_T ~ N((1-t)X_0 + t X_T, sigma^2 t(1-t) I)
    with T=1.
    """
    tc = t[:, None]
    mean = (1.0 - tc) * x0 + tc * xT
    std = sigma * jnp.sqrt(tc * (1.0 - tc))
    return mean + std * jax.random.normal(key, x0.shape)


def bms_independent_coupling_target_control(
    x0: jnp.ndarray,
    xT: jnp.ndarray,
    xt: jnp.ndarray,
    t: jnp.ndarray,
    sigma: float,
    prior_var: float,
) -> jnp.ndarray:
    """
    Independent-coupling BMS target with c(t)=gamma(t)=t, T=1,
    reference dX = sigma dB.

      sigma^{-1} xi =
          grad_x0 log p_prior(x0)
        + grad_xT log p_target(xT)
        - grad_xt log P_{t|0}(xt | x0)

    where grad_xt log P_{t|0}(xt | x0) = -(xt-x0)/(sigma^2 t).
    """
    tc = t[:, None]

    prior_score = -x0 / prior_var
    terminal_score = target_score(xT)
    grad_log_pt_given_0 = -(xt - x0) / (sigma**2 * tc)

    return sigma * (prior_score + terminal_score - grad_log_pt_given_0)


# -----------------------------
# Jitted training step
# -----------------------------


@partial(jax.jit, static_argnames=("batch_size", "use_importance_weights"))
def train_step(
    state: train_state.TrainState,
    ref_params,
    key,
    batch_size: int,
    sigma: float,
    t_eps: float,
    prior_var: float,
    eta: float,
    entropy_coef: float,
    var_reg: float,
    code_reg: float,
    use_importance_weights: bool,
    max_log_weight_span: float,
):
    key_x0, key_xT, key_t, key_bridge = jax.random.split(key, 4)

    dim = TARGET_DIM
    x0 = jnp.sqrt(prior_var) * jax.random.normal(key_x0, (batch_size, dim))

    # Endpoint samples from the current neural-generated terminal GMM.
    # They are treated as data for this SGD step.
    xT = sample_terminal_gmm(state.apply_fn, state.params, key_xT, batch_size)

    t = jax.random.uniform(
        key_t,
        (batch_size,),
        minval=t_eps,
        maxval=1.0 - t_eps,
    )

    xt = sample_brownian_bridge(x0, xT, t, key_bridge, sigma)

    xi = bms_independent_coupling_target_control(
        x0=x0,
        xT=xT,
        xt=xt,
        t=t,
        sigma=sigma,
        prior_var=prior_var,
    )
    xi = jax.lax.stop_gradient(xi)

    u_ref = state.apply_fn({"params": ref_params}, xt, t, sigma)
    u_ref = jax.lax.stop_gradient(u_ref)

    # Optional endpoint importance weighting.
    if use_importance_weights:
        log_q = terminal_gmm_log_prob(state.apply_fn, state.params, xT)
        log_w = target_log_rho(xT) - log_q
        log_w = jax.lax.stop_gradient(log_w)
        log_w = log_w - jnp.max(log_w)
        log_w = jnp.maximum(log_w, -max_log_weight_span)
        w = jnp.exp(log_w)
        weights = w / (jnp.sum(w) + 1.0e-12)
    else:
        weights = jnp.full((batch_size,), 1.0 / batch_size, dtype=xT.dtype)
    weights = jax.lax.stop_gradient(weights)
    weight_ess = 1.0 / (jnp.sum(weights**2) + 1.0e-12)
    weight_max = jnp.max(weights)

    def loss_fn(params):
        u_pred = state.apply_fn({"params": params}, xt, t, sigma)

        per_sample_drift = 0.5 * jnp.sum((u_pred - xi) ** 2, axis=-1)
        per_sample_damping = 0.5 * eta * jnp.sum((u_pred - u_ref) ** 2, axis=-1)
        drift_mse = jnp.sum(weights * per_sample_drift)
        damping = jnp.sum(weights * per_sample_damping)

        logits, means_T, vars_T, raw_logvars_T = get_terminal_params(state.apply_fn, params)
        pi = jax.nn.softmax(logits)

        # Mild regularizers: no mode information, only numerical stabilization.
        entropy_loss = entropy_coef * jnp.sum(pi * jnp.log(pi + 1.0e-8))
        var_loss = var_reg * jnp.mean(jnp.log(vars_T) ** 2)
        code_loss = code_reg * jnp.mean(params["component_codes"] ** 2)

        total = drift_mse + damping + entropy_loss + var_loss + code_loss

        metrics = {
            "loss": total,
            "drift_mse": drift_mse,
            "damping": damping,
            "entropy": -jnp.sum(pi * jnp.log(pi + 1.0e-8)),
            "mean_norm": jnp.mean(jnp.linalg.norm(means_T, axis=-1)),
            "avg_var": jnp.mean(vars_T),
            "weight_ess": weight_ess,
            "weight_max": weight_max,
        }
        return total, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, metrics


# -----------------------------
# Plotting
# -----------------------------


def save_outputs(state, init_params, key, args):
    out_dir = Path.cwd() / "tmp"
    out_dir.mkdir(parents=True, exist_ok=True)

    key_init_samples, key_final_samples = jax.random.split(key)

    init_samples = np.asarray(
        sample_terminal_gmm(state.apply_fn, init_params, key_init_samples, args.n_plot_samples)
    )
    final_samples = np.asarray(
        sample_terminal_gmm(state.apply_fn, state.params, key_final_samples, args.n_plot_samples)
    )

    init_logits, init_means_T, init_vars_T, _ = get_terminal_params(state.apply_fn, init_params)
    init_pi = np.asarray(jax.nn.softmax(init_logits))
    init_means_np = np.asarray(init_means_T)
    init_vars_np = np.asarray(init_vars_T)

    logits, means_T, vars_T, _ = get_terminal_params(state.apply_fn, state.params)
    pi = np.asarray(jax.nn.softmax(logits))
    means_np = np.asarray(means_T)
    vars_np = np.asarray(vars_T)

    np.savez(
        out_dir / "learned_gmm_params.npz",
        weights=pi,
        means=means_np,
        diag_vars=vars_np,
        init_weights=init_pi,
        init_means=init_means_np,
        init_diag_vars=init_vars_np,
    )

    if TARGET_DIM != 2:
        print("Skipping 2D plots because TARGET_DIM != 2.")
        print(f"Saved GMM params to: {out_dir / 'learned_gmm_params.npz'}")
        return

    lim = args.plot_lim
    grid_n = args.grid_n
    xs = np.linspace(-lim, lim, grid_n)
    ys = np.linspace(-lim, lim, grid_n)
    xx, yy = np.meshgrid(xs, ys)
    pts = jnp.asarray(np.stack([xx.ravel(), yy.ravel()], axis=-1))

    log_rho = np.asarray(target_log_rho(pts)).reshape(grid_n, grid_n)
    dens = np.exp(log_rho - np.max(log_rho))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    axes[0].imshow(
        dens,
        extent=[-lim, lim, -lim, lim],
        origin="lower",
        aspect="equal",
    )
    axes[0].set_title("Target Boltzmann density $\\rho(x)=e^{-U(x)}$")
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")

    axes[1].scatter(init_samples[:, 0], init_samples[:, 1], s=2, alpha=0.25)
    axes[1].scatter(init_means_np[:, 0], init_means_np[:, 1], s=80, marker="x")
    axes[1].set_xlim(-lim, lim)
    axes[1].set_ylim(-lim, lim)
    axes[1].set_aspect("equal")
    axes[1].set_title("Initial neural terminal GMM $q_T$ before training")
    axes[1].set_xlabel("$x_1$")
    axes[1].set_ylabel("$x_2$")

    axes[2].scatter(final_samples[:, 0], final_samples[:, 1], s=2, alpha=0.25)
    axes[2].scatter(means_np[:, 0], means_np[:, 1], s=80, marker="x")
    axes[2].set_xlim(-lim, lim)
    axes[2].set_ylim(-lim, lim)
    axes[2].set_aspect("equal")
    axes[2].set_title("Learned neural terminal GMM $q_T$ after training")
    axes[2].set_xlabel("$x_1$")
    axes[2].set_ylabel("$x_2$")

    fig.savefig(out_dir / "target_vs_mixture_samples.png", dpi=200)
    plt.close(fig)

    print(f"\nSaved image to: {out_dir / 'target_vs_mixture_samples.png'}")
    print(f"Saved GMM params to: {out_dir / 'learned_gmm_params.npz'}")


# -----------------------------
# Main
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1.0e-3)

    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--prior_var", type=float, default=1.0)
    parser.add_argument("--t_eps", type=float, default=2.0e-2)

    parser.add_argument("--eta", type=float, default=0.10)
    parser.add_argument("--outer_every", type=int, default=250)

    parser.add_argument("--var_floor", type=float, default=1.0e-3)
    parser.add_argument("--entropy_coef", type=float, default=1.0e-3)
    parser.add_argument("--var_reg", type=float, default=1.0e-3)
    parser.add_argument("--code_reg", type=float, default=1.0e-6)

    # Optional endpoint importance weighting.
    parser.add_argument(
        "--use_importance_weights",
        dest="use_importance_weights",
        action="store_true",
        help="Use self-normalized endpoint weights proportional to rho_target(x_T) / q_current(x_T).",
    )
    parser.add_argument(
        "--no_importance_weights",
        dest="use_importance_weights",
        action="store_false",
        help="Disable endpoint importance weights.",
    )
    parser.set_defaults(use_importance_weights=False)
    parser.add_argument("--max_log_weight_span", type=float, default=20.0)

    # Neural GMM parameter-generator hyperparameters.
    parser.add_argument("--code_dim", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--code_init_std", type=float, default=1.0)
    parser.add_argument("--init_mean_scale", type=float, default=12.0)
    parser.add_argument("--init_terminal_std", type=float, default=2.5)
    parser.add_argument("--mean_head_init_std", type=float, default=0.7)
    parser.add_argument("--logvar_output_scale", type=float, default=0.25)
    parser.add_argument("--logit_output_scale", type=float, default=0.1)

    parser.add_argument("--log_every", type=int, default=250)

    parser.add_argument("--plot_lim", type=float, default=18.5)
    parser.add_argument("--grid_n", type=int, default=350)
    parser.add_argument("--n_plot_samples", type=int, default=30000)

    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path.cwd() / "tmp"
    out_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(args.seed)
    key_init, key_train, key_plot = jax.random.split(key, 3)

    model = NeuralGMMPath(
        k=args.k,
        dim=TARGET_DIM,
        var_floor=args.var_floor,
        prior_var=args.prior_var,
        code_dim=args.code_dim,
        hidden=args.hidden,
        depth=args.depth,
        code_init_std=args.code_init_std,
        init_mean_scale=args.init_mean_scale,
        init_terminal_std=args.init_terminal_std,
        mean_head_init_std=args.mean_head_init_std,
        logvar_output_scale=args.logvar_output_scale,
        logit_output_scale=args.logit_output_scale,
    )

    dummy_x = jnp.zeros((4, TARGET_DIM), dtype=jnp.float32)
    dummy_t = jnp.ones((4,), dtype=jnp.float32) * 0.5

    variables = model.init(key_init, dummy_x, dummy_t, args.sigma)

    tx = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adamw(args.lr, weight_decay=1.0e-5),
    )

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )

    init_params = jax.tree_util.tree_map(lambda z: z.copy(), state.params)
    ref_params = state.params

    print("Starting neural-GMM-parameterized BMS-style training.")
    print(f"Importance weights: {'ON' if args.use_importance_weights else 'OFF'}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Output directory: {out_dir}")

    for step in range(1, args.steps + 1):
        key_train, subkey = jax.random.split(key_train)

        if (step - 1) % args.outer_every == 0:
            ref_params = jax.tree_util.tree_map(lambda z: z, state.params)

        state, metrics = train_step(
            state=state,
            ref_params=ref_params,
            key=subkey,
            batch_size=args.batch_size,
            sigma=args.sigma,
            t_eps=args.t_eps,
            prior_var=args.prior_var,
            eta=args.eta,
            entropy_coef=args.entropy_coef,
            var_reg=args.var_reg,
            code_reg=args.code_reg,
            use_importance_weights=args.use_importance_weights,
            max_log_weight_span=args.max_log_weight_span,
        )

        if step == 1 or step % args.log_every == 0:
            m = jax.device_get(metrics)
            print(f"step {step:6d} | "
                  f"loss {float(m['loss']):10.4f} | "
                  f"mse {float(m['drift_mse']):10.4f} | "
                  f"damp {float(m['damping']):9.4f} | "
                  f"H(pi) {float(m['entropy']):7.3f} | "
                  f"|mu| {float(m['mean_norm']):7.3f} | "
                  f"avg_var {float(m['avg_var']):7.3f} | "
                  f"ESS {float(m['weight_ess']):8.1f} | "
                  f"maxw {float(m['weight_max']):9.6f}")

    save_outputs(state, init_params, key_plot, args)


if __name__ == "__main__":
    main()
