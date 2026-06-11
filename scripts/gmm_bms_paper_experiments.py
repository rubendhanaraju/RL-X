#!/usr/bin/env python3
"""
Paper-style benchmark for explicit GMM learning with GMM-constrained BMS.

Compares:
  - our_gmm: corrected Brownian-bridge/SI direct-GMM BMS
  - our_gmm_iw: same, with endpoint self-normalized importance weights
  - pure_bms: regular BMS sampler endpoint samples
  - bms_posthoc_kmeans: train BMS, fit diagonal GMM to BMS samples with kmeans++ EM
  - bms_posthoc_ourinit: train BMS, fit diagonal GMM to BMS samples initialized from our GMM init
  - oracle_em: fit diagonal GMM directly to true target samples (sanity upper baseline)
  - weighted_em_broad: weighted EM using samples from a broad Gaussian proposal

Outputs:
  <out_dir>/per_run_results.csv
  <out_dir>/summary_results.csv
  <out_dir>/config.json
  <out_dir>/plots/*.png  (only for d=2 if --save_plots)

Install:
  pip install "jax[cuda12]" flax optax matplotlib pandas
  # or: pip install "jax[cpu]" flax optax matplotlib pandas

Example quick smoke test:
  python gmm_bms_paper_experiments.py \
    --dims 2 \
    --seeds 0,1 \
    --methods our_gmm,bms_posthoc_ourinit,oracle_em \
    --steps 1000 --our_steps 1000 \
    --batch_size 512 --our_batch_size 1024 \
    --fit_samples 5000 --eval_samples 5000

Example stronger 2D run:
  python gmm_bms_paper_experiments.py \
    --dims 2 --seeds 0,1,2,3,4 \
    --methods pure_bms,bms_posthoc_kmeans,bms_posthoc_ourinit,our_gmm,oracle_em \
    --steps 20000 --our_steps 20000 \
    --gmm_components 32 --fit_samples 50000 --eval_samples 20000

Example high-dimensional run:
  python gmm_bms_paper_experiments.py \
    --dims 16,32,64,100 \
    --seeds 0,1,2 \
    --methods bms_posthoc_kmeans,bms_posthoc_ourinit,our_gmm,our_gmm_iw,oracle_em \
    --target_modes 20 --mode_scale 20 --target_var 1.0 \
    --gmm_components 32 \
    --steps 50000 --our_steps 50000 \
    --fit_samples 30000 --eval_samples 20000
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def parse_int_list(s: str):
    if isinstance(s, (list, tuple)):
        return list(map(int, s))
    return [int(x) for x in s.split(',') if x.strip()]


def parse_method_list(s: str):
    return [x.strip() for x in s.split(',') if x.strip()]


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def summarize_rows(rows):
    if not rows:
        return []
    group_keys = ['method', 'dim', 'target_modes', 'gmm_components', 'em_init', 'importance_weights']
    groups = {}
    for r in rows:
        key = tuple(r.get(k, '') for k in group_keys)
        groups.setdefault(key, []).append(r)
    out = []
    for key, rs in groups.items():
        row = {k: v for k, v in zip(group_keys, key)}
        row['n_seeds'] = len(rs)
        numeric = []
        for k in sorted(set().union(*[r.keys() for r in rs])):
            vals = []
            for r in rs:
                try:
                    v = float(r[k])
                    if np.isfinite(v):
                        vals.append(v)
                except Exception:
                    pass
            if vals and k not in ['seed', 'dim', 'target_modes', 'gmm_components']:
                row[f'{k}_mean'] = float(np.mean(vals))
                row[f'{k}_std'] = float(np.std(vals, ddof=0))
        out.append(row)
    return out


# -----------------------------------------------------------------------------
# Target GMM generation and evaluation
# -----------------------------------------------------------------------------


def make_target_np(seed: int,
                   dim: int,
                   modes: int,
                   mode_scale: float,
                   target_var: float,
                   weights_kind: str = 'uniform'):
    rng = np.random.default_rng(seed)
    means = rng.uniform(-mode_scale, mode_scale, size=(modes, dim)).astype(np.float32)
    vars_ = np.full((modes, dim), target_var, dtype=np.float32)
    if weights_kind == 'uniform':
        weights = np.full(modes, 1.0 / modes, dtype=np.float32)
    elif weights_kind == 'dirichlet':
        weights = rng.dirichlet(np.ones(modes)).astype(np.float32)
    else:
        raise ValueError(f'Unknown weights_kind={weights_kind}')
    return {'weights': weights, 'means': means, 'vars': vars_}


def sample_target_np(rng, target, n: int):
    weights = np.asarray(target['weights'], dtype=np.float64)
    means = np.asarray(target['means'], dtype=np.float64)
    vars_ = np.asarray(target['vars'], dtype=np.float64)
    comp = rng.choice(means.shape[0], size=n, p=weights / weights.sum())
    eps = rng.normal(size=(n, means.shape[1]))
    return (means[comp] + np.sqrt(vars_[comp]) * eps).astype(np.float32)


def _logsumexp_np(a, axis=-1, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True) + 1e-300)
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def log_diag_gmm_np(x, weights, means, vars_):
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    vars_ = np.asarray(vars_, dtype=np.float64)
    d = means.shape[1]
    diff = x[:, None, :] - means[None, :, :]
    log_comp = (np.log(weights[None, :] + 1e-300) -
                0.5 * np.sum(np.log(2.0 * np.pi * vars_)[None, :, :] + diff**2 / vars_[None, :, :], axis=-1))
    return _logsumexp_np(log_comp, axis=1)


def target_log_prob_np(x, target):
    return log_diag_gmm_np(x, target['weights'], target['means'], target['vars'])


def target_arrays_jax(target):
    return (jnp.asarray(target['weights'], dtype=jnp.float32), jnp.asarray(target['means'], dtype=jnp.float32),
            jnp.asarray(target['vars'], dtype=jnp.float32))


def target_log_prob_jax(x, weights, means, vars_):
    d = means.shape[-1]
    diff = x[..., None, :] - means
    quad = jnp.sum(diff**2 / vars_, axis=-1)
    log_det = jnp.sum(jnp.log(vars_), axis=-1)
    log_comp = jnp.log(weights + 1e-30) - 0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det + quad)
    return jax.nn.logsumexp(log_comp, axis=-1)


def target_score_jax(x, weights, means, vars_):
    d = means.shape[-1]
    diff = x[:, None, :] - means[None, :, :]
    quad = jnp.sum(diff**2 / vars_[None, :, :], axis=-1)
    log_det = jnp.sum(jnp.log(vars_), axis=-1)
    log_comp = jnp.log(weights[None, :] + 1e-30) - 0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det[None, :] + quad)
    resp = jax.nn.softmax(log_comp, axis=-1)
    comp_scores = -diff / vars_[None, :, :]
    return jnp.sum(resp[:, :, None] * comp_scores, axis=1)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def mode_histogram(samples, target):
    centers = np.asarray(target['means'], dtype=np.float64)
    samples = np.asarray(samples, dtype=np.float64)
    d2 = np.sum((samples[:, None, :] - centers[None, :, :])**2, axis=-1)
    assign = np.argmin(d2, axis=1)
    hist = np.bincount(assign, minlength=centers.shape[0]).astype(np.float64)
    return hist / max(hist.sum(), 1.0)


def mode_tvd(samples, target):
    emp = mode_histogram(samples, target)
    true = np.asarray(target['weights'], dtype=np.float64)
    true = true / true.sum()
    return float(0.5 * np.sum(np.abs(emp - true)))


def sliced_w2_np(x, y, n_proj=100, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    n = min(len(x), len(y))
    x = np.asarray(x[:n], dtype=np.float64)
    y = np.asarray(y[:n], dtype=np.float64)
    dirs = rng.normal(size=(n_proj, x.shape[1]))
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    vals = []
    for v in dirs:
        xs = np.sort(x @ v)
        ys = np.sort(y @ v)
        vals.append(np.mean((xs - ys)**2))
    return float(np.sqrt(np.mean(vals)))


def sliced_tvd_np(x, y, n_proj=100, bins=50, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dirs = rng.normal(size=(n_proj, x.shape[1]))
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    tvds = []
    for v in dirs:
        xp = x @ v
        yp = y @ v
        lo = min(np.min(xp), np.min(yp))
        hi = max(np.max(xp), np.max(yp))
        if hi <= lo:
            tvds.append(0.0)
            continue
        hx, edges = np.histogram(xp, bins=bins, range=(lo, hi), density=False)
        hy, _ = np.histogram(yp, bins=edges, density=False)
        px = hx / max(hx.sum(), 1)
        py = hy / max(hy.sum(), 1)
        tvds.append(0.5 * np.sum(np.abs(px - py)))
    return float(np.mean(tvds))


def estimate_forward_kl_to_gmm(rng, target, gmm, n=20000):
    x = sample_target_np(rng, target, n)
    logp = target_log_prob_np(x, target)
    logq = log_diag_gmm_np(x, gmm['weights'], gmm['means'], gmm['vars'])
    return float(np.mean(logp - logq))


def estimate_reverse_kl_to_target(rng, target, gmm, n=20000):
    x = sample_diag_gmm_np(rng, gmm, n)
    logq = log_diag_gmm_np(x, gmm['weights'], gmm['means'], gmm['vars'])
    logp = target_log_prob_np(x, target)
    return float(np.mean(logq - logp))


def estimate_importance_ess(rng, target, gmm, n=20000):
    x = sample_diag_gmm_np(rng, gmm, n)
    logp = target_log_prob_np(x, target)
    logq = log_diag_gmm_np(x, gmm['weights'], gmm['means'], gmm['vars'])
    lw = logp - logq
    lw = lw - np.max(lw)
    w = np.exp(lw)
    ess = (np.sum(w)**2) / (np.sum(w**2) + 1e-300)
    return float(ess / n), float(np.max(w) / (np.sum(w) + 1e-300)), float(np.std(lw))


def evaluate_samples_and_density(rng,
                                 target,
                                 samples,
                                 gmm=None,
                                 target_samples=None,
                                 n_proj=100,
                                 bins=50,
                                 kl_samples=20000):
    if target_samples is None:
        target_samples = sample_target_np(rng, target, len(samples))
    out = {
        'mode_tvd': mode_tvd(samples, target),
        'sliced_w2': sliced_w2_np(samples, target_samples, n_proj=n_proj, rng=rng),
        'sliced_tvd': sliced_tvd_np(samples, target_samples, n_proj=n_proj, bins=bins, rng=rng),
    }
    if gmm is not None:
        fkl = estimate_forward_kl_to_gmm(rng, target, gmm, n=kl_samples)
        rkl = estimate_reverse_kl_to_target(rng, target, gmm, n=kl_samples)
        ess, maxw, lwstd = estimate_importance_ess(rng, target, gmm, n=kl_samples)
        out.update({
            'forward_kl': fkl,
            'reverse_kl': rkl,
            'importance_ess_frac': ess,
            'importance_max_weight_frac': maxw,
            'importance_logw_std': lwstd,
        })
    return out


# -----------------------------------------------------------------------------
# EM baselines
# -----------------------------------------------------------------------------


def kmeanspp_init(rng, x, k):
    n = x.shape[0]
    means = np.empty((k, x.shape[1]), dtype=np.float64)
    idx = rng.integers(n)
    means[0] = x[idx]
    dist2 = np.sum((x - means[0])**2, axis=1)
    for i in range(1, k):
        probs = dist2 / (dist2.sum() + 1e-300)
        idx = rng.choice(n, p=probs)
        means[i] = x[idx]
        dist2 = np.minimum(dist2, np.sum((x - means[i])**2, axis=1))
    return means


def fit_diag_gmm_em(x, k, rng, iters=100, var_floor=1e-3, init='kmeanspp', init_gmm=None, sample_weights=None):
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x).all(axis=1)
    x = x[finite]
    if sample_weights is not None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)[finite]
        sample_weights = np.maximum(sample_weights, 0.0)
        if sample_weights.sum() <= 0:
            sample_weights = None
        else:
            sample_weights = sample_weights / sample_weights.sum()
    if x.shape[0] < k:
        raise ValueError(f'Need at least k finite samples; got {x.shape[0]} for k={k}')
    n, d = x.shape
    if sample_weights is None:
        sample_weights = np.full(n, 1.0 / n, dtype=np.float64)

    if init == 'our_init':
        if init_gmm is None:
            raise ValueError("init='our_init' requires init_gmm")
        weights = np.asarray(init_gmm['weights'], dtype=np.float64).copy()
        means = np.asarray(init_gmm['means'], dtype=np.float64).copy()
        vars_ = np.asarray(init_gmm['vars'], dtype=np.float64).copy()
        if means.shape != (k, d):
            raise ValueError(f'our_init means shape {means.shape}, expected {(k, d)}')
        weights = np.maximum(weights, 1e-12)
        weights = weights / weights.sum()
        vars_ = np.maximum(vars_, var_floor)
    else:
        if init == 'random':
            idx = rng.choice(n, size=k, replace=False)
            means = x[idx].copy()
        else:
            means = kmeanspp_init(rng, x, k)
        weights = np.full(k, 1.0 / k, dtype=np.float64)
        global_var = np.var(x, axis=0) + var_floor
        vars_ = np.tile(global_var[None, :], (k, 1))

    prev_ll = -np.inf
    for _ in range(iters):
        diff = x[:, None, :] - means[None, :, :]
        log_comp = (np.log(weights[None, :] + 1e-300) -
                    0.5 * np.sum(np.log(2.0 * np.pi * vars_)[None, :, :] + diff**2 / vars_[None, :, :], axis=-1))
        log_norm = _logsumexp_np(log_comp, axis=1, keepdims=True)
        resp = np.exp(log_comp - log_norm)
        weighted_resp = sample_weights[:, None] * resp
        nk = weighted_resp.sum(axis=0) + 1e-12
        weights = nk / nk.sum()
        means = (weighted_resp.T @ x) / nk[:, None]
        diff = x[:, None, :] - means[None, :, :]
        vars_ = np.sum(weighted_resp[:, :, None] * diff**2, axis=0) / nk[:, None]
        vars_ = np.maximum(vars_, var_floor)
        ll = float(np.sum(sample_weights * np.squeeze(log_norm)))
        if abs(ll - prev_ll) < 1e-7:
            break
        prev_ll = ll
    weights = np.maximum(weights, 1e-12)
    weights = weights / weights.sum()
    return {'weights': weights.astype(np.float32), 'means': means.astype(np.float32), 'vars': vars_.astype(np.float32)}


def sample_diag_gmm_np(rng, gmm, n):
    weights = np.asarray(gmm['weights'], dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    weights = weights / weights.sum()
    means = np.asarray(gmm['means'], dtype=np.float64)
    vars_ = np.asarray(gmm['vars'], dtype=np.float64)
    comp = rng.choice(means.shape[0], size=n, p=weights)
    eps = rng.normal(size=(n, means.shape[1]))
    return (means[comp] + np.sqrt(vars_[comp]) * eps).astype(np.float32)


# -----------------------------------------------------------------------------
# Pure BMS neural drift baseline
# -----------------------------------------------------------------------------


def time_features(t: jnp.ndarray, num_freqs: int) -> jnp.ndarray:
    t = t[:, None]
    freqs = (2.0**jnp.arange(num_freqs, dtype=t.dtype))[None, :]
    angles = 2.0 * jnp.pi * t * freqs
    return jnp.concatenate([t, jnp.sin(angles), jnp.cos(angles)], axis=-1)


class ResidualBlock(nn.Module):
    hidden: int

    @nn.compact
    def __call__(self, h):
        z = nn.Dense(self.hidden)(h)
        z = nn.silu(z)
        z = nn.Dense(self.hidden)(z)
        return nn.silu(h + z / np.sqrt(2.0))


class BMSDriftNet(nn.Module):
    dim: int
    hidden: int = 256
    depth: int = 4
    time_freqs: int = 8
    x_scale: float = 20.0
    zero_last: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        tf = time_features(t, self.time_freqs)
        h = jnp.concatenate([x / self.x_scale, tf], axis=-1)
        h = nn.Dense(self.hidden)(h)
        h = nn.silu(h)
        for _ in range(self.depth):
            h = ResidualBlock(self.hidden)(h)
        if self.zero_last:
            return nn.Dense(self.dim, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)(h)
        return nn.Dense(self.dim)(h)


@partial(jax.jit, static_argnames=('apply_fn', 'n_steps'))
def simulate_sde(apply_fn, params, x0, key, sigma: float, n_steps: int, state_clip: float, control_clip: float):
    dt = 1.0 / float(n_steps)
    sqrt_dt = jnp.sqrt(dt)
    keys = jax.random.split(key, n_steps)

    def step(carry, subkey):
        x, i = carry
        t = jnp.full((x.shape[0],), (i + 0.5) * dt, dtype=x.dtype)
        u = apply_fn({'params': params}, x, t)
        u_norm = jnp.linalg.norm(u, axis=-1, keepdims=True)
        u = u * jnp.minimum(1.0, control_clip / (u_norm + 1e-6))
        noise = jax.random.normal(subkey, x.shape, dtype=x.dtype)
        x = x + sigma * u * dt + sigma * sqrt_dt * noise
        x = jnp.nan_to_num(x, nan=0.0, posinf=state_clip, neginf=-state_clip)
        x = jnp.clip(x, -state_clip, state_clip)
        return (x, i + 1), None

    (xT, _), _ = jax.lax.scan(step, (x0, 0), keys)
    return xT


def refresh_endpoint_buffer(state, key, args, dim: int, buffer_size: int):
    out = []
    n_done = 0
    key_loop = key
    while n_done < buffer_size:
        n = min(args.sample_batch, buffer_size - n_done)
        key_loop, key_x0, key_sde = jax.random.split(key_loop, 3)
        x0 = args.prior_std * jax.random.normal(key_x0, (n, dim), dtype=jnp.float32)
        xT = simulate_sde(state.apply_fn, state.params, x0, key_sde, args.sigma, args.sde_steps, args.state_clip,
                          args.control_clip)
        out.append(np.asarray(jax.device_get(xT)))
        n_done += n
    arr = np.concatenate(out, axis=0).astype(np.float32)
    finite = np.isfinite(arr).all(axis=1)
    if finite.sum() == 0:
        arr = np.zeros((buffer_size, dim), dtype=np.float32)
    else:
        arr = arr[finite]
        if arr.shape[0] < buffer_size:
            reps = int(np.ceil(buffer_size / arr.shape[0]))
            arr = np.tile(arr, (reps, 1))[:buffer_size]
    return jnp.asarray(arr[:buffer_size])


def sample_brownian_bridge(x0, xT, t, key, sigma: float):
    tc = t[:, None]
    mean = (1.0 - tc) * x0 + tc * xT
    std = sigma * jnp.sqrt(tc * (1.0 - tc))
    return mean + std * jax.random.normal(key, x0.shape, dtype=x0.dtype)


def bms_target_control(x0, xT, xt, t, sigma: float, prior_var: float, weights, means, vars_):
    tc = t[:, None]
    prior_score = -x0 / prior_var
    terminal_score = target_score_jax(xT, weights, means, vars_)
    grad_log_pt_given_0 = -(xt - x0) / (sigma**2 * tc)
    return sigma * (prior_score + terminal_score - grad_log_pt_given_0)


@partial(jax.jit, static_argnames=('batch_size',))
def train_step_bms(state, ref_params, endpoint_buffer, key, batch_size: int, sigma: float, prior_std: float, eta: float,
                   t_eps: float, xi_clip: float, weights, means, vars_):
    key_x0, key_idx, key_t, key_bridge = jax.random.split(key, 4)
    x0 = prior_std * jax.random.normal(key_x0, (batch_size, means.shape[-1]), dtype=jnp.float32)
    idx = jax.random.randint(key_idx, (batch_size,), 0, endpoint_buffer.shape[0])
    xT = endpoint_buffer[idx]
    t = jax.random.uniform(key_t, (batch_size,), minval=t_eps, maxval=1.0 - t_eps)
    xt = sample_brownian_bridge(x0, xT, t, key_bridge, sigma)
    xi = bms_target_control(x0, xT, xt, t, sigma, prior_std**2, weights, means, vars_)
    xi_norm = jnp.linalg.norm(xi, axis=-1, keepdims=True)
    xi = xi * jnp.minimum(1.0, xi_clip / (xi_norm + 1e-6))
    xi = jax.lax.stop_gradient(xi)
    u_ref = state.apply_fn({'params': ref_params}, xt, t)
    u_ref = jax.lax.stop_gradient(u_ref)

    def loss_fn(params):
        u = state.apply_fn({'params': params}, xt, t)
        per_mse = 0.5 * jnp.sum((u - xi)**2, axis=-1)
        per_damp = 0.5 * eta * jnp.sum((u - u_ref)**2, axis=-1)
        loss = jnp.mean(per_mse + per_damp)
        metrics = {
            'loss': loss,
            'mse': jnp.mean(per_mse),
            'damp': jnp.mean(per_damp),
            'u_norm': jnp.mean(jnp.linalg.norm(u, axis=-1)),
            'xi_norm': jnp.mean(jnp.linalg.norm(xi, axis=-1))
        }
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads_finite = jax.tree_util.tree_reduce(
        lambda a, b: a & b,
        jax.tree_util.tree_map(lambda g: jnp.all(jnp.isfinite(g)), grads),
        True,
    )
    finite = jnp.isfinite(loss) & grads_finite
    state = jax.lax.cond(finite, lambda s: s.apply_gradients(grads=grads), lambda s: s, state)
    metrics['finite'] = finite
    return state, metrics


def train_pure_bms(args, key, target, dim):
    weights, means, vars_ = target_arrays_jax(target)
    key_init, key_buf, key_train = jax.random.split(key, 3)
    model = BMSDriftNet(dim=dim,
                        hidden=args.hidden,
                        depth=args.depth,
                        time_freqs=args.time_freqs,
                        x_scale=args.prior_std,
                        zero_last=True)
    dummy_x = jnp.zeros((4, dim), dtype=jnp.float32)
    dummy_t = jnp.ones((4,), dtype=jnp.float32) * 0.5
    variables = model.init(key_init, dummy_x, dummy_t)
    tx = optax.chain(optax.clip_by_global_norm(args.grad_clip), optax.adamw(args.lr, weight_decay=args.weight_decay))
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)
    ref_params = state.params
    endpoint_buffer = refresh_endpoint_buffer(state, key_buf, args, dim, args.buffer_size)
    start = time.time()
    last_metrics = None
    for step in range(1, args.steps + 1):
        key_train, subkey = jax.random.split(key_train)
        if (step - 1) % args.outer_every == 0:
            ref_params = jax.tree_util.tree_map(lambda z: z, state.params)
        if step == 1 or (step - 1) % args.refresh_every == 0:
            key_train, key_refresh = jax.random.split(key_train)
            endpoint_buffer = refresh_endpoint_buffer(state, key_refresh, args, dim, args.buffer_size)
        state, metrics = train_step_bms(state, ref_params, endpoint_buffer, subkey, args.batch_size, args.sigma,
                                        args.prior_std, args.eta, args.t_eps, args.xi_clip, weights, means, vars_)
        last_metrics = metrics
        if args.verbose and (step == 1 or step % args.log_every == 0):
            m = jax.device_get(metrics)
            print(
                f'[BMS d={dim} step={step:6d}] loss={float(m["loss"]):10.4f} mse={float(m["mse"]):10.4f} finite={bool(m["finite"])}'
            )
    return state, time.time() - start, last_metrics


# -----------------------------------------------------------------------------
# Our corrected Brownian-bridge/SI GMM-BMS
# -----------------------------------------------------------------------------


def positive_terminal_vars(raw_logvars: jnp.ndarray, var_floor: float) -> jnp.ndarray:
    return var_floor + jnp.exp(jnp.clip(raw_logvars, -7.0, 4.0))


def terminal_params_from_flax_params(params, var_floor: float):
    logits = params['logits']
    means_T = params['means_T']
    vars_T = positive_terminal_vars(params['raw_logvars_T'], var_floor)
    return logits, means_T, vars_T


def terminal_gmm_log_prob_jax(params, x, var_floor):
    logits, means_T, vars_T = terminal_params_from_flax_params(params, var_floor)
    diff = x[:, None, :] - means_T[None, :, :]
    log_weights = jax.nn.log_softmax(logits)
    log_comp = log_weights[None, :] - 0.5 * jnp.sum(
        jnp.log(2.0 * jnp.pi * vars_T)[None, :, :] + diff**2 / vars_T[None, :, :], axis=-1)
    return jax.nn.logsumexp(log_comp, axis=-1)


def sample_terminal_gmm_jax(params, key, n: int, var_floor: float):
    logits, means_T, vars_T = terminal_params_from_flax_params(params, var_floor)
    key_comp, key_noise = jax.random.split(key)
    comp = jax.random.categorical(key_comp, logits, shape=(n,))
    eps = jax.random.normal(key_noise, (n, means_T.shape[-1]), dtype=means_T.dtype)
    return means_T[comp] + jnp.sqrt(vars_T[comp]) * eps


def gmm_path_control_from_raw(logits, means_T, raw_logvars_T, x, t, sigma, var_floor, prior_var):
    """Corrected direct-GMM BMS/SI control u_phi(x,t).

    Terminal model:
        q_phi,1(x) = sum_k pi_k N(x; mu_k, diag(v_k)).

    The BMS bridge / linear stochastic-interpolant samples are
        X_t = (1-t)X_0 + tX_1 + sigma*sqrt(t(1-t))*Z,
    with X_0 ~ N(0, prior_var I) and X_1|C=k ~ N(mu_k, diag(v_k)).
    Therefore the component marginal path is
        m_k(t) = t mu_k,
        S_k(t) = (1-t)^2 prior_var I + t^2 diag(v_k)
                 + sigma^2 t(1-t) I.

    The physical forward SDE drift preserving this path is
        b_k(x,t) = mu_k + A_k(t)(x - t mu_k),
        A_k(t) = 0.5*(dS_k/dt - sigma^2 I)*S_k(t)^{-1}
               = [t(diag(v_k)-sigma^2 I) - (1-t)prior_var I] S_k(t)^{-1}.

    The returned value is the BMS control u_phi=b_phi/sigma, where
        b_phi(x,t)=sum_k r_k(x,t)b_k(x,t).
    """
    # Training already samples t in [t_eps, 1-t_eps]; this guard only prevents
    # accidental endpoint singularities in responsibilities and drift factors.
    t = jnp.clip(t, 1e-4, 1.0 - 1e-4)
    t3 = t[:, None, None]
    vars_T = positive_terminal_vars(raw_logvars_T, var_floor)

    mu_t = t3 * means_T[None, :, :]
    dmu_dt = means_T[None, :, :]

    # Correct Brownian-bridge / stochastic-interpolant covariance path.
    # Do NOT use the old linear covariance (1-t)*prior_var + t*vars_T here;
    # that is not the marginal covariance of the bridge used by BMS.
    var_t = ((1.0 - t3)**2) * prior_var + (t3**2) * vars_T[None, :, :] + (sigma**2) * t3 * (1.0 - t3)

    # Equivalent to 0.5*(dvar_dt - sigma^2)/var_t for the corrected var_t.
    A_diag = (t3 * (vars_T[None, :, :] - sigma**2) - (1.0 - t3) * prior_var) / var_t

    x_centered = x[:, None, :] - mu_t
    component_drift = dmu_dt + A_diag * x_centered
    log_weights = jax.nn.log_softmax(logits)
    log_comp = log_weights[None, :] - 0.5 * jnp.sum(
        jnp.log(2.0 * jnp.pi * var_t) + x_centered**2 / var_t,
        axis=-1,
    )
    resp = jax.nn.softmax(log_comp, axis=-1)
    actual_drift = jnp.sum(resp[:, :, None] * component_drift, axis=1)
    return actual_drift / sigma


class DirectGMMPath(nn.Module):
    k: int
    dim: int
    var_floor: float = 1e-3
    prior_var: float = 1.0
    init_mean_scale: float = 12.0
    init_terminal_std: float = 2.5

    @nn.compact
    def __call__(self, x, t, sigma):
        logits = self.param('logits', nn.initializers.zeros, (self.k,))
        means_T = self.param('means_T', nn.initializers.normal(self.init_mean_scale), (self.k, self.dim))

        def raw_logvar_init(key, shape, dtype=jnp.float32):
            del key
            init_var_minus_floor = jnp.maximum(jnp.asarray(self.init_terminal_std**2 - self.var_floor, dtype=dtype),
                                               jnp.asarray(1e-6, dtype=dtype))
            return jnp.full(shape, jnp.log(init_var_minus_floor), dtype=dtype)

        raw_logvars_T = self.param('raw_logvars_T', raw_logvar_init, (self.k, self.dim))
        return gmm_path_control_from_raw(logits, means_T, raw_logvars_T, x, t, sigma, self.var_floor, self.prior_var)


@partial(jax.jit, static_argnames=('batch_size', 'use_importance_weights'))
def train_step_our_gmm(state, ref_params, key, batch_size: int, sigma: float, prior_var: float, eta: float,
                       t_eps: float, xi_clip: float, var_floor: float, entropy_coef: float, var_reg: float,
                       use_importance_weights: bool, max_log_weight_span: float, weights, means, vars_):
    key_x0, key_xT, key_t, key_bridge = jax.random.split(key, 4)
    x0 = jnp.sqrt(prior_var) * jax.random.normal(key_x0, (batch_size, means.shape[-1]), dtype=jnp.float32)
    # Fixed-point BMS builds the bridge measure from the frozen previous iterate
    # q_{bar phi,1}; the sampled endpoint must not depend on the candidate params
    # being optimized in this gradient step.
    xT = sample_terminal_gmm_jax(ref_params, key_xT, batch_size, var_floor)
    xT = jax.lax.stop_gradient(xT)
    t = jax.random.uniform(key_t, (batch_size,), minval=t_eps, maxval=1.0 - t_eps)
    xt = sample_brownian_bridge(x0, xT, t, key_bridge, sigma)
    xi = bms_target_control(x0, xT, xt, t, sigma, prior_var, weights, means, vars_)
    xi_norm = jnp.linalg.norm(xi, axis=-1, keepdims=True)
    xi = xi * jnp.minimum(1.0, xi_clip / (xi_norm + 1e-6))
    xi = jax.lax.stop_gradient(xi)
    u_ref = state.apply_fn({'params': ref_params}, xt, t, sigma)
    u_ref = jax.lax.stop_gradient(u_ref)

    if use_importance_weights:
        # Importance weights must use the proposal that actually generated xT,
        # namely the frozen reference endpoint law q_{bar phi,1}.
        log_q = terminal_gmm_log_prob_jax(ref_params, xT, var_floor)
        log_w = target_log_prob_jax(xT, weights, means, vars_) - log_q
        log_w = jax.lax.stop_gradient(log_w)
        log_w = log_w - jnp.max(log_w)
        log_w = jnp.maximum(log_w, -max_log_weight_span)
        w = jnp.exp(log_w)
        sample_weights = w / (jnp.sum(w) + 1e-12)
    else:
        sample_weights = jnp.full((batch_size,), 1.0 / batch_size, dtype=xT.dtype)
    sample_weights = jax.lax.stop_gradient(sample_weights)
    weight_ess = 1.0 / (jnp.sum(sample_weights**2) + 1e-12)
    weight_max = jnp.max(sample_weights)

    def loss_fn(params):
        u = state.apply_fn({'params': params}, xt, t, sigma)
        per_mse = 0.5 * jnp.sum((u - xi)**2, axis=-1)
        per_damp = 0.5 * eta * jnp.sum((u - u_ref)**2, axis=-1)
        drift_mse = jnp.sum(sample_weights * per_mse)
        damping = jnp.sum(sample_weights * per_damp)
        logits, means_T, vars_T = terminal_params_from_flax_params(params, var_floor)
        pi = jax.nn.softmax(logits)
        entropy_loss = entropy_coef * jnp.sum(pi * jnp.log(pi + 1e-8))
        var_loss = var_reg * jnp.mean(jnp.log(vars_T)**2)
        loss = drift_mse + damping + entropy_loss + var_loss
        metrics = {
            'loss': loss,
            'mse': drift_mse,
            'damp': damping,
            'entropy': -jnp.sum(pi * jnp.log(pi + 1e-8)),
            'mean_norm': jnp.mean(jnp.linalg.norm(means_T, axis=-1)),
            'avg_var': jnp.mean(vars_T),
            'weight_ess': weight_ess,
            'weight_max': weight_max
        }
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads_finite = jax.tree_util.tree_reduce(lambda a, b: a & b,
                                             jax.tree_util.tree_map(lambda g: jnp.all(jnp.isfinite(g)), grads), True)
    finite = jnp.isfinite(loss) & grads_finite
    state = jax.lax.cond(finite, lambda s: s.apply_gradients(grads=grads), lambda s: s, state)
    metrics['finite'] = finite
    return state, metrics


def gmm_params_to_np(params, var_floor):
    logits, means_T, vars_T = terminal_params_from_flax_params(params, var_floor)
    return {
        'weights': np.asarray(jax.nn.softmax(logits), dtype=np.float32),
        'means': np.asarray(means_T, dtype=np.float32),
        'vars': np.asarray(vars_T, dtype=np.float32)
    }


def make_initial_our_gmm(args, key, dim):
    k = args.gmm_components
    key_init, _ = jax.random.split(key)
    model = DirectGMMPath(k=k,
                          dim=dim,
                          var_floor=args.our_var_floor,
                          prior_var=args.our_prior_var,
                          init_mean_scale=args.our_init_mean_scale,
                          init_terminal_std=args.our_init_terminal_std)
    dummy_x = jnp.zeros((4, dim), dtype=jnp.float32)
    dummy_t = jnp.ones((4,), dtype=jnp.float32) * 0.5
    variables = model.init(key_init, dummy_x, dummy_t, args.our_sigma)
    return gmm_params_to_np(variables['params'], args.our_var_floor)


def train_our_gmm(args, key, target, dim, use_importance_weights=False):
    weights, means, vars_ = target_arrays_jax(target)
    key_init, key_train = jax.random.split(key)
    model = DirectGMMPath(k=args.gmm_components,
                          dim=dim,
                          var_floor=args.our_var_floor,
                          prior_var=args.our_prior_var,
                          init_mean_scale=args.our_init_mean_scale,
                          init_terminal_std=args.our_init_terminal_std)
    dummy_x = jnp.zeros((4, dim), dtype=jnp.float32)
    dummy_t = jnp.ones((4,), dtype=jnp.float32) * 0.5
    variables = model.init(key_init, dummy_x, dummy_t, args.our_sigma)
    tx = optax.chain(optax.clip_by_global_norm(args.our_grad_clip),
                     optax.adamw(args.our_lr, weight_decay=args.weight_decay))
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)
    ref_params = state.params
    start = time.time()
    last_metrics = None
    for step in range(1, args.our_steps + 1):
        key_train, subkey = jax.random.split(key_train)
        if (step - 1) % args.outer_every == 0:
            ref_params = jax.tree_util.tree_map(lambda z: z, state.params)
        state, metrics = train_step_our_gmm(state, ref_params, subkey, args.our_batch_size, args.our_sigma,
                                            args.our_prior_var, args.our_eta, args.our_t_eps, args.our_xi_clip,
                                            args.our_var_floor, args.our_entropy_coef, args.our_var_reg,
                                            use_importance_weights, args.our_max_log_weight_span, weights, means, vars_)
        last_metrics = metrics
        if args.verbose and (step == 1 or step % args.log_every == 0):
            m = jax.device_get(metrics)
            print(
                f'[OUR-GMM d={dim} step={step:6d} iw={use_importance_weights}] loss={float(m["loss"]):10.4f} mse={float(m["mse"]):10.4f} ESS={float(m["weight_ess"]):8.1f}'
            )
    return state, time.time() - start, last_metrics


# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------


def plot_2d(args, target, samples_by_method, gmms_by_method, out_path):
    dim = target['means'].shape[1]
    if dim != 2:
        return
    lim = args.plot_lim
    grid_n = args.grid_n
    xs = np.linspace(-lim, lim, grid_n)
    ys = np.linspace(-lim, lim, grid_n)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
    logp = target_log_prob_np(pts, target).reshape(grid_n, grid_n)
    dens = np.exp(logp - np.max(logp))
    methods = list(samples_by_method.keys())
    ncols = 1 + len(methods)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5), constrained_layout=True)
    axes[0].imshow(dens, extent=[-lim, lim, -lim, lim], origin='lower', aspect='equal')
    axes[0].set_title('Target')
    axes[0].set_xlim(-lim, lim)
    axes[0].set_ylim(-lim, lim)
    for ax, m in zip(axes[1:], methods):
        s = samples_by_method[m]
        ax.scatter(s[:, 0], s[:, 1], s=2, alpha=0.25)
        if m in gmms_by_method and gmms_by_method[m] is not None:
            ax.scatter(gmms_by_method[m]['means'][:, 0], gmms_by_method[m]['means'][:, 1], s=70, marker='x')
        ax.set_title(m)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_one(args, dim: int, seed: int, methods):
    rng = np.random.default_rng(seed)
    target = make_target_np(seed + args.target_seed_offset, dim, args.target_modes, args.mode_scale, args.target_var,
                            args.target_weights)
    key = jax.random.PRNGKey(seed)
    key_bms, key_our, key_eval, key_oracle, key_weighted = jax.random.split(key, 5)
    target_eval_samples = sample_target_np(rng, target, args.eval_samples)

    rows = []
    samples_for_plot = {}
    gmms_for_plot = {}

    needs_bms = any(m in methods for m in ['pure_bms', 'bms_posthoc_kmeans', 'bms_posthoc_ourinit'])
    bms_state = None
    bms_train_time = np.nan
    bms_samples_fit = None
    bms_samples_eval = None
    if needs_bms:
        print(f'=== Training pure BMS dim={dim} seed={seed} ===')
        bms_state, bms_train_time, _ = train_pure_bms(args, key_bms, target, dim)
        key_eval, key_fit, key_bms_eval = jax.random.split(key_eval, 3)
        bms_samples_fit = np.asarray(refresh_endpoint_buffer(bms_state, key_fit, args, dim, args.fit_samples))
        bms_samples_eval = np.asarray(refresh_endpoint_buffer(bms_state, key_bms_eval, args, dim, args.eval_samples))
        if 'pure_bms' in methods:
            metrics = evaluate_samples_and_density(rng,
                                                   target,
                                                   bms_samples_eval,
                                                   gmm=None,
                                                   target_samples=target_eval_samples,
                                                   n_proj=args.sliced_projections,
                                                   bins=args.sliced_bins,
                                                   kl_samples=args.eval_samples)
            row = base_row(args, 'pure_bms', dim, seed, False, '')
            row.update(metrics)
            row['train_time_sec'] = bms_train_time
            rows.append(row)
            if args.save_plots and dim == 2:
                samples_for_plot['pure_bms'] = bms_samples_eval[:args.plot_samples]

    initial_our_gmm = make_initial_our_gmm(args, key_our, dim)

    for m in methods:
        if m == 'our_gmm' or m == 'our_gmm_iw':
            use_iw = (m == 'our_gmm_iw')
            print(f'=== Training {m} dim={dim} seed={seed} ===')
            offset = 777 if use_iw else 0
            state, train_time, _ = train_our_gmm(args,
                                                 jax.random.fold_in(key_our, offset),
                                                 target,
                                                 dim,
                                                 use_importance_weights=use_iw)
            gmm = gmm_params_to_np(state.params, args.our_var_floor)
            samples = sample_diag_gmm_np(rng, gmm, args.eval_samples)
            metrics = evaluate_samples_and_density(rng,
                                                   target,
                                                   samples,
                                                   gmm=gmm,
                                                   target_samples=target_eval_samples,
                                                   n_proj=args.sliced_projections,
                                                   bins=args.sliced_bins,
                                                   kl_samples=args.eval_samples)
            row = base_row(args, m, dim, seed, use_iw, '')
            row.update(metrics)
            row['train_time_sec'] = train_time
            rows.append(row)
            if args.save_plots and dim == 2:
                samples_for_plot[m] = samples[:args.plot_samples]
                gmms_for_plot[m] = gmm

        elif m == 'bms_posthoc_kmeans' or m == 'bms_posthoc_ourinit':
            if bms_samples_fit is None:
                raise RuntimeError(f'{m} requires BMS training')
            init = 'kmeanspp' if m == 'bms_posthoc_kmeans' else 'our_init'
            print(f'=== Fitting {m} dim={dim} seed={seed} init={init} ===')
            t0 = time.time()
            gmm = fit_diag_gmm_em(bms_samples_fit,
                                  args.gmm_components,
                                  rng,
                                  iters=args.em_iters,
                                  var_floor=args.em_var_floor,
                                  init=init,
                                  init_gmm=initial_our_gmm)
            fit_time = time.time() - t0
            samples = sample_diag_gmm_np(rng, gmm, args.eval_samples)
            metrics = evaluate_samples_and_density(rng,
                                                   target,
                                                   samples,
                                                   gmm=gmm,
                                                   target_samples=target_eval_samples,
                                                   n_proj=args.sliced_projections,
                                                   bins=args.sliced_bins,
                                                   kl_samples=args.eval_samples)
            row = base_row(args, m, dim, seed, False, init)
            row.update(metrics)
            row['train_time_sec'] = bms_train_time
            row['em_fit_time_sec'] = fit_time
            rows.append(row)
            if args.save_plots and dim == 2:
                samples_for_plot[m] = samples[:args.plot_samples]
                gmms_for_plot[m] = gmm

        elif m == 'oracle_em':
            print(f'=== Fitting oracle EM dim={dim} seed={seed} ===')
            t0 = time.time()
            x = sample_target_np(rng, target, args.fit_samples)
            gmm = fit_diag_gmm_em(x,
                                  args.gmm_components,
                                  rng,
                                  iters=args.em_iters,
                                  var_floor=args.em_var_floor,
                                  init='kmeanspp')
            fit_time = time.time() - t0
            samples = sample_diag_gmm_np(rng, gmm, args.eval_samples)
            metrics = evaluate_samples_and_density(rng,
                                                   target,
                                                   samples,
                                                   gmm=gmm,
                                                   target_samples=target_eval_samples,
                                                   n_proj=args.sliced_projections,
                                                   bins=args.sliced_bins,
                                                   kl_samples=args.eval_samples)
            row = base_row(args, m, dim, seed, False, 'target_samples')
            row.update(metrics)
            row['em_fit_time_sec'] = fit_time
            rows.append(row)
            if args.save_plots and dim == 2:
                samples_for_plot[m] = samples[:args.plot_samples]
                gmms_for_plot[m] = gmm

        elif m == 'weighted_em_broad':
            print(f'=== Fitting weighted EM broad proposal dim={dim} seed={seed} ===')
            t0 = time.time()
            x = args.broad_proposal_std * rng.normal(size=(args.fit_samples, dim)).astype(np.float32)
            logp = target_log_prob_np(x, target)
            logq = -0.5 * np.sum(np.log(2 * np.pi * args.broad_proposal_std**2) + x**2 / (args.broad_proposal_std**2),
                                 axis=1)
            lw = logp - logq
            lw = lw - np.max(lw)
            w = np.exp(np.maximum(lw, -args.max_log_weight_span))
            w = w / (w.sum() + 1e-300)
            gmm = fit_diag_gmm_em(x,
                                  args.gmm_components,
                                  rng,
                                  iters=args.em_iters,
                                  var_floor=args.em_var_floor,
                                  init='kmeanspp',
                                  sample_weights=w)
            fit_time = time.time() - t0
            samples = sample_diag_gmm_np(rng, gmm, args.eval_samples)
            metrics = evaluate_samples_and_density(rng,
                                                   target,
                                                   samples,
                                                   gmm=gmm,
                                                   target_samples=target_eval_samples,
                                                   n_proj=args.sliced_projections,
                                                   bins=args.sliced_bins,
                                                   kl_samples=args.eval_samples)
            row = base_row(args, m, dim, seed, False, 'broad_weighted')
            row.update(metrics)
            row['em_fit_time_sec'] = fit_time
            row['proposal_ess_frac'] = float(1.0 / (np.sum(w**2) + 1e-300) / len(w))
            rows.append(row)
            if args.save_plots and dim == 2:
                samples_for_plot[m] = samples[:args.plot_samples]
                gmms_for_plot[m] = gmm

    if args.save_plots and dim == 2 and samples_for_plot:
        plot_dir = Path(args.out_dir) / 'plots'
        ensure_dir(plot_dir)
        plot_2d(args, target, samples_for_plot, gmms_for_plot, plot_dir / f'dim{dim}_seed{seed}.png')

    return rows


def base_row(args, method, dim, seed, importance_weights, em_init):
    return {
        'method': method,
        'dim': dim,
        'seed': seed,
        'target_modes': args.target_modes,
        'mode_scale': args.mode_scale,
        'target_var': args.target_var,
        'gmm_components': args.gmm_components,
        'importance_weights': bool(importance_weights),
        'em_init': em_init,
        'steps': args.steps,
        'our_steps': args.our_steps,
        'fit_samples': args.fit_samples,
        'eval_samples': args.eval_samples,
        'sigma': args.sigma,
        'our_sigma': args.our_sigma,
        'eta': args.eta,
        'our_eta': args.our_eta,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', type=str, default='tmp/gmm_bms_paper_experiments')
    p.add_argument('--seeds', type=str, default='0,1,2,3,4')
    p.add_argument('--dims', type=str, default='2')
    p.add_argument(
        '--methods',
        type=str,
        default='pure_bms,bms_posthoc_kmeans,bms_posthoc_ourinit,our_gmm,oracle_em',
        help=
        'Comma-separated methods: pure_bms,bms_posthoc_kmeans,bms_posthoc_ourinit,our_gmm,our_gmm_iw,oracle_em,weighted_em_broad'
    )
    p.add_argument('--save_plots', action='store_true')
    p.add_argument('--verbose', action='store_true')

    # Target
    p.add_argument('--target_modes', type=int, default=8)
    p.add_argument('--mode_scale', type=float, default=20.0)
    p.add_argument('--target_var', type=float, default=1.0)
    p.add_argument('--target_weights', choices=['uniform', 'dirichlet'], default='uniform')
    p.add_argument('--target_seed_offset', type=int, default=12345)

    # Pure BMS
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--time_freqs', type=int, default=8)
    p.add_argument('--sigma', type=float, default=2.5)
    p.add_argument('--prior_std', type=float, default=20.0)
    p.add_argument('--sde_steps', type=int, default=100)
    p.add_argument('--eta', type=float, default=10.0)
    p.add_argument('--outer_every', type=int, default=250)
    p.add_argument('--refresh_every', type=int, default=500)
    p.add_argument('--buffer_size', type=int, default=30000)
    p.add_argument('--sample_batch', type=int, default=4096)
    p.add_argument('--t_eps', type=float, default=2e-2)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--state_clip', type=float, default=1e4)
    p.add_argument('--control_clip', type=float, default=1e3)
    p.add_argument('--xi_clip', type=float, default=1e4)
    p.add_argument('--log_every', type=int, default=250)

    # Our method
    p.add_argument('--gmm_components', type=int, default=32)
    p.add_argument('--our_steps', type=int, default=20000)
    p.add_argument('--our_batch_size', type=int, default=4096)
    p.add_argument('--our_lr', type=float, default=1e-3)
    p.add_argument('--our_sigma', type=float, default=1.0)
    p.add_argument('--our_prior_var', type=float, default=1.0)
    p.add_argument('--our_t_eps', type=float, default=2e-2)
    p.add_argument('--our_xi_clip', type=float, default=1e12)
    p.add_argument('--our_grad_clip', type=float, default=10.0)
    p.add_argument('--our_eta', type=float, default=0.1)
    p.add_argument('--our_init_mean_scale', type=float, default=12.0)
    p.add_argument('--our_init_terminal_std', type=float, default=2.5)
    p.add_argument('--our_var_floor', type=float, default=1e-3)
    p.add_argument('--our_entropy_coef', type=float, default=1e-3)
    p.add_argument('--our_var_reg', type=float, default=1e-4)
    p.add_argument('--our_max_log_weight_span', type=float, default=20.0)

    # EM / eval
    p.add_argument('--fit_samples', type=int, default=50000)
    p.add_argument('--em_iters', type=int, default=100)
    p.add_argument('--em_var_floor', type=float, default=1e-3)
    p.add_argument('--eval_samples', type=int, default=20000)
    p.add_argument('--sliced_projections', type=int, default=100)
    p.add_argument('--sliced_bins', type=int, default=50)
    p.add_argument('--broad_proposal_std', type=float, default=20.0)
    p.add_argument('--max_log_weight_span', type=float, default=20.0)

    # 2D plot
    p.add_argument('--plot_samples', type=int, default=30000)
    p.add_argument('--plot_lim', type=float, default=25.0)
    p.add_argument('--grid_n', type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    dims = parse_int_list(args.dims)
    seeds = parse_int_list(args.seeds)
    methods = parse_method_list(args.methods)
    ensure_dir(args.out_dir)
    with open(Path(args.out_dir) / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    print('JAX devices:', jax.devices())
    print('Output:', args.out_dir)
    print('Methods:', methods)
    print('Dims:', dims, 'Seeds:', seeds)

    all_rows = []
    for dim in dims:
        for seed in seeds:
            print('=' * 80)
            print(f'Running dim={dim} seed={seed}')
            rows = run_one(args, dim, seed, methods)
            all_rows.extend(rows)
            write_csv(Path(args.out_dir) / 'per_run_results.csv', all_rows)
            write_csv(Path(args.out_dir) / 'summary_results.csv', summarize_rows(all_rows))

    write_csv(Path(args.out_dir) / 'per_run_results.csv', all_rows)
    summary = summarize_rows(all_rows)
    write_csv(Path(args.out_dir) / 'summary_results.csv', summary)
    print('\nSummary rows:')
    for r in summary:
        print(r)


if __name__ == '__main__':
    main()
