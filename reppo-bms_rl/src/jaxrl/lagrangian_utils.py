import jax
import jax.numpy as jnp
from typing import Callable, Tuple
from functools import partial
import time

import numpy as np
from jaxopt import ScipyBoundedMinimize, LBFGS, LBFGSB


def bracket_search_minimizer(
        fn: Callable[[jnp.ndarray], jnp.ndarray],
        lower_bound: float,
        upper_bound: float,
        max_iterations: int = 100,
        parallel_evals: int = 10000,
        tolerance: float = 1e-6
) -> Tuple[float, float]:
    """
    Implements a bracket search minimization algorithm using JAX.

    Args:
        fn: Function to minimize
        lower_bound: Lower bound of search interval
        upper_bound: Upper bound of search interval
        max_iterations: Maximum number of iterations
        parallel_evals: Number of parallel evaluations per step
        tolerance: Convergence tolerance

    Returns:
        Tuple of (minimizer_x, minimum_value)
    """

    @partial(jax.jit, static_argnums=(0,))
    def search_step(fn, lb, ub):
        eval_points = jnp.linspace(lb, ub, parallel_evals, endpoint=True)
        fn_vals = jax.vmap(fn)(eval_points[:, None])
        min_idx = jnp.argmin(fn_vals)

        # Get new bounds centered around minimum
        # interval_width = (ub - lb) / parallel_evals
        new_lb = eval_points[jnp.maximum(0, min_idx - 1)]
        new_ub = eval_points[jnp.minimum(parallel_evals - 1, min_idx + 1)]

        return new_lb, new_ub, eval_points[min_idx], fn_vals[min_idx]

    def minimizer_loop(fn, lb, ub):
        def cond_fun(state):
            i, lb, ub, best_x, best_val = state
            return (i < max_iterations) & ((ub - lb) > tolerance)

        def body_fun(state):
            i, lb, ub, best_x, best_val = state
            new_lb, new_ub, x, val = search_step(fn, lb, ub)
            return i + 1, new_lb, new_ub, x, val

        initial_state = (0, lb, ub, (lb + ub) / 2, fn(jnp.array([(lb + ub) / 2])))
        final_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)
        return final_state[3], final_state[4]  # best_x, best_val

    return minimizer_loop(fn, lower_bound, upper_bound)


def compute_tr_lm(log_weights, dual_fn):
    opt_lm, _ = bracket_search_minimizer(lambda lm: dual_fn(lm, log_weights), lower_bound=0,
                                         upper_bound=10000)
    return opt_lm


def tr_dual_fn(lm, log_w, tr_bound):
    scaled_path_log_weight = (1. / (1. + lm)) * log_w
    dual_val = (1. + lm) * (
            jax.scipy.special.logsumexp(scaled_path_log_weight) - jnp.log(log_w.shape[0])) + lm * tr_bound
    return dual_val.squeeze()

def compute_ent_lm(Q_vals, model_log_prob, dual_fn):
    opt_lm, _ = bracket_search_minimizer(lambda lm: dual_fn(lm, Q_vals, model_log_prob), lower_bound=0,
                                         upper_bound=1e20)
    return opt_lm

def ent_dual_fn(lm, Q_vals, model_log_prob, ent_bound):
    tempered_Q_vals =  (1. / (1. + lm)) * Q_vals
    log_weights = tempered_Q_vals - model_log_prob
    dual_val = (1. + lm) *  (jax.scipy.special.logsumexp(log_weights) - jnp.log(Q_vals.shape[0])) - lm * ent_bound
    return dual_val.squeeze()



# def compute_tr_ent_lm(Q_vals, log_w, ent_bound, dual_fn):
#     # Optional: stop grads if you don't want to backprop through Q/log_w
#     log_w = jax.lax.stop_gradient(log_w)
#     Q_vals = jax.lax.stop_gradient(Q_vals)
#
#     # dual_fn is assumed to have signature dual_fn(lms, log_w=..., Q_vals=..., ent_bound=...)
#     def dual(lms, log_w, Q_vals, ent_bound):
#         return dual_fn(lms, log_w=log_w, Q_vals=Q_vals, ent_bound=ent_bound)
#
#     # JAX-native L-BFGS-B solver with box constraints
#     lbfgsb = LBFGSB(fun=dual, maxiter=100)
#
#     # Initial guess
#     lms0 = 10*jnp.ones(2)
#
#     # Box constraints
#     lower_bounds = jnp.ones(2) * 1e-6
#     upper_bounds = jnp.ones(2) * 1e8
#     bounds = (lower_bounds, upper_bounds)
#
#     # IMPORTANT: bounds is passed as first *arg to run() (then extra args to fun)
#     opt_lms, state = lbfgsb.run(lms0, bounds, log_w, Q_vals, ent_bound)
#
#     dual_val = state.value
#     opt_lm_tr = opt_lms[0]
#     opt_lm_ent = opt_lms[1]
#     return opt_lm_tr, opt_lm_ent, dual_val, state


def compute_tr_ent_lm(Q_vals, log_w, log_p0, ent_bound, ent_lb_bound, dual_fn):
    # Optional: stop grads if you don't want to backprop through Q/log_w
    log_w = jax.lax.stop_gradient(log_w)
    Q_vals = jax.lax.stop_gradient(Q_vals)

    # dual_fn is assumed to have signature dual_fn(lms, log_w=..., Q_vals=..., ent_bound=...)
    def dual(lms, log_w, Q_vals, log_p0, ent_bound, ent_lb_bound):
        return dual_fn(lms, log_w=log_w, Q_vals=Q_vals, log_p0=log_p0, ent_bound=ent_bound, ent_lb_bound=ent_lb_bound)

    # JAX-native L-BFGS-B solver with box constraints
    lbfgsb = LBFGSB(fun=dual, maxiter=100)

    # Initial guess
    lms0 = 10*jnp.ones(3)
    # lms0 = 10*jnp.ones(2)
    # lms0 = 10*jnp.ones(1)

    # Box constraints
    # lower_bounds = jnp.array([1.0e-4, 1.0e-4])
    # upper_bounds = jnp.array([1.0e6, 1.0e6])
    lower_bounds = jnp.array([1.0e-4, 1.0e-4, 1.0e-4])
    upper_bounds = jnp.ones(3) * 1e20
    bounds = (lower_bounds, upper_bounds)

    # IMPORTANT: bounds is passed as first *arg to run() (then extra args to fun)
    opt_lms, state = lbfgsb.run(lms0, bounds, log_w, Q_vals, log_p0, ent_bound, ent_lb_bound)

    dual_val = state.value
    opt_lm_tr = opt_lms[0]
    # opt_lm_tr = 0.0
    opt_lm_ent = opt_lms[1]
    opt_lm_ent_lb = opt_lms[2]
    # opt_lm_ent = 0.0
    return opt_lm_tr, opt_lm_ent, opt_lm_ent_lb, dual_val, state

# def tr_ent_dual_fn(lms, log_w, Q_vals, log_p0, ent_bound, kl_bound):
#     lm_tr = lms[0]
#     lm_ent = lms[1]
#     # lm_ent = 0.0
#     weights = log_w + log_p0
#     tempered_Q_vals = (1. / (1. + lm_tr + lm_ent)) * Q_vals
#     # tempered_weights = ((lm_ent+1) / (1. + lm_tr + lm_ent)) * log_w
#     tempered_weights = ((lm_ent+1) / (1. + lm_tr + lm_ent)) * weights
#     # tempered_weights = ((lm_ent+1) / (1. + lm_tr + lm_ent)) * log_w + ((lm_ent+1) / (1. + lm_tr + lm_ent))*log_p0
#     work_functional = tempered_Q_vals + tempered_weights
#     log_sum_exp = jax.scipy.special.logsumexp(work_functional)
#     dual_val = ((1. + lm_tr + lm_ent) * (log_sum_exp - jnp.log(Q_vals.shape[0]))) + lm_tr * kl_bound + lm_ent * ent_bound
#     dual_val -= (lm_ent+1) * log_p0.mean()
#     # jax.debug.print(
#     #     "lm_ent={:.4e}, logsumexp={:.6f}",
#     #     lm_ent,
#     #     log_sum_exp,
#     # )
#     # jax.debug.print(
#     #     "dual_val={}, lm_ent={}, diff={}, scaled_diff={}, lm_ent_ent_bound={}",
#     #     dual_val,
#     #     lm_ent,
#     #     log_sum_exp - jnp.log(Q_vals.shape[0]),
#     #     (1. + lm_tr + lm_ent) * (log_sum_exp - jnp.log(Q_vals.shape[0])),
#     #     lm_ent * ent_bound
#     # )
#     return dual_val.squeeze()



# def tr_ent_dual_fn(lms, log_w, Q_vals, log_p0, ent_bound, ent_lb_bound, kl_bound):
#     lm_tr = lms[0]
#     lm_ent = lms[1]
#     lm_ent_lb = lms[2]
#     # lm_ent = 0.0
#     weights = log_w + log_p0
#     tempered_Q_vals = (1. / (1. + lm_tr + lm_ent + lm_ent_lb)) * Q_vals
#     # tempered_weights = ((lm_ent+1) / (1. + lm_tr + lm_ent)) * log_w
#     tempered_weights = ((lm_ent + lm_ent_lb + 1) / (1. + lm_tr + lm_ent+ lm_ent_lb)) * weights
#     # tempered_weights = ((lm_ent+1) / (1. + lm_tr + lm_ent)) * log_w + ((lm_ent+1) / (1. + lm_tr + lm_ent))*log_p0
#     work_functional = tempered_Q_vals + tempered_weights
#     log_sum_exp = jax.scipy.special.logsumexp(work_functional)
#     dual_val = ((1. + lm_tr + lm_ent+ lm_ent_lb) * (log_sum_exp - jnp.log(Q_vals.shape[0]))) + lm_tr * kl_bound + lm_ent * ent_bound + lm_ent_lb * ent_lb_bound
#     dual_val -= (lm_ent+lm_ent_lb+1) * log_p0.mean()
#     # jax.debug.print(
#     #     "lm_ent={:.4e}, logsumexp={:.6f}",
#     #     lm_ent,
#     #     log_sum_exp,
#     # )
#     # jax.debug.print(
#     #     "dual_val={}, lm_ent={}, diff={}, scaled_diff={}, lm_ent_ent_bound={}",
#     #     dual_val,
#     #     lm_ent,
#     #     log_sum_exp - jnp.log(Q_vals.shape[0]),
#     #     (1. + lm_tr + lm_ent) * (log_sum_exp - jnp.log(Q_vals.shape[0])),
#     #     lm_ent * ent_bound
#     # )
#     return dual_val.squeeze()



def tr_ent_dual_fn(lms, log_w, Q_vals, log_p0, ent_bound, ent_lb_bound, kl_bound):
    lm_tr = lms[0]
    lm_ent = lms[1]
    lm_ent_lb = lms[2]
    # lm_ent = 0.0
    weights = log_w # + log_p0
    tempered_Q_vals = (1. / (lm_tr + lm_ent + lm_ent_lb)) * Q_vals
    tempered_weights = ((lm_ent+ lm_ent_lb) / (lm_tr + lm_ent+ lm_ent_lb)) * weights
    work_functional = tempered_Q_vals + tempered_weights
    log_sum_exp = jax.scipy.special.logsumexp(work_functional)
    # dual_val = ((lm_tr + lm_ent + lm_ent_lb) * (log_sum_exp - jnp.log(Q_vals.shape[0]))) + lm_tr * kl_bound + lm_ent * ent_bound + lm_ent_lb * ent_lb_bound - (lm_ent_lb+lm_ent_lb)*jnp.log(Q_vals.shape[0])
    dual_val = ((lm_tr + lm_ent + lm_ent_lb) * (log_sum_exp - jnp.log(Q_vals.shape[0]))) + lm_tr * kl_bound + lm_ent * ent_bound + lm_ent_lb * ent_lb_bound - (lm_ent_lb+lm_ent_lb)*jnp.log(2)
    #dual_val -= (lm_ent+lm_ent_lb) * log_p0.mean()
    # jax.debug.print(
    #         "dual_val={}, lm_kl={}, lm_ent={}, lm_ent_lb={}, ent_bound={}, ent_lb_bound={}",
    #         dual_val,
    #         lm_tr,
    #         lm_ent,
    #         lm_ent_lb,
    #         ent_bound,
    #         ent_lb_bound
    #     )
    return dual_val.squeeze()

# def tr_ent_dual_fn(lms, log_w, Q_vals, log_p0, ent_bound, ent_lb_bound, kl_bound):
#     lm_ent = lms[0]
#     lm_ent_lb = lms[1]
#     tempered_Q_vals = (1. / (lm_ent + lm_ent_lb)) * Q_vals
#     work_functional = tempered_Q_vals
#     log_sum_exp = jax.scipy.special.logsumexp(work_functional)
#     # dual_val = ((lm_tr + lm_ent + lm_ent_lb) * (log_sum_exp - jnp.log(Q_vals.shape[0]))) + lm_tr * kl_bound + lm_ent * ent_bound + lm_ent_lb * ent_lb_bound - (lm_ent_lb+lm_ent_lb)*jnp.log(Q_vals.shape[0])
#     dual_val = ((lm_ent + lm_ent_lb) * (log_sum_exp - jnp.log(Q_vals.shape[0])))+ lm_ent * ent_bound + lm_ent_lb * ent_lb_bound - (lm_ent_lb+lm_ent_lb)*jnp.log(2)
#     #dual_val -= (lm_ent+lm_ent_lb) * log_p0.mean()
#     # jax.debug.print(
#     #         "dual_val={}, lm_ent={}, lm_ent_lb={}, ent_bound={}, ent_lb_bound={}",
#     #         dual_val,
#     #         lm_ent,
#     #         lm_ent_lb,
#     #         ent_bound,
#     #         ent_lb_bound
#     #     )
#     return dual_val.squeeze()



# def tr_ent_dual_fn(lms, log_w, Q_vals, ent_bound, kl_bound):
#     lm_tr = lms[0]
#     lm_ent = lms[1]
#     tempered_Q_vals = (1. / (1. + lm_tr + lm_ent)) * Q_vals
#     tempered_weights = ((lm_ent+1) / (1. + lm_tr + lm_ent)) * log_w
#     work_functional = tempered_Q_vals + tempered_weights
#     log_sum_exp = jax.scipy.special.logsumexp(work_functional)
#     dual_val = ((1. + lm_tr + lm_ent) * (log_sum_exp - jnp.log(Q_vals.shape[0]))) + lm_tr * kl_bound + lm_ent * ent_bound
#     return dual_val.squeeze()


def reciprocal_tr_ent_dual_fn(lms, w_t, old_ctrl, ctrl_target, log_p_T_ref, gamma, kl_bound, norm_weights=True):
    lmbda = lms[0]
    eta = lms[1]
    eps=kl_bound
    ctrl_target_norm = jnp.sum(ctrl_target**2,axis=-1)
    old_ctrl_norm = jnp.sum(old_ctrl**2,axis=-1)
    ctrl_diff_norm = jnp.sum((old_ctrl-ctrl_target)**2,axis=-1)
    dual = (1/(2*(w_t+lmbda+eta)))*(lmbda*w_t*ctrl_diff_norm + lmbda*eta*old_ctrl_norm + w_t*ctrl_target_norm) + eta*log_p_T_ref.squeeze() -lmbda*eps +eta*gamma
    return -jnp.mean(dual)


def compute_reciprocal_tr_ent_lm(w_t, old_ctrl, ctrl_target, log_p_T_ref, gamma, dual_fn):
    def dual(lms, w_t, old_ctrl, ctrl_target, gamma):
        return dual_fn(lms, w_t=w_t, old_ctrl=old_ctrl, ctrl_target=ctrl_target, log_p_T_ref=log_p_T_ref, gamma=gamma)

    # JAX-native L-BFGS-B solver with box constraints
    lbfgsb = LBFGSB(fun=dual, maxiter=10000)

    # Initial guess
    lms0 = 10*jnp.ones(2)

    # Box constraints
    lower_bounds = jnp.array([1.0e-6, 1.0e-6])
    upper_bounds = jnp.array([1.0e6, 1.0e6])
    bounds = (lower_bounds, upper_bounds)

    # IMPORTANT: bounds is passed as first *arg to run() (then extra args to fun)
    opt_lms, state = lbfgsb.run(lms0, bounds, w_t, old_ctrl, ctrl_target, gamma)

    dual_val = state.value
    opt_lm_tr = opt_lms[0]
    opt_lm_ent = opt_lms[1]
    return opt_lm_tr, opt_lm_ent, dual_val, state



import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=['num_iters'])
def find_alpha_bisection(
        q_values: jnp.ndarray,
        behavior_log_weights: jnp.ndarray,
        target_entropy: float,
        min_log_alpha: float = -5.0,  # T approx 148
        max_log_alpha: float = 25.0,  # T approx 1e-11
        num_iters: int = 30  # 30 iters reduces error by factor of ~1 billion
):
    """
    Finds alpha using Bisection Search.
    Guaranteed convergence for monotonic functions.
    """

    def compute_entropy(log_alpha):
        alpha = jnp.exp(log_alpha)

        # --- Stability Shift ---
        q_max = jnp.max(q_values)
        shifted_Q = q_values - q_max

        # log_w = alpha * shifted_Q - log_pi
        log_w_shifted = alpha * shifted_Q - behavior_log_weights
        log_Z_shifted = jax.scipy.special.logsumexp(log_w_shifted)

        # Normalized weights
        log_norm_weights = log_w_shifted - log_Z_shifted
        normalized_weights = jnp.exp(log_norm_weights)

        # Reconstruct Log Probs
        # log_prob = alpha * Q - log_Z
        log_probs_target = alpha * shifted_Q - log_Z_shifted + jnp.log(q_values.shape[0])

        # Mask tiny weights to prevent 0 * -inf
        valid_mask = normalized_weights > 1e-30
        entropy_terms = jnp.where(valid_mask, normalized_weights * log_probs_target, 0.0)

        return -jnp.sum(entropy_terms)

    def body_fn(i, val):
        low, high = val
        mid = (low + high) / 2.0

        current_ent = compute_entropy(mid)

        # Logic for Monotonic Decreasing function (Higher Alpha -> Lower Entropy)
        # If Current Ent > Target -> We have too much entropy -> Need higher alpha -> Move Low up
        # If Current Ent < Target -> We have too little entropy -> Need lower alpha -> Move High down

        # We use jnp.where to differentiably update bounds
        # condition: current_ent > target_entropy
        condition = current_ent > target_entropy

        new_low = jnp.where(condition, mid, low)
        new_high = jnp.where(condition, high, mid)

        return (new_low, new_high)

    # Initial bounds
    init_val = (min_log_alpha, max_log_alpha)

    # Run Bisection
    final_low, final_high = jax.lax.fori_loop(0, num_iters, body_fn, init_val)

    # Result is the midpoint of the final bounds
    final_log_alpha = (final_low + final_high) / 2.0

    # Calculate final error for logging
    final_entropy = compute_entropy(final_log_alpha)
    final_diff = final_entropy - target_entropy

    return jnp.exp(final_log_alpha), final_diff


# Example usage
def example():
    # Define a simple function to minimize (x^2 + 2x + 1)
    @jax.jit
    def objective_fn(x):
        return x ** 2 + 2 * x + 1

    # Initialize random key
    key = jax.random.PRNGKey(0)

    # Run minimizer
    x_min, f_min = bracket_search_minimizer(
        objective_fn,
        lower_bound=-5.0,
        upper_bound=5.0,
        max_iterations=50,
        parallel_evals=1000,
        tolerance=1e-6
    )

    print(f"Found minimum at x = {x_min:.6f}")
    print(f"Minimum value = {f_min}")
    print(f"True minimum at x = -1.0")  # Analytical solution for this quadratic


if __name__ == "__main__":
    with jax.disable_jit():
        example()