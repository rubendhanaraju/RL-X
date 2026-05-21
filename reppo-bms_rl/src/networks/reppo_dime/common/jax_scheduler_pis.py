from typing import NamedTuple, Callable

import jax.numpy as jnp


class NoiseScheduler(NamedTuple):
    sigma_t: Callable
    sigma_t_0: Callable
    sigma_T_0: Callable
    sigma_t_T: Callable
    sigma_t_0T: Callable
    mu_t_0T_scale: Callable
    xT_0T_scale: Callable
    x0_0T_scale: Callable
    noise_0T_scale: Callable


def get_constant_scheduler(sigma):
    def sigma_t(t):
        return sigma

    def sigma_t_0(t):
        return sigma * jnp.sqrt(t)

    def sigma_T_0():
        return sigma

    def sigma_t_T(t):
        return jnp.sqrt(sigma_t_0(1.0) ** 2 - sigma_t_0(t) ** 2)

    def sigma_t_0T(t):
        return sigma * jnp.sqrt(t * (1 - t))

    def mu_t_0T_scale(t):
        return sigma_t_0(t) ** 2 / sigma_T_0() ** 2

    def xT_0T_scale(t):
        return sigma_t_0(t) ** 2 / sigma_T_0() ** 2

    def x0_0T_scale(t):
        return 1 - xT_0T_scale(t)

    def noise_0T_scale(t):
        sigma_t_0_val = sigma_t_0(t) ** 2
        return jnp.sqrt(sigma_t_0_val * (sigma_T_0() ** 2 - sigma_t_0_val) / sigma_T_0() ** 2)

    return NoiseScheduler(
        sigma_t=sigma_t,
        sigma_t_0=sigma_t_0,
        sigma_T_0=sigma_T_0,
        sigma_t_T=sigma_t_T,
        sigma_t_0T=sigma_t_0T,
        mu_t_0T_scale=mu_t_0T_scale,
        xT_0T_scale=xT_0T_scale,
        x0_0T_scale=x0_0T_scale,
        noise_0T_scale=noise_0T_scale,
    )


def get_cosine_sq_scheduler(sigma_max, sigma_min):
    a = sigma_max - sigma_min
    b = jnp.pi / 2
    c = sigma_min

    def sigma_t(t):
        return a * jnp.cos(b * t) ** 2 + c

    def sigma_t_0(t):
        term1 = a**2 * (3 * t / 8 + jnp.sin(2 * b * t) / (4 * b) + jnp.sin(4 * b * t) / (32 * b))
        term2 = 2 * a * c * (t / 2 + jnp.sin(2 * b * t) / (4 * b))
        term3 = c**2 * t
        return jnp.sqrt(term1 + term2 + term3)

    sigma_T_0_ = sigma_t_0(1.0)

    def sigma_T_0():
        return sigma_T_0_

    def sigma_t_T(t):
        return jnp.sqrt(sigma_t_0(1.0) ** 2 - sigma_t_0(t) ** 2)

    def sigma_t_0T(t):
        sigma_t_0_ = sigma_t_0(t)
        sigma_ratio = sigma_t_0_ / sigma_T_0_
        return sigma_t_0_ * jnp.sqrt(1 - sigma_ratio**2)

    def mu_t_0T_scale(t):
        return sigma_t_0(t) ** 2 / sigma_T_0() ** 2

    def xT_0T_scale(t):
        return sigma_t_0(t) ** 2 / sigma_T_0() ** 2

    def x0_0T_scale(t):
        return 1 - xT_0T_scale(t)

    def noise_0T_scale(t):
        sigma_t_0_val = sigma_t_0(t) ** 2
        return jnp.sqrt(sigma_t_0_val * (sigma_T_0() ** 2 - sigma_t_0_val) / sigma_T_0() ** 2)

    return NoiseScheduler(
        sigma_t=sigma_t,
        sigma_t_0=sigma_t_0,
        sigma_T_0=sigma_T_0,
        sigma_t_T=sigma_t_T,
        sigma_t_0T=sigma_t_0T,
        mu_t_0T_scale=mu_t_0T_scale,
        xT_0T_scale=xT_0T_scale,
        x0_0T_scale=x0_0T_scale,
        noise_0T_scale=noise_0T_scale,
    )


def get_geometric_scheduler(sigma_max, sigma_min):
    sigma_min = sigma_min
    sigma_max = sigma_max
    r = sigma_max / sigma_min
    log_r = jnp.log(r)

    def sigma_t(t):
        return sigma_min * r ** (1 - t) * jnp.sqrt(2 * log_r)

    def sigma_t_0(t):  # cumulative std from t=0 to t
        return jnp.sqrt(sigma_max**2 * (1 - r ** (-2 * t)))

    sigma_T_0_ = sigma_t_0(1.0)

    def sigma_T_0():
        return sigma_T_0_

    def sigma_t_T(t):
        return jnp.sqrt(sigma_t_0(1.0) ** 2 - sigma_t_0(t) ** 2)

    def sigma_t_0T(t):  # STD at time step t given 0 and T
        sigma_t_0_ = sigma_t_0(t)
        sigma_ratio = sigma_t_0_ / sigma_T_0_
        return sigma_t_0_ * jnp.sqrt(1 - sigma_ratio**2)

    def mu_t_0T_scale(t):  # Mean von p(t|0,T)
        return sigma_t_0(t) ** 2 / sigma_T_0() ** 2

    def xT_0T_scale(t):
        return sigma_t_0(t) ** 2 / sigma_T_0() ** 2

    def x0_0T_scale(t):
        return 1 - xT_0T_scale(t)

    def noise_0T_scale(t):
        sigma_t_0_val = sigma_t_0(t) ** 2
        return jnp.sqrt(sigma_t_0_val * (sigma_T_0() ** 2 - sigma_t_0_val) / sigma_T_0() ** 2)

    return NoiseScheduler(
        sigma_t=sigma_t,
        sigma_t_0=sigma_t_0,
        sigma_T_0=sigma_T_0,
        sigma_t_T=sigma_t_T,
        sigma_t_0T=sigma_t_0T,
        mu_t_0T_scale=mu_t_0T_scale,
        xT_0T_scale=xT_0T_scale,
        x0_0T_scale=x0_0T_scale,
        noise_0T_scale=noise_0T_scale,
    )


if __name__ == "__main__":

    from scipy.integrate import quad
    import matplotlib.pyplot as plt

    # Numerical integration using scipy.quad
    def integrated_scheduler_numerical(t, schedule):
        result, _ = quad(lambda t: schedule(t) ** 2, 0, t)
        return jnp.sqrt(result)

    # scheduler = get_constant_scheduler(3.)
    # scheduler = get_cosine_sq_scheduler(1., 0.1)
    scheduler = get_geometric_scheduler(3.0, 0.001)

    # Evaluate over time
    ts = jnp.linspace(0, 1.0, 100)
    sigma_sq_analytical = jnp.array([scheduler.sigma_t_0(t) ** 2 for t in ts])
    sigma_sq_numerical = jnp.array([integrated_scheduler_numerical(t, scheduler.sigma_t) ** 2 for t in ts])
    abs_error = jnp.abs(sigma_sq_analytical - sigma_sq_numerical)

    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(ts, sigma_sq_analytical, label="Analytical Σ(t)", lw=2)
    plt.plot(ts, sigma_sq_numerical, "--", label="Numerical Σ(t) (quad)", lw=2)
    plt.plot(ts, scheduler.sigma_t(ts), "--", label="Schedule", lw=2)
    plt.fill_between(ts, sigma_sq_analytical, sigma_sq_numerical, color="gray", alpha=0.3)
    plt.xlabel("t")
    plt.ylabel("Σ(t)")
    plt.title("Comparison of Analytical vs Numerical Integration")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print max error
    print(f"Max absolute error: {jnp.max(abs_error):.6e}")
