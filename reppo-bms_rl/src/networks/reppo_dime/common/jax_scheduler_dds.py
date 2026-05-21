import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

def create_cosine_quartic_scheduler(C_start, C_end, T):
    """
    Create a cosine-quartic (cosine^4) monotonically decreasing scheduler.

    Args:
        C_start (float): Starting diffusion coefficient
        C_end (float): Ending diffusion coefficient
        T (float): Total time duration

    Returns:
        Tuple of three functions:
        1. Scheduler function that returns the diffusion coefficient at time t
        2. Forward integrated scheduler function that returns \\int_0^s scheduler(t) dt
        3. Backward integrated scheduler function that returns \\int_s^T scheduler(t) dt
    """

    def scheduler(t):
        """Linear scheduler that decreases from C_start to C_end over time [0, T]"""
        return C_start + (C_end - C_start) * (t / T)

    def forward_integrated_scheduler(t):
        """
        Compute the forward integral of the scheduler from 0 to t.
        This is the analytical integral of the linear scheduler from 0 to t.
        """
        return C_start * t + 0.5 * (C_end - C_start) * (t ** 2 / T)

    def delta_integrated_scheduler(t_prev, t):
        """
        Compute the integral of the scheduler from t_prev to t.
        This is the analytical integral of the cosine-squared scheduler from t_prev to t.
        """
        # We can use the forward integrated scheduler and take the difference
        return forward_integrated_scheduler(t) - forward_integrated_scheduler(t_prev)
        # return backward_integrated_scheduler(t_prev) - backward_integrated_scheduler(t)

    def backward_integrated_scheduler(s):
        """
        Compute the backward integral of the scheduler from s to T.
        This is the analytical integral of the linear scheduler from s to T.
        """
        return C_start * (T - s) + 0.5 * (C_end - C_start) * ((T ** 2 - s ** 2) / T)

    return scheduler, forward_integrated_scheduler, delta_integrated_scheduler, backward_integrated_scheduler


def create_cos_square_scheduler(C_start, C_end, T):
    """
    Create a cosine-squared scheduler.

    Args:
        C_start (float): Starting diffusion coefficient
        C_end (float): Ending diffusion coefficient
        T (float): Total time duration

    Returns:
        Tuple of three functions:
        1. Scheduler function that returns the diffusion coefficient at time t
        2. Forward integrated scheduler function that returns \\int_0^s scheduler(t) dt
        3. Backward integrated scheduler function that returns \\int_s^T scheduler(t) dt
    """
    a = (C_start - C_end)
    b = jnp.pi / (2 * T)
    c = C_end

    def scheduler(t):
        """
        Cosine-squared scheduler that decreases smoothly from C_start to C_end
        Follows the shape: C_end + (C_start - C_end) * cos(π * t / T)^2
        """
        return a * (jnp.cos(b * t) ** 2) + c

    def forward_integrated_scheduler(t):
        """
        Compute the forward integral of the scheduler from 0 to t.
        This is the analytical integral of the linear scheduler from 0 to t.
        """
        return (a * jnp.sin(2 * b * t) / (4 * b)) + (a * t / 2) + (c * t)

    def delta_integrated_scheduler(t_prev, t):
        """
        Compute the integral of the scheduler from t_prev to t.
        This is the analytical integral of the cosine-squared scheduler from t_prev to t.
        """
        # We can use the forward integrated scheduler and take the difference
        forward_t = (a * jnp.sin(2 * b * t) / (4 * b)) + (a * t / 2) + (c * t)
        forward_t_prev = (a * jnp.sin(2 * b * t_prev) / (4 * b)) + (a * t_prev / 2) + (c * t_prev)
        return forward_t - forward_t_prev

    def backward_integrated_scheduler(s):
        """
        Compute the backward integral of the scheduler from s to T.
        This is the analytical integral of the linear scheduler from s to T.
        """
        return -(a * (jnp.sin(2 * b * s)) / (4 * b)) + (a * (T - s) / 2) + (c * (T - s))

    return scheduler, forward_integrated_scheduler, delta_integrated_scheduler, backward_integrated_scheduler


def create_cos_pow4_scheduler(C_start, C_end, T):
    """
    Create a cosine-squared scheduler.

    Args:
        C_start (float): Starting diffusion coefficient
        C_end (float): Ending diffusion coefficient
        T (float): Total time duration

    Returns:
        Tuple of three functions:
        1. Scheduler function that returns the diffusion coefficient at time t
        2. Forward integrated scheduler function that returns \\int_0^s scheduler(t) dt
        3. Backward integrated scheduler function that returns \\int_s^T scheduler(t) dt
    """
    a = (C_start - C_end)
    b = jnp.pi / (2 * T)
    c = C_end

    def scheduler(t):
        """
        Cosine-squared scheduler that decreases smoothly from C_start to C_end
        Follows the shape: C_end + (C_start - C_end) * cos(π * t / T)^2
        """
        return a * (jnp.cos(b * t) ** 4) + c

    def forward_integrated_scheduler(t):
        """
        Compute the forward integral of the scheduler from 0 to t.
        This is the analytical integral of the linear scheduler from 0 to t.
        """
        linear_term = (((3 * a) + (8 * c)) * 4 * b * t)
        sin_terms = 8 * a * jnp.sin(2 * b * t) + a * jnp.sin(4 * b * t)
        return (linear_term + sin_terms) / (32 * b)

    def backward_integrated_scheduler(s):
        """
        Compute the backward integral of the scheduler from s to T.
        This is the analytical integral of the linear scheduler from s to T.
        """
        linear_term = (((3 * a) + (8 * c)) * 4 * b * (T-s))
        sin_terms = 8 * a * jnp.sin(2 * b * s) + a * jnp.sin(4 * b * s)
        return (linear_term - sin_terms) / (32 * b)
        # return -(a ** 2 * (jnp.sin(2 * b * s)) / (4 * b)) + (a ** 2 * (T - s) / 2) + (c * (T - s))

    return scheduler, forward_integrated_scheduler, backward_integrated_scheduler


def create_constant_scheduler(C_constant, T):
    """
    Create a constant scheduler.

    Args:
        C_constant (float): Constant diffusion coefficient
        T (float): Total time duration

    Returns:
        Tuple of three functions:
        1. Scheduler function that returns the constant diffusion coefficient
        2. Forward integrated scheduler function that returns \\int_0^s scheduler(t) dt
        3. Backward integrated scheduler function that returns \\int_s^T scheduler(t) dt
    """

    def scheduler(t):
        """Constant scheduler that returns the same value for all t"""
        return C_constant

    def forward_integrated_scheduler(t):
        """
        Compute the forward integral of the constant scheduler from 0 to t.
        This is simply the constant value multiplied by t.
        """
        return C_constant * t

    def delta_integrated_scheduler(t_prev, t):
        """
        Compute the forward integral of the constant scheduler from 0 to t.
        This is simply the constant value multiplied by t.
        """
        return C_constant * (t - t_prev)

    def backward_integrated_scheduler(s):
        """
        Compute the backward integral of the constant scheduler from s to T.
        This is the constant value multiplied by the remaining time.
        """
        return C_constant * (T - s)

    return scheduler, forward_integrated_scheduler, delta_integrated_scheduler, backward_integrated_scheduler


def create_cosine_scheduler(C_start=1., C_end=0.001, s=0.008, pow=2, T=1.0):
    """
    Creates a cosine scheduler based on the "Improved DDPM" paper,
    along with its analytical integrals.
    
    This function *requires* pow=2 to use the analytical integral.
    """
    if pow != 2:
        raise ValueError(f"This function requires pow=2 for the analytical integral, but got pow={pow}")
    
    # Constants for the schedule
    A = C_start - C_end
    B = C_end
    offset = 1. + s

    # Constants for the integral derived from the math
    # f(t) = A * cos^2(k1*t + k2) + B
    k1 = jnp.pi / (2 * T * offset)
    k2 = (jnp.pi * s) / (2 * offset)
    
    # Constant terms from the integral
    sin_2k2 = jnp.sin(2 * k2)
    four_k1 = 4 * k1

    def scheduler(t):
        """
        The new cosine schedule function (beta_t).
        t should be in forward time [0, T].
        """
        # We use t/T as the normalized time progression from 0 to 1
        norm_t = t / T
        cos_arg = jnp.pi * (s + norm_t) / (2 * offset)
        return A * (jnp.cos(cos_arg) ** 2) + B

    def forward_integrated_scheduler(t):
        """
        Compute the analytical forward integral of the scheduler from 0 to t (alpha_t).
        F(t) = (1-min) * [t/2 + (sin(2*k1*t + 2*k2) - sin(2*k2)) / (4*k1)] + min*t
        """
        integral_cos_part = (t / 2) + (jnp.sin(2 * k1 * t + 2 * k2) - sin_2k2) / four_k1
        return A * integral_cos_part + B * t

    def delta_integrated_scheduler(t_prev, t):
        """
        Compute the integral of the scheduler from t_prev to t.
        This is just F(t) - F(t_prev).
        """
        return forward_integrated_scheduler(t) - forward_integrated_scheduler(t_prev)

    def backward_integrated_scheduler(s):
        """
        Compute the backward integral of the scheduler from s to T.
        This is F(T) - F(s).
        """
        return forward_integrated_scheduler(T) - forward_integrated_scheduler(s)

    return scheduler, forward_integrated_scheduler, delta_integrated_scheduler, backward_integrated_scheduler


# Example usage and visualization
def main():
    # Parameters
    C_start = 1.0  # Starting diffusion coefficient
    C_end = 0.01  # Ending diffusion coefficient
    C_constant = 8.
    T = 8.0  # Total time duration

    # C_start = 1.0  # Starting diffusion coefficient
    # C_end = 0.01  # Ending diffusion coefficient
    # C_constant = 8.
    # T = 1.0  # Total time duration

    # Create scheduler functions
    # scheduler, forward_integrated_scheduler, backward_integrated_scheduler = create_linear_scheduler(C_start, C_end, T)
    # scheduler, forward_integrated_scheduler, delta_integrated_scheduler, backward_integrated_scheduler = create_cosine_quartic_scheduler(C_start, C_end, T)
    scheduler, forward_integrated_scheduler, delta_integrated_scheduler, backward_integrated_scheduler = create_cos_square_scheduler(C_start, C_end, T)
    # scheduler, forward_integrated_scheduler, delta_integrated_scheduler, backward_integrated_scheduler = create_cos_pow4_scheduler(C_start, C_end, T)
    # scheduler, forward_integrated_scheduler, backward_integrated_scheduler = create_constant_scheduler(C_constant, T)

    # scheduler, forward_integrated_scheduler, delta_integrated_scheduler, backward_integrated_scheduler = create_cosine_scheduler(
    #     C_start=1, C_end=0.001, s=0.008, pow=2, T=T
    # )

    # Define number of steps for discrete alpha calculation
    num_steps = int(T * 10)  # Use 10 steps per unit time, or adjust as needed
    dt = T / num_steps

    alpha_vals = 1 - jnp.exp(
        -2
        * jnp.array(
            [
                delta_integrated_scheduler(step * dt, (step + 1) * dt)
                for step in range(num_steps)
            ]
        )
    )



    # Vectorize the functions for plotting
    vec_scheduler = jax.vmap(scheduler)
    vec_forward_integrated_scheduler = jax.vmap(forward_integrated_scheduler)
    vec_backward_integrated_scheduler = jax.vmap(backward_integrated_scheduler)

    # Create time points
    t_points = jnp.linspace(0, T, 100)

    # Compute scheduler and integrated values
    scheduler_values = vec_scheduler(t_points)
    forward_integrated_values = vec_forward_integrated_scheduler(t_points)
    backward_integrated_values = vec_backward_integrated_scheduler(t_points)

    # Plotting
    plt.figure(figsize=(15, 5))

    # Scheduler plot
    plt.subplot(1, 4, 1)
    plt.plot(t_points, scheduler_values, label='Scheduler')
    plt.title('$\\beta(t)$')
    plt.xlabel('$t$')
    plt.legend()
    plt.grid(True)

    # Forward integrated scheduler plot
    plt.subplot(1, 4, 2)
    plt.plot(t_points, jnp.exp(-backward_integrated_values), label='Forward Integrated Scheduler', color='red')
    plt.title('$exp(-\\alpha(t))$')
    plt.xlabel('$t$')
    plt.legend()
    plt.grid(True)


    # Backward integrated scheduler plot
    plt.subplot(1, 4, 3)
    plt.plot(t_points, jnp.sqrt(2 * scheduler_values), label='Backward Integrated Scheduler', color='green')
    plt.title('$\\sigma(t)$')
    plt.xlabel('$t$')
    plt.legend()
    plt.grid(True)

    # Backward integrated scheduler plot
    plt.subplot(1, 4, 4)
    plt.plot(t_points, jnp.sqrt(2 * scheduler_values) * jnp.exp(-backward_integrated_values),
             label='Backward Integrated Scheduler', color='green')
    plt.title('$\\sigma(t) * exp(-\\alpha(t))$')
    plt.xlabel('$t$')
    plt.legend()
    plt.grid(True)

    # Forward integrated scheduler plot
    plt.subplot(1, 4, 2)
    plt.plot(t_points, forward_integrated_values, label='Forward Integrated Scheduler', color='red')
    plt.title('Cumulative Integral from 0 to t')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Integral')
    plt.legend()
    plt.grid(True)
    
    # Backward integrated scheduler plot
    plt.subplot(1, 4, 3)
    plt.plot(t_points, backward_integrated_values, label='Backward Integrated Scheduler', color='green')
    plt.title('Cumulative Integral from t to T')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Integral')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    # Save the plot instead of showing it (since we're in non-interactive mode)
    output_file = "scheduler_verification_2.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()

    # Plot alpha_vals
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Alpha values over steps
    plt.subplot(1, 3, 1)
    steps = jnp.arange(len(alpha_vals))
    plt.plot(steps, alpha_vals, 'o-', linewidth=2, markersize=4)
    plt.title('$\\alpha$ Values per Step', fontsize=14, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('$\\alpha$ Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative alpha
    plt.subplot(1, 3, 2)
    cumulative_alpha = jnp.cumsum(alpha_vals)
    plt.plot(steps, cumulative_alpha, 'o-', color='red', linewidth=2, markersize=4)
    plt.title('Cumulative $\\alpha$ Values', fontsize=14, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Cumulative $\\alpha$', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: 1 - alpha (bar noise retention)
    plt.subplot(1, 3, 3)
    bar_alpha = 1 - alpha_vals
    plt.plot(steps, bar_alpha, 'o-', color='green', linewidth=2, markersize=4)
    plt.title('$\\bar{\\alpha}$ = 1 - $\\alpha$ (Signal Retention)', fontsize=14, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('$\\bar{\\alpha}$ Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save alpha plot
    alpha_output_file = "alpha_vals_verification.png"
    plt.savefig(alpha_output_file, dpi=150, bbox_inches='tight')
    print(f"Alpha plot saved to {alpha_output_file}")
    plt.close()

    # Verification
    print("Verification:")
    print("\nForward Integration:")
    print(f"Forward integral at t=0: {forward_integrated_scheduler(0.0)}")
    print(f"Forward integral at t=T: {forward_integrated_scheduler(T)}")

    print("\nBackward Integration:")
    print(f"Backward integral at t=0: {backward_integrated_scheduler(0.0)}")
    print(f"Backward integral at t=T: {backward_integrated_scheduler(T)}")

    # Additional verification
    # Check that forward + backward integrals sum to the total area
    total_area = C_start * T + 0.5 * (C_end - C_start) * (T ** 2 / T)
    half_time = T / 2
    forward_half = forward_integrated_scheduler(half_time)
    backward_half = backward_integrated_scheduler(half_time)

    print("\nTotal Area Verification:")
    print(f"Total area from analytical calculation: {total_area}")
    print(f"Sum of forward and backward integrals at t={half_time}: {forward_half + backward_half}")
    print(f"Difference from total area: {abs(total_area - (forward_half + backward_half))}")

    print("\nAlpha Values Verification:")
    print(f"Number of alpha values: {len(alpha_vals)}")
    print(f"Min alpha: {jnp.min(alpha_vals):.6f}")
    print(f"Max alpha: {jnp.max(alpha_vals):.6f}")
    print(f"Mean alpha: {jnp.mean(alpha_vals):.6f}")
    print(f"First 5 alpha values: {alpha_vals[:5]}")
    print(f"Last 5 alpha values: {alpha_vals[-5:]}")
    print(f"Sum of all alphas: {jnp.sum(alpha_vals):.6f}")


if __name__ == "__main__":
    main()