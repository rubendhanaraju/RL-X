import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit
from typing import Callable, Tuple, Optional, NamedTuple
import matplotlib.pyplot as plt
import numpy as np


class ConditionalProblem(NamedTuple):
    sample: Callable
    log_prob: Callable
    prob: Callable
    visualize: Callable
    entropy: Callable
    get_Q: Callable

def get_cond_gaussian(mean_func, sigma):

    # @jit
    def sample(key: jax.Array, c: jax.Array, n_samples: int = 1) -> jax.Array:
        """
        Sample from the conditional distribution p(x|c).

        Args:
            key: JAX random key
            c: Conditioning variable(s)
            n_samples: Number of samples per conditioning value

        Returns:
            Samples from p(x|c)
        """
        mu = mean_func(c)

        # Handle both scalar and array conditioning variables
        if jnp.ndim(c) == 0:
            # Single conditioning value
            return mu + sigma * random.normal(key, (n_samples,))
        else:
            # Multiple conditioning values
            keys = random.split(key, len(c))
            samples = jax.vmap(lambda k, m: m + sigma * random.normal(k, (n_samples,)))(keys, mu)
            return samples

    # @jit
    def log_prob(x: jax.Array, c: jax.Array) -> jax.Array:
        """
        Compute log probability log p(x|c).

        Args:
            x: Sample values
            c: Conditioning variable(s)

        Returns:
            Log probabilities
        """
        mu = mean_func(c)
        return -0.5 * jnp.log(2 * jnp.pi * sigma ** 2) - 0.5 * ((x - mu) / sigma) ** 2

    def get_Q(x, c):
        mu = mean_func(c)
        return - 0.5 * ((x - mu) / sigma) ** 2

    # @jit
    def prob(x: jax.Array, c: jax.Array) -> jax.Array:
        """
        Compute probability density p(x|c).

        Args:
            x: Sample values
            c: Conditioning variable(s)

        Returns:
            Probability densities
        """
        return jnp.exp(log_prob(x, c))

    # @jit
    def entropy() -> jax.Array:
        """
            Compute conditional entropy H[X | C=c].

            For a Gaussian with fixed variance sigma^2, the entropy does not
            depend on c or the mean function.

            Args:
                c: Conditioning variable(s) (unused, kept for API consistency)

            Returns:
                Conditional entropy (scalar or broadcasted)
            """
        return 0.5 * jnp.log(2.0 * jnp.pi * jnp.e * sigma ** 2)

    def visualize_2d_contour(
            c_samples,
            x_samples,
            c_range: Tuple[float, float] = (-3, 3),
            x_range: Tuple[float, float] = (-5, 5),
            n_points: int = 100,
            log_scale: bool = True,
            show_mean: bool = True,
            show_samples: bool = False,
            n_samples: int = 50,
            figsize: Tuple[int, int] = (10, 8),
            title: Optional[str] = None,
            show=True) -> plt.Figure:
        """
        Create a 2D contour plot visualization of the conditional Gaussian distribution.

        Args:
            c_range: Range of conditioning variable c (min, max)
            x_range: Range of x values (min, max)
            n_points: Number of grid points for each dimension
            log_scale: If True, plot log probabilities; if False, plot probabilities
            show_mean: If True, overlay the mean function μ(c)
            show_samples: If True, overlay sample points
            n_samples: Number of samples to show if show_samples=True
            figsize: Figure size (width, height)
            title: Plot title

        Returns:
            matplotlib Figure object
        """
        # Create meshgrid for contour plot
        c_grid = jnp.linspace(c_range[0], c_range[1], n_points)
        x_grid = jnp.linspace(x_range[0], x_range[1], n_points)
        C, X = jnp.meshgrid(c_grid, x_grid)

        # Compute log probabilities or probabilities
        if log_scale:
            Z = log_prob(X, C)
            cb_label = 'Log Probability log p(x|c)'
        else:
            Z = prob(X, C)
            cb_label = 'Probability Density p(x|c)'

        # Create figure and plot
        fig, ax = plt.subplots(figsize=figsize)

        # Create contour plot
        if log_scale:
            # For log scale, use filled contours with appropriate levels
            levels = jnp.linspace(jnp.max(Z) - 6, jnp.max(Z), 20)
            contour = ax.contourf(np.array(C), np.array(X), np.array(Z),
                                  levels=np.array(levels), cmap='viridis', alpha=0.8)
            # Add contour lines
            ax.contour(np.array(C), np.array(X), np.array(Z),
                       levels=np.array(levels[::3]), colors='white', alpha=0.3, linewidths=0.5)
        else:
            # For probability scale, use standard contours
            contour = ax.contourf(np.array(C), np.array(X), np.array(Z),
                                  levels=20, cmap='viridis', alpha=0.8)
            ax.contour(np.array(C), np.array(X), np.array(Z),
                       levels=10, colors='white', alpha=0.3, linewidths=0.5)

        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label(cb_label, rotation=270, labelpad=20)

        # Overlay mean function if requested
        if show_mean:
            mean_values = mean_func(c_grid)
            ax.plot(np.array(c_grid), np.array(mean_values), 'r-', linewidth=3,
                    label=f'Mean μ(c)', alpha=0.9)
            # Add confidence bands (±1σ, ±2σ)
            ax.fill_between(np.array(c_grid),
                            np.array(mean_values - sigma),
                            np.array(mean_values + sigma),
                            alpha=0.2, color='red', label='±1σ')
            ax.fill_between(np.array(c_grid),
                            np.array(mean_values - 2 * sigma),
                            np.array(mean_values + 2 * sigma),
                            alpha=0.1, color='red', label='±2σ')

        ax.scatter(np.array(c_samples), np.array(x_samples),
                   c='white', s=30, alpha=0.8, edgecolors='black', linewidth=0.5,
                   label='Samples', zorder=5)

        # Formatting
        ax.set_xlabel('Conditioning Variable c', fontsize=12)
        ax.set_ylabel('x', fontsize=12)
        ax.set_xlim(c_range)
        ax.set_ylim(x_range)
        ax.grid(True, alpha=0.3)

        if show_mean or show_samples:
            ax.legend(loc='upper right', framealpha=0.9)

        # Set title
        if title is None:
            scale_type = "Log-Probability" if log_scale else "Probability"
            title = f'Conditional Gaussian Distribution - {scale_type} Density'
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()
        if show:
            plt.show()

        return fig


    return ConditionalProblem(sample=sample,
                              log_prob=log_prob,
                              prob=prob,
                              visualize=visualize_2d_contour,
                              entropy=entropy,
                              get_Q=get_Q)


# Define different mean functions
@jit
def linear_mean(c: jax.Array, a: float = 2.0, b: float = 1.0) -> jax.Array:
    """Linear dependence: μ(c) = a*c + b"""
    return a * c + b


@jit
def sinusoidal_mean(c: jax.Array, A: float = 3.0, freq: float = 1.0, phase: float = 0.0) -> jax.Array:
    """Sinusoidal dependence: μ(c) = A * sin(freq * c + phase)"""
    return A * jnp.sin(freq * c + phase)


@jit
def quadratic_mean(c: jax.Array, a: float = 0.5, b: float = 1.0, d: float = 0.0) -> jax.Array:
    """Quadratic dependence: μ(c) = a*c² + b*c + d"""
    return a * c ** 2 + b * c + d


@jit
def exponential_mean(c: jax.Array, A: float = 2.0, decay: float = 0.5) -> jax.Array:
    """Exponential dependence: μ(c) = A * exp(-decay * |c|)"""
    return A * jnp.exp(-decay * jnp.abs(c))


def get_problem(mean_func):
    """Demonstrate the conditional Gaussian with different mean functions."""

    # Initialize random key
    # Initialize random key
    key = random.PRNGKey(42)

    # Define conditioning variable range
    c_range = jnp.linspace(-3, 3, 50)

    # Create conditional Gaussians with different mean functions
    linear_gaussian = get_cond_gaussian(
        mean_func=lambda c: linear_mean(c, a=2.0, b=1.0),
        sigma=0.5
    )

    const_gaussian = get_cond_gaussian(
        mean_func=lambda c: jnp.zeros_like(c), sigma=1.0,
    )

    sinusoidal_gaussian = get_cond_gaussian(
        mean_func=lambda c: sinusoidal_mean(c, A=2.0, freq=1.5, phase=0.0),
        sigma=0.3
    )

    quadratic_gaussian = get_cond_gaussian(
        mean_func=lambda c: quadratic_mean(c, a=0.3, b=0.0, d=0.0),
        sigma=0.4
    )

    exponential_gaussian = get_cond_gaussian(
        mean_func=lambda c: exponential_mean(c, A=2.5, decay=0.8),
        sigma=0.3
    )

    gaussians = {
        'Const': const_gaussian,
        'Linear': linear_gaussian,
        'Sinusoidal': sinusoidal_gaussian,
        'Quadratic': quadratic_gaussian,
        'Exponential': exponential_gaussian
    }
    return gaussians[mean_func]


# Example usage and demonstration
def demo_conditional_gaussian():
    """Demonstrate the conditional Gaussian with different mean functions."""

    # Initialize random key
    key = random.PRNGKey(42)

    # Define conditioning variable range
    c_range = jnp.linspace(-3, 3, 50)

    # Create conditional Gaussians with different mean functions
    const_gaussian = get_cond_gaussian(
        mean_func=lambda c: linear_mean(c, a=2.0, b=1.0),
        sigma=0.5
    )

    # Create conditional Gaussians with different mean functions
    linear_gaussian = get_cond_gaussian(
        mean_func=lambda c: linear_mean(c, a=2.0, b=1.0),
        sigma=0.5
    )

    sinusoidal_gaussian = get_cond_gaussian(
        mean_func=lambda c: sinusoidal_mean(c, A=2.0, freq=1.5, phase=0.0),
        sigma=0.3
    )

    quadratic_gaussian = get_cond_gaussian(
        mean_func=lambda c: quadratic_mean(c, a=0.3, b=0.0, d=0.0),
        sigma=0.4
    )

    exponential_gaussian = get_cond_gaussian(
        mean_func=lambda c: exponential_mean(c, A=2.5, decay=0.8),
        sigma=0.3
    )

    gaussians = {
        'Const': const_gaussian,
        'Linear': linear_gaussian,
        'Sinusoidal': sinusoidal_gaussian,
        'Quadratic': quadratic_gaussian,
        'Exponential': exponential_gaussian
    }

    # Generate samples and compute probabilities
    results = {}

    for name, gaussian in gaussians.items():
        key, subkey = random.split(key)

        # Sample from the conditional distribution
        samples = gaussian.sample(subkey, c_range, n_samples=10)

        # # Compute mean function values
        # mean_values = gaussian.mean_func(c_range)

        # Compute probability density for visualization
        x_grid = jnp.linspace(-5, 5, 100)
        c_test = jnp.array([0.0, 1.0, -1.0])  # Test at specific c values

        prob_densities = []
        for c_val in c_test:
            probs = gaussian.prob(x_grid, c_val)
            prob_densities.append(probs)

        results[name] = {
            'samples': samples,
            'c_range': c_range,
            'prob_densities': prob_densities,
            'x_grid': x_grid,
            'c_test': c_test
        }

    return results, gaussians


# Visualization demo function
def demo_visualizations():
    """Demonstrate the 2D contour visualizations for different mean functions."""

    print("Creating 2D contour visualizations...")

    # Create conditional Gaussians with different mean functions
    gaussians = {
        'Linear': get_cond_gaussian(
            mean_func=lambda c: linear_mean(c, a=1.5, b=0.5),
            sigma=0.4
        ),
        'Sinusoidal': get_cond_gaussian(
            mean_func=lambda c: sinusoidal_mean(c, A=2.0, freq=1.2, phase=0.0),
            sigma=0.3
        ),
        'Quadratic': get_cond_gaussian(
            mean_func=lambda c: quadratic_mean(c, a=0.4, b=0.0, d=0.0),
            sigma=0.35
        ),
        'Exponential': get_cond_gaussian(
            mean_func=lambda c: exponential_mean(c, A=2.0, decay=0.6),
            sigma=0.25
        )
    }

    # Create visualizations
    figures = {}

    for name, gaussian in gaussians.items():
        print(f"Creating visualization for {name} mean function...")

        # Create log-probability contour plot with mean overlay and samples
        fig = gaussian.visualize(
            c_range=(-3, 3),
            x_range=(-4, 4),
            n_points=150,
            log_scale=True,
            show_mean=True,
            show_samples=True,
            n_samples=100,
            figsize=(12, 8),
            title=f'Conditional Gaussian - {name} Mean Function'
        )

        figures[name] = fig

        # Display the plot (in a real environment, you might save these)
        plt.show()

    return figures


if __name__ == "__main__":
    print("Conditional Gaussian Distribution Demo")
    print("=" * 40)

    # Run demonstration
    results, gaussians = demo_visualizations()

    # # Print some statistics
    # for name, result in results.items():
    #     print(f"\n{name} Mean Function:")
    #     print(f"  Sample shape: {result['samples'].shape}")
    #     print(f"  Mean range: [{result['mean_values'].min():.2f}, {result['mean_values'].max():.2f}]")
    #
    # # Run parameter learning example
    # print("\n" + "=" * 40)
    # print("Parameter Learning Example")
    # x_obs, c_obs = learn_parameters_example()
    #
    # # Run visualization demo
    # print("\n" + "=" * 40)
    # print("2D Contour Visualization Demo")
    # viz_figures = demo_visualizations()
    #
    # # Example of using the visualization method directly
    # print("\n" + "=" * 40)
    # print("Direct Visualization Usage Example")
    #
    # # Create a specific example
    # example_gaussian = ConditionalGaussian(
    #     mean_func=lambda c: sinusoidal_mean(c, A=1.5, freq=2.0, phase=jnp.pi / 4),
    #     sigma=0.4
    # )
    #
    # # Create different types of visualizations
    # print("Creating log-probability visualization...")
    # fig_log = example_gaussian.visualize_2d_contour(
    #     c_range=(-2, 2),
    #     x_range=(-3, 3),
    #     log_scale=True,
    #     show_mean=True,
    #     show_samples=True,
    #     n_samples=75,
    #     title='Example: Sinusoidal Mean with Log-Probability Scale'
    # )
    #
    # print("Creating probability density visualization...")
    # fig_prob = example_gaussian.visualize_2d_contour(
    #     c_range=(-2, 2),
    #     x_range=(-3, 3),
    #     log_scale=False,
    #     show_mean=True,
    #     show_samples=False,
    #     title='Example: Sinusoidal Mean with Probability Density Scale'
    # )
    #
    # plt.show()
    # 
    # print("\nDemo completed! Check the generated plots.")
    # print("The visualize_2d_contour method supports:")
    # print("- Log-probability or probability density scales")
    # print("- Overlay of mean function μ(c)")
    # print("- Confidence bands (±1σ, ±2σ)")
    # print("- Sample points overlay")
    # print("- Customizable ranges and resolution")
