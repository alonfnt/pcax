from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


class PCAState(NamedTuple):
    """Stores the state of a fitted PCA model.

    Attributes:
        components: Principal components (right singular vectors).
        means: Mean of each feature in the original data.
        explained_variance: Variance explained by each component.
    """

    components: jax.Array
    means: jax.Array
    explained_variance: jax.Array


def transform(state: PCAState, x: jax.Array) -> jax.Array:
    """
    Projects data into the PCA space defined by the fitted components.

    Args:
        state: The fitted PCA state.
        x: Input data of shape (n_samples, n_features).

    Returns:
        Transformed data in the reduced PCA space.
    """
    x = x - state.means
    return jnp.dot(x, jnp.transpose(state.components))


def recover(state: PCAState, x: jax.Array) -> jax.Array:
    """
    Reconstructs data from its PCA-projected representation.

    Args:
        state: The fitted PCA state.
        x: Transformed data of shape (n_samples, n_components).

    Returns:
        Approximate reconstruction of the original input.
    """
    return jnp.dot(x, state.components) + state.means


def fit(
    x: jax.Array, n_components: int, solver: str = "full", rng: jax.Array | None = None
) -> PCAState:
    """
    Fits PCA on the input data using either full or randomized solver.

    Args:
        x: Input data of shape (n_samples, n_features).
        n_components: Number of principal components to retain.
        solver: Either "full" or "randomized".
        rng: PRNG key for randomized solver.

    Returns:
        The learned PCA transformation state.

    Raises:
        ValueError: If an invalid solver name is provided.
    """
    if solver == "full":
        return _fit_full(x, n_components)
    elif solver == "randomized":
        if rng is None:
            rng = jax.random.PRNGKey(n_components)
        return _fit_randomized(x, n_components, rng)
    else:
        raise ValueError("Invalid solver: must be 'full' or 'randomized'")


@partial(jax.jit, static_argnames="n_components")
def _fit_full(x: jax.Array, n_components: int) -> PCAState:
    """
    Performs exact PCA using full SVD on centered input.
    Used internally when `solver='full'`.
    """

    n_samples, n_features = x.shape

    # Subtract the mean of the input data
    means = x.mean(axis=0, keepdims=True)
    x = x - means

    # Factorize the data matrix with singular value decomposition.
    U, S, Vt = jax.scipy.linalg.svd(x, full_matrices=False)

    # Compute the explained variance
    explained_variance = (S[:n_components] ** 2) / (n_samples - 1)

    # Return the transformation matrix
    A = Vt[:n_components]
    return PCAState(components=A, means=means, explained_variance=explained_variance)


def _fit_randomized(
    x: jax.Array, n_components: int, rng: jax.Array, n_iter: int = 5
) -> PCAState:
    """
    Randomized PCA approximation using power iterations.
    Based on Halko et al., [https://doi.org/10.48550/arXiv.1007.5510].
    Used internally when `solver='randomized'`.
    """
    n_samples, n_features = x.shape
    means = jnp.mean(x, axis=0, keepdims=True)
    x = x - means

    # Generate n_features normal vectors of the given size
    size = jnp.minimum(2 * n_components, n_features)
    Q = jax.random.normal(rng, shape=(n_features, size))

    def step_fn(q, _):
        q, _ = jax.scipy.linalg.lu(x @ q, permute_l=True)
        q, _ = jax.scipy.linalg.lu(x.T @ q, permute_l=True)
        return q, None

    Q, _ = jax.lax.scan(step_fn, init=Q, xs=None, length=n_iter)
    Q, _ = jax.scipy.linalg.qr(x @ Q, mode="economic")
    B = Q.T @ x

    _, S, Vt = jax.scipy.linalg.svd(B, full_matrices=False)

    explained_variance = (S[:n_components] ** 2) / (n_samples - 1)
    A = Vt[:n_components]
    return PCAState(components=A, means=means, explained_variance=explained_variance)
