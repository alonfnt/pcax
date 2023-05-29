from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


class PCAState(NamedTuple):
    components: jax.Array
    means: jax.Array
    explained_variance: jax.Array


def transform(state, x):
    x = x - state.means
    return jnp.dot(x, jnp.transpose(state.components))


def recover(state, x):
    return jnp.dot(x, state.components) + state.means


def fit(x, n_components, solver="full", rng=None):
    if solver == "full":
        return _fit_full(x, n_components)
    elif solver == "randomized":
        if rng is None:
            rng = jax.random.PRNGKey(n_components)
        return _fit_randomized(x, n_components, rng)
    else:
        raise ValueError("solver parameter is not correct")


@partial(jax.jit, static_argnames="n_components")
def _fit_full(x, n_components):
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


def _fit_randomized(x, n_components, rng, n_iter=5):
    """Randomized PCA based on Halko et al [https://doi.org/10.48550/arXiv.1007.5510]."""
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
