from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

class PCAState(NamedTuple):
    components: jax.Array
    means: jax.Array
    explained_variance: jax.Array


def fit(x, n_components, solver="full"):
    if solver == "full":
        return _fit_full(x, n_components)
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
    explained_variance = (S**2) / (n_samples - 1)

    # Return the transformation matrix
    A = Vt[:n_components]
    return PCAState(components=A, means=means, explained_variance=explained_variance)


def transform(state, x):
    x = x - state.means
    return jnp.dot(x, jnp.transpose(state.components))


def recover(state, x):
    return jnp.dot(x, state.components) + state.means
