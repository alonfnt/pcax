import pytest
import jax
import jax.numpy as jnp

from pcax import fit, transform, recover

KEY = jax.random.PRNGKey(42)


def test_fit_invalid_solver():
    with pytest.raises(ValueError):
        fit(jnp.zeros((10, 5)), n_components=2, solver="invalid_solver")


@pytest.mark.parametrize("n_components", [1, 2, 5, 10])
@pytest.mark.parametrize("n_entries", [100, 200, 300])
@pytest.mark.parametrize("solver", ["full", "randomized"])
def test_fit_output_shapes(n_entries, n_components, solver):
    x = jax.random.normal(KEY, shape=(n_entries, 50))
    rng, _ = jax.random.split(KEY)

    state = fit(x, n_components=n_components, solver=solver, rng=rng)

    assert state.components.shape == (n_components, x.shape[1])
    assert state.means.shape == (1, x.shape[1])
    assert state.explained_variance.shape == (n_components,)


def test_fit_zero_mean():
    x = jax.random.normal(KEY, shape=(100, 50))
    n_components = 5
    state = fit(x, n_components=n_components, solver="full")
    x_zero_mean = x - state.means
    x_pca = jnp.dot(x_zero_mean, state.components.T)
    x_pca2 = transform(state, x)
    assert jnp.allclose(x_pca.mean(axis=0), jnp.zeros(n_components), atol=1e-5)
    assert jnp.allclose(x_pca, x_pca2)


@pytest.mark.parametrize("n_components", [50])
@pytest.mark.parametrize("n_entries", [300, 500])
@pytest.mark.parametrize("solver", ['full', 'randomized'])
def test_reconstruction(n_entries, n_components, solver):
    x = jax.random.normal(KEY, shape=(n_entries, 50))
    state = fit(x, n_components=n_components, solver=solver)
    x_pca = transform(state, x)
    x_recovered = recover(state, x_pca)
    assert x_recovered.shape == x.shape
    assert jnp.allclose(x, x_recovered, atol=1e-1)
