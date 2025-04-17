<p align=center>
  <img src="https://github.com/user-attachments/assets/4f48d642-ca12-42c4-91a3-32f2ef464b3a" width="640" />
</p>

[![tests](https://github.com/alonfnt/bayex/actions/workflows/tests.yml/badge.svg)](https://github.com/alonfnt/pcax/actions/workflows/pytest.yml)
[![PyPI](https://img.shields.io/pypi/v/bayex.svg)](https://pypi.org/project/pcax/)

`pcax` is a minimal PCA implementation in JAX that’s both GPU/TPU/CPU‑native and fully differentiable.
It keeps data and computation on-device with zero-copy transfers, lets you backpropagate through your dimensionality reduction step, and plugs directly into [JAX](https://github.com/jax-ml/jax) workflows loops for seamless model integration.

## Usage
```python
import pcax

# Fit the PCA model with 3 components on your data X
state = pcax.fit(X, n_components=3)

# Transform X to its principal components
X_pca = pcax.transform(state, X)

# Recover the original X from its principal components
X_recover = pcax.recover(state, X_pca)
```

## Installation
`pcax` can be installed from PyPI via `pip`
```
pip install pcax
```

## Citation
If you use `pcax` in your research and need to reference it, please cite it as follows:
```
@software{alonso_pcax,
  author = {Alonso, Albert},
  title = {pcax: Minimal Principal Component Analysis (PCA) Implementation in JAX},
  url = {https://github.com/alonfnt/pcax},
  version = {0.2.2}
}
```
