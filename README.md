# pcax

**PCAx** is a minimal Principal Component Analysis (PCA) written in Jax.
The aim of this project is to provide a JAX-based PCA implementation, eliminating the need for unnecessary data transfer to CPU or conversions to Numpy. This can provide performance benefits when working with large datasets or in GPU-intensive workflow

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
  version = {0.0.1}
}
```
