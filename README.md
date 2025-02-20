# pcax
**PCAx** is a lightweight, differentiable PCA implementation in JAX, designed for seamless GPU acceleration.
It eliminates unnecessary CPU transfers and conversions, optimizing performance for large datasets and GPU-heavy workflows.

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
  version = {0.1.0}
}
```
