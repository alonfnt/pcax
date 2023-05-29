# pcax
Principal Component Analsys implementation using jax.

## Usage
```python
import pcax

state = pcax.fit(X, n_components=3)
X_pca = pcax.transform(state, X)
X_recover = pcax.recover(state, X_pca)
```

## Installation
`pcax` can be installed from PyPI via `pip`
```
pip install pcax
```

or directly from the github repository
```
pip install git+git://github.com/alonfnt/pcax.git
```

