<div align="center">

# DetHDC

### Deterministic Hyperdimensional Learning with Rank Refinement

[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-6A5ACD.svg)](https://ojs.aaai.org/index.php/AAAI/article/view/42253)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Official research implementation of the AAAI 2026 paper  
“Deterministic Hyperdimensional Learning with Rank Refinement.”**

[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/42253) · [Quick Start](#-quick-start) · [Citation](#-citation)

</div>

---

## ✨ Overview

**DetHDC** is a lightweight PyTorch library for deterministic hyperdimensional learning using:

- **Sobol quasi-random projections**
- **position–value binding**
- **prototype-based HDC classification**
- **rank-based refinement**
- **CPU and CUDA execution**
- **reproducible training through explicit random seeds**

The repository turns the method from our AAAI 2026 paper into a reusable Python API instead of a dataset-specific research script.

---

## 🚀 Quick Start

### Install from source

```bash
git clone https://github.com/Abu-Kaisar-Mohammad-Masum/DetHDC.git
cd DetHDC
pip install -e .
```

### Minimal example

```python
from dethdc import DetHDC

model = DetHDC(
    dimensions=10000,
    epochs=5,
    lr=0.01,
    margin=0.2,
    seed=42,
    device="auto",
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = model.score(X_test, y_test)

print(f"Accuracy: {accuracy * 100:.2f}%")
```

---

## 🔥 Why DetHDC?

Traditional HDC implementations often rely on randomly generated hypervectors. That can introduce run-to-run variation and make reproducibility harder.

DetHDC instead uses **Sobol-based deterministic projections**, giving a controlled projection space while retaining the efficiency and robustness properties of hyperdimensional learning.

The library also exposes the refinement stage directly:

```python
model = DetHDC(
    dimensions=10000,
    refinement=True,
    epochs=5,
)
```

For a base model without rank refinement:

```python
model = DetHDC(
    dimensions=10000,
    refinement=False,
)
```

---

## 📦 Library API

```python
from dethdc import DetHDC, SobolEncoder
```

### `DetHDC`

```python
DetHDC(
    dimensions=10000,
    epochs=5,
    lr=0.01,
    margin=0.2,
    refinement=True,
    scramble=True,
    seed=42,
    device="auto",
)
```

Main methods:

```python
model.fit(X, y)
model.encode(X)
model.predict(X)
model.predict_similarity(X)
model.score(X, y)
model.get_config()
model.save("model.pt")
DetHDC.load("model.pt")
```

---

## 🧪 MNIST Example

Run:

```bash
python examples/mnist.py
```

The example:

1. downloads MNIST,
2. flattens native \(28 \times 28\) images,
3. constructs Sobol projection matrices,
4. encodes the training/test sets,
5. builds class prototypes,
6. performs rank-based refinement,
7. reports classification accuracy.

---

## 🗂 Repository Structure

```text
DetHDC/
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── src/
│   └── dethdc/
│       ├── __init__.py
│       ├── classifier.py
│       └── encoder.py
├── examples/
│   └── mnist.py
├── benchmarks/
│   └── reproduce_aaai2026.py
└── tests/
    ├── test_encoder.py
    └── test_classifier.py
```

---

## ⚡ GPU Support

DetHDC automatically uses CUDA when available:

```python
model = DetHDC(device="auto")
```

You can also force a backend:

```python
model = DetHDC(device="cpu")
model = DetHDC(device="cuda")
```

Classification stays in PyTorch and avoids unnecessary GPU → CPU → NumPy transfers.

---

## 🔬 Reproducibility

DetHDC exposes the seed explicitly:

```python
model = DetHDC(seed=42)
```

You can inspect the complete configuration:

```python
print(model.get_config())
```

Example:

```python
{
    "dimensions": 10000,
    "epochs": 5,
    "lr": 0.01,
    "margin": 0.2,
    "refinement": True,
    "scramble": True,
    "seed": 42,
    "device": "cuda"
}
```

---

## 🧪 Testing

```bash
pip install -e ".[test]"
pytest
```

The test suite checks:

- output dimensions,
- deterministic Sobol generation,
- fitting and prediction,
- model save/load,
- CPU execution,
- CUDA execution when available.

---

## 🧪 Library Sanity Check

Using the reproducible MNIST example included in this repository:

| Configuration | Result |
|---|---:|
| Dimension | 10,000 |
| Refinement iterations | 5 |
| Seed | 42 |
| Test split | 30% stratified |
| Accuracy | **95.46%** |

The packaged example uses a fixed deterministic split and seed and is
intended as a reproducible usage example rather than an exact recreation
of the paper's experimental split.

## 📄 Paper

**Deterministic Hyperdimensional Learning with Rank Refinement**  
Abu Kaisar Mohammad Masum and Sercan Aygun  
*Proceedings of the AAAI Conference on Artificial Intelligence*, 2026.

Paper:

https://ojs.aaai.org/index.php/AAAI/article/view/42253

---

## 📚 Citation

If this repository helps your research, please cite our AAAI paper:

```bibtex
@inproceedings{masum2026deterministic,
  title={Deterministic hyperdimensional learning with rank refinement (student abstract)},
  author={Masum, Abu Kaisar Mohammad and Aygun, Sercan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={48},
  pages={41313--41315},
  year={2026}
}
```


## 📜 License

Released under the [MIT License](LICENSE).

---
### Iris Benchmark

We evaluate DetHDC on the Iris dataset using repeated stratified
cross-validation (5 folds × 10 repeats, 50 evaluations).

| Metric | Mean ± Std |
|---|---:|
| Accuracy | **96.67 ± 2.49%** |
| Precision (Macro) | **96.98 ± 2.29%** |
| Recall (Macro) | **96.67 ± 2.49%** |
| Macro F1 | **96.65 ± 2.51%** |

**Configuration:** D = 5,000, refinement epochs = 5, learning rate = 0.01,
margin = 0.2, model seed = 42.


</div>
