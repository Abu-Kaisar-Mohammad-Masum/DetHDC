import numpy as np
import torch

from dethdc import SobolEncoder


def test_encoder_shape():
    X = np.random.RandomState(0).rand(8, 16).astype("float32")
    encoder = SobolEncoder(dimensions=128, seed=42, device="cpu")
    H = encoder.fit_transform(X)
    assert H.shape == (8, 128)


def test_same_seed_same_projection():
    X = np.random.RandomState(1).rand(4, 8).astype("float32")

    a = SobolEncoder(dimensions=64, seed=7, device="cpu").fit_transform(X)
    b = SobolEncoder(dimensions=64, seed=7, device="cpu").fit_transform(X)

    assert torch.allclose(a, b)
