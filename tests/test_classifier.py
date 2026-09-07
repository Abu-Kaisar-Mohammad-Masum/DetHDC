import numpy as np

from dethdc import DetHDC


def test_fit_predict():
    rng = np.random.RandomState(42)

    X0 = rng.normal(0.2, 0.05, size=(20, 8)).astype("float32")
    X1 = rng.normal(0.8, 0.05, size=(20, 8)).astype("float32")

    X = np.vstack([X0, X1])
    y = np.array([0] * 20 + [1] * 20)

    model = DetHDC(
        dimensions=256,
        epochs=1,
        refinement=True,
        seed=42,
        device="cpu",
    )

    model.fit(X, y)
    pred = model.predict(X)

    assert pred.shape == y.shape
    assert model.score(X, y) >= 0.5
