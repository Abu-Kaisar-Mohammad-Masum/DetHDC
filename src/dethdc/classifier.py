from pathlib import Path

import torch
import torch.nn.functional as F

from .encoder import SobolEncoder


class DetHDC:
    """Deterministic HDC classifier with optional rank-based refinement."""

    def __init__(
        self,
        dimensions=10000,
        epochs=5,
        lr=0.01,
        margin=0.2,
        refinement=True,
        scramble=True,
        seed=42,
        device="auto",
    ):
        self.dimensions = int(dimensions)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.margin = float(margin)
        self.refinement = bool(refinement)
        self.scramble = bool(scramble)
        self.seed = int(seed)

        self.encoder = SobolEncoder(
            dimensions=self.dimensions,
            scramble=self.scramble,
            seed=self.seed,
            device=device,
        )
        self.device = self.encoder.device

        self.class_hypervectors_ = None
        self.classes_ = None

    def encode(self, X):
        return self.encoder.transform(X)

    def _initialize_prototypes(self, encoded, y):
        y = torch.as_tensor(y, dtype=torch.long, device=self.device)
        self.classes_ = torch.unique(y, sorted=True)

        if not torch.equal(
            self.classes_,
            torch.arange(len(self.classes_), device=self.device),
        ):
            raise ValueError(
                "Current implementation expects integer class labels 0..C-1."
            )

        prototypes = torch.zeros(
            (len(self.classes_), self.dimensions),
            dtype=torch.float32,
            device=self.device,
        )

        prototypes.index_add_(0, y, encoded)
        return F.normalize(prototypes, p=2, dim=1, eps=1e-12)

    def _rank_refine(self, encoded, y):
        y = torch.as_tensor(y, dtype=torch.long, device=self.device)

        prototypes = self.class_hypervectors_.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([prototypes], lr=self.lr)

        for _ in range(self.epochs):
            order = torch.randperm(
                len(encoded),
                generator=torch.Generator(device="cpu").manual_seed(self.seed),
            )

            for idx in order.tolist():
                optimizer.zero_grad()

                x = encoded[idx : idx + 1]
                target = int(y[idx].item())

                sims = F.cosine_similarity(x, prototypes)
                pred = int(torch.argmax(sims).item())

                if pred == target:
                    continue

                loss = torch.clamp(
                    self.margin - (sims[target] - sims[pred]),
                    min=0.0,
                )

                if loss.item() > 0:
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        prototypes[:] = F.normalize(
                            prototypes, p=2, dim=1, eps=1e-12
                        )

        self.class_hypervectors_ = prototypes.detach()

    def fit(self, X, y):
        encoded = self.encoder.fit_transform(X)
        self.class_hypervectors_ = self._initialize_prototypes(encoded, y)

        if self.refinement and self.epochs > 0:
            self._rank_refine(encoded, y)

        return self

    def predict_similarity(self, X):
        self._check_fitted()
        encoded = self.encode(X)
        encoded = F.normalize(encoded, p=2, dim=1, eps=1e-12)
        prototypes = F.normalize(
            self.class_hypervectors_, p=2, dim=1, eps=1e-12
        )
        return encoded @ prototypes.T

    def predict(self, X):
        similarities = self.predict_similarity(X)
        return torch.argmax(similarities, dim=1).detach().cpu().numpy()

    def score(self, X, y):
        pred = torch.as_tensor(self.predict(X))
        target = torch.as_tensor(y).cpu()
        return float((pred == target).float().mean().item())

    def get_config(self):
        return {
            "dimensions": self.dimensions,
            "epochs": self.epochs,
            "lr": self.lr,
            "margin": self.margin,
            "refinement": self.refinement,
            "scramble": self.scramble,
            "seed": self.seed,
            "device": str(self.device),
        }

    def save(self, path):
        self._check_fitted()
        payload = {
            "config": self.get_config(),
            "classes": self.classes_.detach().cpu(),
            "class_hypervectors": self.class_hypervectors_.detach().cpu(),
            "projection": self.encoder.projection_.detach().cpu(),
            "input_dim": self.encoder.input_dim_,
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path, device="auto"):
        payload = torch.load(Path(path), map_location="cpu")
        cfg = dict(payload["config"])
        cfg["device"] = device

        model = cls(**cfg)
        model.classes_ = payload["classes"].to(model.device)
        model.class_hypervectors_ = payload["class_hypervectors"].to(model.device)
        model.encoder.projection_ = payload["projection"].to(model.device)
        model.encoder.input_dim_ = payload["input_dim"]
        return model

    def _check_fitted(self):
        if self.class_hypervectors_ is None:
            raise RuntimeError("Call fit() before prediction.")
