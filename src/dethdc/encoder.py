import math
import torch
import torch.nn.functional as F
from torch.quasirandom import SobolEngine


class SobolEncoder:
    """Sobol-based deterministic position/value hypervector encoder."""

    def __init__(
        self,
        dimensions=10000,
        scramble=True,
        seed=42,
        device="auto",
    ):
        self.dimensions = int(dimensions)
        self.scramble = bool(scramble)
        self.seed = int(seed)
        self.device = self._resolve_device(device)
        self.projection_ = None
        self.input_dim_ = None

    @staticmethod
    def _resolve_device(device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _build_projection(self, input_dim):
        pos_engine = SobolEngine(
            dimension=input_dim,
            scramble=self.scramble,
            seed=self.seed,
        )
        val_engine = SobolEngine(
            dimension=input_dim,
            scramble=self.scramble,
            seed=self.seed + 1,
        )

        position = 2.0 * pos_engine.draw(self.dimensions) - 1.0
        value = 2.0 * val_engine.draw(self.dimensions) - 1.0

        self.projection_ = torch.stack(
            [position, value], dim=0
        ).to(self.device, dtype=torch.float32)

        self.input_dim_ = input_dim

    def fit(self, X):
        X = torch.as_tensor(X)
        input_dim = int(X[0].numel()) if X.ndim > 1 else int(X.numel())
        self._build_projection(input_dim)
        return self

    def transform(self, X):
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        X = X.reshape(X.shape[0], -1)

        if self.projection_ is None:
            self._build_projection(X.shape[1])

        if X.shape[1] != self.input_dim_:
            raise ValueError(
                f"Expected input dimension {self.input_dim_}, got {X.shape[1]}."
            )

        # Match the original image preprocessing behavior when inputs are [0,1].
        if torch.min(X) >= 0 and torch.max(X) <= 1:
            X = X * 2.0 - 1.0

        position = self.projection_[0]
        value = self.projection_[1]

        inv_sqrt_l = 1.0 / math.sqrt(float(self.input_dim_))

        pos = torch.einsum("bi,di->bd", X, position) * inv_sqrt_l
        val = torch.einsum("bi,di->bd", X, value) * inv_sqrt_l

        hv = pos * val
        return F.normalize(hv, p=2, dim=1, eps=1e-12)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
