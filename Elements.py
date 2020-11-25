import torch
import torch.nn as nn

from Maps import DriftMap, QuadKick


class Element(nn.Module):
    def __init__(self, dim: int, order=2, dtype=torch.float32):
        super().__init__()

        self.dim = dim
        self.dtype = dtype

        if order == 2:
            self.coeffC = [1 / 2, 1 / 2]
            self.coeffD = [1, 0]
        elif order == 3:
            self.coeffC = [2 / 3, -2 / 3, 1]
            self.coeffD = [7 / 24, 3 / 4, -1 / 24]
        elif order == 4:
            self.coeffC = [0.6756, 0.411, 0.411, 0.6756]
            self.coeffD = [0, 0.82898, 0.72991, 0.82898]
        else:
            raise NotImplementedError("order {} not implemented".format(order))

        return

    def forward(self, x):
        for m in self.maps:
            x = m(x)

        return x

    def driftMap(self, length: float, dim: int):
        # drift = torch.tensor([[1, length, 0, 0], [0, 1, 0, 0], [0, 0, 1, length], [0, 0, 0, 1]], dtype=self.dtype)
        #
        # driftMap = nn.Linear(dim, dim, bias=False)
        # driftMap.weight = nn.Parameter(drift)

        driftMap = DriftMap(length, dim, self.dtype)
        return driftMap

    def rMatrix(self):
        rMatrix = torch.eye(self.dim, dtype=self.dtype)

        for m in self.maps:
            rMatrix = torch.matmul(m.rMatrix(), rMatrix)

        return rMatrix


class Drift(Element):
    def __init__(self, length: float, dim: int, dtype: torch.dtype):
        super(Drift, self).__init__(dim=dim, dtype=dtype)

        self.map = self.driftMap(length, dim)

        self.maps = nn.ModuleList([self.map])
        return

    def forward(self, x):
        return self.map(x)


class Quadrupole(Element):
    def __init__(self, length: float, k1: float, dim: int, dtype: torch.dtype):
        super().__init__(dim=dim, dtype=dtype)

        self.maps = nn.ModuleList()
        for c, d in zip(self.coeffC, self.coeffD):
            if c:
                self.maps.append(self.driftMap(c * length, dim))
            if d:
                self.maps.append(self.quadMap(length, k1, dim))

        return

    def quadMap(self, length: float, k1: float, dim: int):
        # quad = torch.tensor([[1, 0, 0, 0], [-1 * length * k1, 1, 0, 0], [0, 0, 1, 0], [0, 0, length * k1, 1]], dtype=self.dtype)
        #
        # quadMap = nn.Linear(dim, dim, bias=False)
        # quadMap.weight = nn.Parameter(quad)
        quadMap = QuadKick(length, k1, dim, self.dtype)
        return quadMap


if __name__ == "__main__":
    drift = Drift(3)
    quad = Quadrupole(1, 0.5)

    # create particle
    x0 = torch.tensor([[1e-3, 2e-3, 1e-3, 0],])

    # track
    x = drift(x0)
    print(x)

