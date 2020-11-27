import torch
import torch.nn as nn

from Maps import DriftMap, QuadKick


class Element(nn.Module):
    def __init__(self, dim: int, slices: int = 1, order: int = 2, dtype=torch.float32):
        super().__init__()

        self.dim = dim
        self.slices = slices
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

    def rMatrix(self):
        rMatrix = torch.eye(self.dim, dtype=self.dtype)

        for m in self.maps:
            rMatrix = torch.matmul(m.rMatrix(), rMatrix)

        return rMatrix


class Drift(Element):
    def __init__(self, length: float, dim: int, slices: int, order: int, dtype: torch.dtype):
        super(Drift, self).__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        # ignore split scheme and slices for increased performance
        self.map = DriftMap(length, dim, self.dtype)

        self.maps = nn.ModuleList([self.map])
        return

    def forward(self, x):
        return self.map(x)


class Quadrupole(Element):
    def __init__(self, length: float, k1: float, dim: int, slices: int, order: int, dtype: torch.dtype):
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        self.maps = nn.ModuleList()
        for slice in range(slices):
            for c, d in zip(self.coeffC, self.coeffD):
                if c:
                    self.maps.append(DriftMap(c * length / slices, dim, self.dtype))
                if d:
                    self.maps.append(QuadKick(d * length / slices, k1, dim, self.dtype))

        return


if __name__ == "__main__":
    dim = 4
    slices = 1
    dtype = torch.float32

    drift = Drift(3, dim=dim, slices=slices, dtype=dtype)
    quad = Quadrupole(1, 0.3, dim=dim, slices=slices, dtype=dtype)

    # create particle
    x0 = torch.tensor([[1e-3, 2e-3, 1e-3, 0], ])

    # track
    x = quad(x0)
    print(x)

    print(quad.rMatrix())
