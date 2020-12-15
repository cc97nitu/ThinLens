import torch
import torch.nn as nn

from ThinLens.Maps import DriftMap, DipoleKick, QuadKick, EdgeKick


class Element(nn.Module):
    def __init__(self, dim: int, slices: int = 1, order: int = 2, dtype=torch.float32):
        super().__init__()

        self.dim = dim
        self.slices = slices
        self.dtype = dtype
        self.order = order

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
        self.length = length

        # ignore split scheme and slices for increased performance
        self.map = DriftMap(length, dim, self.dtype)

        self.maps = nn.ModuleList([self.map])
        return

    def forward(self, x):
        return self.map(x)


class Monitor(Drift):
    """Special drift."""


class KickElement(Element):
    """Base class for elements consisting of both drift and kicks."""

    def __init__(self, length: float, kickMap, dim: int, slices: int, order: int, dtype: torch.dtype):
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)
        self.length = length

        # split scheme for hamiltonian
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

        # same map for each slice
        self.maps = list()

        for c, d in zip(self.coeffC, self.coeffD):
            if c:
                self.maps.append(DriftMap(c * length / slices, dim, self.dtype))
            if d:
                self.maps.append(kickMap(d * length / slices))

        self.maps = nn.ModuleList(self.maps * slices)
        return


class SBen(KickElement):
    """Horizontal sector bending magnet."""

    def __init__(self, length: float, angle: float, dim: int, slices: int, order: int, dtype: torch.dtype,
                 e1: float = 0, e2: float = 0):
        kickMap = lambda length: DipoleKick(length, angle / slices, dim, dtype)

        super().__init__(length=length, kickMap=kickMap, dim=dim, slices=slices, order=order, dtype=dtype)

        # edges present?
        if e1:
            self.maps.insert(0, EdgeKick(length, angle, e1, dim, self.dtype))

        if e2:
            self.maps.append(EdgeKick(length, angle, e2, dim, self.dtype))

        return


class RBen(SBen):
    """Horizontal rectangular bending magnet."""

    def __init__(self, length: float, angle: float, dim: int, slices: int, order: int, dtype: torch.dtype,
                 e1: float = 0, e2: float = 0):
        # modify edges
        e1 += angle / 2
        e2 += angle / 2

        super().__init__(length, angle, dim, slices, order, dtype, e1=e1, e2=e2)
        return


class Quadrupole(KickElement):
    def __init__(self, length: float, k1: float, dim: int, slices: int, order: int, dtype: torch.dtype):
        kickMap = lambda length: QuadKick(length, k1, dim, self.dtype)

        super().__init__(length=length, kickMap=kickMap, dim=dim, slices=slices, order=order, dtype=dtype)

        return


if __name__ == "__main__":
    dim = 4
    order = 2
    slices = 1
    dtype = torch.float32

    drift = Drift(3, dim=dim, order=order, slices=slices, dtype=dtype)
    quad = Quadrupole(1, 0.3, dim=dim, order=order, slices=slices, dtype=dtype)

    # create particle
    x0 = torch.tensor([[1e-3, 2e-3, 1e-3, 0], ])

    # track
    x = quad(x0)
    print(x)

    print(quad.rMatrix())
