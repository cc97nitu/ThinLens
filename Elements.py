import torch
import torch.nn as nn

from ThinLens.Maps import DriftMap, DipoleKick, EdgeKick, MultipoleKick


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


class Dummy(Drift):
    def __init__(self, length: float, dim: int, slices: int, order: int, dtype: torch.dtype):
        super(Dummy, self).__init__(length, dim, slices, order, dtype)
        return


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


class MultipoleKickElement(Element):
    """Base class for elements consisting of both drift and kicks."""

    def __init__(self, length: float, kn: list, ks: list, dim: int, slices: int, order: int, dtype: torch.dtype):
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)
        self.length = length

        # split scheme for hamiltonian
        if order == 2:
            self.coeffC = [1 / 2, 1 / 2]
            self.coeffD = [1, 0]
        else:
            raise NotImplementedError("order {} not implemented".format(order))

        # register common weights
        for i in range(1, 4):
            normWeight = torch.tensor([kn[i - 1]], dtype=dtype)
            self.register_parameter("k{}n".format(i), nn.Parameter(normWeight))
            skewWeight = torch.tensor([ks[i - 1]], dtype=dtype)
            self.register_parameter("k{}s".format(i), nn.Parameter(skewWeight))

        # same map for each slice
        self.maps = list()

        for c, d in zip(self.coeffC, self.coeffD):
            if c:
                self.maps.append(DriftMap(c * length / slices, dim, self.dtype))
            if d:
                self.maps.append(MultipoleKick(d * length / slices, dim, self.dtype))

        # use common weights
        for m in self.maps:
            if type(m) is DriftMap:
                continue

            m.k1n = self.k1n
            m.k2n = self.k2n
            m.k3n = self.k3n
            m.k1s = self.k1s
            m.k2s = self.k2s
            m.k3s = self.k3s

        self.maps = nn.ModuleList(self.maps * slices)
        return


class Quadrupole(MultipoleKickElement):
    def __init__(self, length: float, k1: float, dim: int, slices: int, order: int, dtype: torch.dtype):

        kn = [k1, 0, 0]
        ks = [0, 0, 0]
        super().__init__(length=length, kn=kn, ks=ks, dim=dim, slices=slices, order=order, dtype=dtype)

        return


class Sextupole(MultipoleKickElement):
    def __init__(self, length: float, k2: float, dim: int, slices: int, order: int, dtype: torch.dtype):

        kn = [0, k2, 0]
        ks = [0, 0, 0]
        super().__init__(length=length, kn=kn, ks=ks, dim=dim, slices=slices, order=order, dtype=dtype)

        return


if __name__ == "__main__":
    dim = 6
    order = 2
    slices = 4
    dtype = torch.float32

    drift = Drift(3, dim=dim, order=order, slices=slices, dtype=dtype)
    quad = Sextupole(1, 0.3, dim=dim, order=order, slices=slices, dtype=dtype)

    # create particle
    x0 = torch.tensor([[1e-3, 2e-3, 1e-3, 0, 0, 0, 0, 1, 1], ])

    # track
    x = quad(x0)
    print(x)

