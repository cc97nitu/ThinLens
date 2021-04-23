import torch
import torch.nn as nn

from ThinLens.Maps import DriftMap, DipoleKick, EdgeKick, MultipoleKick


class Element(nn.Module):
    def __init__(self, slices: int = 1, order: int = 2):
        super().__init__()

        self.slices = slices
        self.order = order

        return

    def forward(self, bunch: tuple):
        for m in self.maps:
            bunch = m(bunch)

        return bunch

    def rMatrix(self):
        rMatrix = torch.eye(self.dim, dtype=self.dtype)

        for m in self.maps:
            rMatrix = torch.matmul(m.rMatrix(), rMatrix)

        return rMatrix

    def getWeights(self) -> list:
        weights = list()
        for m in self.maps:
            weights.append(m.weight.tolist())

        return weights

    def setWeights(self, weights):
        weights = iter(weights)

        for m in self.maps:
            weight = torch.tensor(next(weights), dtype=self.dtype)
            m.weight = nn.Parameter(weight)

        return


class Drift(Element):
    def __init__(self, length: float, slices: int, order: int):
        super(Drift, self).__init__(slices=slices, order=order)
        self.length = length

        # ignore split scheme and slices for increased performance
        self.map = DriftMap(length)

        self.maps = nn.ModuleList([self.map, ])
        return

    def forward(self, bunch: tuple):
        return self.map(bunch)


class Dummy(Drift):
    def __init__(self, length: float, slices: int, order: int):
        super(Dummy, self).__init__(length, slices, order)
        return


class Monitor(Drift):
    """Special drift."""


class KickElement(Element):
    """Base class for elements consisting of both drift and kicks."""

    def __init__(self, length: float, kickMap, slices: int, order: int):
        super().__init__(slices=slices, order=order)
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
                self.maps.append(DriftMap(c * length / slices))
            if d:
                self.maps.append(kickMap(d * length / slices))

        self.maps = nn.ModuleList(self.maps * slices)
        return


class SBen(KickElement):
    """Horizontal sector bending magnet."""

    def __init__(self, length: float, angle: float, slices: int, order: int, e1: float = 0, e2: float = 0):
        kickMap = lambda length: DipoleKick(length, angle / slices)

        super().__init__(length=length, kickMap=kickMap, slices=slices, order=order)

        # edges present?
        if e1:
            self.maps.insert(0, EdgeKick(length, angle, e1))

        if e2:
            self.maps.append(EdgeKick(length, angle, e2))

        return


class RBen(SBen):
    """Horizontal rectangular bending magnet."""

    def __init__(self, length: float, angle: float, slices: int, order: int,
                 e1: float = 0, e2: float = 0):
        # modify edges
        e1 += angle / 2
        e2 += angle / 2

        super().__init__(length, angle, slices, order, e1=e1, e2=e2)
        return


class MultipoleKickElement(Element):
    """Base class for elements consisting of both drift and kicks."""

    def __init__(self, length: float, kn: list, ks: list, slices: int, order: int):
        super().__init__(slices=slices, order=order)
        self.length = length

        # split scheme for hamiltonian
        if order == 2:
            self.coeffC = [1 / 2, 1 / 2]
            self.coeffD = [1, 0]
        else:
            raise NotImplementedError("order {} not implemented".format(order))

        # register common weights
        for i in range(1, 4):
            normWeight = torch.tensor([kn[i - 1]], dtype=torch.double)
            self.register_parameter("k{}n".format(i), nn.Parameter(normWeight))
            skewWeight = torch.tensor([ks[i - 1]], dtype=torch.double)
            self.register_parameter("k{}s".format(i), nn.Parameter(skewWeight))

        # same map for each slice
        self.maps = list()

        for c, d in zip(self.coeffC, self.coeffD):
            if c:
                self.maps.append(DriftMap(c * length / slices))
            if d:
                self.maps.append(MultipoleKick(d * length / slices))

        # use common weights
        self.shareWeights()

        self.maps = nn.ModuleList(self.maps * slices)
        return

    def getWeights(self) -> dict:
        """Return dictionary containing multipole strengths."""
        weights = dict()
        weights["k1n"] = self.k1n.item()
        weights["k2n"] = self.k2n.item()
        weights["k3n"] = self.k3n.item()

        weights["k1s"] = self.k1s.item()
        weights["k2s"] = self.k2s.item()
        weights["k3s"] = self.k3s.item()

        return weights

    def setWeights(self, weights: dict):
        """Apply multipole strengths to element."""
        self.k1n = nn.Parameter(torch.tensor([weights["k1n"], ], dtype=torch.double))
        self.k2n = nn.Parameter(torch.tensor([weights["k2n"], ], dtype=torch.double))
        self.k3n = nn.Parameter(torch.tensor([weights["k3n"], ], dtype=torch.double))

        self.k1s = nn.Parameter(torch.tensor([weights["k1s"], ], dtype=torch.double))
        self.k2s = nn.Parameter(torch.tensor([weights["k2s"], ], dtype=torch.double))
        self.k3s = nn.Parameter(torch.tensor([weights["k3s"], ], dtype=torch.double))

        self.shareWeights()
        return

    def shareWeights(self):
        """Make maps use weights of parent element."""
        for m in self.maps:
            if type(m) is DriftMap:
                continue

            m.k1n = self.k1n
            m.k2n = self.k2n
            m.k3n = self.k3n
            m.k1s = self.k1s
            m.k2s = self.k2s
            m.k3s = self.k3s

        return

    def suppressMultipoles(self, threshold: float):
        """Remove any multipole component smaller than threshold."""
        if torch.abs(self.k1n.item()) < threshold:
            self.k1n = nn.Parameter(torch.zeros(1, dtype=torch.double), requires_grad=self.k1n.requires_grad)
        if torch.abs(self.k2n.item()) < threshold:
            self.k2n = nn.Parameter(torch.zeros(1, dtype=torch.double), requires_grad=self.k2n.requires_grad)
        if torch.abs(self.k3n.item()) < threshold:
            self.k3n = nn.Parameter(torch.zeros(1, dtype=torch.double), requires_grad=self.k3n.requires_grad)
        if torch.abs(self.k1s.item()) < threshold:
            self.k1s = nn.Parameter(torch.zeros(1, dtype=torch.double), requires_grad=self.k1s.requires_grad)
        if torch.abs(self.k2s.item()) < threshold:
            self.k2s = nn.Parameter(torch.zeros(1, dtype=torch.double), requires_grad=self.k2s.requires_grad)
        if torch.abs(self.k3s.item()) < threshold:
            self.k3s = nn.Parameter(torch.zeros(1, dtype=torch.double), requires_grad=self.k3s.requires_grad)

        self.shareWeights()
        return


class Quadrupole(MultipoleKickElement):
    def __init__(self, length: float, k1: float, slices: int, order: int):
        kn = [k1, 0, 0]
        ks = [0, 0, 0]
        super().__init__(length=length, kn=kn, ks=ks, slices=slices, order=order)

        return


class Sextupole(MultipoleKickElement):
    def __init__(self, length: float, k2: float, slices: int, order: int):
        kn = [0, k2, 0]
        ks = [0, 0, 0]
        super().__init__(length=length, kn=kn, ks=ks, slices=slices, order=order)

        return


# re-implement octupoles in Transformations first
# class Octupole(MultipoleKickElement):
#     def __init__(self, length: float, k3n: float, dim: int, slices: int, order: int, dtype: torch.dtype):
#         kn = [0, 0, k3n]
#         ks = [0, 0, 0]
#         super().__init__(length=length, kn=kn, ks=ks, dim=dim, slices=slices, order=order, dtype=dtype)
#
#         return


if __name__ == "__main__":
    import time
    from Beam import Beam

    order = 2
    slices = 4

    drift = Drift(3, order=order, slices=slices)
    quad = Quadrupole(0.1, 0.5, order=order, slices=slices)
    sext = Sextupole(0.1, -0.2, order=order, slices=slices)
    bend = RBen(2.13, 0.15, order=order, slices=slices)

    # create a beam
    beam = Beam(mass=18.798, energy=19.0, exn=1.258e-1, eyn=2.005e-1, sigt=0.01, sige=0.005, particles=int(1e3))
    bunch = beam.bunch.double().requires_grad_(True)
    print(bunch.shape)
    bunch0 = bunch.detach().clone()
    loseBunch = bunch.transpose(1, 0).unbind(0)

    # track
    t0 = time.time()

    turns = 2000
    x = loseBunch
    for i in range(turns):
        x = drift(x)

    inverseDrift = Drift(-3, order=order, slices=slices, )
    for i in range(turns):
        x = inverseDrift(x)

    print("tracking completed within {:.2f}s".format(time.time() - t0))

    # drifting forward and backward should result in the initial coordinates
    finalBunch = torch.stack(x, dim=1)
    print(torch.norm(bunch0 - finalBunch))

    # test some elements
    x = quad(x)
    x = sext(x)
    x = bend(x)

