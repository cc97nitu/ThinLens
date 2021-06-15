import math
import torch
import torch.nn as nn

import ThinLens.Transformations


class Map(nn.Module):
    def __init__(self):
        super().__init__()
        self.length = 0.0
        return

    def madX(self) -> str:
        """Express this map via "arbitrary matrix" element from MAD-X."""
        rMatrix = self.rMatrix()

        elementDefinition = "MATRIX, L={}".format(self.length)

        for i in range(len(rMatrix)):
            for j in range(len(rMatrix[0])):
                elementDefinition += ", RM{}{}={}".format(i + 1, j + 1, rMatrix[i, j])

        elementDefinition += ";"
        return elementDefinition


class DriftMap(Map):
    """Propagate bunch along drift section."""

    def __init__(self, length: float):
        super().__init__()
        self.length = length

        # set up weights
        kernel = torch.tensor(self.length, dtype=torch.double)
        self.weight = nn.Parameter(kernel)

        return

    def forward(self, bunch: tuple):
        return ThinLens.Transformations.Drift.apply(*bunch, self.weight)

    def rMatrix(self):
        rMatrix = torch.eye(6, dtype=torch.double)
        rMatrix[0, 1] = self.weight
        rMatrix[2, 3] = self.weight

        return rMatrix


class DipoleKick(Map):
    """Apply an horizontal dipole kick."""

    def __init__(self, length: float, angle: float):
        super().__init__()
        self.dipoleLength = length  # used to calculate k0L for Mad-X
        self.lengthTensor = torch.tensor(self.dipoleLength, dtype=torch.double)  # input to phase-space transformation

        # initialize weight
        curvature = angle / length
        kernel = torch.tensor(curvature, dtype=torch.double)
        self.weight = nn.Parameter(kernel)

        return

    def forward(self, bunch: tuple):
        return ThinLens.Transformations.DipoleKick.apply(*bunch, self.lengthTensor, self.weight)

    def rMatrix(self):
        rMatrix = torch.eye(6, dtype=torch.double)
        rMatrix[1, 0] = -1 * self.weight ** 2 * self.dipoleLength
        rMatrix[4, 0] = -1 * self.weight * self.dipoleLength

        return rMatrix

    def thinMultipoleElement(self):
        k0L = self.weight.item() * self.dipoleLength  # horizontal bending angle
        return "LRAD={length}, KNL={{{k0L}}}".format(length=self.dipoleLength, k0L=k0L)


class EdgeKick(Map):
    """Dipole edge effects."""

    def __init__(self, length: float, bendAngle: float, edgeAngle: float):
        super().__init__()

        self.curvature = bendAngle / length
        self.curvatureTensor = torch.tensor(self.curvature, dtype=torch.double)

        self.edgeAngle = edgeAngle

        # initialize weight
        kernel = torch.tensor(math.tan(self.edgeAngle), dtype=torch.double)
        self.weight = nn.Parameter(kernel)

        return

    def forward(self, bunch: tuple):
        return ThinLens.Transformations.EdgeKick.apply(*bunch, self.weight, self.curvatureTensor)

    def rMatrix(self):
        rMatrix = torch.eye(6, dtype=torch.double)

        rMatrix[1, 0] = self.curvature * math.tan(self.edgeAngle)
        rMatrix[3, 2] = -1 * self.curvature * math.tan(self.edgeAngle)

        return rMatrix

    def thinMultipoleElement(self):
        return "H={}, E1={}".format(self.curvature, self.edgeAngle)


class MultipoleKick(Map):
    def __init__(self, length: float, kn: list = None, ks: list = None):
        super().__init__()
        self.length = 0.0  # dummy length used to locate kick / drift locations along the ring
        self.kickLength = torch.tensor(length,
                                       dtype=torch.double)  # length used to calculate integrated multipole strengths

        # register multipole strengths as weights <- one-day octupoles will be re-implemented
        if kn:
            for i in range(1, 4):
                weight = torch.tensor([kn[i - 1]], dtype=torch.double)
                self.register_parameter("k{}n".format(i), nn.Parameter(weight))
        else:
            self.k1n = nn.Parameter(torch.tensor([0, ], dtype=torch.double))
            self.k2n = nn.Parameter(torch.tensor([0, ], dtype=torch.double))
            self.k3n = nn.Parameter(torch.tensor([0, ], dtype=torch.double))

        if ks:
            for i in range(1, 4):
                weight = torch.tensor([ks[i - 1]], dtype=torch.double)
                self.register_parameter("k{}s".format(i), nn.Parameter(weight))
        else:
            self.k1s = nn.Parameter(torch.tensor([0, ], dtype=torch.double))
            self.k2s = nn.Parameter(torch.tensor([0, ], dtype=torch.double))
            self.k3s = nn.Parameter(torch.tensor([0, ], dtype=torch.double))

        return

    def forward(self, bunch: tuple):
        return ThinLens.Transformations.ThinMultipole.apply(*bunch, self.kickLength, self.k1n, self.k2n, self.k1s,
                                                            self.k2s)

    def rMatrix(self):
        """Calculate transfer matrix considering only linear optics."""
        rMatrix = torch.eye(6, dtype=torch.double)

        rMatrix[1, 0] = -1 * self.kickLength * self.k1n
        rMatrix[3, 2] = self.kickLength * self.k1n

        rMatrix[1, 2] = self.kickLength * self.k1s
        rMatrix[3, 0] = self.kickLength * self.k1s

        return rMatrix

    def thinMultipoleElement(self, nameVariables: bool = False):
        """
        Represent this map as Mad-X thin-multipole element.
        :param nameVariables: Replace numerical values by names of Mad-X variables.
        :return: Mad-X element definition.
        """
        integratedMultipoleStrengths = dict()

        integratedMultipoleStrengths["k1nl"] = self.kickLength.item() * self.k1n.item()
        integratedMultipoleStrengths["k2nl"] = self.kickLength.item() * self.k2n.item()
        integratedMultipoleStrengths["k3nl"] = self.kickLength.item() * self.k3n.item()

        integratedMultipoleStrengths["k1sl"] = self.kickLength.item() * self.k1s.item()
        integratedMultipoleStrengths["k2sl"] = self.kickLength.item() * self.k2s.item()
        integratedMultipoleStrengths["k3sl"] = self.kickLength.item() * self.k3s.item()

        if nameVariables:
            if self.k1n.item() > 0:
                integratedMultipoleStrengths["k1nl"] = "k1fl"
            elif self.k1n.item() < 0:
                integratedMultipoleStrengths["k1nl"] = "k1dl"
            else:
                integratedMultipoleStrengths["k1nl"] = 0.0

            if self.k2n.item() > 0:
                integratedMultipoleStrengths["k2nl"] = "k2fl"
            elif self.k2n.item() < 0:
                integratedMultipoleStrengths["k2nl"] = "k2dl"
            else:
                integratedMultipoleStrengths["k2nl"] = 0.0

        return "KNL:={{0.0, {k1nl}, {k2nl}, {k3nl}}}, KSL:={{0.0, {k1sl}, {k2sl}, {k3sl}}}".format(
            **integratedMultipoleStrengths)


if __name__ == "__main__":
    from Beam import Beam

    # create a beam
    beam = Beam(mass=18.798, energy=19.0, exn=1.258e-1, eyn=2.005e-1, sigt=0.01, sige=0.005, particles=int(1e3))
    bunch = beam.bunch.double().requires_grad_(True)
    loseBunch = bunch[:1].transpose(1, 0).unbind(0)

    # test drift
    drift = DriftMap(2.5)
    res = drift(loseBunch)

    # test dipole
    dipole = DipoleKick(2.13, 0.15)
    res = dipole(res)

    # test dipole edge
    edge = EdgeKick(2.13, 0.15, 0.1)
    res = edge(res)

    # test multipole
    magnet = MultipoleKick(0.1, kn=[0.5, -0.2, 0], ks=[0.05, 0.1, 0])
    res = magnet(res)
