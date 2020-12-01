import math
import torch
import torch.nn as nn

import ThinLens.Elements as Elements


class Model(nn.Module):
    def __init__(self, dim: int = 4, slices: int = 1, order: int = 2, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.generalProperties: dict = {"dim": dim, "dtype": dtype, "slices": slices, "order": order}
        self.dim = dim
        self.dtype = dtype

        # log element positions
        self.positions = list()
        self.endPositions = list()

        # must be set by child classes
        self.elements = None
        self.totalLen: float = 0

        return

    def forward(self, x, nTurns: int = 1, outputPerElement: bool = False, outputAtBPM: bool = False):
        if outputPerElement:
            outputs = list()
            for turn in range(nTurns):
                for e in self.elements:
                    x = e(x)
                    outputs.append(x)

            return torch.stack(outputs).permute(1, 2, 0)  # particle, dim, element
        # elif outputAtBPM:
        #     outputs = list()
        #     for turn in range(nTurns):
        #         for m in self.maps:
        #             x = m(x)
        #
        #             if type(m.element) is elements.Monitor:
        #                 outputs.append(x)
        #
        #     return torch.stack(outputs).permute(1, 2, 0)  # particle, dim, element
        else:
            for turn in range(nTurns):
                for e in self.elements:
                    x = e(x)

            return x

    def logElementPositions(self):
        """Store beginning and end of each element."""
        self.positions = list()
        self.endPositions = list()
        totalLength = 0

        for element in self.elements:
            self.positions.append(totalLength)
            totalLength += element.length
            self.endPositions.append(totalLength)

        self.totalLen = totalLength

        return

    def rMatrix(self):
        """Obtain linear transfer matrix."""
        rMatrix = torch.eye(self.dim, dtype=self.dtype)

        for element in self.elements:
            rMatrix = torch.matmul(element.rMatrix(), rMatrix)

        return rMatrix

    def getTunes(self) -> list:
        """Calculate tune from one-turn map."""
        oneTurnMap = self.rMatrix()

        xTrace = oneTurnMap[:2, :2].trace()
        xTune = torch.acos(1 / 2 * xTrace).item() / (2 * math.pi)

        if self.dim == 4 or self.dim == 6:
            yTrace = oneTurnMap[2:4, 2:4].trace()
            yTune = torch.acos(1 / 2 * yTrace).item() / (2 * math.pi)

            return [xTune, yTune]

        return [xTune, ]


class F0D0Model(Model):
    def __init__(self, k1: float, dim: int = 4, slices: int = 1, order: int = 2, dtype: torch.dtype = torch.float32):
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        # define elements
        d1 = Elements.Drift(1, **self.generalProperties)
        qf = Elements.Quadrupole(1, k1, **self.generalProperties)
        d2 = Elements.Drift(2, **self.generalProperties)
        qd = Elements.Quadrupole(1, -k1, **self.generalProperties)

        # add them to the model
        self.elements = nn.ModuleList([d1, qf, d2, qd])
        self.logElementPositions()
        return


class RBendLine(Model):
    def __init__(self, angle: float, e1: float, e2: float, dim: int = 4, slices: int = 1, order: int = 2,
                 dtype: torch.dtype = torch.float32):
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        # define beam line
        d1 = Elements.Drift(1, **self.generalProperties)
        rb1 = Elements.RBen(0.1, angle, e1=e1, e2=e2, **self.generalProperties)
        d2 = Elements.Drift(1, **self.generalProperties)

        # beam line
        self.elements = nn.ModuleList([d1, rb1, d2])
        self.logElementPositions()
        return


class SIS18_Cell_minimal(Model):
    def __init__(self, dim: int = 4, slices: int = 1, order: int = 2, dtype: torch.dtype = torch.float32):
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        # specify beam line elements
        rb1 = Elements.RBen(length=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354,
                            **self.generalProperties)
        rb2 = Elements.RBen(length=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354,
                            **self.generalProperties)

        k1f = 3.12391e-01  # tune: 4.2 (whole ring)
        k1d = -4.78047e-01  # tune: 3.3
        qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **self.generalProperties)
        qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **self.generalProperties)
        qs3t = Elements.Quadrupole(length=0.4804, k1=2 * k1f, **self.generalProperties)

        d1 = Elements.Drift(0.645, **self.generalProperties)
        d2 = Elements.Drift(0.9700000000000002, **self.generalProperties)
        d3 = Elements.Drift(6.839011704000001, **self.generalProperties)
        d4 = Elements.Drift(0.5999999999999979, **self.generalProperties)
        d5 = Elements.Drift(0.7097999999999978, **self.generalProperties)
        d6 = Elements.Drift(0.49979999100000283, **self.generalProperties)

        # set up beam line
        self.cell = [d1, rb1, d2, rb2, d3, qs1f, d4, qs2d, d5, qs3t, d6]

        # beam line
        self.elements = nn.ModuleList(self.cell)
        self.logElementPositions()
        return


class SIS18_Lattice_minimal(Model):
    def __init__(self, k1f: float = 3.12391e-01, k1d: float = -4.78047e-01, dim: int = 4, slices: int = 1,
                 order: int = 2, dtype: torch.dtype = torch.float32):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * 4

        # SIS18 consists of 12 identical cells
        beamline = list()
        for i in range(12):
            # specify beam line elements
            rb1 = Elements.RBen(length=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354,
                                **self.generalProperties)
            rb2 = Elements.RBen(length=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354,
                                **self.generalProperties)

            qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
            qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
            qs3t = Elements.Quadrupole(length=0.4804, k1=2 * k1f, **quadrupoleGeneralProperties)

            d1 = Elements.Drift(0.645, **self.generalProperties)
            d2 = Elements.Drift(0.9700000000000002, **self.generalProperties)
            d3 = Elements.Drift(6.839011704000001, **self.generalProperties)
            d4 = Elements.Drift(0.5999999999999979, **self.generalProperties)
            d5 = Elements.Drift(0.7097999999999978, **self.generalProperties)
            d6 = Elements.Drift(0.49979999100000283, **self.generalProperties)

            beamline.append([d1, rb1, d2, rb2, d3, qs1f, d4, qs2d, d5, qs3t, d6])

        # flatten beamline
        flattenedBeamline = list()
        for cell in beamline:
            for element in cell:
                flattenedBeamline.append(element)

        self.elements = nn.ModuleList(flattenedBeamline)
        self.logElementPositions()
        return


if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=True)

    dtype = torch.double

    angle = 0.2
    e1 = 0.1
    e2 = 0.1
    # model = RBendLine(angle, e1=e1, e2=e2, slices=5, dtype=dtype)
    # model.requires_grad_(False)

    model = SIS18_Lattice_minimal(slices=4, dtype=dtype)
    # model = SIS18_Cell_minimal(dtype=dtype)

    # create particle
    x0 = torch.tensor([[1e-3, 2e-3, 1e-3, 0], ], dtype=dtype)  # x, xp, y, yp
    # x0 = torch.tensor([[1e-3, 1e-3, 2e-3, 0],])  # x, y, xp, yp

    # track
    x = model(x0, outputPerElement=True)
    print(x)

    # test symplecticity
    sym = torch.tensor([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=dtype)

    rMatrix = model.rMatrix()
    res = torch.matmul(rMatrix.transpose(1, 0), torch.matmul(sym, rMatrix)) - sym
    print("sym penalty: {}".format(torch.norm(res)))

    print("tunes: {}".format(model.getTunes()))

    ############ compare with TorchOcelot
    import sys

    sys.path.append("../TorchOcelot/src/")
    from Simulation.Lattice import DummyLattice
    from Simulation.Models import LinearModel

    # Ocelot
    print("ocelot")

    lattice = DummyLattice(angle=angle, e1=e1, e2=e2)
    ocelotModel = LinearModel(lattice)

    print(ocelotModel(torch.as_tensor(x0, dtype=torch.float32)))
    print(ocelotModel.oneTurnMap())
