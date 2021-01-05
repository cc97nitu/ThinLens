import math
import torch
import torch.nn as nn

import ThinLens.Elements as Elements
import ThinLens.Maps as Maps


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
        elif outputAtBPM:
            outputs = list()
            for turn in range(nTurns):
                for e in self.elements:
                    x = e(x)

                    if type(e) is Elements.Monitor:
                        outputs.append(x)

            return torch.stack(outputs).permute(1, 2, 0)  # particle, dim, element
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

    def madX(self) -> str:
        """Provide a sequence and necessary templates in order to import this model into Mad-X."""
        # get templates for every map
        elementMaps = list()

        identifier = 0
        for element in self.elements:
            for m in element.maps:
                elementMaps.append(
                    tuple([m.length, m.madX(), identifier]))  # (length, madX element template, identifier)
                identifier += 1

        # create single string containing whole sequence
        templates = ""
        sequence = "sis18: sequence, l = {};\n".format(self.totalLen)

        # add templates and elements
        beginPos = 0  # location of beginning of element (slice) of current map
        for length, template, identifier in elementMaps:
            refPos = beginPos + length / 2  # center of element

            templates += "map{}: ".format(identifier) + template + "\n"
            sequence += "map{}, at={};\n".format(identifier, refPos)

            beginPos += length

        sequence += "\nendsequence;"

        # lattice contains templates and sequence
        lattice = templates + "\n" + sequence
        return lattice

    def thinMultipoleMadX(self):
        # create single string containing whole sequence
        templates = ""
        sequence = "sis18: sequence, l = {};\n".format(self.totalLen)

        currentPos = 0.0
        kickIdentifier = 0
        for element in self.elements:
            if type(element) is Elements.Drift or type(element) is Elements.Monitor or type(Elements) is Elements.Dummy:
                # drifts are added automatically by Mad-X
                currentPos += element.length
                continue

            for m in element.maps:
                currentPos += m.length

                if type(m) is Maps.DriftMap:
                    # drifts are added automatically by Mad-X
                    continue

                if type(m) is Maps.EdgeKick:
                    # dipole edges cannot be expressed as thin multipoles
                    templates += "map{}: ".format(kickIdentifier) + m.madX() + "\n"
                    sequence += "map{}, at={};\n".format(kickIdentifier, currentPos)
                else:
                    # add template
                    templates += "kick{}: MULTIPOLE, ".format(kickIdentifier) + m.thinMultipoleElement() + ";\n"
                    sequence += "kick{}, at={};\n".format(kickIdentifier, currentPos)

                kickIdentifier += 1

        sequence += "\nendsequence;"

        # lattice contains templates and sequence
        lattice = templates + "\n" + sequence
        return lattice

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
    def __init__(self, k1f: float = 3.12391e-01, k1d: float = -4.78047e-01, dim: int = 4, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, dtype: torch.dtype = torch.float32):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        # specify beam line elements
        rb1 = Elements.RBen(length=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354,
                            **self.generalProperties)
        rb2 = Elements.RBen(length=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354,
                            **self.generalProperties)

        d1 = Elements.Drift(0.645, **self.generalProperties)
        d2 = Elements.Drift(0.9700000000000002, **self.generalProperties)
        d3 = Elements.Drift(6.839011704000001, **self.generalProperties)
        d4 = Elements.Drift(0.5999999999999979, **self.generalProperties)
        d5 = Elements.Drift(0.7097999999999978, **self.generalProperties)
        d6 = Elements.Drift(0.49979999100000283, **self.generalProperties)

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * quadSliceMultiplicity

        qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
        qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
        qs3t = Elements.Quadrupole(length=0.4804, k1=2 * k1f, **quadrupoleGeneralProperties)

        # set up beam line
        self.cell = [d1, rb1, d2, rb2, d3, qs1f, d4, qs2d, d5, qs3t, d6]

        # beam line
        self.elements = nn.ModuleList(self.cell)
        self.logElementPositions()
        return


class SIS18_DoubleCell_minimal(Model):
    def __init__(self, k1f: float = 3.12391e-01, k1d: float = -4.78047e-01, dim: int = 4, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, dtype: torch.dtype = torch.float32):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * quadSliceMultiplicity

        # SIS18 consists of 12 identical cells
        beamline = list()
        for i in range(2):
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


class SIS18_DoubleCell_minimal_identical(Model):
    def __init__(self, k1f: float = 3.12391e-01, k1d: float = -4.78047e-01, dim: int = 4, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, dtype: torch.dtype = torch.float32):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        # specify beam line elements
        rb1 = Elements.RBen(length=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354,
                            **self.generalProperties)
        rb2 = Elements.RBen(length=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354,
                            **self.generalProperties)

        d1 = Elements.Drift(0.645, **self.generalProperties)
        d2 = Elements.Drift(0.9700000000000002, **self.generalProperties)
        d3 = Elements.Drift(6.839011704000001, **self.generalProperties)
        d4 = Elements.Drift(0.5999999999999979, **self.generalProperties)
        d5 = Elements.Drift(0.7097999999999978, **self.generalProperties)
        d6 = Elements.Drift(0.49979999100000283, **self.generalProperties)

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * quadSliceMultiplicity

        qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
        qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
        qs3t = Elements.Quadrupole(length=0.4804, k1=2 * k1f, **quadrupoleGeneralProperties)

        # set up beam line
        self.cell = [d1, rb1, d2, rb2, d3, qs1f, d4, qs2d, d5, qs3t, d6]

        # beam line
        self.elements = nn.ModuleList(self.cell)
        self.logElementPositions()
        return


class SIS18_Cell(Model):
    def __init__(self, k1f: float = 3.12391e-01, k1d: float = -4.78047e-01, dim: int = 4, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, dtype: torch.dtype = torch.float32):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)
        self.quadSliceMultiplicity = quadSliceMultiplicity

        # define beam line elements
        rb1a = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0.1274090354, e2=0,
                             **self.generalProperties)
        rb1b = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0, e2=0.1274090354,
                             **self.generalProperties)
        rb2a = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0.1274090354, e2=0,
                             **self.generalProperties)
        rb2b = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0, e2=0.1274090354,
                             **self.generalProperties)

        # one day there will be sextupoles
        ks1c = Elements.Dummy(length=0.32, **self.generalProperties)
        ks3c = Elements.Dummy(length=0.32, **self.generalProperties)

        # one day there will be correctors
        hKick1 = Elements.Dummy(0, **self.generalProperties)
        hKick2 = Elements.Dummy(0, **self.generalProperties)
        vKick = Elements.Dummy(0, **self.generalProperties)

        hMon = Elements.Monitor(0.13275, **self.generalProperties)
        vMon = Elements.Monitor(0.13275, **self.generalProperties)

        d1 = Elements.Drift(0.2, **self.generalProperties)
        d2 = Elements.Drift(0.9700000000000002, **self.generalProperties)
        d3a = Elements.Drift(6.345, **self.generalProperties)
        d3b = Elements.Drift(0.175, **self.generalProperties)
        d4 = Elements.Drift(0.5999999999999979, **self.generalProperties)
        d5a = Elements.Drift(0.195, **self.generalProperties)
        d5b = Elements.Drift(0.195, **self.generalProperties)
        d6a = Elements.Drift(0.3485, **self.generalProperties)
        d6b = Elements.Drift(0.3308, **self.generalProperties)

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * self.quadSliceMultiplicity

        qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
        qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
        qs3t = Elements.Quadrupole(length=0.4804, k1=2 * k1f, **quadrupoleGeneralProperties)

        # set up beam line
        self.cell = [d1, rb1a, hKick1, rb1b, d2, rb2a, hKick2, rb2b, d3a, ks1c, d3b, qs1f, vKick, d4, qs2d, d5a, ks3c,
                     d5b,
                     qs3t, d6a, hMon, vMon, d6b]

        # beam line
        self.elements = nn.ModuleList(self.cell)
        self.logElementPositions()
        return


class SIS18_Lattice_minimal(Model):
    def __init__(self, k1f: float = 3.12391e-01, k1d: float = -4.78047e-01, dim: int = 4, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, dtype: torch.dtype = torch.float32):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * quadSliceMultiplicity

        # SIS18 consists of 12 identical cells
        self.cells = list()
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

            cell = [d1, rb1, d2, rb2, d3, qs1f, d4, qs2d, d5, qs3t, d6]
            self.cells.append(cell)
            beamline.append(cell)

        # flatten beamline
        flattenedBeamline = list()
        for cell in beamline:
            for element in cell:
                flattenedBeamline.append(element)

        self.elements = nn.ModuleList(flattenedBeamline)
        self.logElementPositions()
        return


class SIS18_Lattice(Model):
    def __init__(self, k1f: float = 3.12391e-01, k1d: float = -4.78047e-01, dim: int = 4, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, dtype: torch.dtype = torch.float32, cellsIdentical: bool = False):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)
        self.quadSliceMultiplicity = quadSliceMultiplicity

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * quadSliceMultiplicity

        # SIS18 consists of 12 identical cells
        self.cells = list()
        beamline = list()
        if cellsIdentical:
            # specify beam line elements
            rb1a = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0.1274090354, e2=0,
                                 **self.generalProperties)
            rb1b = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0, e2=0.1274090354,
                                 **self.generalProperties)
            rb2a = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0.1274090354, e2=0,
                                 **self.generalProperties)
            rb2b = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0, e2=0.1274090354,
                                 **self.generalProperties)

            # one day there will be sextupoles
            ks1c = Elements.Drift(length=0.32, **self.generalProperties)
            ks3c = Elements.Drift(length=0.32, **self.generalProperties)

            # one day there will be correctors
            hKick1 = Elements.Drift(0, **self.generalProperties)
            hKick2 = Elements.Drift(0, **self.generalProperties)
            vKick = Elements.Drift(0, **self.generalProperties)

            hMon = Elements.Monitor(0.13275, **self.generalProperties)
            vMon = Elements.Monitor(0.13275, **self.generalProperties)

            d1 = Elements.Drift(0.2, **self.generalProperties)
            d2 = Elements.Drift(0.9700000000000002, **self.generalProperties)
            d3a = Elements.Drift(6.345, **self.generalProperties)
            d3b = Elements.Drift(0.175, **self.generalProperties)
            d4 = Elements.Drift(0.5999999999999979, **self.generalProperties)
            d5a = Elements.Drift(0.195, **self.generalProperties)
            d5b = Elements.Drift(0.195, **self.generalProperties)
            d6a = Elements.Drift(0.3485, **self.generalProperties)
            d6b = Elements.Drift(0.3308, **self.generalProperties)

            # quadrupoles shall be sliced more due to their strong influence on tunes
            quadrupoleGeneralProperties = dict(self.generalProperties)
            quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * self.quadSliceMultiplicity

            qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
            qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
            qs3t = Elements.Quadrupole(length=0.4804, k1=2 * k1f, **quadrupoleGeneralProperties)

            for i in range(12):
                cell = [d1, rb1a, hKick1, rb1b, d2, rb2a, hKick2, rb2b, d3a, ks1c, d3b, qs1f, vKick, d4, qs2d, d5a, ks3c,
                     d5b,
                     qs3t, d6a, hMon, vMon, d6b]

                self.cells.append(cell)
                beamline.append(cell)

        else:
            for i in range(12):
                # specify beam line elements
                rb1a = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0.1274090354, e2=0,
                                     **self.generalProperties)
                rb1b = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0, e2=0.1274090354,
                                     **self.generalProperties)
                rb2a = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0.1274090354, e2=0,
                                     **self.generalProperties)
                rb2b = Elements.RBen(length=2.617993878 / 2, angle=0.2617993878 / 2, e1=0, e2=0.1274090354,
                                     **self.generalProperties)

                # one day there will be sextupoles
                ks1c = Elements.Drift(length=0.32, **self.generalProperties)
                ks3c = Elements.Drift(length=0.32, **self.generalProperties)

                # one day there will be correctors
                hKick1 = Elements.Drift(0, **self.generalProperties)
                hKick2 = Elements.Drift(0, **self.generalProperties)
                vKick = Elements.Drift(0, **self.generalProperties)

                hMon = Elements.Monitor(0.13275, **self.generalProperties)
                vMon = Elements.Monitor(0.13275, **self.generalProperties)

                d1 = Elements.Drift(0.2, **self.generalProperties)
                d2 = Elements.Drift(0.9700000000000002, **self.generalProperties)
                d3a = Elements.Drift(6.345, **self.generalProperties)
                d3b = Elements.Drift(0.175, **self.generalProperties)
                d4 = Elements.Drift(0.5999999999999979, **self.generalProperties)
                d5a = Elements.Drift(0.195, **self.generalProperties)
                d5b = Elements.Drift(0.195, **self.generalProperties)
                d6a = Elements.Drift(0.3485, **self.generalProperties)
                d6b = Elements.Drift(0.3308, **self.generalProperties)

                # quadrupoles shall be sliced more due to their strong influence on tunes
                quadrupoleGeneralProperties = dict(self.generalProperties)
                quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * self.quadSliceMultiplicity

                qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
                qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **quadrupoleGeneralProperties)
                qs3t = Elements.Quadrupole(length=0.4804, k1=2 * k1f, **quadrupoleGeneralProperties)

                cell = [d1, rb1a, hKick1, rb1b, d2, rb2a, hKick2, rb2b, d3a, ks1c, d3b, qs1f, vKick, d4, qs2d, d5a, ks3c,
                     d5b,
                     qs3t, d6a, hMon, vMon, d6b]

                self.cells.append(cell)
                beamline.append(cell)

        # flatten beamline
        flattenedBeamline = list()
        for cell in beamline:
            for element in cell:
                flattenedBeamline.append(element)

        self.elements = nn.ModuleList(flattenedBeamline)
        self.logElementPositions()
        return


if __name__ == "__main__":
    import ThinLens.Maps

    torch.set_printoptions(precision=4, sci_mode=True)

    dtype = torch.double

    # set up models
    mod1 = SIS18_Cell(dtype=dtype,)

    # get locations
    print(mod1.madX())
