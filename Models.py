import math
import json
import typing
import collections

import numpy as np

import torch
import torch.nn as nn

import ThinLens.Elements as Elements
import ThinLens.Maps as Maps

import sixtracklib as stl


class TwissFailed(ValueError):
    """Indicate a problem with twiss-calculation."""

    def __init__(self, message):
        self.message = message
        super(TwissFailed, self).__init__(self.message)
        return


class Model(nn.Module):
    class MonitorDummy(object):
        """Used to represent monitors in case of concatenated drifts."""

        def __init__(self):
            super().__init__()
            self.length = 0.0
            return

    def __init__(self, slices: int = 1, order: int = 2):
        super().__init__()
        self.generalProperties: dict = {"slices": slices, "order": order}

        self.modelType = {"type": type(self).__name__, "slices": slices, "order": order}

        # log element positions
        self.positions = list()
        self.endPositions = list()

        # must be set by child classes
        self.elements = None
        self.totalLen: float = 0

        # placeholder for merged lattice
        self.mergedMaps = None

        return

    def forward(self, x: torch.tensor, nTurns: int = 1, outputPerElement: bool = False, outputAtBPM: bool = False,
                rotate: typing.Union[None, int] = None, mergeDrifts: bool = False):
        # create lose bunch
        x = x.transpose(1, 0).unbind(0)

        if rotate is None:
            elements = self.elements
        else:
            elements = collections.deque(self.elements)
            elements.rotate(rotate)

        if mergeDrifts:
            if outputPerElement:
                raise ValueError("cannot produce output per element if drifts are merged")

            if not self.mergedMaps:
                self.mergedMaps = self.mergeDrifts()

            if rotate is not None:
                mergedMaps = self.mergeDrifts(elements)
            else:
                mergedMaps = self.mergedMaps

        if outputPerElement:
            outputs = list()
            for turn in range(nTurns):
                for e in elements:
                    x = e(x)
                    outputs.append(x)

            # merge coordinates into single bunch
            for i in range(len(outputs)):
                outputs[i] = torch.stack(outputs[i], dim=1)

            return torch.stack(outputs).permute(1, 2, 0)  # particle, dim, element
        elif outputAtBPM:
            outputs = list()
            for turn in range(nTurns):
                if mergeDrifts:
                    for m in mergedMaps:
                        if type(m) is Model.MonitorDummy:
                            outputs.append(x)
                        else:
                            x = m(x)
                else:
                    for e in elements:
                        x = e(x)

                        if type(e) is Elements.Monitor:
                            outputs.append(x)

            # merge coordinates into single bunch
            for i in range(len(outputs)):
                outputs[i] = torch.stack(outputs[i], dim=1)

            return torch.stack(outputs).permute(1, 2, 0)  # particle, dim, element
        else:
            if mergeDrifts:
                for turn in range(nTurns):
                    for m in mergedMaps:
                        if type(m) is Model.MonitorDummy:
                            continue
                        x = m(x)
            else:
                for turn in range(nTurns):
                    for e in elements:
                        x = e(x)

            return torch.stack(x, dim=1)

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
        rMatrix = torch.eye(6, dtype=torch.double)

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

    def thinMultipoleMadX(self, nameVariables: bool = False):
        """Export as Mad-X sequence consisting of thin-multipole and dipole edge elements."""
        # create single string containing whole sequence
        templates = ""
        sequence = "sis18: sequence, l = {};\n".format(self.totalLen)

        currentPos = 0.0
        kickIdentifier = 0
        monitorIdentifier = 0
        for element in self.elements:
            if type(element) is Elements.Drift or type(Elements) is Elements.Dummy:
                # drifts are added automatically by Mad-X
                currentPos += element.length
                continue
            elif type(element) is Elements.Monitor:
                # update current position
                currentPos += element.length

                # add monitor to sequence
                templates += "BPM{}: MONITOR, L=0".format(monitorIdentifier) + ";\n"
                sequence += "BPM{}, at={};\n".format(monitorIdentifier, currentPos)  # add Monitor to end of BPM

                monitorIdentifier += 1
                continue

            for m in element.maps:
                currentPos += m.length

                if type(m) is Maps.DriftMap:
                    # drifts are added automatically by Mad-X
                    continue

                if type(m) is Maps.EdgeKick:
                    # dipole edges cannot be expressed as thin multipoles
                    templates += "dipedge{}: dipedge, ".format(kickIdentifier) + m.thinMultipoleElement() + ";\n"
                    sequence += "dipedge{}, at={};\n".format(kickIdentifier, currentPos)
                else:
                    # add template
                    if type(m) is Maps.MultipoleKick:
                        templates += "kick{}: MULTIPOLE, ".format(kickIdentifier) + m.thinMultipoleElement(
                            nameVariables=nameVariables) + ";\n"
                    else:
                        templates += "kick{}: MULTIPOLE, ".format(kickIdentifier) + m.thinMultipoleElement() + ";\n"

                    sequence += "kick{}, at={};\n".format(kickIdentifier, currentPos)

                kickIdentifier += 1

        sequence += "\nendsequence;"

        # lattice contains templates and sequence
        lattice = templates + "\n" + sequence
        return lattice

    def mergeDrifts(self, elements: typing.Union[None, list] = None):
        """Create a map-based model with consecutive drifts merged."""
        if elements is None:
            elements = self.elements

        maps = list()
        for element in elements:
            for m in element.maps:
                maps.append(m)

            if type(element) is Elements.Monitor:
                maps.append(Model.MonitorDummy())

        # remove consecutive drifts
        for i in reversed(range(len(maps) - 1)):
            try:
                curMap, prevMap = maps[i], maps[i + 1]
            except IndexError:
                break

            if type(curMap) is Maps.DriftMap and type(prevMap) is Maps.DriftMap:
                maps[i] = Maps.DriftMap(curMap.length + prevMap.length)
                del maps[i + 1]

        return maps

    def sixTrackLib(self, numStores: int = 1, installBPMs: bool = True, finalPhaseSpace: bool = False):
        """Export model to SixTrackLib."""
        myElem = stl.Elements()

        if not installBPMs and not finalPhaseSpace:
            raise ValueError("no output specified")

        # build map-wise
        maps = self.mergeDrifts()

        # add maps to SixTrackLib
        for m in maps:
            if type(m) is Maps.DriftMap:
                myElem.DriftExact(length=m.length)
            elif type(m) is Maps.DipoleKick:
                k0L = m.weight.item() * m.dipoleLength

                myElem.Multipole(length=m.dipoleLength, hxl=m.angle, knl=[k0L, ])
            elif type(m) is Maps.EdgeKick:
                myElem.DipoleEdge(h=m.curvature, e1=m.edgeAngle)
            elif type(m) is Maps.MultipoleKick:
                k1nl = m.kickLength.item() * m.k1n.item()
                k2nl = m.kickLength.item() * m.k2n.item()
                k1sl = m.kickLength.item() * m.k1s.item()
                k2sl = m.kickLength.item() * m.k2s.item()

                knl = [0.0, k1nl, k2nl]
                ksl = [0.0, k1sl, k2sl]
                myElem.Multipole(knl=knl, ksl=ksl)
            elif type(m) is Model.MonitorDummy:
                if installBPMs:
                    myElem.BeamMonitor(num_stores=numStores)
                else:
                    continue
            else:
                raise NotImplementedError()

        if finalPhaseSpace:
            myElem.BeamMonitor(num_stores=numStores)

        return myElem

    def trackWithSTL(self, beam, nTurns, outputAtBPM=True, finalPhaseSpace=False):
        # track with BPMs
        elements = self.sixTrackLib(nTurns, installBPMs=outputAtBPM, finalPhaseSpace=finalPhaseSpace)
        particles = beam.sixTrackLibParticles()

        jobBPM = stl.TrackJob(elements, particles, device=None)

        jobBPM.track_until(nTurns)
        jobBPM.collect()

        # bring tracking results into same shape as model output
        spatial = list()
        for bpm in jobBPM.output.particles:
            x = bpm.x.reshape(-1, len(beam.bunch))
            px = bpm.px.reshape(-1, len(beam.bunch))
            y = bpm.y.reshape(-1, len(beam.bunch))
            py = bpm.py.reshape(-1, len(beam.bunch))
            zeta = bpm.zeta.reshape(-1, len(beam.bunch))
            delta = bpm.delta.reshape(-1, len(beam.bunch))
            velocityRatio = 1 / bpm.rvv.reshape(-1, len(beam.bunch))  # beta0 / beta

            sigma = zeta * velocityRatio  # <=> zeta / (beta0 / beta)

            spatialCoordinates = np.stack([x, px, y, py, sigma, delta, velocityRatio])
            spatial.append(spatialCoordinates)  # (dim, turn, particle)

        spatial = np.stack(spatial)  # bpm, dim, turn, particle
        output = [spatial[:, :, i, :] for i in range(spatial.shape[2])]
        output = np.concatenate(output)  # bpm, dim, particle

        return torch.as_tensor(np.transpose(output, (2, 1, 0)), ), bpm  # particle, dim, bpm

    def getTunes(self) -> list:
        """Calculate tune from one-turn map."""
        oneTurnMap = self.rMatrix()

        xTrace = oneTurnMap[:2, :2].trace()
        xTune = torch.acos(1 / 2 * xTrace).item() / (2 * math.pi)

        yTrace = oneTurnMap[2:4, 2:4].trace()
        yTune = torch.acos(1 / 2 * yTrace).item() / (2 * math.pi)

        return [xTune, yTune]

    def getInitialTwiss(self):
        """Calculate twiss parameters of periodic solution at lattice start."""
        oneTurnMap = self.rMatrix()

        # verify absence of coupling
        xyCoupling = oneTurnMap[:2, 2:4]
        yxCoupling = oneTurnMap[2:4, :2]
        couplingIndicator = torch.norm(xyCoupling) + torch.norm(yxCoupling)

        if couplingIndicator != 0:
            raise TwissFailed("coupled motion detected")

        # does a stable solution exist?
        cosMuX = 1 / 2 * oneTurnMap[:2, :2].trace()

        if torch.abs(cosMuX) > 1:
            raise TwissFailed("no periodic solution, cosine(phaseAdvance) out of bounds")

        # calculate twiss from one-turn map
        sinMuX = torch.sign(oneTurnMap[0, 1]) * torch.sqrt(1 - cosMuX ** 2)
        betaX0 = oneTurnMap[0, 1] / sinMuX
        alphaX0 = 1 / (2 * sinMuX) * (oneTurnMap[0, 0] - oneTurnMap[1, 1])

        cosMuY = 1 / 2 * oneTurnMap[2:4, 2:4].trace()

        if torch.abs(cosMuX) > 1:
            raise ValueError("no periodic solution, cosine(phaseAdvance) out of bounds")

        sinMuY = torch.sign(oneTurnMap[2, 3]) * torch.sqrt(1 - cosMuY ** 2)
        betaY0 = oneTurnMap[2, 3] / sinMuY
        alphaY0 = 1 / (2 * sinMuY) * (oneTurnMap[2, 2] - oneTurnMap[3, 3])

        return tuple([betaX0, alphaX0]), tuple([betaY0, alphaY0])

    def twissTransportMatrix(self, rMatrix: torch.Tensor):
        """Convert transport matrix into twiss transport matrix."""
        # x-plane
        xMat = rMatrix[:2, :2]
        c, cp, s, sp = xMat[0, 0], xMat[1, 0], xMat[0, 1], xMat[1, 1]

        twissTransportX = torch.tensor([[c ** 2, -2 * s * c, s ** 2],
                                        [-1 * c * cp, s * cp + sp * c, -1 * s * sp],
                                        [cp ** 2, -2 * sp * cp, sp ** 2], ], dtype=torch.double)

        # y-plane
        yMat = rMatrix[2:4, 2:4]
        c, cp, s, sp = yMat[0, 0], yMat[1, 0], yMat[0, 1], yMat[1, 1]

        twissTransportY = torch.tensor([[c ** 2, -2 * s * c, s ** 2],
                                        [-1 * c * cp, s * cp + sp * c, -1 * s * sp],
                                        [cp ** 2, -2 * sp * cp, sp ** 2], ], dtype=torch.double)

        return twissTransportX, twissTransportY

    def getTwiss(self):
        # get initial twiss
        twissX0, twissY0 = self.getInitialTwiss()

        pos = [0, ]
        betaX, alphaX, betaY, alphaY = [twissX0[0]], [twissX0[1]], [twissY0[0]], [twissY0[1]]
        twissX0 = torch.tensor([betaX[-1], alphaX[-1], (1 + alphaX[-1] ** 2) / betaX[-1]], dtype=torch.double)
        twissY0 = torch.tensor([betaY[-1], alphaY[-1], (1 + alphaY[-1] ** 2) / betaY[-1]], dtype=torch.double)

        lengths = [0, ]
        mux = [0, ]

        # calculate twiss along lattice
        rMatrix = torch.eye(6, dtype=torch.double)

        for element in self.elements:
            for m in element.maps:
                # update position
                pos.append(pos[-1] + m.length)

                # update twiss
                rMatrix = torch.matmul(m.rMatrix(), rMatrix)
                twissTransportX, twissTransportY = self.twissTransportMatrix(rMatrix)

                twissX = torch.matmul(twissTransportX, twissX0)
                twissY = torch.matmul(twissTransportY, twissY0)

                betaX.append(twissX[0].item())
                alphaX.append(twissX[1].item())
                betaY.append(twissY[0].item())
                alphaY.append(twissY[0].item())

                # # update phase advance
                # lengths.append(m.length)
                # betaXMean = 1/2 * (betaX[-1] + betaX[-2])
                # mux.append(1/betaX[-1] * m.length)

        # store results
        twiss = dict()
        twiss["s"] = torch.tensor(pos, dtype=torch.double)

        twiss["betx"] = torch.tensor(betaX, dtype=torch.double)
        twiss["alfx"] = torch.tensor(alphaX, dtype=torch.double)
        twiss["bety"] = torch.tensor(betaY, dtype=torch.double)
        twiss["alfy"] = torch.tensor(alphaY, dtype=torch.double)

        # # calculate phase advance
        # twiss["mux"] = torch.cumsum(torch.tensor(mux, dtype=torch.double), dim=0)
        # twiss["muy"] = torch.cumsum(1/twiss["bety"], dim=0)
        # # twiss["mux"] = torch.tensor(mux, dtype=torch.double)
        #
        # lengths = torch.tensor(lengths, dtype=torch.double)
        # mux = [torch.trapz(1/twiss["betx"][:i+1], lengths[:i+1]) for i in range(len(twiss["betx"]))]
        # twiss["mux"] = torch.tensor(mux, dtype=torch.double)

        return twiss

    def dumpJSON(self, fileHandler: typing.TextIO):
        """Save model to disk."""
        modelDescription = self.toJSON()

        fileHandler.write(modelDescription)
        return

    def loadJSON(self, fileHandler: typing.TextIO):
        """Load model from disk."""
        modelDescription = fileHandler.read()

        self.fromJSON(modelDescription)
        return

    def toJSON(self):
        """Return model description as string."""
        # store all weights
        weights = [self.modelType, ]
        for e in self.elements:
            weights.append(e.getWeights())

        # return as JSON string
        return json.dumps(weights)

    def fromJSON(self, description: str):
        """Load model from string."""
        weights = iter(json.loads(description))

        # check if model dump is of same type as self
        modelType = next(weights)
        if not modelType == self.modelType:
            print(modelType)
            print(self.modelType)
            raise IOError("file contains wrong model type")

        for e in self.elements:
            e.setWeights(next(weights))

        return


class F0D0Model(Model):
    def __init__(self, k1: float, slices: int = 1, order: int = 2):
        super().__init__(slices=slices, order=order)

        # define elements
        d1 = Elements.Drift(1, **self.generalProperties)
        qf = Elements.Quadrupole(0.3, k1, **self.generalProperties)
        d2 = Elements.Drift(1, **self.generalProperties)
        qd = Elements.Quadrupole(0.3, -k1, **self.generalProperties)

        # add them to the model
        self.elements = nn.ModuleList([d1, qf, d2, qd])
        self.logElementPositions()
        return


class RBendLine(Model):
    def __init__(self, angle: float, e1: float, e2: float, slices: int = 1, order: int = 2):
        super().__init__(slices=slices, order=order)

        # define beam line
        d1 = Elements.Drift(1, **self.generalProperties)
        rb1 = Elements.RBen(0.1, angle, e1=e1, e2=e2, **self.generalProperties)
        d2 = Elements.Drift(1, **self.generalProperties)

        # beam line
        self.elements = nn.ModuleList([d1, rb1, d2])
        self.logElementPositions()
        return


class SBendLine(Model):
    def __init__(self, angle: float, e1: float, e2: float, slices: int = 1, order: int = 2):
        super().__init__(slices=slices, order=order)

        # define beam line
        d1 = Elements.Drift(0.5, **self.generalProperties)
        rb1 = Elements.SBen(0.1, angle, e1=e1, e2=e2, **self.generalProperties)
        d2 = Elements.Drift(0.5, **self.generalProperties)

        # beam line
        self.elements = nn.ModuleList([d1, rb1, d2])
        self.logElementPositions()
        return


class SIS18_Cell_minimal(Model):
    def __init__(self, k1f: float = 0.3525911342676681, k1d: float = -0.3388671731064351,
                 k1f_support: typing.Union[None, float] = 0, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.4
        super().__init__(slices=slices, order=order)

        # specify beam line elements
        bendingAngle = 0.2617993878
        rb1 = Elements.RBen(length=2.617993878, angle=bendingAngle, e1=0, e2=0,
                            **self.generalProperties)
        rb2 = Elements.RBen(length=2.617993878, angle=bendingAngle, e1=0, e2=0,
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

        if k1f_support is not None:
            qs3t = Elements.Quadrupole(length=0.4804, k1=k1f_support, **quadrupoleGeneralProperties)
        else:
            qs3t = Elements.Drift(length=0.4804, **self.generalProperties)

        # set up beam line
        self.cell = [d1, rb1, d2, rb2, d3, qs1f, d4, qs2d, d5, qs3t, d6]

        # beam line
        self.elements = nn.ModuleList(self.cell)
        self.logElementPositions()
        return


class SIS18_Cell_minimal_noDipoles(Model):
    def __init__(self, k1f: float = 0.3525911342676681, k1d: float = -0.3388671731064351,
                 k1f_support: typing.Union[None, float] = 0, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(slices=slices, order=order)

        # specify beam line elements
        d3 = Elements.Drift(2 * 2.617993878 + 0.645 + 0.9700000000000002 + 6.839011704000001,
                            **self.generalProperties)  # replace beginning of SIS18_Cell_minimal by a long drift
        d4 = Elements.Drift(0.5999999999999979, **self.generalProperties)
        d5 = Elements.Drift(0.7097999999999978, **self.generalProperties)
        d6 = Elements.Drift(0.49979999100000283, **self.generalProperties)

        # quadrupoles shall be sliced more due to their strong influence on tunes
        quadrupoleGeneralProperties = dict(self.generalProperties)
        quadrupoleGeneralProperties["slices"] = self.generalProperties["slices"] * quadSliceMultiplicity

        qs1f = Elements.Quadrupole(length=1.04, k1=k1f, **quadrupoleGeneralProperties)
        qs2d = Elements.Quadrupole(length=1.04, k1=k1d, **quadrupoleGeneralProperties)

        if k1f_support is not None:
            qs3t = Elements.Quadrupole(length=0.4804, k1=k1f_support, **quadrupoleGeneralProperties)
        else:
            qs3t = Elements.Drift(length=0.4804, **self.generalProperties)

        # set up beam line
        self.cell = [d3, qs1f, d4, qs2d, d5, qs3t, d6]

        # beam line
        self.elements = nn.ModuleList(self.cell)
        self.logElementPositions()
        return


class SIS18_Cell(Model):
    def __init__(self, k1f: float = 0.3525911342676681, k1d: float = -0.3388671731064351,
                 k1f_support: typing.Union[None, float] = 0, k2f: float = 0, k2d: float = 0, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(slices=slices, order=order)
        self.quadSliceMultiplicity = quadSliceMultiplicity

        # define beam line elements
        bendingAngle = 0.2617993878
        rb1a = Elements.RBen(length=2.617993878 / 2, angle=bendingAngle / 2, e1=0, e2=0,
                             **self.generalProperties)
        rb1b = Elements.RBen(length=2.617993878 / 2, angle=bendingAngle / 2, e1=0, e2=0,
                             **self.generalProperties)
        rb2a = Elements.RBen(length=2.617993878 / 2, angle=bendingAngle / 2, e1=0, e2=0,
                             **self.generalProperties)
        rb2b = Elements.RBen(length=2.617993878 / 2, angle=bendingAngle / 2, e1=0, e2=0,
                             **self.generalProperties)

        # sextupoles
        ks1c = Elements.Sextupole(length=0.32, k2=k2f, **self.generalProperties)
        ks3c = Elements.Sextupole(length=0.32, k2=k2d, **self.generalProperties)

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

        if k1f_support is not None:
            qs3t = Elements.Quadrupole(length=0.4804, k1=k1f_support, **quadrupoleGeneralProperties)
        else:
            qs3t = Elements.Drift(length=0.4804, **self.generalProperties)

        # set up beam line
        self.cell = [d1, rb1a, hKick1, rb1b, d2, rb2a, hKick2, rb2b, d3a, ks1c, d3b, qs1f, vKick, d4, qs2d, d5a, ks3c,
                     d5b,
                     qs3t, d6a, hMon, vMon, d6b]

        # beam line
        self.elements = nn.ModuleList(self.cell)
        self.logElementPositions()
        return


class SIS18_Cell_noDipoles(Model):
    def __init__(self, k1f: float = 0.3525911342676681, k1d: float = -0.3388671731064351,
                 k1f_support: typing.Union[None, float] = 0, k2f: float = 0, k2d: float = 0, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(slices=slices, order=order)
        self.quadSliceMultiplicity = quadSliceMultiplicity

        # define beam line elements
        rb1a = Elements.Drift(length=2.617993878 / 2,
                              **self.generalProperties)
        rb1b = Elements.Drift(length=2.617993878 / 2,
                              **self.generalProperties)
        rb2a = Elements.Drift(length=2.617993878 / 2,
                              **self.generalProperties)
        rb2b = Elements.Drift(length=2.617993878 / 2,
                              **self.generalProperties)

        # sextupoles
        ks1c = Elements.Sextupole(length=0.32, k2=k2f, **self.generalProperties)
        ks3c = Elements.Sextupole(length=0.32, k2=k2d, **self.generalProperties)

        # one day there will be correctors
        hKick1 = Elements.Dummy(0, **self.generalProperties)
        hKick2 = Elements.Dummy(0, **self.generalProperties)
        vKick = Elements.Dummy(0, **self.generalProperties)

        hMon = Elements.Monitor(0.13275, **self.generalProperties)
        vMon = Elements.Monitor(0.13275, **self.generalProperties)

        # d1 = Elements.Drift(0.2, **self.generalProperties)
        d1 = Elements.Drift(0.2 + 2.617993878 / 2, **self.generalProperties)
        d2 = Elements.Drift(0.9700000000000002 + 2.617993878, **self.generalProperties)
        d3a = Elements.Drift(6.345 + 2.617993878 / 2, **self.generalProperties)
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

        if k1f_support is not None:
            qs3t = Elements.Quadrupole(length=0.4804, k1=k1f_support, **quadrupoleGeneralProperties)
        else:
            qs3t = Elements.Drift(length=0.4804, **self.generalProperties)

        # set up beam line
        self.cell = [d1, hKick1, d2, hKick2, d3a, ks1c, d3b, qs1f, vKick, d4, qs2d, d5a, ks3c,
                     d5b,
                     qs3t, d6a, hMon, vMon, d6b]

        # beam line
        self.elements = nn.ModuleList(self.cell)
        self.logElementPositions()
        return


class SIS18_Cell_oneBPM(Model):
    def __init__(self, k1f: float = 0.3525911342676681, k1d: float = -0.3388671731064351,
                 k1f_support: typing.Union[None, float] = 0, k2f: float = 0, k2d: float = 0, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(slices=slices, order=order)
        self.quadSliceMultiplicity = quadSliceMultiplicity

        # define beam line elements
        bendingAngle = 0.2617993878
        rb1a = Elements.RBen(length=2.617993878 / 2, angle=bendingAngle / 2, e1=0, e2=0,
                             **self.generalProperties)
        rb1b = Elements.RBen(length=2.617993878 / 2, angle=bendingAngle / 2, e1=0, e2=0,
                             **self.generalProperties)
        rb2a = Elements.RBen(length=2.617993878 / 2, angle=bendingAngle / 2, e1=0, e2=0,
                             **self.generalProperties)
        rb2b = Elements.RBen(length=2.617993878 / 2, angle=bendingAngle / 2, e1=0, e2=0,
                             **self.generalProperties)

        # sextupoles
        ks1c = Elements.Sextupole(length=0.32, k2=k2f, **self.generalProperties)
        ks3c = Elements.Sextupole(length=0.32, k2=k2d, **self.generalProperties)

        # one day there will be correctors
        hKick1 = Elements.Dummy(0, **self.generalProperties)
        hKick2 = Elements.Dummy(0, **self.generalProperties)
        vKick = Elements.Dummy(0, **self.generalProperties)

        hMon = Elements.Monitor(0.13275, **self.generalProperties)
        vMonDrift = Elements.Drift(0.13275, **self.generalProperties)

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

        if k1f_support is not None:
            qs3t = Elements.Quadrupole(length=0.4804, k1=k1f_support, **quadrupoleGeneralProperties)
        else:
            qs3t = Elements.Drift(length=0.4804, **self.generalProperties)

        # set up beam line
        self.cell = [d1, rb1a, hKick1, rb1b, d2, rb2a, hKick2, rb2b, d3a, ks1c, d3b, qs1f, vKick, d4, qs2d, d5a, ks3c,
                     d5b,
                     qs3t, d6a, hMon, vMonDrift, d6b]

        # beam line
        self.elements = nn.ModuleList(self.cell)
        self.logElementPositions()
        return


class SIS18_Lattice_minimal_noDipoles(Model):
    def __init__(self, k1f: float = 0.3525911342676681, k1d: float = -0.3388671731064351,
                 k1f_support: typing.Union[None, float] = 0, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, cellsIdentical: bool = False):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.4
        super().__init__(slices=slices, order=order)

        # SIS18 consists of 12 cells
        self.cells = list()
        beamline = list()

        if cellsIdentical:
            cell = SIS18_Cell_minimal_noDipoles(k1f=k1f, k1d=k1d, k1f_support=k1f_support, slices=slices, order=order,
                                                quadSliceMultiplicity=quadSliceMultiplicity)

            for i in range(12):
                self.cells.append(cell)
                beamline += cell.elements
        else:
            for i in range(12):
                # create cell
                cell = SIS18_Cell_minimal_noDipoles(k1f=k1f, k1d=k1d, k1f_support=k1f_support, slices=slices,
                                                    order=order, quadSliceMultiplicity=quadSliceMultiplicity)
                self.cells.append(cell)
                beamline += cell.elements

        self.elements = nn.ModuleList(beamline)
        self.logElementPositions()
        return


class SIS18_Lattice_minimal(Model):
    def __init__(self, k1f: float = 0.3525911342676681, k1d: float = -0.3388671731064351,
                 k1f_support: typing.Union[None, float] = 0, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4, cellsIdentical: bool = False):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.4
        super().__init__(slices=slices, order=order)

        # SIS18 consists of 12 cells
        self.cells = list()
        beamline = list()

        if cellsIdentical:
            cell = SIS18_Cell_minimal(k1f=k1f, k1d=k1d, k1f_support=k1f_support, slices=slices, order=order,
                                      quadSliceMultiplicity=quadSliceMultiplicity)

            for i in range(12):
                self.cells.append(cell)
                beamline += cell.elements
        else:
            for i in range(12):
                # create cell
                cell = SIS18_Cell_minimal(k1f=k1f, k1d=k1d, k1f_support=k1f_support, slices=slices, order=order,
                                          quadSliceMultiplicity=quadSliceMultiplicity)
                self.cells.append(cell)
                beamline += cell.elements

        self.elements = nn.ModuleList(beamline)
        self.logElementPositions()
        return


class SIS18_Lattice_noDipoles(Model):
    def __init__(self, k1f: float = 0.3525911342676681, k1d: float = -0.3388671731064351, k1f_support: float = 0,
                 k2f: float = 0, k2d: float = 0, slices: int = 1, order: int = 2, quadSliceMultiplicity: int = 4,
                 cellsIdentical: bool = False):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(slices=slices, order=order)
        self.quadSliceMultiplicity = quadSliceMultiplicity

        # SIS18 consists of 12 cells
        self.cells = list()
        beamline = list()

        if cellsIdentical:
            cell = SIS18_Cell_noDipoles(k1f=k1f, k1d=k1d, k1f_support=k1f_support, k2f=k2f, k2d=k2d, slices=slices,
                                        order=order,
                                        quadSliceMultiplicity=quadSliceMultiplicity)

            for i in range(12):
                self.cells.append(cell)
                beamline += cell.elements
        else:
            for i in range(12):
                cell = SIS18_Cell_noDipoles(k1f=k1f, k1d=k1d, k1f_support=k1f_support, k2f=k2f, k2d=k2d, slices=slices,
                                            order=order, quadSliceMultiplicity=quadSliceMultiplicity)

                self.cells.append(cell)
                beamline += cell.elements

        self.elements = nn.ModuleList(beamline)
        self.logElementPositions()
        return


class SIS18_Lattice_oneBPM(Model):
    def __init__(self, k1f: float = 0.3525911342676681, k1d: float = -0.3388671731064351, k1f_support: float = 0,
                 k2f: float = 0, k2d: float = 0, slices: int = 1, order: int = 2, quadSliceMultiplicity: int = 4,
                 cellsIdentical: bool = False):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(slices=slices, order=order)
        self.quadSliceMultiplicity = quadSliceMultiplicity

        # SIS18 consists of 12 cells
        self.cells = list()
        beamline = list()

        if cellsIdentical:
            cell = SIS18_Cell_oneBPM(k1f=k1f, k1d=k1d, k1f_support=k1f_support, k2f=k2f, k2d=k2d, slices=slices,
                                     order=order,
                                     quadSliceMultiplicity=quadSliceMultiplicity)

            for i in range(12):
                self.cells.append(cell)
                beamline += cell.elements
        else:
            for i in range(12):
                cell = SIS18_Cell_oneBPM(k1f=k1f, k1d=k1d, k1f_support=k1f_support, k2f=k2f, k2d=k2d, slices=slices,
                                         order=order, quadSliceMultiplicity=quadSliceMultiplicity)

                self.cells.append(cell)
                beamline += cell.elements

        self.elements = nn.ModuleList(beamline)
        self.logElementPositions()
        return


class SIS18_Lattice(Model):
    def __init__(self, k1f: float = 0.3525911342676681, k1d: float = -0.3388671731064351, k1f_support: float = 0,
                 k2f: float = 0, k2d: float = 0, slices: int = 1, order: int = 2, quadSliceMultiplicity: int = 4,
                 cellsIdentical: bool = False):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(slices=slices, order=order)
        self.quadSliceMultiplicity = quadSliceMultiplicity

        # SIS18 consists of 12 cells
        self.cells = list()
        beamline = list()

        if cellsIdentical:
            cell = SIS18_Cell(k1f=k1f, k1d=k1d, k1f_support=k1f_support, k2f=k2f, k2d=k2d, slices=slices, order=order,
                              quadSliceMultiplicity=quadSliceMultiplicity)

            for i in range(12):
                self.cells.append(cell)
                beamline += cell.elements
        else:
            for i in range(12):
                cell = SIS18_Cell(k1f=k1f, k1d=k1d, k1f_support=k1f_support, k2f=k2f, k2d=k2d, slices=slices,
                                  order=order, quadSliceMultiplicity=quadSliceMultiplicity)

                self.cells.append(cell)
                beamline += cell.elements

        self.elements = nn.ModuleList(beamline)
        self.logElementPositions()

        # prepare merged drifts
        self.mergedMaps = self.mergeDrifts()
        return


class SIS18_Cell_S(Model):
    def __init__(self, k1f: float = 0.3525911342676681, k1d: float = -0.3388671731064351,
                 k1f_support: typing.Union[None, float] = 0, k2f: float = 0, k2d: float = 0, slices: int = 1,
                 order: int = 2, quadSliceMultiplicity: int = 4):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(slices=slices, order=order)
        self.quadSliceMultiplicity = quadSliceMultiplicity

        # define beam line elements
        bendingAngle = 0.2617993878
        rb1a = Elements.SBen(length=2.617993878 / 2, angle=bendingAngle / 2, e1=0, e2=0,
                             **self.generalProperties)
        rb1b = Elements.SBen(length=2.617993878 / 2, angle=bendingAngle / 2, e1=0, e2=0,
                             **self.generalProperties)
        rb2a = Elements.SBen(length=2.617993878 / 2, angle=bendingAngle / 2, e1=0, e2=0,
                             **self.generalProperties)
        rb2b = Elements.SBen(length=2.617993878 / 2, angle=bendingAngle / 2, e1=0, e2=0,
                             **self.generalProperties)

        # sextupoles
        ks1c = Elements.Sextupole(length=0.32, k2=k2f, **self.generalProperties)
        ks3c = Elements.Sextupole(length=0.32, k2=k2d, **self.generalProperties)

        # one day there will be correctors
        hKick1 = Elements.Dummy(0, **self.generalProperties)
        hKick2 = Elements.Dummy(0, **self.generalProperties)
        vKick = Elements.Dummy(0, **self.generalProperties)

        hMon = Elements.Monitor(0.13275, **self.generalProperties)
        vMonDrift = Elements.Drift(0.13275, **self.generalProperties)

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

        if k1f_support is not None:
            qs3t = Elements.Quadrupole(length=0.4804, k1=k1f_support, **quadrupoleGeneralProperties)
        else:
            qs3t = Elements.Drift(length=0.4804, **self.generalProperties)

        # set up beam line
        self.cell = [d1, rb1a, hKick1, rb1b, d2, rb2a, hKick2, rb2b, d3a, ks1c, d3b, qs1f, vKick, d4, qs2d, d5a, ks3c,
                     d5b,
                     qs3t, d6a, hMon, vMonDrift, d6b]

        # beam line
        self.elements = nn.ModuleList(self.cell)
        self.logElementPositions()
        return


class SIS18_Lattice_S(Model):
    def __init__(self, k1f: float = 0.3525911342676681, k1d: float = -0.3388671731064351, k1f_support: float = 0,
                 k2f: float = 0, k2d: float = 0, slices: int = 1, order: int = 2, quadSliceMultiplicity: int = 4,
                 cellsIdentical: bool = False):
        # default values for k1f, k1d correspond to a tune of 4.2, 3.3
        super().__init__(slices=slices, order=order)
        self.quadSliceMultiplicity = quadSliceMultiplicity

        # SIS18 consists of 12 cells
        self.cells = list()
        beamline = list()

        if cellsIdentical:
            cell = SIS18_Cell_S(k1f=k1f, k1d=k1d, k1f_support=k1f_support, k2f=k2f, k2d=k2d, slices=slices, order=order,
                              quadSliceMultiplicity=quadSliceMultiplicity)

            for i in range(12):
                self.cells.append(cell)
                beamline += cell.elements
        else:
            for i in range(12):
                cell = SIS18_Cell_S(k1f=k1f, k1d=k1d, k1f_support=k1f_support, k2f=k2f, k2d=k2d, slices=slices,
                                  order=order, quadSliceMultiplicity=quadSliceMultiplicity)

                self.cells.append(cell)
                beamline += cell.elements

        self.elements = nn.ModuleList(beamline)
        self.logElementPositions()

        # prepare merged drifts
        self.mergedMaps = self.mergeDrifts()
        return


if __name__ == "__main__":
    import timeit
    import matplotlib.pyplot as plt

    torch.set_printoptions(precision=4, sci_mode=True)

    # set up models
    mod1 = SIS18_Lattice(slices=4, quadSliceMultiplicity=4)

    # dump to string
    modelDescription = mod1.toJSON()
    mod1.fromJSON(modelDescription)

    # dump to file
    with open("/dev/shm/modelDump.json", "w") as f:
        mod1.dumpJSON(f)

    with open("/dev/shm/modelDump.json", "r") as f:
        mod1.loadJSON(f)

    # check map concatenation
    x = lambda: mod1.mergeDrifts()
    x()
    duration = timeit.timeit(x, number=100)
    print(duration)
