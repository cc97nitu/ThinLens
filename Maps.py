import math
import torch
import torch.nn as nn


class DriftMap(nn.Conv1d):
    """Propagate bunch along drift section."""

    def __init__(self, length: float, dim: int, dtype: torch.dtype):
        super().__init__(1, 1, 3, padding=1, bias=False)
        self.dim = dim
        self.dtype = dtype
        self.length = length

        # set up weights
        kernel = torch.tensor([[[length, 0, length], ], ], dtype=self.dtype)
        self.weight = nn.Parameter(kernel)

        #
        self.forward = self.forward4D
        return

    def forward4D(self, x):
        # get momenta in reversed order
        momenta = x[:, [1, 3]].flip(1).unsqueeze(1)

        # get updated position
        pos = super().forward(momenta).squeeze(1)  # feed momenta only
        pos = pos + x[:, [0, 2]]

        # update phase space vector
        xT = x.transpose(1, 0)
        posT = pos.transpose(1, 0)

        x = torch.stack([posT[0], xT[1], posT[1], xT[3]], ).transpose(1, 0)

        return x

    def rMatrix(self):
        momentaRows = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 1]], dtype=self.dtype)

        positionRows = torch.tensor(
            [[1, self.weight[0, 0, 2], 0, self.weight[0, 0, 1]], [0, self.weight[0, 0, 1], 1, self.weight[0, 0, 0]]],
            dtype=self.dtype)

        rMatrix = torch.stack([positionRows[0], momentaRows[0], positionRows[1], momentaRows[1]])
        return rMatrix

    def madX(self) -> str:
        """Express this map via "arbitrary matrix" element from MAD-X."""
        rMatrix = self.rMatrix()

        elementDefinition = "MATRIX, L={}".format(self.length)

        for i in range(len(rMatrix)):
            for j in range(len(rMatrix[0])):
                elementDefinition += ", RM{}{}={}".format(i + 1, j + 1, rMatrix[i, j])

        elementDefinition += ";"
        return elementDefinition



class KickMap(nn.Conv1d):
    """Apply an update to momenta."""

    def __init__(self, dim: int, dtype: torch.dtype):
        super().__init__(1, 1, 3, padding=1, bias=False)
        self.dim = dim
        self.dtype = dtype
        self.length = 0

        return

    def forward(self, x):
        # get positions in reversed order
        pos = x[:, [0, 2]].flip(1).unsqueeze(1)

        # get updated momenta
        momenta = super().forward(pos).squeeze(1)
        momenta = momenta + x[:, [1, 3]]

        # update phase space vector
        xT = x.transpose(1, 0)
        momentaT = momenta.transpose(1, 0)

        x = torch.stack([xT[0], momentaT[0], xT[2], momentaT[1]], ).transpose(1, 0)
        return x

    def rMatrix(self):
        positionRows = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=self.dtype)

        momentaRows = torch.tensor(
            [[self.weight[0, 0, 2], 1, self.weight[0, 0, 1], 0], [self.weight[0, 0, 1], 0, self.weight[0, 0, 0], 1]],
            dtype=self.dtype)

        rMatrix = torch.stack([positionRows[0], momentaRows[0], positionRows[1], momentaRows[1]])
        return rMatrix

    def madX(self) -> str:
        """Express this map via "arbitrary matrix" element from MAD-X."""
        rMatrix = self.rMatrix()

        elementDefinition = "MATRIX, L=0"

        for i in range(len(rMatrix)):
            for j in range(len(rMatrix[0])):
                elementDefinition += ", RM{}{}={}".format(i + 1, j + 1, rMatrix[i, j])

        elementDefinition += ";"
        return elementDefinition



# class QuadKick(KickMap):
#     """Apply a quadrupole kick."""
#
#     def __init__(self, length: float, k1: float, dim: int, dtype: torch.dtype):
#         super().__init__(dim=dim, dtype=dtype)
#
#         # initialize weight
#         kernel = torch.tensor([[[length * k1, 0, -1 * length * k1], ], ], dtype=dtype)
#         self.weight = nn.Parameter(kernel)
#         return


class QuadKick(nn.Module):
    """Decoupled planes."""

    def __init__(self, length: float, k1: float, dim: int, dtype: torch.dtype):
        super().__init__()
        self.dim = dim
        self.dtype = dtype
        self.length = 0.0

        weight = torch.tensor([-1 * length * k1, length * k1], dtype=dtype)
        self.register_parameter("weight", nn.Parameter(weight))


        return

    def forward(self, x):
        # get positions in reversed order
        pos = x[:, [0, 2]]

        # get updated momenta
        momenta = self.weight * pos
        momenta = momenta + x[:, [1, 3]]

        # update phase space vector
        xT = x.transpose(1, 0)
        momentaT = momenta.transpose(1, 0)

        x = torch.stack([xT[0], momentaT[0], xT[2], momentaT[1]], ).transpose(1, 0)
        return x

    def rMatrix(self):
        positionRows = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=self.dtype)

        momentaRows = torch.tensor(
            [[self.weight[0], 1, 0, 0], [0, 0, self.weight[1], 1]],
            dtype=self.dtype)

        rMatrix = torch.stack([positionRows[0], momentaRows[0], positionRows[1], momentaRows[1]])
        return rMatrix

    def madX(self) -> str:
        """Express this map via "arbitrary matrix" element from MAD-X."""
        rMatrix = self.rMatrix()

        elementDefinition = "MATRIX, L=0"

        for i in range(len(rMatrix)):
            for j in range(len(rMatrix[0])):
                elementDefinition += ", RM{}{}={}".format(i + 1, j + 1, rMatrix[i, j])

        elementDefinition += ";"
        return elementDefinition



class DipoleKick(KickMap):
    """Apply an horizontal dipole kick."""

    def __init__(self, length: float, angle: float, dim: int, dtype: torch.dtype):
        super().__init__(dim=dim, dtype=dtype)

        # initialize weight
        curvature = angle / length

        kernel = torch.tensor([[[0, 0, -1 * curvature ** 2 * length], ], ], dtype=dtype)
        self.weight = nn.Parameter(kernel)

        return


class EdgeKick(KickMap):
    """Dipole edge effects."""

    def __init__(self, length: float, bendAngle: float, edgeAngle: float, dim: int, dtype: torch.dtype):
        super().__init__(dim=dim, dtype=dtype)

        # initialize weight
        curvature = bendAngle / length

        kernel = torch.tensor([[[-1 * curvature * math.tan(edgeAngle), 0, curvature * math.tan(edgeAngle)], ], ],
                              dtype=dtype)
        self.weight = nn.Parameter(kernel)

        return


if __name__ == "__main__":
    dim = 4
    dtype = torch.double

    # set up quad
    quad = QuadKick(1, 0.1, dim, dtype)

    # track
    x = torch.randn(2, dim, dtype=dtype)

    print(x)

    print("standard quad")
    print(quad(x))
    print(quad.rMatrix())
    print(quad.madX())

