import math
import torch
import torch.nn as nn


class Map(nn.Module):
    def __init__(self, dim: int, dtype: torch.dtype):
        super().__init__()
        self.dim = dim
        self.dtype = dtype
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

    def __init__(self, length: float, dim: int, dtype: torch.dtype):
        super().__init__(dim, dtype)
        self.length = length

        if dim == 4:
            # set up weights
            kernel = torch.tensor([self.length, self.length], dtype=self.dtype)
            self.weight = nn.Parameter(kernel)

            self.forward = self.forward4D
        elif dim == 6:
            # set up weights
            kernel = torch.tensor([self.length, self.length], dtype=self.dtype)
            self.weight = nn.Parameter(kernel)

            self.forward = self.forward6D
        else:
            raise NotImplementedError("dim {} not supported".format(dim))

        return

    def forward4D(self, x):
        # get momenta
        momenta = x[:, [1, 3]]

        # get updated momenta
        pos = self.length * momenta
        pos = pos + x[:, [0, 2]]

        # update phase space vector
        xT = x.transpose(1, 0)
        posT = pos.transpose(1, 0)

        x = torch.stack([posT[0], xT[1], posT[1], xT[3]], ).transpose(1, 0)

        return x

    def forward6D(self, x):
        # get momenta
        momenta = x[:, [1, 3, ]]
        velocityRatio = x[:, 8]

        # get updated momenta
        pos = self.weight * momenta
        sigma = self.length - self.length * velocityRatio
        pos = pos + x[:, [0, 2]]
        sigma = sigma + x[:, 4]

        # update phase space vector
        xT = x.transpose(1, 0)
        posT = pos.transpose(1, 0)

        x = torch.stack([posT[0], xT[1], posT[1], xT[3], sigma, *xT[5:]], ).transpose(1, 0)
        return x

    def rMatrix(self):
        if self.dim == 4:
            rMatrix = torch.eye(4, dtype=self.dtype)
            rMatrix[0, 1] = self.weight[0]
            rMatrix[2, 3] = self.weight[1]
        else:
            rMatrix = torch.eye(6, dtype=self.dtype)
            rMatrix[0, 1] = self.weight[0]
            rMatrix[2, 3] = self.weight[1]

        return rMatrix


class QuadKick(Map):
    """Decoupled planes."""

    def __init__(self, length: float, k1: float, dim: int, dtype: torch.dtype):
        super().__init__(dim, dtype)
        self.length = 0.0

        weight = torch.tensor([length * k1], dtype=dtype)
        self.register_parameter("weight", nn.Parameter(weight))

        if dim == 4:
            self.forward = self.forward4D
        elif dim == 6:
            self.forward = self.forward6D
        else:
            raise NotImplementedError("dim {} not supported".format(dim))

        return

    def forward4D(self, x):
        # get positions in reversed order
        # pos = x[:, [0, 2]]
        xPos, yPos = x[:, 0], x[:, 2]

        # get updated momenta
        # momenta = self.weight * pos
        # momenta = momenta + x[:, [1, 3]]
        dPx = -1 * self.weight * xPos
        dPy = self.weight * yPos
        px = x[:, 1] + dPx
        py = x[:, 3] + dPy

        # update phase space vector
        xT = x.transpose(1, 0)
        # momentaT = momenta.transpose(1, 0)

        # x = torch.stack([xT[0], momentaT[0], xT[2], momentaT[1]], ).transpose(1, 0)
        x = torch.stack([xT[0], px, xT[2], py], ).transpose(1, 0)
        return x

    def forward6D(self, x):
        # get positions
        # pos = x[:, [0, 2]]
        xPos, yPos = x[:, 0], x[:, 2]
        # invDelta = x[:, 7].unsqueeze(1)
        invDelta = x[:, 7]

        # get updated momenta
        # momenta = self.weight * invDelta * pos
        # momenta = momenta + x[:, [1, 3]]
        dPx = -1 * self.weight * invDelta * xPos
        dPy = self.weight * invDelta * yPos
        px = x[:, 1] + dPx
        py = x[:, 3] + dPy


        # update phase space vector
        xT = x.transpose(1, 0)
        # momentaT = momenta.transpose(1, 0)

        x = torch.stack([xT[0], px, xT[2], py, *xT[4:]]).transpose(1, 0)
        return x

    def rMatrix(self):
        # if self.dim == 4:
        #     rMatrix = torch.eye(4, dtype=self.dtype)
        #     rMatrix[1, 0] = self.weight[0]
        #     rMatrix[3, 2] = self.weight[1]
        # else:
        #     rMatrix = torch.eye(6, dtype=self.dtype)
        #     rMatrix[1, 0] = self.weight[0]
        #     rMatrix[3, 2] = self.weight[1]

        if self.dim == 4:
            rMatrix = torch.eye(4, dtype=self.dtype)
            rMatrix[1, 0] = -1 * self.weight
            rMatrix[3, 2] = self.weight
        else:
            rMatrix = torch.eye(6, dtype=self.dtype)
            rMatrix[1, 0] = -1 * self.weight
            rMatrix[3, 2] = self.weight

        return rMatrix


class DipoleKick(Map):
    """Apply an horizontal dipole kick."""

    def __init__(self, length: float, angle: float, dim: int, dtype: torch.dtype):
        super().__init__(dim=dim, dtype=dtype)

        # initialize weight
        curvature = angle / length

        if dim == 4:
            kernel = torch.tensor([-1 * curvature ** 2 * length], dtype=dtype)
            self.weight = nn.Parameter(kernel)

            self.forward = self.forward4D
        elif dim == 6:
            kernel = torch.tensor([-1 * curvature ** 2 * length, -1 * curvature * length,], dtype=dtype)
            self.weight = nn.Parameter(kernel)

            self.forward = self.forward6D
        else:
            raise NotImplementedError("dim {} not supported".format(dim))

        return

    def forward4D(self, x):
        # get horizontal position
        pos = x[:, 0]

        # get updated momenta
        momenta = self.weight * pos
        momenta = momenta + x[:, 1]

        # update phase space vector
        xT = x.transpose(1, 0)

        x = torch.stack([xT[0], momenta, xT[2], xT[3]], ).transpose(1, 0)
        return x

    def forward6D(self, x):
        # get x and sigma
        pos = x[:, [0, 4]]

        delta = x[:, 6]
        velocityRatio = x[:, 8]

        # get updates
        px = self.weight[0] * pos[:, 0] + -1 * self.weight[1] * delta
        px = x[:, 1] + px

        sigma = self.weight[1] * velocityRatio
        sigma = x[:, 4] + sigma

        # update phase space vector
        xT = x.transpose(1, 0)

        x = torch.stack([xT[0], px, xT[2], xT[3], sigma, *xT[5:]]).transpose(1, 0)
        return x

    def rMatrix(self):
        if self.dim == 4:
            rMatrix = torch.eye(4, dtype=self.dtype)
            rMatrix[1, 0] = self.weight[0]
        else:
            rMatrix = torch.eye(6, dtype=self.dtype)
            rMatrix[1, 0] = self.weight[0]
            rMatrix[4, 0] = self.weight[1]

        return rMatrix


class EdgeKick(Map):
    """Dipole edge effects."""

    def __init__(self, length: float, bendAngle: float, edgeAngle: float, dim: int, dtype: torch.dtype):
        super().__init__(dim=dim, dtype=dtype)

        # initialize weight
        curvature = bendAngle / length

        kernel = torch.tensor([curvature * math.tan(edgeAngle), -1 * curvature * math.tan(edgeAngle), curvature * length], dtype=dtype)
        self.weight = nn.Parameter(kernel)

        if dim == 4:
            self.forward = self.forward4D
        elif dim == 6:
            self.forward = self.forward6D
        else:
            raise NotImplementedError("dim {} not supported".format(dim))

        return

    def forward4D(self, x):
        # get positions
        pos = x[:, [0, 2]]

        # get updated momenta
        momenta = self.weight[:2] * pos
        momenta = momenta + x[:, [1, 3]]

        # update phase space vector
        xT = x.transpose(1, 0)
        momentaT = momenta.transpose(1, 0)

        x = torch.stack([xT[0], momentaT[0], xT[2], momentaT[1]], ).transpose(1, 0)
        return x

    def forward6D(self, x):
        # get positions
        pos = x[:, [0, 2]]
        velocityRatio = x[:, 8]

        # get updated momenta
        momenta = self.weight[:2] * pos
        momenta = momenta + x[:, [1, 3]]

        # sigma = -1 * velocityRatio * pos[0] * self.weight[2]
        # sigma = sigma + x[:, 4]

        # update phase space vector
        xT = x.transpose(1, 0)
        momentaT = momenta.transpose(1, 0)

        x = torch.stack([xT[0], momentaT[0], xT[2], momentaT[1], *xT[4:],]).transpose(1, 0)
        return x


    def rMatrix(self):
        if self.dim == 4:
            rMatrix = torch.eye(4, dtype=self.dtype)
            rMatrix[1, 0] = self.weight[0]
            rMatrix[3, 2] = self.weight[1]
        else:
            rMatrix = torch.eye(6, dtype=self.dtype)
            rMatrix[1, 0] = self.weight[0]
            rMatrix[3, 2] = self.weight[1]

        return rMatrix



if __name__ == "__main__":
    dim = 6
    dtype = torch.double

    # set up quad
    quad = DipoleKick(1, 0.1, dim, dtype)

    # track
    x = torch.randn(2, dim, dtype=dtype)

    # matrix
    print("rMatrix")
    print(quad.rMatrix())

