import torch
import torch.nn as nn


class DriftMap(nn.Linear):
    def __init__(self, length: float, dim: int, dtype: torch.dtype):
        super(DriftMap, self).__init__(dim // 2, dim // 2, bias=False)
        self.dtype = dtype

        drift = torch.tensor([[length, 0], [0, length], ], dtype=dtype)
        self.weight = nn.Parameter(drift)

        return

    def forward(self, x):
        # get updated position
        pos = super().forward(x[:, [1, 3]])  # feed momenta only
        pos = pos + x[:, [0, 2]]

        # update phase space vector
        xT = x.transpose(1, 0)
        posT = pos.transpose(1, 0)

        x = torch.stack([posT[0], xT[1], posT[1], xT[3]], ).transpose(1, 0)

        return x

    def rMatrix(self):
        momentaRows = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 1]], dtype=self.dtype)

        positionRows = torch.tensor(
            [[1, self.weight[0, 0], 0, self.weight[0, 1]], [0, self.weight[1, 0], 1, self.weight[1, 1]]],
            dtype=self.dtype)

        rMatrix = torch.stack([positionRows[0], momentaRows[0], positionRows[1], momentaRows[1]])
        return rMatrix


class QuadKick(nn.Linear):
    def __init__(self, length: float, k1: float, dim: int, dtype: torch.dtype):
        super().__init__(dim, dim // 2, bias=False)
        self.dtype = dtype

        quad = torch.tensor([[-1 * length * k1, 0], [0, length * k1]], dtype=dtype)
        self.weight = nn.Parameter(quad)
        return

    def forward(self, x):
        # get updated momenta
        momenta = super().forward(x[:, [0, 2]])  # feed positions only
        momenta = momenta + x[:, [1, 3]]

        # update phase space vector
        xT = x.transpose(1, 0)
        momentaT = momenta.transpose(1, 0)

        x = torch.stack([xT[0], momentaT[0], xT[2], momentaT[1]], ).transpose(1, 0)
        return x

    def rMatrix(self):
        positionRows = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=self.dtype)

        momentaRows = torch.tensor(
            [[self.weight[0, 0], 1, self.weight[0, 1], 0], [self.weight[1, 0], 0, self.weight[1, 1], 1]],
            dtype=self.dtype)

        rMatrix = torch.stack([positionRows[0], momentaRows[0], positionRows[1], momentaRows[1]])
        return rMatrix
