import torch
import torch.nn as nn


class DriftMap(nn.Conv1d):
    """Propagate bunch along drift section."""
    def __init__(self, length: float, dim: int, dtype: torch.dtype):
        super().__init__(1, 1, 3, padding=1, bias=False)
        self.dim = dim
        self.dtype = dtype

        # set up weights
        kernel = torch.tensor([[[length, 0, length], ], ], dtype=self.dtype)
        self.weight = nn.Parameter(kernel)
        return

    def forward(self, x):
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


class QuadKick(nn.Conv1d):
    """Apply a quadrupole kick."""
    def __init__(self, length: float, k1: float, dim: int, dtype: torch.dtype):
        super().__init__(1, 1, 3, padding=1, bias=False)
        self.dim = dim
        self.dtype = dtype

        # initialize weight
        kernel = torch.tensor([[[length * k1, 0, -1 * length * k1],],], dtype=dtype)
        self.weight = nn.Parameter(kernel)
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


