import torch
import torch.nn as nn


class DriftMap(nn.Linear):
    def __init__(self, length: float, dim: int, dtype: torch.dtype):
        super(DriftMap, self).__init__(dim // 2, dim // 2, bias=False)
        self.dtype = dtype

        drift = torch.tensor([[length, 0], [0, length],], dtype=dtype)
        self.weight = nn.Parameter(drift)

        # identity
        self.register_buffer("ident", torch.ones(dim // 2, dtype=dtype))
        return

    def forward(self, x):
        # get updated position
        # pos = super().forward(x)
        pos = super().forward(x[:,[1,3]])
        pos = pos + self.ident

        # update phase space vector
        # x[:,[0, 2]] = xPos
        xT = x.transpose(1,0)
        posT = pos.transpose(1,0)

        x = torch.stack([posT[0], xT[1], posT[1], xT[3]], ).transpose(1, 0)

        return x

    def rMatrix(self):
        momentaRows = torch.tensor([[0,1,0,0],[0,0,0,1]], dtype=self.dtype)

        rMatrix = torch.stack([self.weight[0], momentaRows[0], self.weight[1], momentaRows[1]])
        return rMatrix


class QuadKick(nn.Linear):
    def __init__(self, length: float, k1: float, dim: int, dtype: torch.dtype):
        super().__init__(dim, dim // 2, bias=False)
        self.dtype = dtype

        quad = torch.tensor([[-1 * length * k1, 1, 0, 0], [0, 0, length * k1, 1]], dtype=dtype)
        # quad = torch.tensor([[-1 * length * k1, 0, 1, 0], [0, length * k1, 0, 1]], dtype=dtype)
        self.weight = nn.Parameter(quad)
        return

    def forward(self, x):
        # get updated momenta
        momenta = super().forward(x)

        # update phase space vector
        # x[:, [1, 3]] = momenta

        xT = x.transpose(1,0)
        momentaT = momenta.transpose(1,0)

        x = torch.stack([xT[0], momentaT[0], xT[2], momentaT[1]],).transpose(1, 0)
        return x

    def rMatrix(self):
        positionRows = torch.tensor([[1,0,0,0],[0,0,1,0]], dtype=self.dtype)

        rMatrix = torch.stack([positionRows[0], self.weight[0], positionRows[1], self.weight[1]],)
        return rMatrix

