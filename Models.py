import torch
import torch.nn as nn

import Elements


class F0D0Model(nn.Module):
    def __init__(self, k1: float, dim: int = 4, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.dim = dim
        self.dtype = dtype

        d1 = Elements.Drift(1, dim=dim, dtype=dtype)
        qf = Elements.Quadrupole(1, k1, dim=dim, dtype=dtype)
        d2 = Elements.Drift(2, dim=dim, dtype=dtype)
        qd = Elements.Quadrupole(1, -k1, dim=dim, dtype=dtype)

        self.elements = nn.ModuleList([d1, qf, d2, qd])
        return

    def forward(self, x):
        for e in self.elements:
            x = e(x)

        return x

    def rMatrix(self):
        rMatrix = torch.eye(self.dim, dtype=self.dtype)

        for element in self.elements:
            rMatrix = torch.matmul(element.rMatrix(), rMatrix)

        return rMatrix



if __name__ == "__main__":
    dtype = torch.double
    model = F0D0Model(1.3, dtype=dtype)
    model.requires_grad_(False)

    # create particle
    x0 = torch.tensor([[1e-3, 2e-3, 1e-3, 0],], dtype=dtype)  # x, xp, y, yp
    # x0 = torch.tensor([[1e-3, 1e-3, 2e-3, 0],])  # x, y, xp, yp

    # track
    x = model(x0)
    print(x)

    # # test symplecticity
    # sym = torch.tensor([[0,1,0,0],[-1,0,0,0],[0,0,0,1],[0,0,-1,0]], dtype=dtype)
    #
    # rMatrix = model.rMatrix()
    # res = torch.matmul(rMatrix.transpose(1,0), torch.matmul(sym, rMatrix)) - sym
    # print("sym penalty: {}".format(torch.norm(res)))
