import torch
import torch.nn as nn

import Elements


class Model(nn.Module):
    def __init__(self, dim: int = 4, slices: int = 1, order: int = 2, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.generalProperties: dict = {"dim": dim, "dtype": dtype, "slices": slices, "order": order}
        self.dim = dim
        self.dtype = dtype

        # needs to be set by child classes
        self.elements = None
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
        return


class RBendLine(Model):
    def __init__(self, angle: float, e1: float, e2: float, dim: int = 4, slices: int = 1, order: int = 2, dtype: torch.dtype = torch.float32):
        super().__init__(dim=dim, slices=slices, order=order, dtype=dtype)

        # define beam line
        d1 = Elements.Drift(1, **self.generalProperties)
        rb1 = Elements.RBen(0.1, angle, e1=e1, e2=e2, **self.generalProperties)
        d2 = Elements.Drift(1, **self.generalProperties)

        # beam line
        self.elements = nn.ModuleList([d1, rb1, d2])
        return


if __name__ == "__main__":
    torch.set_printoptions(precision=4, sci_mode=False)

    dtype = torch.double

    angle = 0.2
    e1 = 0.1
    e2 = 0.1
    model = RBendLine(angle, e1=e1, e2=e2, slices=5, dtype=dtype)
    model.requires_grad_(False)

    # create particle
    x0 = torch.tensor([[1e-3, 2e-3, 1e-3, 0],], dtype=dtype)  # x, xp, y, yp
    # x0 = torch.tensor([[1e-3, 1e-3, 2e-3, 0],])  # x, y, xp, yp

    # track
    x = model(x0)
    print(x)

    # test symplecticity
    sym = torch.tensor([[0,1,0,0],[-1,0,0,0],[0,0,0,1],[0,0,-1,0]], dtype=dtype)

    rMatrix = model.rMatrix()
    res = torch.matmul(rMatrix.transpose(1,0), torch.matmul(sym, rMatrix)) - sym
    print("sym penalty: {}".format(torch.norm(res)))

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
