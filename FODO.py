import numpy as np
import torch


# compare with TorchOcelot
import sys
sys.path.append("../TorchOcelot/src/")
from Simulation.Lattice import DummyLattice
from Simulation.Models import LinearModel


def driftMap(length: float):
    m = np.array([[1, length, 0, 0],[0,1,0,0],[0,0,1,length],[0,0,0,1]])
    return m


def quadMap(length: float, k1: float):
    m = np.array([[1,0,0,0],[-1*length*k1,1,0,0],[0,0,1,0],[0,0,length*k1,1]])
    return m


class Drift(object):
    def __init__(self, length: float):
        self.drift1 = driftMap(length)
        return

    def __call__(self, *args, **kwargs):
        return np.matmul(self.drift1, args[0])


class Quadrupole(object):
    def __init__(self, length, k1):
        self.drift1 = driftMap(length / 2)
        self.kick1 = quadMap(length, k1)
        self.drift2 = driftMap(length / 2)
        return

    def __call__(self, *args, **kwargs):
        x = args[0]
        x = np.matmul(self.drift1, x)
        x = np.matmul(self.kick1, x)
        x = np.matmul(self.drift2, x)
        return x


class QuadrupoleN(object):
    def __init__(self, length, k1, order=2):
        if order == 2:
            coeffC = [1/2, 1/2]
            coeffD = [1,0]
        elif order == 3:
            coeffC = [2/3, -2/3, 1]
            coeffD = [7/24, 3/4, -1/24]
        elif order == 4:
            coeffC = [0.6756,0.411, 0.411, 0.6756]
            coeffD = [0,0.82898,0.72991,0.82898]

        self.maps = list()
        for c, d in zip(coeffC, coeffD):
            if c:
                self.maps.append(driftMap(c * length))
            if d:
                self.maps.append(quadMap(d*length, k1))

        return

    def __call__(self, *args, **kwargs):
        x = args[0]
        for m in self.maps:
            x = np.matmul(m, x)

        return x


if __name__ == "__main__":
    # pretty print of ndarray
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    # define FODO
    k1 = 0.3

    d1 = Drift(1)
    qf = Quadrupole(1, k1)
    d2 = Drift(2)
    qd = Quadrupole(1, -k1)

    order = 2
    qfN = QuadrupoleN(1, k1, order=order)
    qdN = QuadrupoleN(1, -k1, order=order)

    fodo = lambda x: qd(d2(qf(d1(x))))
    fodoN = lambda x: qdN(d2(qfN(d1(x))))

    # create particle
    x0 = np.array([1e-3, 2e-3, 1e-3, 0])

    # track
    print("thin lens")
    print("fodo: {}".format(fodo(x0)))
    print("fodoN: {}".format(fodoN(x0)))

    #####################3
    # Ocelot
    print("ocelot")

    lattice = DummyLattice(angle=0.1)
    ocelotModel = LinearModel(lattice)

    print(ocelotModel(torch.as_tensor(x0, dtype=torch.float32)))
    print(ocelotModel.oneTurnMap())