import time
import numpy as np
import matplotlib.pyplot as plt

import torch

from Models import SIS18_Lattice_minimal


def track(model, bunch, turns: int):
    device = next(model.parameters()).device
    bunch.to(device)

    # track
    with torch.no_grad():
        t0 = time.time()

        multiTurnOutput = list()
        y = bunch
        for i in range(turns):
            # y = model(y)
            # multiTurnOutput.append(y)

            y = model(y, outputPerElement=True)
            multiTurnOutput.append(y)
            y = y[:, :, -1]

    # prepare tracks for plotting
    trackResults = torch.cat(multiTurnOutput, 2)  # indices: particle, dim, element
    # trackResults = multiTurnOutput[0]  # restrict to first turn, indices: element, particle, dim
    return trackResults


def plotTrajectories(ax, trackResults, lattice):
    """Plot individual trajectories."""
    pos = [lattice.endPositions[i % len(lattice.endPositions)] + i // len(lattice.endPositions) * lattice.totalLen
           for i in range(trackResults.size(2))]

    for particle in trackResults:
        # x-coord
        ax.plot(pos, particle[0])

    return


def plotBeamCentroid(ax, trackResults, lattice):
    """Plot beam centroid."""
    pos = [lattice.endPositions[i % len(lattice.endPositions)] + i // len(lattice.endPositions) * lattice.totalLen
           for i in range(trackResults.size(2))]

    trackResults = trackResults.permute((1, 2, 0))  # dim, element, particle
    beamCentroid = torch.mean(trackResults, dim=2)

    ax.plot(pos, beamCentroid[0].to("cpu").numpy())
    return


def plotBeamSigma(ax, trackResults, lattice):
    """Plot beam size as standard deviation of position."""
    pos = [lattice.endPositions[i % len(lattice.endPositions)] + i // len(lattice.endPositions) * lattice.totalLen
           for i in range(trackResults.size(2))]

    trackResults = trackResults.permute((1, 2, 0))  # dim, element, particle
    beamSigma = torch.std(trackResults, dim=2)

    # plt.plot(pos, beamSigma[0].numpy())
    # plt.show()
    # plt.close()
    ax.plot(pos, beamSigma[0].to("cpu").numpy())
    return


if __name__ == "__main__":
    # create model of SIS18
    print("building model")
    dim = 4
    dtype = torch.float32
    model = SIS18_Lattice_minimal(dim=dim, dtype=dtype, )
    model.logElementPositions()

    # load bunch
    print("loading bunch")
    bunch = np.loadtxt("../TorchOcelot/res/bunch_6d_n=1e5.txt.gz")
    bunch = torch.as_tensor(bunch, dtype=dtype)[:10,:4]
    bunch = bunch - bunch.permute(1, 0).mean(dim=1)  # set bunch centroid to 0 for each dim
    # bunch = bunch + torch.tensor([1e-3, 0, 1e-3, 0, 0, 0], dtype=torch.double)  # bunch has transverse offset

    # visualize accelerator
    trackResults = track(model, bunch, 6)

    fig, axes = plt.subplots(3, sharex=True)
    plotTrajectories(axes[0], trackResults, model)
    plotBeamCentroid(axes[1], trackResults, model)
    plotBeamSigma(axes[2], trackResults, model)

    plt.show()
    plt.close()

    ########################################################################
    bunchCentroid = bunch.permute(1, 0).mean(dim=1)  # dim, particle
    print(bunchCentroid)

    bunchSigma = bunch.permute(1, 0).std(dim=1)  # dim, particle
    print(bunchSigma)