import torch
import torch.nn as nn
import torch.optim as optim

import Elements
import Maps
from Models import F0D0Model, SIS18_Lattice_minimal

import matplotlib.pyplot as plt
import Plot as PlotTrajectory

# general
torch.set_printoptions(precision=4, sci_mode=False)

dim = 4
slices = 4
dtype = torch.double
outputPerElement = True

# set up models
# model = F0D0Model(k1=0.3, slices=slices, dim=dim, dtype=dtype)
# perturbedModel = F0D0Model(k1=0.2, dim=dim, slices=slices, dtype=dtype)

k1d = -4.78047e-01
model = SIS18_Lattice_minimal(k1d=k1d, dim=dim, slices=slices, dtype=dtype)
perturbedModel = SIS18_Lattice_minimal(k1d=0.95*k1d, dim=dim, slices=slices, dtype=dtype)

model.requires_grad_(False)
perturbedModel.requires_grad_(False)

# train set
bunch = torch.tensor([[1e-3, 2e-3, 1e-3, 0], [-1e-3,1e-3,0,1e-3]], dtype=dtype)
label = perturbedModel(bunch, outputPerElement=outputPerElement)


# plot initial trajectories
fig, axes = plt.subplots(3, sharex=True)
PlotTrajectory.plotTrajectories(axes[0], PlotTrajectory.track(model, bunch, 1), model)
axes[0].set_ylabel("ideal")

PlotTrajectory.plotTrajectories(axes[1], PlotTrajectory.track(perturbedModel, bunch, 1), perturbedModel)
axes[1].set_ylabel("perturbed")

# test symplecticity
sym = torch.tensor([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=dtype)

rMatrix = model.rMatrix()
res = torch.matmul(rMatrix.transpose(1, 0), torch.matmul(sym, rMatrix)) - sym
print("sym penalty before training: {}".format(torch.norm(res)))

# activate gradients on kick maps
for m in model.modules():
    if type(m) is Elements.Quadrupole:
        for mod in m.modules():
            if type(mod) is Maps.QuadKick:
                mod.requires_grad_(True)

# set up optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# train loop
for epoch in range(100):
    optimizer.zero_grad()

    out = model(bunch, outputPerElement=outputPerElement)
    loss = criterion(out, label)

    loss.backward()
    optimizer.step()

    if epoch % 20 == 19:
        print(loss.item())


# plot final trajectories
PlotTrajectory.plotTrajectories(axes[2], PlotTrajectory.track(model, bunch, 1), model)
axes[2].set_ylabel("trained")

axes[2].set_xlabel("pos / m")

plt.show()
plt.close()

# test symplecticity
sym = torch.tensor([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=dtype)

rMatrix = model.rMatrix()
res = lambda x: torch.matmul(x.transpose(1, 0), torch.matmul(sym, x)) - sym

print("sym penalty after training: {}".format(torch.norm(res(rMatrix))))

print("transport matrix after training:")
print(model.rMatrix())

print("determinant after training: {}".format(torch.det(model.rMatrix())))
# # look at maps
# print("trained model:")
# quad = model.elements[1]
# print(quad.rMatrix())
# # for m in quad.maps:
# #     print(m.rMatrix())
#
# print("perturbed model first quad:")
# print(perturbedModel.elements[1].rMatrix())

# print tunes
print("perturbed model tunes: {}".format(perturbedModel.getTunes()))
print("trained model tunes: {}".format(model.getTunes()))