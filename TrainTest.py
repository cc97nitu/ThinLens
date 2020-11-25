import torch
import torch.nn as nn
import torch.optim as optim

import Elements
import Maps
from Models import F0D0Model


# general
dim = 4
dtype = torch.double

# set up models
model = F0D0Model(k1=0.1, dim=dim, dtype=dtype)
perturbedModel = F0D0Model(k1=0.2, dim=dim, dtype=dtype)

model.requires_grad_(False)
perturbedModel.requires_grad_(False)

# test symplecticity
sym = torch.tensor([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=dtype)

# rMatrix = model.rMatrix()
# res = torch.matmul(rMatrix.transpose(1, 0), torch.matmul(sym, rMatrix)) - sym
# print("sym penalty: {}".format(torch.norm(res)))


# activate gradients on kick maps
for m in model.modules():
    if type(m) is Elements.Quadrupole:
        for mod in m.modules():
            if type(mod) is Maps.QuadKick:
                mod.requires_grad_(True)

# train set
bunch = torch.tensor([[1e-3, 2e-3, 1e-3, 0],], dtype=dtype)
label = perturbedModel(bunch)

# set up optimizer
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

# train loop
for epoch in range(300):
    optimizer.zero_grad()

    out = model(bunch)
    loss = criterion(out, label)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 9:
        print(loss.item())

# # test symplecticity
# sym = torch.tensor([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], dtype=dtype)
#
# rMatrix = model.rMatrix()
# print(rMatrix)
# res = lambda x: torch.matmul(x.transpose(1, 0), torch.matmul(sym, x)) - sym
#
# print("sym penalty: {}".format(torch.norm(res(rMatrix))))

