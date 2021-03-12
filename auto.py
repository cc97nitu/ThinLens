import torch
import torch.autograd

import time

from Maps import DriftMap

# set up particle
# part = torch.tensor([[0,0,0,0,0,0,0,0,0],
#                      [1,1,0,0,0,0,0,1,1],
#                      [0,0,0,0,0,0,0,0,0],
#                      [1,1,1,1,0,0,0,1,1],], dtype=torch.double, requires_grad=True)

part = torch.tensor([[0,0,0,0,0,0,0,1,1],
                     [0,0,0,0,0,0,0,1,1],
                     [0,0,0,0,0,0,0,1,1],
                     [0,0,0,0,0,0,0,1,1],], dtype=torch.double, requires_grad=True)

# set up drift
driftLength = 3
drift = DriftMap(driftLength, dim=6, dtype=torch.double)

# track and backprop
y = drift(part)
z = y.sum()

y.retain_grad()
z.retain_grad()

z.backward(retain_graph=True)


# perform grad check?
test = torch.autograd.gradcheck(drift, part)
print("result of gradcheck: ", test)


# implement with autograd
class autoDrift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # get momenta
        momenta = x[:, [1, 3, ]]
        velocityRatio = x[:, 8]

        # get updated momenta
        pos = weight * momenta
        sigma = weight * (1 - velocityRatio)
        pos = pos + x[:, [0, 2]]
        sigma = sigma + x[:, 4]

        # update phase space vector
        xT = x.transpose(1, 0)
        posT = pos.transpose(1, 0)

        x = torch.stack([posT[0], xT[1], posT[1], xT[3], sigma, *xT[5:]], ).transpose(1, 0)

        # store weight
        ctx.save_for_backward(x, weight)
        return x

    @staticmethod
    def backward(ctx, gradOutputs: torch.tensor):
        batch, weight, = ctx.saved_tensors

        # calculate gradient for phase space
        px = weight * gradOutputs[:, 0]
        py = weight * gradOutputs[:, 2]
        velocityRatio = -1 * weight * gradOutputs[:, 8]

        gradInputs = torch.stack([gradOutputs[:,0], gradOutputs[:,1]+px, gradOutputs[:,2], gradOutputs[:,3]+py,
                                  gradOutputs[:,4], gradOutputs[:,5], gradOutputs[:,6],
                                  gradOutputs[:,7], gradOutputs[:,8]+velocityRatio]).transpose(1, 0)

        # calculate gradient for weight
        gradOutputs = gradOutputs.transpose(1, 0)  # convenient to access columns
        first = batch[:, 1] * gradOutputs[0]
        third = batch[:, 3] * gradOutputs[2]
        fifth = (1 - batch[:, 8]) * gradOutputs[4]

        # empty = torch.zeros(len(gradOutputs), dtype=gradOutputs.dtype)
        # gradWeight = torch.stack([first, empty, third, empty, fifth, empty, empty])

        gradWeight = first + third + fifth

        print(gradWeight)
        return gradInputs, gradWeight


myWeight = torch.tensor([3.], requires_grad=True)
myDrift = autoDrift.apply

# monitor memory consumption
with torch.autograd.profiler.profile() as prof:
    for _ in range(1000):
        y = myDrift(part, myWeight)

    y.sum().backward()

# benchmark
t0 = time.time()
model = list()
for i in range(100000):
    # model.append(DriftMap(driftLength, dim=6, dtype=torch.double))

    w = torch.tensor([3.], dtype=torch.double, requires_grad=True)
    model.append(lambda x: autoDrift.apply(x, w))

y = part
for m in model:
    y = m(y)

print("completed within {:.2f}s".format(time.time() - t0))