import torch
import torch.nn
import torch.autograd

import time

from Beam import Beam


# define drift transformation
class DriftTrafo(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, xp, y, yp, sigma, pSigma, delta, invDelta, vR, length):
        # save inputs for backward pass
        ctx.save_for_backward(length, xp, yp, vR)

        # update phase space coordinates
        newX = x + length * xp
        newY = y + length * yp
        newSigma = sigma + (1 - vR) * length

        return newX, xp, newY, yp, newSigma, pSigma, delta, invDelta, vR

    @staticmethod
    def backward(ctx, gradX, gradXp, gradY, gradYp, gradSigma, gradPSigma, gradDelta, gradInvDelta, gradVR):
        # old phase space
        length, xp, yp, vR = ctx.saved_tensors

        # calculate gradients
        newGradXp = gradXp + length * gradX
        newGradYp = gradYp + length * gradY
        newGradVR = gradVR - length * gradSigma

        # gradLength = xp + yp + (1 - vR)
        gradLength = xp * gradXp + yp * gradYp + (1 - vR) * gradVR


        return gradX, newGradXp, gradY, newGradYp, gradSigma, gradPSigma, gradDelta, gradInvDelta, newGradVR, gradLength


if __name__ == "__main__":
    from torch.autograd import gradcheck
    from Maps import DriftMap

    # create a beam
    beam = Beam(mass=18.798, energy=19.0, exn=1.258e-6, eyn=2.005e-6, sigt=0.01, sige=0.005, particles=int(1e3))
    bunch = beam.bunch.requires_grad_(True)
    bunch2 = bunch.detach().clone().requires_grad_(True)
    loseBunch = bunch2.transpose(1, 0).unbind(0)

    # transport bunch through drift
    length = -0.4

    oldDrift = DriftMap(length, dim=6, dtype=torch.double)
    oldResult = oldDrift(bunch)

    newDrift = DriftTrafo.apply
    lengthParameter = torch.nn.Parameter(torch.tensor(length))
    newResult = newDrift(*loseBunch, lengthParameter)

    # resemble bunch
    finalBunch = torch.stack(newResult).transpose(1, 0)
    print(torch.allclose(oldResult, finalBunch))

    # max deviation?
    deviation = oldResult - finalBunch
    print("max deviation: {}".format(deviation.max()))


    #######################################################
    # look at gradients
    oldResult.sum().backward()
    finalBunch.sum().backward()

    print("deviation for length gradient: {}".format(oldDrift.weight.grad - lengthParameter.grad))

    # track phase space coordinates which require grad
    bunch.grad = None
    bunch.requires_grad_(True)
    loseBunch = bunch.transpose(1, 0).unbind(0)

    part = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1], ], dtype=torch.double, requires_grad=True)

    part2 = part.detach().clone().requires_grad_(True)
    losePart = part2.transpose(1, 0).unbind(0)

    oldResult = oldDrift(part)
    newResult = newDrift(*losePart, lengthParameter)

    oldResult.sum().backward()
    finalBunch = torch.stack(newResult).transpose(1, 0)
    finalBunch.sum().backward()

    gradDiff = part.grad - part2.grad
    print("max deviation for phase space gradient: {}".format(gradDiff.max()))


    #########################
    # calculate jacobian
    jaco = torch.autograd.functional.jacobian(oldDrift, part[0].unsqueeze(0))
    print(jaco)

    part.grad = None

    # verify gradient
    inp = part[0].unsqueeze(0)

    checkOld = gradcheck(oldDrift, inp, eps=1e-6, atol=1e-4)
    checkNew = gradcheck(lambda x: newDrift(*x, lengthParameter), inp.unbind(0), eps=1e-6, atol=1e-4)
    print("check old: {}, check new: {}".format(checkOld, checkNew))



