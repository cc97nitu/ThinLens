"""Implement phase-space transformations as autograd functions."""
import torch.autograd


class Drift(torch.autograd.Function):
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

        gradLength = xp * gradX + yp * gradY + (1 - vR) * gradSigma

        return gradX, newGradXp, gradY, newGradYp, gradSigma, gradPSigma, gradDelta, gradInvDelta, newGradVR, gradLength


class ThinMultipole(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, xp, y, yp, sigma, pSigma, delta, invDelta, vR, length, k1n, k2n, k1s, k2s):
        # save inputs for backward pass
        ctx.save_for_backward(length, k1n, k2n, k1s, k2s, x, y, invDelta)

        # get updated momenta
        quadDPx = k1n * x - k1s * y
        quadDPy = k1n * y + k1s * x

        sextDPx = k2n * 1 / 2 * (x ** 2 - y ** 2) - k2s * x * y
        sextDPy = k2n * x * y + k2s * 1 / 2 * (x ** 2 + y ** 2)

        # care about octupoles later
        # octDPx = k3n * (1 / 6 * x ** 3 - 1 / 2 * x * y ** 2) + k3s * (
        #         1 / 6 * y ** 3 - 1 / 2 * x ** 2 * y)
        # octDPy = k3n * (-1 / 6 * y ** 3 + 1 / 2 * x ** 2 * y) + k3s * (
        #         1 / 6 * x ** 3 - 1 / 2 * x * y ** 2)

        newXp = xp - length * invDelta * (quadDPx + sextDPx)
        newYp = yp + length * invDelta * (quadDPy + sextDPy)

        return x, newXp, y, newYp, sigma, pSigma, delta, invDelta, vR

    @staticmethod
    def backward(ctx, gradX, gradXp, gradY, gradYp, gradSigma, gradPSigma, gradDelta, gradInvDelta, gradVR):
        # old phase space
        length, k1n, k2n, k1s, k2s, x, y, invDelta = ctx.saved_tensors

        quadDPx = k1n * x - k1s * y
        quadDPy = k1n * y + k1s * x

        sextDPx = k2n * 1 / 2 * (x ** 2 - y ** 2) - k2s * x * y
        sextDPy = k2n * x * y + k2s * 1 / 2 * (x ** 2 + y ** 2)

        # calculate phase-space gradients
        newGradX = gradX - length * invDelta * ((k1n + k2n * x - k2s * y) * gradXp + (k1s + k2n * y + k2s * x) * gradYp)
        newGradY = gradY + length * invDelta * ((-k1s - k2n * y - k2s * x) * gradXp + (k1n + k2n * x + k2s * y) * gradYp)
        newGradInvDelta = gradInvDelta - length * (quadDPx + sextDPx) * gradXp + length * (quadDPy + sextDPy) * gradYp

        # calculate gradients for weights
        gradK1n = -1 * length * invDelta * (x * gradXp + y * gradYp)
        gradK1s = -1 * length * invDelta * (-y * gradXp + x * gradYp)
        gradK2n = -1 * length * invDelta * (1 / 2 * (x ** 2 + y ** 2) * gradXp + x * y * gradYp)
        gradK2s = -1 * length * invDelta * (-1 * x * y * gradXp + 1 / 2 * (x ** 2 + y ** 2) * gradYp)

        if ctx.needs_input_grad[0]:
            gradLength = -1 * invDelta * (quadDPx + sextDPx) * gradXp + invDelta * (quadDPy + sextDPy) * gradYp
        else:
            gradLength = None

        return newGradX, gradXp, newGradY, gradYp, gradSigma, gradPSigma, gradDelta, newGradInvDelta, gradVR, gradLength, gradK1n, gradK2n, gradK1s, gradK2s


if __name__ == "__main__":
    import torch
    from torch.autograd import gradcheck

    from Beam import Beam

    # create a beam
    beam = Beam(mass=18.798, energy=19.0, exn=1.258e-1, eyn=2.005e-1, sigt=0.01, sige=0.005, particles=int(1e3))
    bunch = beam.bunch.double().requires_grad_(True)
    loseBunch = bunch[:1].transpose(1, 0).unbind(0)

    # perform numerical gradcheck for Drift
    length = torch.tensor(3.0, dtype=torch.double, requires_grad=True)

    myMap = Drift.apply
    inp = [*loseBunch, length]
    checkNew = gradcheck(myMap, inp, eps=1e-6, atol=1e-4)

    print("check result for Drift: {}".format(checkNew))

    # perform numerical gradcheck for ThinMultipole
    bunch.grad = None
    bunch.requires_grad_(False)

    paramGrad = True
    mPLength = torch.tensor(0.1, dtype=torch.double, requires_grad=paramGrad)
    k1n = torch.tensor(0.5, dtype=torch.double, requires_grad=paramGrad)
    k1s = torch.tensor(0.0, dtype=torch.double, requires_grad=paramGrad)
    k2n = torch.tensor(0.0, dtype=torch.double, requires_grad=paramGrad)
    k2s = torch.tensor(0.0, dtype=torch.double, requires_grad=paramGrad)

    tPMap = ThinMultipole.apply
    inp = tuple([*loseBunch, mPLength, k1n, k2n, k1s, k2s])

    jaco = torch.autograd.functional.jacobian(tPMap, inp)
    print("jacobian:", torch.tensor(jaco))

    checkTP = gradcheck(tPMap, inp, eps=1e-4, atol=1e-4)

    print("check result for ThinMultipole: {}".format(checkTP))
