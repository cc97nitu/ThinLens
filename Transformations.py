"""Implement phase-space transformations as autograd functions."""
import torch.autograd


class Drift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, xp, y, yp, sigma, pSigma, delta, invDelta, vR, length):
        # save inputs for backward pass
        ctx.save_for_backward(length, xp, yp, invDelta, vR)

        # update phase space coordinates
        newX = x + length * xp * invDelta
        newY = y + length * yp * invDelta
        newSigma = sigma + (1 - vR * (1 + 1/2 * invDelta**2 * (xp**2 + yp**2))) * length

        return newX, xp, newY, yp, newSigma, pSigma, delta, invDelta, vR

    @staticmethod
    def backward(ctx, gradX, gradXp, gradY, gradYp, gradSigma, gradPSigma, gradDelta, gradInvDelta, gradVR):
        # old phase space
        length, xp, yp, invDelta, vR = ctx.saved_tensors

        # calculate gradients
        newGradXp = gradXp + length * invDelta * (gradX - vR * xp * invDelta * gradSigma)
        newGradYp = gradYp + length * invDelta * (gradY - vR * yp * invDelta * gradSigma)
        newGradInvDelta = gradInvDelta + length * xp * gradX + length * yp * gradY - length * vR * (invDelta * (xp ** 2 + yp ** 2)) * gradSigma

        newGradVR = gradVR - length * (1 + 1/2 * invDelta**2 * (xp**2 + yp**2)) * gradSigma

        if ctx.needs_input_grad[9]:
            gradLength = xp * invDelta * gradX + yp * invDelta * gradY + (1 - vR * (1 + 1/2 * invDelta**2 * (xp**2 + yp**2))) * gradSigma
        else:
            gradLength = None

        return gradX, newGradXp, gradY, newGradYp, gradSigma, gradPSigma, gradDelta, newGradInvDelta, newGradVR, gradLength


class Quadrupole(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, xp, y, yp, sigma, pSigma, delta, invDelta, vR, length, k1n, k1s):
        # save inputs for backward pass
        ctx.save_for_backward(length, k1n, k1s, x, y, invDelta)

        # update momenta
        newXp = xp - length * invDelta * (k1n * x - k1s * y)
        newYp = yp + length * invDelta * (k1n * y + k1s * x)

        return x, newXp, y, newYp, sigma, pSigma, delta, invDelta, vR

    @staticmethod
    def backward(ctx, gradX, gradXp, gradY, gradYp, gradSigma, gradPSigma, gradDelta, gradInvDelta, gradVR):
        # old phase space
        length, k1n, k1s, x, y, invDelta = ctx.saved_tensors

        # phase space gradients
        newGradX = gradX + length * invDelta * (-1 * k1n * gradXp + k1s * gradYp)
        newGradY = gradY + length * invDelta * (k1s * gradXp + k1n * gradYp)
        newGradInvDelta = gradInvDelta + length * (-1 * (k1n * x - k1s * y) * gradXp + (k1n * y + k1s * x) * gradYp)

        # weight gradients
        gradLength = invDelta * (-1 * (k1n * x - k1s * y) * gradXp + (k1n * y + k1s * x) * gradYp)
        gradK1n = length * invDelta * (-1 * x * gradXp + y * gradYp)
        gradK1s = length * invDelta * (y * gradXp + x * gradYp)

        return newGradX, gradXp, newGradY, gradYp, gradSigma, gradPSigma, gradDelta, newGradInvDelta, gradVR, gradLength, gradK1n, gradK1s


class ThinMultipole(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, xp, y, yp, sigma, pSigma, delta, invDelta, vR, length, k1n, k2n, k1s, k2s):
        # save inputs for backward pass
        ctx.save_for_backward(length, k1n, k2n, k1s, k2s, x, y, invDelta)

        # update momenta
        newXp = xp - length * (k1n * x - k1s * y + k2n * 1 / 2 * (x ** 2 - y ** 2) - k2s * x * y)
        newYp = yp + length * (k1n * y + k1s * x + k2s * 1 / 2 * (x ** 2 - y ** 2) + k2n * x * y)

        return x, newXp, y, newYp, sigma, pSigma, delta, invDelta, vR

    @staticmethod
    def backward(ctx, gradX, gradXp, gradY, gradYp, gradSigma, gradPSigma, gradDelta, gradInvDelta, gradVR):
        # old phase space
        length, k1n, k2n, k1s, k2s, x, y, invDelta = ctx.saved_tensors

        # phase space gradients
        newGradX = gradX + length * (
                (k1s + k2s * x + k2n * y) * gradYp + -1 * (k1n + k2n * x - k2s * y) * gradXp)
        newGradY = gradY + length * ((k1s + k2n * x + k2s * y) * gradXp + (k1n - k2s * y + k2n * x) * gradYp)

        focX = (k1n * x - k1s * y + k2n * 1 / 2 * (x ** 2 - y ** 2) - k2s * x * y)
        focY = (k1n * y + k1s * x + k2s * 1 / 2 * (x ** 2 - y ** 2) + k2n * x * y)
        # newGradInvDelta = gradInvDelta + length * (-1 * focX * gradXp + focY * gradYp)

        # weight gradients
        gradLength = gradK1n = gradK2n = gradK1s = gradK2s = None

        if ctx.needs_input_grad[9]:
            gradLength = (-1 * focX * gradXp + focY * gradYp)
        if ctx.needs_input_grad[10]:
            gradK1n = length * (-1 * (x * gradXp) + y * gradYp)
        if ctx.needs_input_grad[11]:
            gradK2n = length * (-1 * (1 / 2 * (x ** 2 - y ** 2) * gradXp) + x * y * gradYp)
        if ctx.needs_input_grad[12]:
            gradK1s = length * (y * gradXp + x * gradYp)
        if ctx.needs_input_grad[13]:
            gradK2s = length * (x * y * gradXp + 1 / 2 * (x ** 2 - y ** 2) * gradYp)

        return newGradX, gradXp, newGradY, gradYp, gradSigma, gradPSigma, gradDelta, gradInvDelta, gradVR, gradLength, gradK1n, gradK2n, gradK1s, gradK2s


class EdgeKick(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, xp, y, yp, sigma, pSigma, delta, invDelta, vR, weight, curvature):
        # save inputs for backward pass
        ctx.save_for_backward(weight, curvature, x, y, torch.sqrt(1 + delta))

        # update momenta
        newXp = xp + weight * curvature * torch.sqrt(1 + delta) * x
        newYp = yp - weight * curvature * torch.sqrt(1 + delta) * y

        return x, newXp, y, newYp, sigma, pSigma, delta, invDelta, vR

    @staticmethod
    def backward(ctx, gradX, gradXp, gradY, gradYp, gradSigma, gradPSigma, gradDelta, gradInvDelta, gradVR):
        # old phase space
        weight, curvature, x, y, sqrtDelta = ctx.saved_tensors

        # phase-space gradients
        newGradX = gradX + weight * curvature * sqrtDelta * gradXp
        newGradY = gradY - weight * curvature * sqrtDelta * gradYp
        newGradDelta = gradDelta + weight * curvature * 1 / (2 * sqrtDelta) * (x * gradXp - y * gradYp)

        # weight gradient
        if ctx.needs_input_grad[9]:
            gradWeight = curvature * sqrtDelta * x * gradXp - curvature * sqrtDelta * y * gradYp
        else:
            gradWeight = None

        return newGradX, gradXp, newGradY, gradYp, gradSigma, gradPSigma, newGradDelta, gradInvDelta, gradVR, gradWeight, None


class DipoleKick(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, xp, y, yp, sigma, pSigma, delta, invDelta, vR, length, curvature):
        # save inputs for backward pass
        ctx.save_for_backward(length, curvature, x, y, delta, vR)

        # update coordinates
        newXp = xp + curvature * length * (delta - curvature * x)
        newSigma = sigma - curvature * length * vR * x

        return x, newXp, y, yp, newSigma, pSigma, delta, invDelta, vR

    @staticmethod
    def backward(ctx, gradX, gradXp, gradY, gradYp, gradSigma, gradPSigma, gradDelta, gradInvDelta, gradVR):
        # old phase space
        length, curvature, x, y, delta, vR = ctx.saved_tensors

        # phase-space gradients
        newGradX = gradX - curvature ** 2 * length * gradXp - curvature * length * vR * gradSigma
        newGradDelta = gradDelta + curvature * length * gradXp
        newGradVR = gradVR - curvature * length * x * gradSigma

        # weight gradients
        gradLength = gradCurvature = None

        if ctx.needs_input_grad[9]:
            gradLength = curvature * (delta - curvature * x) * gradXp - curvature * vR * x * gradSigma
        if ctx.needs_input_grad[10]:
            gradCurvature = (length * (
                    delta - curvature * x) - length * curvature * x) * gradXp - length * vR * x * gradSigma

        return newGradX, gradXp, gradY, gradYp, gradSigma, gradPSigma, newGradDelta, gradInvDelta, newGradVR, gradLength, gradCurvature


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

    # perform gradcheck for Quadrupole
    bunch.grad = None
    bunch.requires_grad_(True)

    paramGrad = True
    quadLength = torch.tensor(0.1, dtype=torch.double, requires_grad=paramGrad)
    k1n = torch.tensor(0.5, dtype=torch.double, requires_grad=paramGrad)
    k1s = torch.tensor(0.5, dtype=torch.double, requires_grad=paramGrad)

    quadMap = Quadrupole.apply
    inp = tuple([*loseBunch, quadLength, k1n, k1s])

    checkQuad = gradcheck(quadMap, inp, eps=1e-6, atol=1e-4)
    print("check result for Quadrupole: {}".format(checkQuad))

    # perform gradcheck for Sextupole
    bunch.grad = None
    bunch.requires_grad_(True)

    paramGrad = True
    sextLength = torch.tensor(0.1, dtype=torch.double, requires_grad=paramGrad)
    k1n = torch.tensor(0.5, dtype=torch.double, requires_grad=paramGrad)
    k2n = torch.tensor(0.5, dtype=torch.double, requires_grad=paramGrad)
    k1s = torch.tensor(-0.5, dtype=torch.double, requires_grad=paramGrad)
    k2s = torch.tensor(0.5, dtype=torch.double, requires_grad=paramGrad)

    sextMap = ThinMultipole.apply
    inp = tuple([*loseBunch, sextLength, k1n, k2n, k1s, k2s])

    checkSext = gradcheck(sextMap, inp, eps=1e-6, atol=1e-4)
    print("check result for ThinMultipole: {}".format(checkSext))

    # perform gradcheck for EdgeKick
    bunch.grad = None
    bunch.requires_grad_(True)

    weight = torch.tensor(0.2, dtype=torch.double, requires_grad=True)
    curvature = torch.tensor(0.1, dtype=torch.double, requires_grad=False)

    edgeMap = EdgeKick.apply
    inp = tuple([*loseBunch, weight, curvature])

    checkEdge = gradcheck(edgeMap, inp, eps=1e-6, atol=1e-4)
    print("check result for EdgeKick: {}".format(checkEdge))

    # perform gradcheck for DipoleKick
    bunch.grad = None
    bunch.requires_grad_(True)

    paramGrad = True
    dipLength = torch.tensor(2.13, dtype=torch.double, requires_grad=paramGrad)
    curvature = torch.tensor(0.2, dtype=torch.double, requires_grad=paramGrad)

    dipMap = DipoleKick.apply
    inp = tuple([*loseBunch, length, curvature])

    # # calculate jacobian
    # jaco = torch.autograd.functional.jacobian(dipMap, inp)
    # print("jacobian", torch.tensor(jaco))

    checkDip = gradcheck(dipMap, inp, eps=1e-6, atol=1e-4)
    print("check result for DipoleKick: {}".format(checkDip))
