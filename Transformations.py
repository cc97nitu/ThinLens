"""Implement phase-space transformations as autograd functions."""
import torch.autograd


class Drift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, px, y, py, sigma, pSigma, delta, pz, vR, length):
        # save inputs for backward pass
        ctx.save_for_backward(length, px, py, delta, pz, vR)

        # update phase space coordinates
        newX = x + length * px / pz
        newY = y + length * py / pz
        newSigma = sigma + (1 - vR * (1 + delta) / pz) * length

        return newX, px, newY, py, newSigma, pSigma, delta, pz, vR

    @staticmethod
    def backward(ctx, gradX, gradPx, gradY, gradPy, gradSigma, gradPSigma, gradDelta, gradPz, gradVR):
        # old phase space
        length, px, py, delta, pz, vR = ctx.saved_tensors

        # calculate gradients
        newGradPx = gradPx + length / pz * gradX
        newGradPy = gradPy + length / pz * gradY

        newGradDelta = gradDelta - length * vR /pz * gradSigma
        newGradPz = gradPz + length / pz**2 * (vR * (1 + delta) * gradSigma - px * gradX - py * gradY)


        newGradVR = gradVR - (1 + delta) / pz * length * gradSigma

        if ctx.needs_input_grad[9]:
            gradLength = 1/pz * (px * gradX + py * gradY) + (1 - vR * (1 + delta) / pz) * gradSigma
        else:
            gradLength = None

        return gradX, newGradPx, gradY, newGradPy, gradSigma, gradPSigma, newGradDelta, newGradPz, newGradVR, gradLength


class ThinMultipole(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, px, y, py, sigma, pSigma, delta, pz, vR, length, k1n, k2n, k1s, k2s):
        # save inputs for backward pass
        ctx.save_for_backward(length, k1n, k2n, k1s, k2s, x, y)

        # update momenta
        newXp = px - length * (k1n * x - k1s * y + k2n * 1 / 2 * (x ** 2 - y ** 2) - k2s * x * y)
        newYp = py + length * (k1n * y + k1s * x + k2s * 1 / 2 * (x ** 2 - y ** 2) + k2n * x * y)

        return x, newXp, y, newYp, sigma, pSigma, delta, pz, vR

    @staticmethod
    def backward(ctx, gradX, gradPx, gradY, gradPy, gradSigma, gradPSigma, gradDelta, gradPz, gradVR):
        # old phase space
        length, k1n, k2n, k1s, k2s, x, y = ctx.saved_tensors

        # phase space gradients
        newGradX = gradX + length * (
                (k1s + k2s * x + k2n * y) * gradPy + -1 * (k1n + k2n * x - k2s * y) * gradPx)
        newGradY = gradY + length * ((k1s + k2n * x + k2s * y) * gradPx + (k1n - k2s * y + k2n * x) * gradPy)

        focX = (k1n * x - k1s * y + k2n * 1 / 2 * (x ** 2 - y ** 2) - k2s * x * y)
        focY = (k1n * y + k1s * x + k2s * 1 / 2 * (x ** 2 - y ** 2) + k2n * x * y)
        # newGradInvDelta = gradInvDelta + length * (-1 * focX * gradXp + focY * gradYp)

        # weight gradients
        gradLength = gradK1n = gradK2n = gradK1s = gradK2s = None

        if ctx.needs_input_grad[9]:
            gradLength = (-1 * focX * gradPx + focY * gradPy)
        if ctx.needs_input_grad[10]:
            gradK1n = length * (-1 * (x * gradPx) + y * gradPy)
        if ctx.needs_input_grad[11]:
            gradK2n = length * (-1 * (1 / 2 * (x ** 2 - y ** 2) * gradPx) + x * y * gradPy)
        if ctx.needs_input_grad[12]:
            gradK1s = length * (y * gradPx + x * gradPy)
        if ctx.needs_input_grad[13]:
            gradK2s = length * (x * y * gradPx + 1 / 2 * (x ** 2 - y ** 2) * gradPy)

        return newGradX, gradPx, newGradY, gradPy, gradSigma, gradPSigma, gradDelta, gradPz, gradVR, gradLength, gradK1n, gradK2n, gradK1s, gradK2s


class EdgeKick(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, px, y, py, sigma, pSigma, delta, pz, vR, weight, curvature):
        # save inputs for backward pass
        ctx.save_for_backward(weight, curvature, x, y, torch.sqrt(1 + delta))

        # update momenta
        newXp = px + weight * curvature * torch.sqrt(1 + delta) * x
        newYp = py - weight * curvature * torch.sqrt(1 + delta) * y

        return x, newXp, y, newYp, sigma, pSigma, delta, pz, vR

    @staticmethod
    def backward(ctx, gradX, gradPx, gradY, gradPy, gradSigma, gradPSigma, gradDelta, gradPz, gradVR):
        # old phase space
        weight, curvature, x, y, sqrtDelta = ctx.saved_tensors

        # phase-space gradients
        newGradX = gradX + weight * curvature * sqrtDelta * gradPx
        newGradY = gradY - weight * curvature * sqrtDelta * gradPy
        newGradDelta = gradDelta + weight * curvature * 1 / (2 * sqrtDelta) * (x * gradPx - y * gradPy)

        # weight gradient
        if ctx.needs_input_grad[9]:
            gradWeight = curvature * sqrtDelta * x * gradPx - curvature * sqrtDelta * y * gradPy
        else:
            gradWeight = None

        return newGradX, gradPx, newGradY, gradPy, gradSigma, gradPSigma, newGradDelta, gradPz, gradVR, gradWeight, None


class DipoleKick(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, px, y, py, sigma, pSigma, delta, pz, vR, length, curvature):
        # save inputs for backward pass
        ctx.save_for_backward(length, curvature, x, y, delta, vR)

        # update coordinates
        newXp = px + curvature * length * (delta - curvature * x)
        newSigma = sigma - curvature * length * vR * x

        return x, newXp, y, py, newSigma, pSigma, delta, pz, vR

    @staticmethod
    def backward(ctx, gradX, gradPx, gradY, gradPy, gradSigma, gradPSigma, gradDelta, gradPz, gradVR):
        # old phase space
        length, curvature, x, y, delta, vR = ctx.saved_tensors

        # phase-space gradients
        newGradX = gradX - curvature ** 2 * length * gradPx - curvature * length * vR * gradSigma
        newGradDelta = gradDelta + curvature * length * gradPx
        newGradVR = gradVR - curvature * length * x * gradSigma

        # weight gradients
        gradLength = gradCurvature = None

        if ctx.needs_input_grad[9]:
            gradLength = curvature * (delta - curvature * x) * gradPx - curvature * vR * x * gradSigma
        if ctx.needs_input_grad[10]:
            gradCurvature = (length * (
                    delta - curvature * x) - length * curvature * x) * gradPx - length * vR * x * gradSigma

        return newGradX, gradPx, gradY, gradPy, gradSigma, gradPSigma, newGradDelta, gradPz, newGradVR, gradLength, gradCurvature


if __name__ == "__main__":
    import torch
    from torch.autograd import gradcheck

    from Beam import Beam

    # create a beam
    beam = Beam(mass=18.798, energy=19.0, exn=1.258e-6, eyn=2.005e-6, sigt=0.01, sige=0.005, particles=int(1e1))

    bunch = beam.bunch.double().requires_grad_(True)
    loseBunch = bunch[:1].transpose(1, 0).unbind(0)

    # perform numerical gradcheck for Drift
    length = torch.tensor(3.0, dtype=torch.double, requires_grad=True)

    myMap = Drift.apply
    inp = [*loseBunch, length]
    checkNew = gradcheck(myMap, inp, eps=1e-6, atol=1e-4)

    print("check result for Drift: {}".format(checkNew))

    # perform gradcheck for ThinMultipole
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
