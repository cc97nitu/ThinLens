"""Implement phase-space transformations as autograd functions."""
import torch.autograd
from torch.utils.cpp_extension import load

Transformations_cpp = load(name="Transformations_cpp", sources=["/home/conrad/ThesisWorkspace/Tracking/ThinLens/Transformations.cpp"], verbose=False)


class Drift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, px, y, py, sigma, delta, vR, length):
        # update phase space coordinates
        newX, px, newY, py, newSigma, delta, vR, pz = Transformations_cpp.drift_forward(x, px, y, py, sigma, delta, vR,
                                                                                        length)

        # save inputs for backward pass
        ctx.save_for_backward(length, px, py, delta, pz, vR)

        return newX, px, newY, py, newSigma, delta, vR

    @staticmethod
    def backward(ctx, gradX, gradPx, gradY, gradPy, gradSigma, gradDelta, gradVR):
        # old phase space
        # length, px, py, delta, pz, vR = ctx.saved_tensors

        gradX, newGradPx, gradY, newGradPy, gradSigma, newGradDelta, newGradVR, gradLength = Transformations_cpp.drift_backward(
            gradX, gradPx, gradY, gradPy, gradSigma, gradDelta, gradVR, *ctx.saved_tensors)

        if not ctx.needs_input_grad[7]:
            gradLength = None

        return gradX, newGradPx, gradY, newGradPy, gradSigma, newGradDelta, newGradVR, gradLength


if __name__ == "__main__":
    import torch
    from torch.autograd import gradcheck

    from Beam import Beam
    import ThinLens.Transformations as PyTrafo

    # fix seed for reproducibility
    torch.manual_seed(12345)

    # create a beam
    beam = Beam(mass=18.798, energy=19.0, exn=1.258e-6, eyn=2.005e-6, sigt=0.01, sige=0.005, particles=int(1e1))

    bunch = beam.bunch.double().requires_grad_(True)
    loseBunch = bunch[:1].transpose(1, 0).unbind(0)

    #######################
    ## check Drift
    #######################
    length = torch.tensor(3.0, dtype=torch.double, requires_grad=True)
    myMap = Drift.apply
    inp = [*loseBunch, length]

    # compute accuracy for forward pass
    pyBunch = PyTrafo.Drift.apply(*inp)
    cppBunch = myMap(*inp)
    print("Drift: accurate implementation of forward pass: {}".format(pyBunch == cppBunch))

    # perform numerical grad check for Drift
    checkDriftGrad = gradcheck(myMap, inp, eps=1e-6, atol=1e-4)
    print("gradient check for Drift: {}".format(checkDriftGrad))
