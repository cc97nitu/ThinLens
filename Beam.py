import math

import torch
import torch.distributions

import sixtracklib as stl


class Beam(object):
    def __init__(self, mass: float, energy: float, exn: float, eyn: float, sigt: float, sige: float, particles: int,
                 charge: int = 1, centroid: list = (0, 0, 0, 0, 0)):
        """
        Set up beam including bunch of individual particles.
        """
        # calculate properties of reference particle
        self.energy = energy  # GeV
        self.mass = mass  # GeV / c**2
        self.charge = charge  # e
        self.exn = exn
        self.eyn = eyn
        self.sigt = sigt
        self.sige = sige
        self.centroid = centroid

        self.gamma = self.energy / self.mass
        self.momentum = math.sqrt(self.energy ** 2 - self.mass ** 2)  # GeV/c

        self.beta = self.momentum / (self.gamma * self.mass)

        # standard deviations assuming round beams
        ex = exn / (self.beta * self.gamma)  # m
        ey = eyn / (self.beta * self.gamma)  # m

        stdX = math.sqrt(ex / math.pi)
        stdY = math.sqrt(ey / math.pi)

        stdE = sige * self.energy  # GeV

        std = torch.FloatTensor([stdX, stdX, stdY, stdY, sigt, stdE])

        # sample particles
        loc = torch.FloatTensor([*centroid, self.energy])
        scaleTril = torch.diag(std ** 2)
        dist = torch.distributions.multivariate_normal.MultivariateNormal(loc, scale_tril=scaleTril)

        # create bunch
        preliminaryBunch = dist.sample((particles,)).double()  # x, xp, y, yp, sigma, totalEnergy

        x = preliminaryBunch[:, 0]
        xp = preliminaryBunch[:, 1]
        y = preliminaryBunch[:, 2]
        yp = preliminaryBunch[:, 3]
        sigma = preliminaryBunch[:, 4]
        energy = preliminaryBunch[:, 5]

        # calculate missing properties for individual particles
        momentum = torch.sqrt(energy ** 2 - self.mass ** 2)

        delta = (momentum - self.momentum) / self.momentum
        gamma = energy / self.mass
        beta = momentum / (gamma * self.mass)
        velocityRatio = self.beta / beta

        # assemble bunch
        self.bunch = torch.stack([x, xp, y, yp, sigma, delta, velocityRatio,]).t()

        # check if nan occurs in bunch <- can be if sige is too large and hence energy is smaller than rest energy
        assert not self.bunch.isnan().any()

        return

    def fromDelta(self, delta: torch.tensor):
        """Set particle momentum deviation to delta and adjust coordinates accordingly."""
        if len(delta) > len(self.bunch):
            raise ValueError("more delta values given than particles")

        # calculate properties
        momentum = self.momentum * delta + self.momentum
        energy = torch.sqrt(momentum ** 2 + self.mass ** 2)
        gamma = energy / self.mass
        beta = momentum / (gamma * self.mass)

        velocityRatio = self.beta / beta

        # select and update particles
        bunch = self.bunch[:len(delta)].t()
        bunch = torch.stack([*bunch[:5], delta, velocityRatio])
        return bunch.t()

    def madX(self):
        """Export as arguments for madx.beam command."""
        return {"mass": self.mass, "charge": self.charge, "exn": self.exn, "eyn": self.eyn, "gamma": self.gamma}

    def toDict(self):
        """Beam properties as dictionary."""
        return {"mass": self.mass, "charge": self.charge, "exn": self.exn, "eyn": self.eyn, "gamma": self.gamma,
                "sigt": self.sigt, "sige": self.sige, "energy": self.energy, "centroid": self.centroid}

    def sixTrackLibParticles(self):
        """Return bunch as stl.Particles."""
        particles = stl.Particles.from_ref(len(self.bunch), self.momentum, )

        # this sets reference momentum, mass and velocity to correct values
        particles.set_reference(p0c=1e9*self.momentum, mass0=1e9*self.mass)

        # load phase space coordinates
        particles.x = self.bunch[:, 0]
        particles.px = self.bunch[:, 1]
        particles.y = self.bunch[:, 2]
        particles.py = self.bunch[:, 3]
        particles.zeta = self.bunch[:, 4]


        # apparently this updates other coordinates related to longitudinal momentum too
        particles.delta = self.bunch[:, 5]

        return particles


if __name__ == "__main__":
    torch.set_printoptions(precision=2, sci_mode=True)

    beam = Beam(mass=18.798, energy=19.0, exn=1.258e-6, eyn=2.005e-6, sigt=0.01, sige=0.005, particles=int(1e1))

    print(beam.gamma)

