import time

import numpy as np
import torch

from Models import SIS18_Lattice_minimal


# set up model
device = torch.device("cuda")
dtype = torch.float32

model = SIS18_Lattice_minimal(slices=4, dtype=dtype).to(device)

# load bunch
print("loading bunch")
bunch = np.loadtxt("../TorchOcelot/res/bunch_6d_n=1e6.txt.gz")
bunch = torch.as_tensor(bunch, dtype=dtype)[:,:4]
bunch = bunch.to(device)

# track
print("started tracking")
t0 = time.time()

with torch.no_grad():
    model(bunch)

# model(bunch)

print("tracking completed within {:.2f}".format(time.time() - t0))
