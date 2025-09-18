import os
import sys

src_aa = os.path.abspath(os.path.join(os.getcwd(), "src"))
sys.path.insert(0, src_aa)

import numpy as np

from pyGroupedTransforms.GroupedTransforms import *

rng = np.random.default_rng()

d = 4
ds = 3

M = 1000
X = rng.random((M, d)) - 0.5

# set up transform ###################################################

F = GroupedTransform("chui3", X, d=d, ds=ds, N=[3, 2, 1])

# compute transform with NFFT ########################################

fhat = GroupedCoefficients(F.settings)
for i in range(len(F.settings)):
    u = F.settings[i].u
    fhat[u] = rng.random(len(fhat[u]))

# arithmetic tests ###################################################

ghat = GroupedCoefficients(F.settings)
for i in range(len(F.settings)):
    u = F.settings[i].u
    ghat[u] = rng.random(len(ghat[u]))

fhat[1]
fhat[1] = 1.0
2 * fhat
fhat + ghat
fhat - ghat
F[(1, 2)]
fhat.set_data(ghat.data)

###

f = F * fhat

# generate random function values ###################################

y = rng.random(M)

fhat = F.adjoint() * y
