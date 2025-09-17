import os
import sys

src_aa = os.path.abspath(os.path.join(os.getcwd(), "src"))
sys.path.insert(0, src_aa)

import numpy as np

from pyGroupedTransforms.GroupedTransforms import *

rng = np.random.default_rng()

d = 4

M = 1000
X = rng.random((M, d)) - 0.5

U = [(), (1,), (1, 2)]

# set up transform ###################################################

F = GroupedTransform("exp", X, U=U, N=[0, 64, 16])
F_direct = F.get_matrix()

# compute transform with NFFT ########################################

fhat = GroupedCoefficients(F.settings)
for i in range(len(F.settings)):
    u = F.settings[i].u
    fhat[u] = rng.random(len(fhat[u])) + 1.0j * rng.random(len(fhat[u]))

# arithmetic tests ###################################################

ghat = GroupedCoefficients(F.settings)
for i in range(len(F.settings)):
    u = F.settings[i].u
    ghat[u] = rng.random(len(ghat[u])) + 1.0j * rng.random(len(ghat[u]))

fhat[1]
fhat[1] = 1.0 + 1.0j
2 * fhat
fhat + ghat
fhat - ghat
F[(1, 2)]
fhat.set_data(ghat.data)

###

f = F * fhat

# compute transform without NFFT #####################################

f_direct = np.matmul(F_direct, fhat.vec())

# compare results ####################################################

error = np.linalg.norm(f - f_direct)
assert error < 1e-5

# generate random function values ####################################

y = rng.random(M) + 1.0j * rng.random(M)

# compute adjoint transform with NFFT ################################

fhat = F.adjoint() * y

# compute adjoint transform without NFFT #############################

fhat_direct = np.matmul(np.matrix(F_direct).H, y)

# compare results ####################################################

error = np.linalg.norm(fhat.vec() - fhat_direct)
assert error < 1e-5
