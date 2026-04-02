import os
import sys

src_aa = os.path.abspath(os.path.join(os.getcwd(), "src"))
sys.path.insert(0, src_aa)

import numpy as np

from pyGroupedTransforms.GroupedTransforms import *

rng = np.random.default_rng(42)

d = 4
ds = 3

basis_vect = ["exp", "alg", "cos", "alg"]

M = 1000
# X in (M, d) format; exp dimension shifted to [-0.5, 0.5)
X = rng.random((M, d))
X[:, 0] -= 0.5

U = [(), (1,), (1, 2)]

# set up transform ###################################################

F = GroupedTransform(
    "mixed", X, U=U, N=[0, 64, 16], basis_vect=basis_vect
)
F_direct = F.get_matrix()

# compute transform with NFMT ########################################

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
F[(1,)]
fhat.set_data(ghat.data)

###

f = F * fhat

# compute transform without NFMT #####################################

f_direct = np.matmul(F_direct, fhat.vec())

# compare results ####################################################

error = np.linalg.norm(f - f_direct)
assert error < 1e-5, f"trafo error {error} >= 1e-5"

# generate random function values ####################################

y = rng.random(M) + 1.0j * rng.random(M)

# compute adjoint transform with NFMT ################################

fhat = F * y

# compute adjoint transform without NFMT #############################

fhat_direct = np.matmul(np.conj(F_direct).T, y)

# compare results ####################################################

error = np.linalg.norm(fhat.vec() - fhat_direct)
assert error < 1e-5, f"adjoint error {error} >= 1e-5"
