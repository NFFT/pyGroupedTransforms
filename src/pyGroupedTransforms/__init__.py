import itertools
# Needed for CWWTtools:
import math
import threading
from itertools import permutations
from math import pi, sqrt

from pyNFFT3.flags import *
from pyNFFT3.NFCT import *
from pyNFFT3.NFFT import *
# Needed for GroupedCoefficients
from scipy.linalg import circulant
from scipy.sparse import coo_matrix

# import modules
from .cardinal_bspline import *
from .DeferredLinearOperator import *
from .GroupedCoefficients import *  # NFFTtools, NFCTtools, NFMTtools and CWWTtools are imported with GroupedCoefficients
from .GroupedTransform import *
from .GroupedTransforms import *

# export

__all__ = [
    "NFFTtools",
    "NFCTtools",
    "NFMTtools",
    "CWWTtools",
    "cardinal_bspline",
    "DeferredLinearOperator",
    "get_IndexSet",
    "get_NumFreq",
    "GC",
    "variances",
    "Setting",
    "GroupedCoefficientsComplex",
    "GroupedCoefficientsReal",
    "get_superposition_set",  # TODO: Exports ab hier testen
    "GroupedTransform",
    "GroupedCoefficients",
]
