import numpy as np

from pyGroupedTransforms import *


def datalength(
    bandwidths: np.ndarray,
) -> int:  # datalength(bandwidths::Vector{Int})::Int
    """
    `N = datalength(bandwidths)`

    # Input:
     * `bandwidths::Vector{Int}`

    # Output:
     * `N::Int` ... length of a Fourier-coefficient with the given bandwidths
    """
    if bandwidths.ndim != 1 or bandwidths.dtype != "int32":
        return "Please use an one-dimensional numpy.array with dtype 'int32' as input"
    else:
        return np.prod(bandwidths - 1)


def index_set_without_zeros(
    bandwidths: np.array,
) -> np.array:  # nfct_index_set_without_zeros(bandwidths::Vector{Int})::Array{Int}
    """
    `freq = nfct_index_set_without_zeros(bandwidths)`

    # Input:
     * `bandwidths::Vector{Int}`

    # Output:
     * `freq::Array{Int}` ... all frequencies of the full cube without any vector having a zero entry
    """

    if bandwidths.ndim > 1 or bandwidths.dtype != "int32":
        return "Please use an zero or one-dimensional numpy.array with dtype 'int32' as input"

    d = len(bandwidths)

    if d == 0:
        return np.array([0])
    if d == 1:
        return np.array([1] + list(range(2, bandwidths[0])))

    bandwidths = bandwidths[::-1]

    tmp = tuple([[1] + list(range(2, bw)) for bw in bandwidths])
    tmp = itertools.product(*(tmp[::-1]))
    freq = np.empty((d, np.prod(bandwidths - 1)), dtype=int)
    for m, x in enumerate(tmp):
        freq[:, m] = x

    return freq


def nfct_index_set(
    bandwidths: np.array,
) -> np.array:  # nfct_index_set(bandwidths::Vector{Int})::Array{Int}
    """
    `freq = nfct_index_set(bandwidths)`

    # Input:
     * `bandwidths::Vector{Int}`

    # Output:
     * `freq::Array{Int}` ... all frequencies of the full cube
    """

    if bandwidths.ndim > 1 or bandwidths.dtype != "int32":
        return "Please use an zero or one-dimensional numpy.array with dtype 'int32' as input"

    d = len(bandwidths)
    if d == 0:
        return np.array([0], dtype="int")
    if d == 1:
        return np.array([i for i in range(0, bandwidths[0])], "int")

    bandwidths = bandwidths[::-1]
    tmp = tuple([list(range(0, bw)) for bw in bandwidths])
    tmp = itertools.product(*(tmp[::-1]))

    freq = np.empty((d, np.prod(bandwidths)), dtype=int)
    for m, x in enumerate(tmp):
        freq[:, m] = x
    return freq


def nfct_mask(
    bandwidths: np.array,
) -> np.array:  # nfft_mask(bandwidths::Vector{Int})::BitArray{1}
    """
    `mask = nfct_index_set(bandwidths)`

    # Input:
     * `bandwidths::Vector{Int}`

    # Output:
     * `mask::BitArray{1}` ... mask with size of the full cube having zeros whereever a frequency has at least one zero-element and vice-versa
    """
    if bandwidths.ndim > 1 or bandwidths.dtype != "int32":
        return "Please use an zero or one-dimensional numpy.array with dtype 'int32' as input"

    freq = nfct_index_set(bandwidths)
    nfft_mask = np.empty((1, 1), dtype="bool")
    if freq.ndim == 1:
        return freq != 0
    else:
        return [0 not in col for col in freq.T]


def get_transform(
    bandwidths: np.array, X: np.array
):  # get_transform(bandwidths::Vector{Int}, X::Array{Float64})::LinearMap
    """
    `F = get_transform(bandwidths, X)

    # Input:
     * `bandwidths::Vector{Int}`
     * `X::Array{Float64}` ... nodes in |u| x M format

    # Output:
     * `F::LinearMap{Float64}` ... Linear map of the Fourier-transform implemented by the NFCT
    """

    if bandwidths.ndim > 1 or bandwidths.dtype != "int32":
        return "Please use an zero or one-dimensional numpy.array with dtype 'int32' as input"

    (M, d) = X.shape

    if len(bandwidths) == 0:
        return DeferredLinearOperator(
            dtype=np.float64,
            shape=(M, 1),
            mfunc=lambda fhat: np.full(M, fhat[0]),
            rmfunc=lambda f: np.array([np.sum(f)]),
        )

    mask = nfct_mask(bandwidths)
    N = bandwidths
    plan = NFCT(N, M)  # ursprünglich in Julia: plan = NFCT(N, M, 2*N, 5)
    plan.x = X
    factor_sqrt = sqrt(2) ** d

    def trafo(fhat):
        plan.fhat = np.zeros(len(mask), "float")
        plan.fhat[mask] = fhat
        plan.fhat *= factor_sqrt
        plan.nfct_trafo()
        return plan.f

    def adjoint(f):
        plan.f = f
        plan.nfct_adjoint()
        return plan.fhat[mask] * factor_sqrt

    N = np.prod(bandwidths - 1)

    return DeferredLinearOperator(
        dtype=np.float64, shape=(M, N), mfunc=trafo, rmfunc=adjoint
    )


def get_multiplier(n):  # get_multiplier(n::Int)::Float64

    if np.isscalar(n):
        return 1.0 if n == 0 else np.sqrt(2)
    else:
        n = np.asarray(n)
        abs_support = np.count_nonzero(n)
        return 1.0 if abs_support == 0 else np.sqrt(2) ** abs_support


def get_matrix(
    bandwidths, X
):  # function get_matrix(bandwidths::Vector{Int}, X::Array{Float64})::Array{Float64}
    """
    `F = get_matrix(bandwidths, X)

    # Input:
     * `bandwidths::Vector{Int}`
     * `X::Array{Float64}` ... nodes in |u| x M format

    # Output:
     * `F::Array{ComplexF64}` ... Matrix of the Fourier-transform
    """

    if X.ndim == 1 or X.shape[0] == 1:
        X = X.flatten()
        d = 1
        M = len(X)
    else:
        d, M = X.shape

    if len(bandwidths) == 0:
        return np.ones((M, 1), dtype="float")
    freq = index_set_without_zeros(np.array(bandwidths, dtype=np.int32))

    if d == 1:
        freq = freq.flatten()
        F_direct = np.array(
            [get_multiplier(n) * np.cos(2 * np.pi * x * n) for x in X for n in freq]
        ).reshape(M, -1)
    else:
        F_direct = np.array(
            [
                get_multiplier(n) * np.prod(np.cos(2 * np.pi * x * n))
                for x in X.T
                for n in freq.T
            ]
        ).reshape(M, -1)

    return F_direct
