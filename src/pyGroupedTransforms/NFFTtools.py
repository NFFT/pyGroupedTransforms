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
) -> np.array:  # nfft_index_set_without_zeros(bandwidths::Vector{Int})::Array{Int}
    """

    `freq = nfft_index_set_without_zeros(bandwidths)`

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
        return np.array(
            [
                list(range(-int(bw // 2), 0)) + list(range(1, int(bw // 2)))
                for bw in bandwidths
            ]
        )

    tmp = tuple(
        [
            list(range(-int(bw // 2), 0)) + list(range(1, int(bw // 2)))
            for bw in bandwidths
        ]
    )
    tmp = itertools.product(*(tmp[::-1]))
    freq = np.empty((d, np.prod(bandwidths - 1)), dtype=int)
    for m, x in enumerate(tmp):
        freq[:, m] = x

    return freq


def nfft_index_set(
    bandwidths: np.array,
) -> np.array:  # nfft_index_set(bandwidths::Vector{Int})::Array{Int}
    """
    `freq = nfft_index_set(bandwidths)`

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
        return np.array(
            list(range(int(-bandwidths[0] / 2), int(bandwidths[0] / 2))), dtype="int"
        )

    tmp = [list(range(int(-bw / 2), int(bw / 2))) for bw in bandwidths]
    tmp = itertools.product(*(tmp[::-1]))

    freq = np.empty((d, np.prod(bandwidths)), dtype=int)
    for m, x in enumerate(tmp):
        freq[:, m] = x
    return freq


def nfft_mask(
    bandwidths: np.array,
) -> np.array:  # nfft_mask(bandwidths::Vector{Int})::BitArray{1}
    """
    `mask = nfft_index_set(bandwidths)`

    # Input:
     * `bandwidths::Vector{Int}`

    # Output:
     * `mask::BitArray{1}` ... mask with size of the full cube having zeros whereever a frequency has at least one zero-element and vice-versa
    """

    if bandwidths.ndim > 1 or bandwidths.dtype != "int32":
        return "Please use an zero or one-dimensional numpy.array with dtype 'int32' as input"

    freq = nfft_index_set(bandwidths)
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
     * `F::LinearMap{ComplexF64}` ... Linear map of the Fourier-transform implemented by the NFFT
    """

    if bandwidths.ndim > 1 or bandwidths.dtype != "int32":
        return "Please use an zero or one-dimensional numpy.array with dtype 'int32' as input"

    (M, d) = np.shape(X)

    if len(bandwidths) == 0:
        return DeferredLinearOperator(
            dtype=np.complex128,
            shape=(M, 1),
            mfunc=lambda fhat: np.full(M, fhat[0]),
            rmfunc=lambda f: np.array([np.sum(f)]),
        )

    mask = nfft_mask(bandwidths)
    N = bandwidths
    plan = NFFT(N, M)  # ursprünglich in Julia: plan = NFFT(N, M, 2*N, 5)
    plan.x = X

    def trafo(fhat):  # function trafo(fhat::Vector{ComplexF64})::Vector{ComplexF64}
        plan.fhat = np.zeros(len(mask), dtype=np.complex128)
        plan.fhat[mask] = fhat
        plan.nfft_trafo()
        return plan.f

    def adjoint(f):  # function adjoint(f::Vector{ComplexF64})::Vector{ComplexF64}
        plan.f = f
        plan.nfft_adjoint()
        return plan.fhat[mask]

    N = np.prod(bandwidths - 1)

    return DeferredLinearOperator(
        dtype=np.complex128, shape=(M, N), mfunc=trafo, rmfunc=adjoint
    )


def get_matrix(
    bandwidths, X
):  # get_matrix(bandwidths::Vector{Int}, X::Array{Float64})::Array{ComplexF64}
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
        return np.ones((M, 1), dtype="complex")

    freq = index_set_without_zeros(np.array(bandwidths, dtype=np.int32))

    if d == 1:
        freq = freq.flatten()
        F_direct = np.array([[np.exp(-2j * np.pi * x * n) for n in freq] for x in X])
    else:
        F_direct = np.array(
            [[np.exp(-2j * np.pi * np.dot(x, n)) for n in freq.T] for x in X.T]
        )

    return F_direct
