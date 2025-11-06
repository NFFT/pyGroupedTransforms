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
     * `N::Int` ... length of a wavelet-coefficient with the given bandwidths
    """
    if bandwidths.size == 0:
        return 1
    elif bandwidths.size == 1:
        return 2 ** (bandwidths[0] + 1) - 1
    elif bandwidths.size == 2:
        return 2 ** (bandwidths[0] + 1) * bandwidths[0] + 1
    elif bandwidths.size == 3:
        n = bandwidths[0]
        return 2**n * n**2 + 2**n * n + 2 ** (n + 1) - 1
    else:
        d = bandwidths.size
        n = bandwidths[0]
        s = 0
        for i in range(n + 1):
            s += 2**i * math.comb(i + d - 1, d - 1)
        return s


def partitions_exact_k(n, k, max_part=None):  # helpfunction for cwwt_index_set

    if max_part is None:
        max_part = n
    if k == 1:
        if n <= max_part:
            yield [n]
        return
    for i in range(min(max_part, n - k + 1), 0, -1):
        for tail in partitions_exact_k(n - i, k - 1, i):
            yield [i] + tail


def partitions(n, k, max_part=None):  # helpfunction for cwwt_index_set
    if max_part is None:
        max_part = n
    if k == 1:
        if n <= max_part:
            yield [n]
        return
    for i in range(min(max_part, n - k + 1), 0, -1):
        for tail in partitions_exact_k(n - i, k - 1, i):
            yield [i] + tail


def cwwt_index_set(n):  # cwwt_index_set(n::Vector{Int})::Array{Int}
    """
    `freq = cwwt_index_set(bandwidths)`

    # Input:
     * `n::Vector{Int}`

    # Output:
     * `freq::Array{Int}` ... all frequencies with |j|<n
    """

    d = len(n)
    if d == 0:
        return np.array([[0]], dtype=int)
    if d == 1:
        return np.arange(n[0] + 1).reshape(1, -1)  # row vector

    freq = np.zeros((d, 1), dtype=int)

    for j in range(1, n[-1] + 1):
        for x in partitions(d + j, d):
            x = [xi - 1 for xi in x]
            if all(xi <= j for xi in x):
                ys = set(permutations(x))  # diffrent Order as in julia
                for y in ys:
                    y_col = np.array(y).reshape(d, 1)
                    freq = np.hstack((freq, y_col))

    return freq


def begin_index2d(j):  # begin_index2d(j::Int)::Int
    """
    Begin of 2-dimensional index sets
    """
    if j == 0:
        ind = 1
    elif j == 1:
        ind = 2
    else:
        ind = (2**j) * (j - 1) + 2
    return ind


def indextoN(j, k):  # indextoN(j::Array{Int}, k::Array{Int})::Array{Int}
    """
    # function from index set of wavelets to natural numbers, i.e.
    # (j,k) maps to N
    # input:
    #       j in  0 ... n
    #       k in  0 ... 2^j-1
    # output:
    #       out  in N
    # creates row vector with entry for every column vector in k
    """
    d = len(j)  # dimension

    d2 = k.shape[0]
    s = k.shape[1]

    if d != d2:
        ValueError("j and k have to have same length, k has to be column vector.")

    out = np.zeros(s)

    if d == 1:
        for i in range(s):
            out[i] = 2 ** (j[0]) + k[0][i]

    elif d == 2:
        level = np.sum(j)
        for i in range(s):  # in julia j[0] ist j(1) ? kommt auch nen fehler raus..
            out[i] = (
                begin_index2d(level) + j[0] * 2**level + 2 ** (j[1]) + k[0, i] + k[1, i]
            )

    return out


# Chui-Wang-Wavelet Function for different orders m :
def Chui_wavelet(x, m):  # Chui_wavelet(x::Array{Float64}, m::Int)::Array{Float64}
    """
    % periodic Chui-Wang-Wavelets,
    % Chui Wang has support [0,2m-1],
    % INPUT:
    % x ... x-values
    % m ... order
    %
    %
    % OUTPUT:
    % psi ... function values at x
    """

    n = np.arange(0, 3 * m - 1, dtype=float)
    q = np.zeros_like(n)

    for ell in range(m + 1):
        q += math.comb(m, ell) * cardinal_bspline(n + 1 - ell - m, 2 * m)

    q *= ((-1) ** n) * (2 ** (1 - m))

    # Compute psi for each x
    psi = np.zeros_like(x, dtype=float)
    for idx, xx in enumerate(x):
        psi[idx] = np.sum(q * cardinal_bspline(2 * xx - n - (m / 2), m))

    return psi


# "periodic 1-d-wavelet for one j and different k:"
def _Chui_periodic_1d(x, m, j, k):
    """
    1D periodic Chui-Wang wavelet

    INPUT:
    - x: shape (M,), 1D input points
    - m: wavelet order
    - j: dilation parameter
    - k: shape (M, mm), translation parameters

    OUTPUT:
    - y: shape (M, mm), wavelet values
    """

    if j == -1:
        return np.ones((len(x), 1))

    mm = k.shape[1]
    y = np.zeros((len(x), mm))

    if (2**j) > (2 * m - 1):
        for i in range(mm):
            l = np.ceil(k[:, i] / 2**j - x)
            arg = 2**j * (x + l) - k[:, i]
            y[:, i] = 2 ** (j / 2) * Chui_wavelet(arg.flatten(), m)
    else:
        for i in range(mm):
            for ll in range(2 * m - 1):
                l = np.ceil(k[:, i] / 2**j - x) + ll
                arg = 2**j * (x + l) - k[:, i]
                y[:, i] += 2 ** (j / 2) * Chui_wavelet(arg.flatten(), m)

    return y


# periodic muliti-d-wavelet :
def Chui_periodic(x, m, j, k):
    """
    For d == 1 use _Chui_periodic_1d, else for d > 1:

    INPUT:
    - x: shape (M, d), M points in d dimensions
    - m: wavelet order
    - j: dilation parameters (list of ints)
    - k: translation parameters, shape (M, mm, d)

    OUTPUT:
    - y: shape (M, mm), wavelet values
    """

    if isinstance(j, int):  # d == 1
        return _Chui_periodic_1d(x, m, j, k)
    else:
        M, d = x.shape
        mm = k.shape[1]
        y = np.ones((M, mm))

        for i in range(mm):
            for dd in range(d):
                x_dd = x[:, dd]
                k_dd = k[:, i, dd][:, None]  # shape (M, 1)
                y[:, i] *= _Chui_periodic_1d(x_dd, m, j[dd], k_dd).flatten()

        return y


def get_transform(
    bandwidths, X, m
):  # get_transform(bandwidths::Vector{Int}, X::Array{Float64}, m::Int)::LinearMap
    """
    `F = get_transform(bandwidths, X, order)

    # Input:
     * `bandwidths::Vector{Int}`
     * `X::Array{Float64}` ... nodes in M x |u| format

    # Output:
     * `F::LinearMap{ComplexF64}` ... Linear maps of the sparse Matrices
    """

    if len(X.shape) == 1 or X.shape[1] == 1:
        X = X.flatten()
        d = 1
        M = len(X)
    else:
        M, d = X.shape

    freq = cwwt_index_set(bandwidths)

    if len(bandwidths) == 0:
        I = np.arange(M)
        J = np.zeros(M)
        V = np.ones(M)

        A = coo_matrix((V, (I.astype(int), J.astype(int))), (M, int(max(J) + 1)))
        B = A.T.conj()

        def trafo(x):
            return A @ x

        def adjoint(x):
            return B @ x

        return DeferredLinearOperator(
            dtype=np.float64, shape=(M, int(max(J) + 1)), mfunc=trafo, rmfunc=adjoint
        )

    if d == 1:
        X_col = X if X.ndim == 1 else X[:, 0]

        I = np.arange(M)
        J = np.zeros(M)
        V = Chui_periodic(X_col, m, 0, np.array([[0]])).flatten()

        for j in range(1, bandwidths[0] + 1):
            num = min(2**j, 2 * m - 1)

            a = (np.floor(2**j * X_col)[:, None] - 2 * m + 2) * np.ones((1, num))
            b = np.ones((M, 1)) * (np.arange(num).T)
            k = np.mod(a + b, 2**j)

            I_new = np.repeat(np.arange(M), num)
            J_new = (k + 2**j).flatten() - 1
            V_new = Chui_periodic(X_col, m, j, k).flatten()

            I = np.concatenate((I, I_new))
            J = np.concatenate((J, J_new))
            V = np.concatenate((V, V_new))

        A = coo_matrix((V, (I.astype(int), J.astype(int))), (M, int(max(J) + 1)))
        A = A.tocsc()
        A = A[:, A.getnnz(0) > 0]
        B = A.T.conj()

        def trafo(x):
            return A @ x

        def adjoint(x):
            return B @ x

        return DeferredLinearOperator(
            dtype=np.float64, shape=(M, A.shape[1]), mfunc=trafo, rmfunc=adjoint
        )

    elif d == 2:
        X1 = X[:, 0]
        X2 = X[:, 1]

        I = np.arange(M)
        J = np.zeros(M)
        V = Chui_periodic(X, m, [0, 0], np.array([[[0, 0]]])).flatten()

        freq = freq[:, 1 : freq.shape[1]]
        ac_co = 2

        for j in freq.T:
            num1 = min(2 ** j[0], 2 * m - 1)
            num2 = min(2 ** j[1], 2 * m - 1)

            k1 = np.mod(
                (np.floor(2 ** j[0] * X1)[:, None] - 2 * m + 2) * np.ones((1, num1))
                + np.ones((M, 1)) * np.arange(num1).T,
                2 ** j[0],
            )
            k2 = np.mod(
                (np.floor(2 ** j[1] * X2)[:, None] - 2 * m + 2) * np.ones((1, num2))
                + np.ones((M, 1)) * np.arange(num2).T,
                2 ** j[1],
            )

            if len(k1.shape) == 1:
                k1 = k1[:, None]
            if len(k2.shape) == 1:
                k2 = k2[:, None]

            k_ind = np.zeros((M, num1 * num2))
            k = np.zeros((M, num1 * num2, d))

            for kk1 in range(num1):
                k_ind[:, kk1 * num2 : (kk1 + 1) * num2] = (
                    2 ** j[1] * k1[:, kk1][:, None] + k2
                )
                k[:, kk1 * num2 : (kk1 + 1) * num2, :] = np.concatenate(
                    (
                        np.expand_dims(k1[:, kk1][:, None] * np.ones(num2), axis=2),
                        np.expand_dims(k2, axis=2),
                    ),
                    axis=2,
                )

            I_new = np.repeat(np.arange(M), num1 * num2)
            J_new = (k_ind + ac_co).flatten() - 1
            V_new = Chui_periodic(X, m, [j[0], j[1]], k).flatten()

            I = np.concatenate((I, I_new))
            J = np.concatenate((J, J_new))
            V = np.concatenate((V, V_new))

            ac_co = ac_co + np.prod(2**j)

        A = coo_matrix((V, (I.astype(int), J.astype(int))), (M, int(max(J) + 1)))
        A = A.tocsc()
        A = A[:, A.getnnz(0) > 0]
        B = A.T.conj()

        def trafo(x):
            return A @ x

        def adjoint(x):
            return B @ x

        return DeferredLinearOperator(
            dtype=np.float64, shape=(M, A.shape[1]), mfunc=trafo, rmfunc=adjoint
        )

    elif d == 3:
        I = np.arange(M)
        J = np.zeros(M)
        V = Chui_periodic(X, m, [0, 0, 0], np.array([[[0, 0, 0]]])).flatten()

        freq = freq[:, 1:]
        ac_co = 2

        for j in freq.T:
            num1 = min(2 ** j[0], 2 * m - 1)
            num2 = min(2 ** j[1], 2 * m - 1)
            num3 = min(2 ** j[2], 2 * m - 1)

            k1 = np.mod(
                (np.floor(2 ** j[0] * X[:, 0])[:, None] - 2 * m + 2) + np.arange(num1),
                2 ** j[0],
            )
            k2 = np.mod(
                (np.floor(2 ** j[1] * X[:, 1])[:, None] - 2 * m + 2) + np.arange(num2),
                2 ** j[1],
            )
            k3 = np.mod(
                (np.floor(2 ** j[2] * X[:, 2])[:, None] - 2 * m + 2) + np.arange(num3),
                2 ** j[2],
            )

            k_ind = np.zeros((M, num1 * num2 * num3))
            k = np.zeros((M, num1 * num2 * num3, d))

            for kk1 in range(num1):
                for kk2 in range(num2):
                    idx_start = (kk1 * num2 + kk2) * num3
                    idx_end = idx_start + num3
                    k_ind[:, idx_start:idx_end] = (
                        2 ** (j[1] + j[2]) * k1[:, kk1][:, None]
                        + 2 ** j[2] * k2[:, kk2][:, None]
                        + k3
                    )
                    k[:, idx_start:idx_end, 0] = k1[:, kk1][:, None]
                    k[:, idx_start:idx_end, 1] = k2[:, kk2][:, None]
                    k[:, idx_start:idx_end, 2] = k3

            I_new = np.repeat(np.arange(M), num1 * num2 * num3)
            J_new = (k_ind + ac_co).flatten() - 1
            V_new = Chui_periodic(X, m, j.tolist(), k).flatten()

            I = np.concatenate((I, I_new))
            J = np.concatenate((J, J_new))
            V = np.concatenate((V, V_new))

            ac_co += 2 ** sum(j)

        A = coo_matrix((V, (I.astype(int), J.astype(int))), (M, int(max(J) + 1)))
        A = A.tocsc()
        A = A[:, A.getnnz(0) > 0]
        B = A.T.conj()

        def trafo(x):
            return A * x

        def adjoint(x):
            return B * x

        return DeferredLinearOperator(
            dtype=np.float64, shape=(M, A.shape[1]), mfunc=trafo, rmfunc=adjoint
        )

    elif d == 4:
        I = np.arange(M)
        J = np.zeros(M)
        V = Chui_periodic(X, m, [0, 0, 0, 0], np.array([[[0, 0, 0, 0]]])).flatten()

        freq = freq[:, 1 : freq.shape[1]]
        ac_co = 2

        for j in freq.T:
            num4 = min(2 ** j[0], 2 * m - 1)
            num1 = min(2 ** j[1], 2 * m - 1)
            num3 = min(2 ** j[2], 2 * m - 1)
            num2 = min(2 ** j[3], 2 * m - 1)

            k1 = np.mod(
                (np.floor(2 ** j[0] * X[:, 0])[:, None] - 2 * m + 2)
                * np.ones((1, num1))
                + np.ones((M, 1)) * np.arange(num1),
                2 ** j[0],
            )
            k2 = np.mod(
                (np.floor(2 ** j[1] * X[:, 1])[:, None] - 2 * m + 2)
                * np.ones((1, num2))
                + np.ones((M, 1)) * np.arange(num2),
                2 ** j[1],
            )
            k3 = np.mod(
                (np.floor(2 ** j[2] * X[:, 2])[:, None] - 2 * m + 2)
                * np.ones((1, num3))
                + np.ones((M, 1)) * np.arange(num3),
                2 ** j[2],
            )
            k4 = np.mod(
                (np.floor(2 ** j[3] * X[:, 3])[:, None] - 2 * m + 2)
                * np.ones((1, num4))
                + np.ones((M, 1)) * np.arange(num4),
                2 ** j[3],
            )

            total_size = num1 * num2 * num3 * num4
            k_ind = np.zeros((M, total_size))
            k = np.zeros((M, total_size, d))

            idx = 0
            for kk1 in range(num1):
                for kk2 in range(num2):
                    for kk3 in range(num3):
                        for kk4 in range(num4):
                            k_ind[:, idx] = (
                                2 ** (j[1] + j[2] + j[3]) * k1[:, kk1]
                                + 2 ** (j[2] + j[3]) * k2[:, kk2]
                                + 2 ** j[3] * k3[:, kk3]
                                + k4[:, kk4]
                            )
                            k[:, idx, 0] = k1[:, kk1]
                            k[:, idx, 1] = k2[:, kk2]
                            k[:, idx, 2] = k3[:, kk3]
                            k[:, idx, 3] = k4[:, kk4]
                            idx += 1

            I_new = np.repeat(np.arange(M), total_size)
            J_new = (k_ind + ac_co).flatten() - 1
            V_new = Chui_periodic(X, m, j.tolist(), k).flatten()

            I = np.concatenate((I, I_new))
            J = np.concatenate((J, J_new))
            V = np.concatenate((V, V_new))

            ac_co += 2 ** sum(j)

        A = coo_matrix((V, (I.astype(int), J.astype(int))), (M, int(max(J) + 1)))
        A = A.tocsc()
        A = A[:, A.getnnz(0) > 0]
        B = A.T.conj()

        def trafo(x):
            return A * x

        def adjoint(x):
            return B * x

        return DeferredLinearOperator(
            dtype=np.float64, shape=(M, A.shape[1]), mfunc=trafo, rmfunc=adjoint
        )

    elif d == 5:
        I = np.arange(M)
        J = np.zeros(M)
        V = Chui_periodic(
            X, m, [0, 0, 0, 0, 0], np.array([[[0, 0, 0, 0, 0]]])
        ).flatten()

        freq = freq[:, 1 : freq.shape[1]]
        ac_co = 2

        for j in freq.T:
            num1 = min(2 ** j[0], 2 * m - 1)
            num2 = min(2 ** j[1], 2 * m - 1)
            num3 = min(2 ** j[2], 2 * m - 1)
            num4 = min(2 ** j[3], 2 * m - 1)
            num5 = min(2 ** j[4], 2 * m - 1)

            k1 = np.mod(
                (np.floor(2 ** j[0] * X[:, 0])[:, None] - 2 * m + 2)
                * np.ones((1, num1))
                + np.ones((M, 1)) * np.arange(num1),
                2 ** j[0],
            )
            k2 = np.mod(
                (np.floor(2 ** j[1] * X[:, 1])[:, None] - 2 * m + 2)
                * np.ones((1, num2))
                + np.ones((M, 1)) * np.arange(num2),
                2 ** j[1],
            )
            k3 = np.mod(
                (np.floor(2 ** j[2] * X[:, 2])[:, None] - 2 * m + 2)
                * np.ones((1, num3))
                + np.ones((M, 1)) * np.arange(num3),
                2 ** j[2],
            )
            k4 = np.mod(
                (np.floor(2 ** j[3] * X[:, 3])[:, None] - 2 * m + 2)
                * np.ones((1, num4))
                + np.ones((M, 1)) * np.arange(num4),
                2 ** j[3],
            )
            k5 = np.mod(
                (np.floor(2 ** j[4] * X[:, 4])[:, None] - 2 * m + 2)
                * np.ones((1, num5))
                + np.ones((M, 1)) * np.arange(num5),
                2 ** j[4],
            )

            total_size = num1 * num2 * num3 * num4 * num5
            k_ind = np.zeros((M, total_size))
            k = np.zeros((M, total_size, d))

            idx = 0
            for kk1 in range(num1):
                for kk2 in range(num2):
                    for kk3 in range(num3):
                        for kk4 in range(num4):
                            for kk5 in range(num5):
                                k_ind[:, idx] = (
                                    2 ** (j[1] + j[2] + j[3] + j[4]) * k1[:, kk1]
                                    + 2 ** (j[2] + j[3] + j[4]) * k2[:, kk2]
                                    + 2 ** (j[3] + j[4]) * k3[:, kk3]
                                    + 2 ** j[4] * k4[:, kk4]
                                    + k5[:, kk5]
                                )
                                k[:, idx, 0] = k1[:, kk1]
                                k[:, idx, 1] = k2[:, kk2]
                                k[:, idx, 2] = k3[:, kk3]
                                k[:, idx, 3] = k4[:, kk4]
                                k[:, idx, 4] = k5[:, kk5]
                                idx += 1

            I_new = np.repeat(np.arange(M), total_size)
            J_new = (k_ind + ac_co).flatten() - 1
            V_new = Chui_periodic(X, m, j.tolist(), k).flatten()

            I = np.concatenate((I, I_new))
            J = np.concatenate((J, J_new))
            V = np.concatenate((V, V_new))

            ac_co += 2 ** sum(j)

        A = coo_matrix((V, (I.astype(int), J.astype(int))), (M, int(max(J) + 1)))
        A = A.tocsc()
        A = A[:, A.getnnz(0) > 0]
        B = A.T.conj()

        def trafo(x):
            return A * x

        def adjoint(x):
            return B * x

        return DeferredLinearOperator(
            dtype=np.float64, shape=(M, A.shape[1]), mfunc=trafo, rmfunc=adjoint
        )
