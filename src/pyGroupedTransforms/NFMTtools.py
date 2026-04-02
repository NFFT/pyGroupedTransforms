import numpy as np
from pyNFFT3.NFMT import BASES

from pyGroupedTransforms import *


def datalength(bandwidths: np.ndarray) -> int:
    """Return the number of mixed-basis Fourier coefficients (zero-frequencies excluded).

    For each dimension j the frequency range used is:
      exp  -> {-N_j/2, ..., -1, 1, ..., N_j/2 - 1}  (N_j - 1 values)
      cos/alg -> {1, 2, ..., N_j - 1}                 (N_j - 1 values)
    Total: prod(N - 1).
    """
    if bandwidths.ndim != 1 or bandwidths.dtype != "int32":
        return "Please use an one-dimensional numpy.array with dtype 'int32' as input"
    if len(bandwidths) == 0:
        return 1
    return int(np.prod(bandwidths - 1))


def nfmt_index_set_without_zeros(bandwidths: np.ndarray, bases) -> np.ndarray:
    """Return the (d, prod(N-1)) integer frequency matrix.

    Mirrors Julia's GroupedTransforms.NFMTtools.nfmt_index_set_without_zeros:
      exp  dimension j: {-N_j/2, ..., -1, 1, ..., N_j/2 - 1}
      cos/alg dimension j: {1, 2, ..., N_j - 1}
    """
    d = len(bandwidths)
    if d == 0:
        return np.array([[0]], dtype=np.int64)

    ranges = []
    for n, basis in zip(bandwidths, bases):
        if BASES[basis] > 0:
            # cos or alg: {1, ..., N-1}
            ranges.append(list(range(1, int(n))))
        else:
            # exp: {-N/2, ..., -1, 1, ..., N/2-1}
            half = int(n) // 2
            ranges.append(list(range(-half, 0)) + list(range(1, half)))

    if d == 1:
        return np.array(ranges[0], dtype=np.int64).reshape(1, -1)

    # Cartesian product; last dimension changes fastest (row-major, matching Julia)
    mesh = np.array(np.meshgrid(*ranges, indexing="ij"), dtype=np.int64)
    return mesh.reshape(d, -1)


def get_matrix(bandwidths, X, bases) -> np.ndarray:
    """Build the evaluation matrix F with F[m, j] = phi(x_m, k_j).

    Mirrors Julia's GroupedTransforms.NFMTtools.get_matrix / get_phi:
      exp:     exp(-2pi i k x)
      cos:     sqrt(2) * cos(pi * k * x)   (k != 0 guaranteed by frequency set)
      alg:     sqrt(2) * cos(k * arccos(2x - 1))

    X is expected in (d, M) format (columns are nodes), consistent with the
    convention used in GroupedTransform.get_matrix.
    """
    if X.ndim == 1 or X.shape[0] == 1:
        X_eval = X.flatten().reshape(-1, 1)  # (M, 1)
        d = 1
        M = X_eval.shape[0]
    else:
        d, M = X.shape
        X_eval = X.T  # (M, d)

    if len(bandwidths) == 0:
        return np.ones((M, 1), dtype=np.complex128)

    freq = nfmt_index_set_without_zeros(
        np.asarray(bandwidths, dtype=np.int32), list(bases)
    )  # (d, nf)

    nf = freq.shape[1]
    F = np.ones((M, nf), dtype=np.complex128)

    for j in range(d):
        n_j = freq[j]  # (nf,)  all nonzero by construction
        x_j = X_eval[:, j][:, None]  # (M, 1)
        if BASES[bases[j]] == 1:
            # cos basis: sqrt(2) * cos(pi * k * x)
            F *= np.sqrt(2.0) * np.cos(np.pi * x_j * n_j)
        elif BASES[bases[j]] == 2:
            # alg (Chebyshev) basis: sqrt(2) * cos(k * arccos(2x - 1))
            F *= np.sqrt(2.0) * np.cos(n_j * np.arccos(2.0 * x_j - 1.0))
        else:
            # exp basis: exp(-2pi i k x)
            F *= np.exp(-2.0j * np.pi * x_j * n_j)

    return F


def get_transform(bandwidths: np.ndarray, X: np.ndarray, bases):
    """Return a DeferredLinearOperator for the mixed-basis grouped transform.

    Uses the same matrix as get_matrix wrapped in a linear operator so that
    trafo and adjoint are consistent by construction.
    X is expected in (M, d) format (rows are nodes), consistent with the
    convention used in GroupedTransform.transforms.
    """
    if bandwidths.ndim > 1 or bandwidths.dtype != "int32":
        return "Please use a zero or one-dimensional numpy.array with dtype 'int32' as input"

    M = X.shape[0]

    if len(bandwidths) == 0:
        return DeferredLinearOperator(
            dtype=np.complex128,
            shape=(M, 1),
            mfunc=lambda fhat: np.full(M, fhat[0], dtype=np.complex128),
            rmfunc=lambda f: np.array([np.sum(f)], dtype=np.complex128),
        )

    # X is (M, d) here; get_matrix expects (d, M)
    mat = get_matrix(bandwidths, X.T, list(bases))  # (M, nf)
    N = int(np.prod(bandwidths - 1))

    return DeferredLinearOperator(
        dtype=np.complex128,
        shape=(M, N),
        mfunc=lambda fhat: mat @ np.asarray(fhat, dtype=np.complex128),
        rmfunc=lambda f: mat.conj().T @ np.asarray(f, dtype=np.complex128),
    )
