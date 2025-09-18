import numpy as np

from pyGroupedTransforms import *


def get_NumFreq(settings):
    if settings[0].mode == CWWTtools:

        def datalength(bandwidths):
            if len(bandwidths) == 0:
                return 1
            elif len(bandwidths) == 1:
                return 2 ** (bandwidths[0] + 1) - 1
            elif len(bandwidths) == 2:
                return 2 ** (bandwidths[0] + 1) * bandwidths[0] + 1
            elif len(bandwidths) == 3:
                n = bandwidths[0]
                return 2**n * n**2 + 2**n * n + 2 ** (n + 1) - 1
            else:
                d = len(bandwidths)
                n = bandwidths[0]
                tmp = 0
                for i in range(0, n + 1):
                    tmp += 2**i * math.comb(i + d - 1, d - 1)

            return tmp  # in julia kriegen wir hier s zurück?

        return sum(datalength(s.bandwidths) for s in settings)
    else:
        return sum([np.prod(s.bandwidths - 1) for s in settings])


def get_IndexSet(settings, d):
    nf = get_NumFreq(settings)
    index_set = np.zeros((d, nf), dtype=np.int64)
    idx = 0

    for s in settings:
        if len(s.u) == 0:
            idx += 1
            continue

        nf_u = np.prod(s.bandwidths - 1)

        if s.mode in [NFFTtools, NFCTtools]:
            index_set_u = s.mode.index_set_without_zeros(s.bandwidths)
        elif s.mode == NFMTtools:
            index_set_u = s.mode.nfmt_index_set_without_zeros(s.bandwidths, s.bases)
        else:
            ValueError(f"Unknown mode: {s.mode}")

        if len(s.u) == 1:
            index_set[s.u[0], idx : idx + nf_u] = index_set_u
        else:
            for i, dim in enumerate(s.u):
                index_set[dim, idx : idx + nf_u] = index_set_u[i]

        idx += nf_u

    return index_set


# get_setting ist in GroupedTransform.py
