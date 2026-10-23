def __getattr__(name):
    # NFMTtools is imported unconditionally at package import time (see
    # GroupedCoefficients.py), so raising here directly would break importing
    # pyGroupedTransforms entirely. Raise lazily instead, only once someone
    # actually tries to use the (currently unimplemented) NFMT/"mixed" system.
    raise RuntimeError("NFMTtools is not implemented yet!")
