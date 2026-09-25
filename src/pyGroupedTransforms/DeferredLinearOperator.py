from scipy.sparse.linalg import LinearOperator


class DeferredLinearOperator(LinearOperator):

    def __init__(
        self,
        dtype="complex",
        shape=None,
        mfunc=None,
        rmfunc=None,
    ):

        self.sh = False
        if shape == None:
            shape = (0, 0)
        else:
            self.sh = True

        self._matvec_func = mfunc
        self._rmatvec_func = rmfunc
        super().__init__(dtype=dtype, shape=shape)

    def set(self, dtype=None, shape=None, mfunc=None, rmfunc=None):
        if dtype != None:
            self.dtype = dtype

        if shape != None:
            self.sh = True
            self.shape = shape

        if mfunc != None:
            self._matvec_func = mfunc

        if rmfunc != None:
            self._rmatvec_func = rmfunc

    def _matvec(self, x):
        if self._matvec_func is None or self.sh == False:
            raise NotImplementedError("LinearOperator is not yet initialized.")

        return self._matvec_func(x)

    def _rmatvec(self, x):
        if self._matvec_func is None or self.sh == False:
            raise NotImplementedError("LinearOperator is not yet initialized.")
        if self._rmatvec_func is None:
            raise NotImplementedError("rmatvec is not yet implemented")

        return self._rmatvec_func(x)
