from scipy.sparse.linalg import LinearOperator


class DeferredLinearOperator(LinearOperator):

    def __init__(
        self,
        dtype="complex",
        shape=None,
        mfunc=lambda x: "not yet initialized",
        rmfunc=lambda x: "not yet initialized",
    ):

        self.sh = False
        if shape == None:
            shape = (0, 0)
        else:
            self.sh = True

        self._matvec_func = mfunc
        self._rmatvec_func = rmfunc
        self._matvec = lambda x: self._matvec_func(x)
        self._rmatvec = lambda x: self._rmatvec_func(x)
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
        if self._matvec_func(0) == "not implemented yet" or self.sh == False:
            raise NotImplementedError("LinearOperator is not yet initialized.")

        print(self.LO @ x)
        return self @ x

    def _rmatvec(self, x):
        if self._matvec_func(0) == "not implemented yet" or self.sh == False:
            raise NotImplementedError("LinearOperator is not yet initialized.")
        if self._rmatvec_func(0) == "not implemented yet":
            raise NotImplementedError("rmatvec is not yet implemented")

        return self.H @ x
