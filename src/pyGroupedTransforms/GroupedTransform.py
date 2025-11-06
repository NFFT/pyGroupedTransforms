import numpy as np

from pyGroupedTransforms import *

# All code that is linked to NFMTtools or to system = "mixed" is not tested yet....

systems = {
    "exp": NFFTtools,
    "cos": NFCTtools,
    "chui1": CWWTtools,
    "chui2": CWWTtools,
    "chui3": CWWTtools,
    "chui4": CWWTtools,
    "mixed": NFMTtools,
}


def get_superposition_set(d, ds):
    """
    get_superposition_set( d::Int, ds::Int )::Vector{Vector{Int}}

    This function returns ``U^{(d,ds)} = \{  \pmb u \subset \{1,2,\dots,d\} : |\pmb u| \leq ds \}``.
    """
    nset = [[j] for j in range(d)]
    returnset = [[]] + nset
    for i in range(ds - 1):
        nextnset = []
        for s in nset:
            for j in range(d):
                if s[-1] < j:
                    nextnset.append(s + [j])
        returnset = returnset + nextnset
        nset = nextnset

    return [tuple(item) for item in returnset]


def get_setting(
    system, N, U=None, d=None, ds=None
):  # I have to add bases and NFMT stuff here!

    if system not in systems:
        raise ValueError("System not found.")

    if d != None:  # Input system::String, d::Int,  ds::Int, N::Vector{Int}

        if len(N) != ds:
            raise ValueError("N must have ds entries.")

        tmp = np.concatenate(([0], N))
        U = get_superposition_set(d, ds)
        bwl = [np.full(len(u), tmp[len(u)], "int32") for u in U]

        if systems[system] == NFMTtools:
            if len(basis_vect) == 0:
                ValueError(
                    "please call get_setting with basis_vect for a NFMT transform."
                )
            if len(basis_vect) < d:
                ValueError("basis_vect must have an entry for every dimension.")
            return [
                Setting(
                    u=U[idx],
                    mode=systems[system],
                    bandwidths=np.array(bwl[idx], "int32"),
                    bases=basis_vect[U[idk]],
                )
                for idx in range(len(U))
            ]

        else:
            return [
                Setting(
                    u=U[idx],
                    mode=systems[system],
                    bandwidths=np.array(bwl[idx], "int32"),
                )
                for idx in range(len(U))
            ]

    if len(N) != len(U):
        raise ValueError("N must have |U| entries.")

    if type(N[0]) == int:  # Input system::String,  U::list{list{Int}},  N::list{Int}

        bwl = [None] * len(U)

        for i, u in enumerate(U):
            if len(u) == 0:
                bwl[i] = [0] * len(u)
            else:
                bwl[i] = np.full(len(u), N[i])

        if systems[system] == NFMTtools:
            if len(basis_vect) == 0:
                ValueError(
                    "please call get_setting with basis_vect for a NFMT transform."
                )
            if len(basis_vect) < max(max(u) for u in U):
                ValueError("basis_vect must have an entry for every dimension.")
            return [
                Setting(
                    u=u,
                    mode=systems[system],
                    bandwidths=np.array(bwl[i], "int32"),
                    bases=basis_vect[u],
                )
                for i, u in enumerate(U)
            ]

        else:
            return [
                Setting(u=u, mode=systems[system], bandwidths=np.array(bwl[i], "int32"))
                for i, u in enumerate(U)
            ]

    elif (
        type(N[0]) == list or type(N[0]) == np.ndarray
    ):  # Input system::String,  U::list{list{Int}},  N::list{list{Int}} oder list{np.ndarray}

        bwl = [None] * len(U)

        for i, u in enumerate(U):
            if len(u) == 0:
                bwl[i] = [0] * len(u)
            else:
                if len(N[i]) != len(u):
                    raise ValueError(
                        "Vector N has for the set", u, "not the right length"
                    )
                bwl[i] = N[i]

        if systems[system] == NFMTtools:
            if len(basis_vect) == 0:
                ValueError(
                    "please call get_setting with basis_vect for a NFMT transform."
                )
            if len(basis_vect) < max(max(u) for u in U):
                ValueError("basis_vect must have an entry for every dimension.")
            return [
                Setting(
                    u=U[idx],
                    mode=systems[system],
                    bandwidths=np.array(bwl[idx], "int32"),
                    bases=basis_vect[U[idk]],
                )
                for idx in range(len(U))
            ]

        else:
            return [
                Setting(
                    u=U[idx],
                    mode=systems[system],
                    bandwidths=np.array(bwl[idx], "int32"),
                )
                for idx in range(len(U))
            ]


###GroupedTransform

# A struct to describe a GroupedTransformation

## Fields
# * `system::String` - choice of `"exp"` or `"cos"` or `"chui1"` or `"chui2"` or `"chui3"` or `"chui4"` or `"mixed"`
# * `setting::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{String}}}}` - vector of the dimensions, mode, bandwidths and bases for each term/group, see also [`get_setting(system::String,d::Int,ds::Int,N::Vector{Int},basis_vect::Vector{String})::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{String}}}}`](@ref) and [`get_setting(system::String,U::Vector{Vector{Int}},N::Vector{Int},basis_vect::Vector{String})::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{String}}}}`](@ref)
# * `X::Array{Float64}` - array of nodes
# * `transforms::Vector{LinearMap}` - holds the low-dimensional sub transformations
# * `basis_vect::Vector{String}` - holds for every dimension if a cosinus basis [true] or exponential basis [false] is used
#
## Constructor
#    GroupedTransform( system, setting, X, basis_vect::Vector{String} = Vector{String}([]) )
#
## Additional Constructor
#    GroupedTransform( system, d, ds, N::Vector{Int}, X, basis_vect::Vector{String} = Vector{String}([]) )
#    GroupedTransform( system, U, N, X, basis_vect::Vector{String} = Vector{String}([]) )


class GroupedTransform:
    def __init__(
        self,
        system,
        X,
        settings=[],
        fastmult=True,
        parallel=True,
        basis_vect=[],
        N=[],
        U=None,
        d=None,
        ds=None,
    ):

        if system not in systems:
            raise ValueError("System not found.")

        if system == "mixed":
            if len(basis_vect) == 0:
                ValueError(
                    "please call GroupedTransform with basis_vect for a NFMT transform."
                )
            if len(basis_vect) != X.shape[1]:
                ValueError("basis_vect must have an entry for every dimension.")

        if system in {"exp", "chui1", "chui2", "chui3", "chui4"}:
            if np.min(X) < -0.5 or np.max(X) >= 0.5:
                raise ValueError("Nodes must be between -0.5 and 0.5.")
        elif system == "cos":
            if np.min(X) < 0 or np.max(X) > 0.5:
                raise ValueError("Nodes must be between 0 and 0.5.")
        elif system == "mixed":
            basis_vals = np.array([NFMTtools.BASES[b] for b in basis_vect])

            cosine_mask = basis_vals > 0
            if np.sum(cosine_mask) > 0:
                if (np.min(X[cosine_mask, :]) < 0) or (np.max(X[cosine_mask, :]) > 1):
                    raise ValueError(
                        "Nodes must be between 0 and 1 for cosine or Chebyshev dimensions."
                    )

            exp_mask = ~cosine_mask
            if np.sum(exp_mask) > 0:
                if (np.min(X[exp_mask, :]) < -0.5) or (np.max(X[exp_mask, :]) > 0.5):
                    raise ValueError(
                        "Nodes must be between -0.5 and 0.5 for exponentional dimensions."
                    )

        if system in {"chui1", "chui2", "chui3", "chui4"}:
            fastmult = True

        self.fastmult = fastmult
        self.basis_vect = basis_vect
        self.system = system
        self.X = X
        self.parallel = parallel

        if len(settings) == 0:
            self.settings = get_setting(system=system, N=N, U=U, d=d, ds=ds)
        else:
            self.settings = settings

        if fastmult:
            self.matrix = np.empty((0, 0), dtype=object)
            self.transforms = [
                DeferredLinearOperator() for i in range(len(self.settings))
            ]
            for idx, s in enumerate(self.settings):
                if len(s.bandwidths) == 0:
                    u = (0,)
                else:
                    u = s.u
                if system.startswith("chui"):
                    Order = int(system[-1])
                    self.transforms[idx] = s.mode.get_transform(
                        bandwidths=s.bandwidths, X=np.copy(X[:, u], order="C"), m=Order
                    )
                elif system == "mixed":
                    self.transforms[idx] = s.mode.get_transform(
                        bandwidths=s.bandwidths,
                        X=np.copy(X[:, u], order="C"),
                        bases=s.bases,
                    )
                else:
                    self.transforms[idx] = s.mode.get_transform(
                        bandwidths=s.bandwidths, X=np.copy(X[:, u], order="C")
                    )
        else:
            self.transforms = []
            s1 = self.settings[0]
            if len(s1.bandwidths) == 0:
                u1 = (0,)
            else:
                u1 = s1.u
            if system.startswith("chui"):
                raise ValueError(
                    "Direct computation with full matrix not supported for wavelet basis."
                )
            elif system == "mixed":
                matrix = np.array(
                    s1.mode.get_matrix(bandwidths=s1.bandwidths, X=X[:, u1].T),
                    bases=s1.bases,
                )
                for s in self.settings[1:]:
                    if len(s.bandwidths) == 0:
                        u = (0,)
                    else:
                        u = s.u
                    matrix = np.hstack(
                        [
                            matrix,
                            np.array(
                                s.mode.get_matrix(s.bandwidths, X[:, u].T),
                                bases=s.bases,
                            ),
                        ]
                    )
            else:
                matrix = np.array(
                    s1.mode.get_matrix(bandwidths=s1.bandwidths, X=X[:, u1].T)
                )
                for s in self.settings[1:]:
                    if len(s.bandwidths) == 0:
                        u = (0,)
                    else:
                        u = s.u
                    matrix = np.hstack(
                        [matrix, np.array(s.mode.get_matrix(s.bandwidths, X[:, u].T))]
                    )
            self.matrix = matrix

    def __mul__(self, other):
        """
        (*( F::GroupedTransform, fhat::GroupedCoefficients )::Vector{<:Number})

        If other (= fhat) is an object of GC, this function
        overloads the `*` notation in order to achieve `f = F*fhat`.

        (*( F::GroupedTransform, f::Vector{<:Number} )::GroupedCoefficients)

        If other (= f) is an numpy.ndarray, this function
        overloads the * notation in order to achieve the adjoint transform `f = F*f`.
        """

        if isinstance(other, np.ndarray):  # `f = F*f`    (f = other)
            if self.fastmult:
                fhat = GroupedCoefficients(self.settings)

                def adjoint_worker(i):
                    adjoint_result = self.transforms[i].H @ other
                    fhat[self.settings[i].u] = adjoint_result

                if self.parallel:
                    threads = []
                    for i in range(len(self.transforms)):
                        t = threading.Thread(target=adjoint_worker, args=(i,))
                        t.start()
                        threads.append(t)

                    for t in threads:
                        t.join()

                else:
                    for i in range(len(self.transforms)):
                        adjoint_worker(i)

                return fhat
            else:
                return GroupedCoefficients(
                    self.settings, (self.matrix.conj()).T @ other
                )
        elif isinstance(other, GC):  # `f = F*fhat`    (fhat = other)
            if self.settings != other.settings:
                raise ValueError(
                    "The GroupedTransform and the GroupedCoefficients have different settings"
                )

            if self.fastmult:
                results = []

                def worker(i):
                    u = self.settings[i].u
                    result = self.transforms[i] @ other[u]
                    results.append(result)

                if self.parallel:
                    threads = []
                    for i in range(len(self.transforms)):
                        t = threading.Thread(target=worker, args=(i,))
                        t.start()
                        threads.append(t)
                    for t in threads:
                        t.join()
                else:
                    for i in range(len(self.transforms)):
                        worker(i)

                return sum(results)
            else:
                return self.matrix @ other.data
        else:
            raise ValueError("Wrong input data type")

    def __matmul__(self, other):
        return self.__mul__(other)

    def adjoint(self):
        """
        adjoint( F::GroupedTransform )::GroupedTransform

        Overloads the `F'` notation and gives back the same GroupdTransform. GroupedTransform decides by the input if it is the normal trafo or the adjoint so this is only for convinience.
        """
        return self

    @property
    def H(self):
        """
        Overloads the `F'` notation and gives back the same GroupdTransform. GroupedTransform decides by the input if it is the normal trafo or the adjoint so this is only for convinience.
        """
        return self

    def __getitem__(self, u):
        """
        F::GroupedTransform[u::Vector{Int}]::LinearMap{<:Number} or SparseArray

        This function overloads getindex of GroupedTransform such that you can do `F[[1,3]]` to obtain the transform of the corresponding ANOVA term defined by `u`.
        """
        idx = next(
            (i for i, s in enumerate(self.settings) if len(s.u) == len(u) and s.u == u),
            None,
        )

        if idx is None:
            raise ValueError("This term is not contained")
        elif self.fastmult:
            return self.transforms[idx]
        else:
            return self.get_matrix()

    def get_matrix(self):
        """
            get_matrix( F::GroupedTransform )::Matrix{<:Number}

        This function returns the actual matrix of the transformation. This is not available for the wavelet basis.
        """
        if self.system in ["chui1", "chui2", "chui3", "chui4"]:
            raise ValueError(
                "Direct computation with full matrix not supported for wavelet basis."
            )
        elif self.system == "mixed":
            s1 = self.settings[0]
            if len(s1.bandwidths) == 0:
                u1 = (0,)
            else:
                u1 = s1.u
            F_direct = s1.mode.get_matrix(s.bdanwidths, self.X[:, u1].T, bases=s.bases)
            for idx, s in enumerate(self.settings):
                if idx == 0:
                    continue
                if len(s.bandwidths) == 0:
                    u = (0,)
                else:
                    u = s.u
                mat = s.mode.get_matrix(s.bandwidths, self.X[:, u].T, s.bases)
                F_direct = np.hstack([F_direct, mat])
        else:
            s1 = self.settings[0]
            if len(s1.bandwidths) == 0:
                u1 = (0,)
            else:
                u1 = s1.u
            F_direct = s1.mode.get_matrix(s1.bandwidths, self.X[:, u1].T)

            for idx, s in enumerate(self.settings):
                if idx == 0:
                    continue
                if len(s.bandwidths) == 0:
                    u = (0,)
                else:
                    u = s.u
                mat = s.mode.get_matrix(s.bandwidths, self.X[:, u].T)
                F_direct = np.hstack((F_direct, mat))

            return F_direct
