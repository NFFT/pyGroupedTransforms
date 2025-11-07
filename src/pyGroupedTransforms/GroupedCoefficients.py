import numpy as np
import scipy

from pyGroupedTransforms import CWWTtools, NFCTtools, NFFTtools, NFMTtools


class GC:  # Superclass of GroupedCoefficientsComplex and GroupedCoefficientsReal.

    def __getitem__(self, idx):
        """
        (fhat::GroupedCoefficients[u::Vector{Int}])

        If idx is a tuple that contains integer,
        this function overloads getitem of GC such that you can do `fhat[(1,3)]` to obtain the basis coefficients of the corresponding ANOVA term defined by `u`.

        (fhat::GroupedCoefficients[idx::Int])

        If idx is an integer,
        this function overloads getitem of GC such that you can do `fhat[1]` to obtain the basis coefficient determined by `idx`.
        """
        if type(idx) == int:
            return self.data[idx]

        elif type(idx) == tuple and (len(idx) == 0 or type(idx[0]) == int):
            start = 0
            for s in self.settings:
                if len(s.u) == len(idx) and s.u == idx:
                    stop = start + s.mode.datalength(s.bandwidths)
                    return np.array(self.data[start:stop])
                else:
                    start += s.mode.datalength(s.bandwidths)

            raise KeyError("This term is not contained")
        else:
            raise TypeError(
                "Index must be an integer or a tuple that contains integer or the empty tuple"
            )

    def __setitem__(self, idx, Number):
        """
        (fhat::GroupedCoefficients[u::Vector{Int}] = fhatu::Union{Vector{ComplexF64},Vector{Float64}})

        If idx is a tuple that contains integer,
        this function overloads setitem of GC such that you can do `fhat[(1,3)] = [1 2 3]` to set the basis coefficients of the corresponding ANOVA term defined by `u`.

        (fhat::GroupedCoefficients[idx::Int] = z::Number)

        If idx is an integer,
        this function overloads setitem of GC such that you can do `fhat[1] = 3` to set the basis coefficient determined by `idx`.
        """
        if type(idx) == int:
            self.data[idx] = Number

        elif type(idx) == tuple and (len(idx) == 0 or type(idx[0]) == int):
            if not isinstance(Number, np.ndarray):
                raise TypeError("Type mismatch.")
            if (
                isinstance(self, GroupedCoefficientsComplex)
                and np.isrealobj(Number)
                or isinstance(self, GroupedCoefficientsReal)
                and np.iscomplexobj(Number)
            ):
                raise TypeError("Type mismatch.")

            start = 0
            for s in self.settings:
                if len(s.u) == len(idx) and s.u == idx:
                    stop = start + s.mode.datalength(s.bandwidths)
                    self.data[start:stop] = Number
                    return None
                else:
                    start += s.mode.datalength(s.bandwidths)

            raise KeyError("This term is not contained")

        else:
            raise TypeError("Index must be an int or a tuple that contains integer")

    def vec(self):
        """
        (vec( fhat::GroupedCoefficients )::Vector{<:Number})

        This function returns the vector of the basis coefficients of self.
        """
        return self.data

    def __rmul__(self, alpha):
        """
        (*( z::Number, fhat::GroupedCoefficients )::GroupedCoefficients)

        This function defines the multiplication of a number with a GC object.
        """
        if isinstance(alpha, (int, float, complex)):
            return GroupedCoefficients(self.settings, alpha * self.data)
        return NotImplemented

    def __mul__(self, alpha):

        if isinstance(alpha, (int, float, complex)):
            return GroupedCoefficients(self.settings, alpha * self.data)
        return NotImplemented

    def __add__(self, other):
        """
        (+( z::Number, fhat::GroupedCoefficients )::GroupedCoefficients)

        This function defines the addition of two GC objects.
        """
        if not isinstance(other, GC):
            return NotImplemented

        if self.settings == other.settings:
            return GroupedCoefficients(self.settings, self.data + other.data)
        else:
            raise ValueError("Settings mismatch.")

    def __sub__(self, other):
        """
        (-( z::Number, fhat::GroupedCoefficients )::GroupedCoefficients)

        This function defines the subtraction of two GC objects.
        """
        return self + (-1 * other)

    def set_data(self, data):
        """
        (set_data!(
        fhat::GroupedCoefficients,
        data::Union{Vector{ComplexF64},Vector{Float64}},))

        With this function one can set the data of a GC object.
        """
        if (
            isinstance(self, GroupedCoefficientsComplex)
            and np.isrealobj(data)
            or isinstance(self, GroupedCoefficientsReal)
            and np.iscomplexobj(data)
        ):
            raise TypeError("Type mismatch.")
        else:
            self.data[:] = data
            return None

    def norms(self, Dict=False, other=None, m=None):
        """
        If other == None:
        (norms(fhat::GroupedCoefficients; dict =false))

        If other != None and m == None:
        (norms(fhat::GroupedCoefficients, what::GroupedCoefficients)::Vector{Float64})

        If other == None and m != None:
        (function norms(fhat::GroupedCoefficients, m::Int ; dict =false)::Union{Vector{Float64},Dict{Vector{Int},Float64}})
        Inputs sind vertauscht leider...
        """

        if other == None:
            if m == None:
                if Dict == False:
                    return [
                        np.linalg.norm(self[self.settings[i].u])
                        for i in range(len(self.settings))
                    ]

                else:
                    dd = {}
                    for s in self.settings:
                        if len(s.u) > 0:
                            dd[tuple(s.u)] = np.linalg.norm(self[s.u])
                    return dd

            else:  # m != None
                if Dict == False:
                    if (
                        self.settings[0].mode == CWWTtools
                    ):  # does not work (in julia and in python... ?)
                        n = np.zeros(len(self.settings))
                        for i in range(len(self.settings)):
                            s = self.settings[i]
                            d = len(self.settings[i].u)
                            ac_in = 1
                            freq = CWWTtools.cwwt_index_set(self.settings[i].bandwidths)
#                            if d == 1:                    TODO: Why????
#                                freq = freq.T

                            for jj in range(freq.shape[1]):
                                j = freq[:, jj]
                                a = self[s.u][ac_in - 1 : ac_in + 2 ** np.sum(j) - 1]
                                Psi = scipy.linalg.circulant(variances(j[0], m))
                                if d > 1:
                                    for dd in range(1, d):
                                        Psi = np.kron(
                                            Psi,
                                            scipy.linalg.circulant(variances(j[dd], m)),
                                        )

                                n[i] += a @ Psi.T @ a.T
                                ac_in = ac_in + 2 ** np.sum(j)
                        return np.sqrt(n)
                    else:
                        return [np.linalg.norm(self[s.u]) for s in self.settings]

                else:  # dict == True
                    dd = {}
                    if self.settings[0].mode == CWWTtools:  # doesn't work... ?
                        n = np.zeros(len(self.settings))
                        for i in range(len(self.settings)):
                            s = self.settings[i]
                            d = len(self.settings[i].u)
                            ac_in = 1
                            freq = CWWTtools.cwwt_index_set(self.settings[i].bandwidths)
#                            if d == 1:
#                                freq = freq.T

                            for jj in range(freq.shape[1]):
                                j = freq[:, jj]
                                a = self[s.u][ac_in - 1 : ac_in + 2 ** np.sum(j) - 1]
                                Psi = scipy.linalg.circulant(variances(j[0], m))
                                if d > 1:
                                    for dd in range(1, d):
                                        Psi = np.kron(
                                            Psi,
                                            scipy.linalg.circulant(variances(j[dd], m)),
                                        )

                                n[i] += a @ Psi.T @ a.T
                                ac_in = ac_in + 2 ** np.sum(j)

                            if len(self.settings[i].u) != 0:
                                dd[self.settings[i].u] = np.sqrt(n[i])
                        return dd
                    else:
                        for i in range(len(self.settings)):
                            if len(self.settings[i].u) != 0:
                                dd[tuple(self.settings[i].u)] = np.linalg.norm(
                                    self[self.settings[i].u]
                                )
                        return dd

        else:  # other != None
            c = GroupedCoefficients(
                self.settings, (np.sqrt(np.real(other.data))) * self.data
            )
            return [np.linalg.norm(c[c.settings[i].u]) for i in range(len(c.settings))]


# Matrix of variances between two basis functions, needed for wavelet basis, since they are not orthonormal
def variances(j, m):
    """
    (variances(j::Int,m::Int)::Vector{Float64})


    INPUT
    j 		... level of wavelet
    m 		...	order of wavelet
    OUTPUT
    y = (<psi_{j,0},psi_{j,k}>)for k = 0...2^j-1
                                                            (psi_{j,k}) are the wavelets, output contains a vector of all scalar products of one level
                                                            for 2^j >2m*1 the same values, but more zeros for higher j.

    """
    if m == 2:
        if j == 0:
            y = [1 / 3]
        elif j == 1:
            y = [0.240740740740715, 0.092592592592576]
        elif j == 2:
            y = [
                0.250000000000097,
                0.046296296296494,
                -0.009259259259238,
                0.046296296296451,
            ]
        else:
            y = np.zeros(2**j)
            y[0:3] = [0.249999999999968, 0.046296296296379, -0.004629629629584]
            y[-3:-1] = [0.046296296296379, -0.004629629629584]

    elif m == 3:
        if j == 0:
            y = [0.133333333333334]
        elif j == 1:
            y = [0.085629629629627, 0.047703703703701]
        elif j == 2:
            y = [
                0.098444444444468,
                0.023851851851835,
                -0.012814814814815,
                0.023851851851891,
            ]
        elif j == 3:
            y = [
                0.098443287037094,
                0.024151620370357,
                -0.006407407407473,
                -2.997685185108751e-04,
                1.157407410282006e-06,
                -2.997685185108751e-04,
                -0.006407407407473,
                0.024151620370,
            ]
        else:
            y = np.zeros(2**j)
            y[0:5] = [
                0.098443287037052,
                0.024151620370458,
                -0.006407407407375,
                -2.997685186147822e-04,
                5.787036288131668e-07,
            ]
            y[-5:-1] = y[1:5]

    elif m == 4:
        if j == 0:
            y = [0.053968253968254]
        elif j == 1:
            y = [0.032014093350, 0.021954160617785]
        elif j == 2:
            y = [
                0.041543521817902,
                0.010977080308839,
                -0.009529428467484,
                0.010977080308902,
            ]
        elif j == 3:
            y = [
                0.041534149423629,
                0.011629855618373,
                -0.004764714233735,
                -6.527753094869201e-04,
                9.372394228547530e-06,
                -6.527753094788768e-04,
                -0.004764714233719,
                0.011629855618379,
            ]
        else:
            y = np.zeros(2**j)
            y[0:7] = [
                0.041534149423653,
                0.011629855618368,
                -0.004764714225921,
                -6.528682451447852e-04,
                4.686197131487609e-06,
                9.293564145017154e-08,
                -7.821712475559538e-12,
            ]
            y[-7:-1] = y[1:7]

    return y


class Setting:
    def __init__(self, u=None, mode=NFFTtools, bandwidths=None, bases=None):
        self.u = u
        self.mode = mode
        self.bandwidths = bandwidths
        self.bases = bases

    def __eq__(self, other):
        if isinstance(other, Setting):
            boollist = [False, False, False, False]
            if self.mode == other.mode:
                boollist[0] = True

            if len(self.u) == 0 == len(other.u):
                boollist[1] = True
            elif len(self.u) == len(other.u) and self.u == other.u:
                boollist[1] = True

            if len(self.bandwidths) == 0 == len(other.bandwidths):
                boollist[2] = True
            elif (
                len(self.bandwidths) == len(other.bandwidths)
                and (self.bandwidths == other.bandwidths).all()
            ):
                boollist[2] = True

            if self.bases == None == other.bases:
                boollist[3] = True
            elif len(self.bases) == len(other.bases) and self.bases == other.bases:
                boollist[3] = True

            return boollist[0] and boollist[2] and boollist[2] and boollist[3]
        else:
            return NotImplemented


# A class to hold complex coefficients belonging to indices in a grouped index set

### Fields:
# * `setting::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{String}}}}` - uniquely describes the setting such as the bandlimits ``N_{\pmb u}``, see also [`get_setting(system::String,d::Int,ds::Int,N::Vector{Int},basis_vect::Vector{String})::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{String}}}}`](@ref) and #[`get_setting(system::String,U::Vector{Vector{Int}},N::Vector{Int},basis_vect::Vector{String})::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{String}}}}`](@ref)
# * `data::Union{Vector{ComplexF64},Nothing}` - the vector of coefficients

## Constructor
#    GroupedCoefficientsComplex( setting, data = nothing )

## Additional Constructor
#    GroupedCoefficients( setting, data = nothing )


class GroupedCoefficientsComplex(GC):
    def __init__(self, settings, data=None):

        try:
            N = sum(s.mode.datalength(s.bandwidths) for s in settings)
        except:
            raise RuntimeError(
                "The mode is not supportet yet or does not have the function datalength."
            )

        if data is None:
            data = np.zeros(N, dtype=complex)
        if len(data) != N:
            raise ValueError("The supplied data vector has the wrong length.")

        self.settings = settings
        self.data = data


# A class to hold real coefficients belonging to indices in a grouped index set

### Fields
# * `setting::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{String}}}}` - uniquely describes the setting such as the bandlimits ``N_{\pmb u}``, see also [`get_setting(system::String,d::Int,ds::Int,N::Vector{Int},basis_vect::Vector{String})::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{String}}}}`](@ref) and [`get_setting(system::String,U::Vector{Vector{Int}},N::Vector{Int},basis_vect::Vector{String})::Vector{NamedTuple{(:u, :mode, :bandwidths, :bases),Tuple{Vector{Int},Module,Vector{Int},Vector{String}}}}`](@ref)
# * `data::Union{Vector{ComplexF64},Nothing}` - the vector of coefficients

## Constructor
#    GroupedCoefficientsComplex( setting, data = nothing )

## Additional Constructor
#    GroupedCoefficients( setting, data = nothing )


class GroupedCoefficientsReal(GC):
    def __init__(self, settings, data=None):

        try:
            N = sum(s.mode.datalength(s.bandwidths) for s in settings)
        except:
            raise RuntimeError(
                "The mode is not supportet yet or does not have the function datalength."
            )

        if data is None:
            data = np.zeros(N, dtype=float)
        if len(data) != N:
            raise ValueError("The supplied data vector has the wrong length.")

        self.settings = settings
        self.data = data


def GroupedCoefficients(settings, data=None):
    if settings[0].mode == NFFTtools:
        return GroupedCoefficientsComplex(settings, data)
    if settings[0].mode == NFCTtools or settings[0].mode == CWWTtools:
        return GroupedCoefficientsReal(settings, data)
