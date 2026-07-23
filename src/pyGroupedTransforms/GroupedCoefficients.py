"""Coefficients belonging to the indices of a grouped index set.

:py:class:`GC` is the shared base class; use the :py:func:`GroupedCoefficients`
factory rather than instantiating :py:class:`GroupedCoefficientsComplex` or
:py:class:`GroupedCoefficientsReal` directly, since it picks the right one from
the mode of the given settings.

A :py:class:`Setting` describes one ANOVA term — its dimensions, transform mode
and bandwidths — and a list of them uniquely fixes the layout of the
coefficient vector.
"""

import numpy as np
import scipy

from pyGroupedTransforms import CWWTtools, NFCTtools, NFFTtools, NFMTtools


class GC:
    r"""Base class for grouped coefficient vectors.

    Superclass of :py:class:`GroupedCoefficientsComplex` and
    :py:class:`GroupedCoefficientsReal`. Holds the coefficients of every ANOVA
    term in one flat vector, together with the settings that describe how that
    vector is partitioned.

    Attributes:
        settings (list[Setting]): One entry per ANOVA term, uniquely
            describing the layout such as the bandwidths.
        data (numpy.ndarray): The flat vector of coefficients, real or complex
            depending on the subclass.
    """

    def __getitem__(self, idx):
        r"""Return coefficients by ANOVA term or by position.

        - a tuple selects a whole ANOVA term: ``fhat[(1, 3)]`` returns the
          basis coefficients of the term defined by ``u``
        - an integer selects a single coefficient: ``fhat[1]``

        Args:
            idx (tuple[int] or int): The ANOVA term, or the position in the
                flat coefficient vector.

        Returns:
            numpy.ndarray or numbers.Number: The coefficients of the term, or
            the single coefficient.

        Raises:
            ValueError: If the requested term is not part of the settings.
            TypeError: If ``idx`` is neither an int nor a tuple of ints.
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
        r"""Set coefficients by ANOVA term or by position.

        - a tuple sets a whole ANOVA term: ``fhat[(1, 3)] = np.array([1, 2, 3])``
        - an integer sets a single coefficient: ``fhat[1] = 3``

        Args:
            idx (tuple[int] or int): The ANOVA term, or the position in the
                flat coefficient vector.
            Number (numpy.ndarray or numbers.Number): The value(s) to store.
                Must match the real/complex type of the object.

        Raises:
            ValueError: If the requested term is not part of the settings.
            TypeError: If ``idx`` has the wrong type, or if a real object is
                given complex data or vice versa.
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
        r"""Return the flat vector of basis coefficients.

        Returns:
            numpy.ndarray: The underlying coefficient vector.
        """
        return self.data

    def __rmul__(self, alpha):
        r"""Multiply the coefficients by a scalar.

        Args:
            alpha (numbers.Number): The scalar factor.

        Returns:
            GC: A new object holding the scaled coefficients.
        """
        if isinstance(alpha, (int, float, complex)):
            return GroupedCoefficients(self.settings, alpha * self.data)
        return NotImplemented

    def __mul__(self, alpha):

        if isinstance(alpha, (int, float, complex)):
            return GroupedCoefficients(self.settings, alpha * self.data)
        return NotImplemented

    def __add__(self, other):
        r"""Add two grouped coefficient objects.

        Args:
            other (GC): The object to add. Must carry the same settings.

        Returns:
            GC: A new object holding the summed coefficients.

        Raises:
            ValueError: If the two objects have different settings.
        """
        if not isinstance(other, GC):
            return NotImplemented

        if self.settings == other.settings:
            return GroupedCoefficients(self.settings, self.data + other.data)
        else:
            raise ValueError("Settings mismatch.")

    def __sub__(self, other):
        r"""Subtract two grouped coefficient objects.

        Args:
            other (GC): The object to subtract. Must carry the same settings.

        Returns:
            GC: A new object holding the difference.

        Raises:
            ValueError: If the two objects have different settings.
        """
        return self + (-1 * other)

    def set_data(self, data):
        r"""Overwrite the coefficient vector in place.

        Args:
            data (numpy.ndarray): The new coefficients. Must match the
                real/complex type of the object.

        Raises:
            TypeError: If a real object is given complex data, or vice versa.
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
        r"""Compute the norm of each ANOVA term's coefficient block.

        These norms are what the variance and sensitivity analyses are built
        on. For the Chui-Wang wavelet bases the basis functions are not
        orthonormal, so passing the wavelet order ``m`` switches to a weighted
        norm using the Gram matrix from :py:func:`variances`.

        Args:
            Dict (bool, optional): Defaults to ``False``. If ``True``, return a dictionary indexed by ANOVA
                term, skipping the empty term. Otherwise return a list in
                settings order.
            other (GC, optional): Compare against a second coefficient object
                instead of taking plain norms.
            m (int, optional): Wavelet order. Required to get correct norms for
                the ``"chui*"`` bases; ignored for the orthonormal bases.

        Returns:
            list or dict: One norm per ANOVA term, keyed by term if ``Dict``
            is set.
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
                                    for kd in range(1, d):
                                        Psi = np.kron(
                                            Psi,
                                            scipy.linalg.circulant(variances(j[kd], m)),
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
                                    for kd in range(1, d):
                                        Psi = np.kron(
                                            Psi,
                                            scipy.linalg.circulant(variances(j[kd], m)),
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
    r"""Return the scalar products between wavelets of one level.

    The Chui-Wang wavelets are not orthonormal, so computing norms of wavelet
    coefficients needs the Gram matrix of the basis. This returns the first row
    of that matrix for level ``j``, from which the full (circulant) matrix is
    reconstructed by :py:meth:`GC.norms`.

    Formally the result is

    .. math::

        y = \left( \langle \psi_{j,0}, \psi_{j,k} \rangle
        \right)_{k=0,\dots,2^j-1},

    where :math:`\psi_{j,k}` are the wavelets of level :math:`j`. For
    :math:`2^j > 2m` the nonzero values repeat and only the number of trailing
    zeros grows with :math:`j`.

    Args:
        j (int): Level of the wavelet.
        m (int): Order of the wavelet, 2 to 4.

    Returns:
        numpy.ndarray: The :math:`2^j` scalar products of level ``j``.
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
    r"""Describes one ANOVA term within a grouped index set.

    A list of these uniquely fixes the layout of a grouped coefficient vector
    and of a :py:class:`GroupedTransform`. Two settings compare equal when
    their mode, dimensions, bandwidths and bases all agree.

    Args:
        u (tuple[int]): The coordinate subset this term covers. The empty
            tuple denotes the constant term.
        mode (module): The transform module handling this term — one of
            :py:mod:`NFFTtools`, :py:mod:`NFCTtools`, :py:mod:`NFMTtools` or
            :py:mod:`CWWTtools`. Defaults to :py:mod:`NFFTtools`.
        bandwidths (numpy.ndarray): Bandwidths for this term, dtype
            ``int32``, one entry per dimension in ``u``.
        bases (list[str], optional): Per-dimension basis, used by the
            ``"mixed"`` system.

    Attributes:
        u (tuple[int]): The coordinate subset.
        mode (module): The transform module.
        bandwidths (numpy.ndarray): The bandwidths.
        bases (list[str]): The per-dimension bases.
    """

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


class GroupedCoefficientsComplex(GC):
    r"""Complex coefficients belonging to a grouped index set.

    Created by the :py:func:`GroupedCoefficients` factory for settings whose
    mode is :py:mod:`NFFTtools`.

    Args:
        settings (list[Setting]): One entry per ANOVA term, fixing the layout
            of the coefficient vector.
        data (numpy.ndarray, optional): The coefficients. If omitted, a zero
            vector of the required length is allocated.

    Raises:
        ValueError: If ``data`` has the wrong length for ``settings``.
        RuntimeError: If a setting's mode does not provide ``datalength``.
    """

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


class GroupedCoefficientsReal(GC):
    r"""Real coefficients belonging to a grouped index set.

    Created by the :py:func:`GroupedCoefficients` factory for settings whose
    mode is :py:mod:`NFCTtools` or :py:mod:`CWWTtools`.

    Args:
        settings (list[Setting]): One entry per ANOVA term, fixing the layout
            of the coefficient vector.
        data (numpy.ndarray, optional): The coefficients. If omitted, a zero
            vector of the required length is allocated.

    Raises:
        ValueError: If ``data`` has the wrong length for ``settings``.
        RuntimeError: If a setting's mode does not provide ``datalength``.
    """

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
    r"""Create a grouped coefficient object matching the given settings.

    Picks :py:class:`GroupedCoefficientsComplex` or
    :py:class:`GroupedCoefficientsReal` from the transform mode of the
    settings. This is the intended way to construct coefficients.

    Args:
        settings (list[Setting]): One entry per ANOVA term.
        data (numpy.ndarray, optional): The coefficients. If omitted, a zero
            vector of the required length is allocated.

    Returns:
        GC: A complex object for :py:mod:`NFFTtools` settings, a real one for
        :py:mod:`NFCTtools` and :py:mod:`CWWTtools`.
    """
    if settings[0].mode == NFFTtools:
        return GroupedCoefficientsComplex(settings, data)
    if settings[0].mode == NFCTtools or settings[0].mode == CWWTtools:
        return GroupedCoefficientsReal(settings, data)
