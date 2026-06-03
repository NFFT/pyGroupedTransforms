GroupedCoefficients
=======================

.. py:class:: GC(settings, data)

   Superclass of GroupedCoefficientsComplex and GroupedCoefficientsReal.
   A class to hold coefficients belonging to indices in a grouped index set.

   .. rubric:: Attributes:

   .. py:attribute:: settings
      
      List of setting objects (see bellow) - uniquely describes the setting such as the bandlimits

   .. py:attribute:: data
      
      Numpy array of dtype float or dtype complex - the vector of coefficients

   .. rubric:: Constructor:

   .. py:method:: GroupedCoefficients( setting, data = nothing )


   .. rubric:: Functions:

   .. py:method:: __getitem__(idx)

      If `idx` is a tuple that contains integer,
      this function overloads getitem of GC such that you can do ``fhat[(1,3)]`` to obtain the basis coefficients of the corresponding ANOVA term defined by `u`.

      If `idx` is an integer,
      this function overloads getitem of GC such that you can do ``fhat[1]`` to obtain the basis coefficient determined by `idx`.

   .. py:method:: __setitem__(idx, Number)

      If `idx` is a tuple that contains integer,
      this function overloads setitem of GC such that you can do ``fhat[(1,3)] = [1 2 3]``.

      If `idx` is an integer,
      this function overloads setitem of GC such that you can do ``fhat[1] = 3``.

   .. py:method:: vec()

      This function returns the vector of the basis coefficients of `self`.

   .. py:method:: __rmul__(alpha)

      This function defines multiplication of a number with a GC object.

   .. py:method:: __add__(other)

      This function defines the addition of two GC objects.

   .. py:method:: __sub__(other)

      This function defines the subtraction of two GC objects.

   .. py:method:: set_data(data)

      With this function one can set the data of a GC object.


.. py:function:: variances(j, m)

   matrix of variances between two basis functions, needed for wavelet basis, since they are not orthonormal.


.. py:class:: GroupedCoefficientsComplex(GC)

   A class to hold complex coefficients belonging to indices in a grouped index set.

   .. rubric:: Constructor:

   .. py:method:: GroupedCoefficientsComplex( setting, data = nothing )


.. py:class:: GroupedCoefficientsReal(GC)

   A class to hold real coefficients belonging to indices in a grouped index set.

   .. rubric:: Constructor:

   .. py:method:: GroupedCoefficientsReal( setting, data = nothing )


.. py:class:: Setting

   .. rubric:: Attributes:

   .. py:attribute:: u
      
      tuple of ints

   .. py:attribute:: mode
      
      string

   .. py:attribute:: bandwidths
      
      numpy array of dtype "int32"


   .. py:attribute:: basis_vect
      
      list of strings
   