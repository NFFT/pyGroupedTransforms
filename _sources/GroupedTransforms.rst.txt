GroupedTransform
=======================

.. py:class:: GroupedTransform

   A class to describe a GroupedTransformation.

   .. rubric:: Attributes:

   .. py:attribute:: system
   
      A string - choice of "exp" or "cos" or "chui1" or "chui2" or "chui3" or "chui4" or "mixed".

   .. py:attribute:: settings
      
      List of setting objects - uniquely describes the setting such as the bandlimits.

   .. py:attribute:: X

      Numpy  array of dtype float of dimension (M,d) - array of nodes.
   
   .. py:attribute:: transforms

      List of DeferredLinearOperator objects - holds the low-dimensional sub transformations.

   .. py:attribute:: basis_vect
   
      List of strings  - holds for every dimension if a cosinus basis [true] or exponential basis [false] is used.
   


   .. rubric:: Constructor:

   .. py:method:: GroupedTransform( system, X, settings = settings, basis_vect = basis_vect)

   .. rubric:: Additional Constructor:

   .. py:method:: GroupedTransform( system, X, d=d, ds = ds, N =N basis_vect = basis_vect)

   .. py:method:: GroupedTransform( system, X, U = U, N = N basis_vect = basis_vect)


   .. rubric:: Functions:

   .. py:method:: `*`
   
      If `F` is a GroupedTransfom object and `f` is a numpy array, 
      this overloads the * notation in order to achieve the adjoint transform `f = F*f`.
      
      If `F` is a GroupedTransform object and fhat is a GroupedCoefficient object,
      this overloads the * notation in order to achieve `f = F*fhat`.

   .. py:method:: adjoint

      Overloads the `F'` notation and gives back the same GroupdTransform. GroupedTransform decides by the input if it is the normal trafo or the adjoint so this is only for convinience.

   .. py:method:: __getitem__

      This function overloads `[]` of GroupedTransform such that you can do `F[(1,3)]` to obtain the transform of the corresponding ANOVA term defined by `u`.