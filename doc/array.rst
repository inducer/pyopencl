Multi-dimensional arrays
========================

.. module:: pyopencl.array

The functionality in this module provides something of a work-alike for
:mod:`numpy` arrays, but with all operations executed on the CL compute device.

Data Types
----------

PyOpenCL provides some amount of integration between the :mod:`numpy`
type system, as represented by :class:`numpy.dtype`, and the types
available in OpenCL. All the simple scalar types map straightforwardly
to their CL counterparts.

.. _vector-types:

Vector Types
^^^^^^^^^^^^

.. class :: vec

    All of OpenCL's supported vector types, such as `float3` and `long4` are
    available as :mod:`numpy` data types within this class. These
    :class:`numpy.dtype` instances have field names of `x`, `y`, `z`, and `w`
    just like their OpenCL counterparts. They will work both for parameter passing
    to kernels as well as for passing data back and forth between kernels and
    Python code. For each type, a `make_type` function is also provided (e.g.
    `make_float3(x,y,z)`).

    If you want to construct a pre-initialized vector type you have three new
    functions to choose from:

    * `zeros_type()`
    * `ones_type()`
    * `filled_type(fill_value)`

    .. versionadded:: 2014.1

    .. versionchanged:: 2014.1
        The `make_type` functions have a default value (0) for each component.
        Relying on the default values has been deprecated. Either specify all
        components or use one of th new flavors mentioned above for constructing
        a vector.

Custom data types
^^^^^^^^^^^^^^^^^

If you would like to use your own (struct/union/whatever) data types in array
operations where you supply operation source code, define those types in the
*preamble* passed to :class:`pyopencl.elementwise.ElementwiseKernel`,
:class:`pyopencl.reduction.ReductionKernel` (or similar), and let PyOpenCL know
about them using this function:

.. currentmodule:: pyopencl.tools

.. autofunction:: get_or_register_dtype

.. exception:: TypeNameNotKnown

    .. versionadded:: 2013.1

.. function:: register_dtype(dtype, name)

    .. versionchanged:: 2013.1
        This function has been deprecated. It is recommended that you develop
        against the new interface, :func:`get_or_register_dtype`.

.. function:: dtype_to_ctype(dtype)

    Returns a C name registered for *dtype*.

    .. versionadded: 2013.1

This function helps with producing C/OpenCL declarations for structured
:class:`numpy.dtype` instances:

.. autofunction:: match_dtype_to_c_struct

A more complete example of how to use custom structured types can be
found in :file:`examples/demo-struct-reduce.py` in the PyOpenCL
distribution.

.. currentmodule:: pyopencl.array

Complex Numbers
^^^^^^^^^^^^^^^

PyOpenCL's :class:`Array` type supports complex numbers out of the box, by
simply using the corresponding :mod:`numpy` types.

If you would like to use this support in your own kernels, here's how to
proceed: Since OpenCL 1.2 (and earlier) do not specify native complex number
support, PyOpenCL works around that deficiency. By saying::

    #include <pyopencl-complex.h>

in your kernel, you get complex types `cfloat_t` and `cdouble_t`, along with
functions defined on them such as `cfloat_mul(a, b)` or `cdouble_log(z)`.
Elementwise kernels automatically include the header if your kernel has
complex input or output.
See the `source file
<https://github.com/inducer/pyopencl/blob/master/pyopencl/cl/pyopencl-complex.h>`_
for a precise list of what's available.

If you need double precision support, please::

    #define PYOPENCL_DEFINE_CDOUBLE

before including the header, as DP support apparently cannot be reliably
autodetected.

Under the hood, the complex types are struct types as defined in the header.
Ideally, you should only access the structs through the provided functions,
never directly.

.. versionadded:: 2012.1

.. versionchanged:: 2015.2

    **[INCOMPATIBLE]** Changed PyOpenCL's complex numbers from ``float2`` and
    ``double2`` OpenCL vector types to custom ``struct``. This was changed
    because it very easily introduced bugs where

    * complex*complex
    * real+complex

    *look* like they may do the right thing, but silently do the wrong thing.

The :class:`Array` Class
------------------------

.. autoclass:: Array

.. autoexception:: ArrayHasOffsetError

Constructing :class:`Array` Instances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: to_device
.. function:: empty(queue, shape, dtype, order="C", allocator=None, data=None)

    A synonym for the :class:`Array` constructor.

.. autofunction:: zeros
.. autofunction:: empty_like
.. autofunction:: zeros_like
.. autofunction:: arange
.. autofunction:: take
.. autofunction:: concatenate
.. autofunction:: stack

Manipulating :class:`Array` instances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: transpose
.. autofunction:: reshape

Conditionals
^^^^^^^^^^^^

.. autofunction:: if_positive
.. autofunction:: maximum
.. autofunction:: minimum

.. _reductions:

Reductions
^^^^^^^^^^

.. autofunction:: sum
.. autofunction:: dot
.. autofunction:: vdot
.. autofunction:: subset_dot
.. autofunction:: max
.. autofunction:: min
.. autofunction:: subset_max
.. autofunction:: subset_min

See also :ref:`custom-reductions`.

Elementwise Functions on :class:`Array` Instances
-------------------------------------------------

.. module:: pyopencl.clmath

The :mod:`pyopencl.clmath` module contains exposes array versions of the C
functions available in the OpenCL standard. (See table 6.8 in the spec.)

.. function:: acos(array, queue=None)
.. function:: acosh(array, queue=None)
.. function:: acospi(array, queue=None)

.. function:: asin(array, queue=None)
.. function:: asinh(array, queue=None)
.. function:: asinpi(array, queue=None)

.. function:: atan(array, queue=None)
.. autofunction:: atan2
.. function:: atanh(array, queue=None)
.. function:: atanpi(array, queue=None)
.. autofunction:: atan2pi

.. function:: cbrt(array, queue=None)
.. function:: ceil(array, queue=None)
.. TODO: copysign

.. function:: cos(array, queue=None)
.. function:: cosh(array, queue=None)
.. function:: cospi(array, queue=None)

.. function:: erfc(array, queue=None)
.. function:: erf(array, queue=None)
.. function:: exp(array, queue=None)
.. function:: exp2(array, queue=None)
.. function:: exp10(array, queue=None)
.. function:: expm1(array, queue=None)

.. function:: fabs(array, queue=None)
.. TODO: fdim
.. function:: floor(array, queue=None)
.. TODO: fma
.. TODO: fmax
.. TODO: fmin

.. function:: fmod(arg, mod, queue=None)

    Return the floating point remainder of the division `arg/mod`,
    for each element in `arg` and `mod`.

.. TODO: fract


.. function:: frexp(arg, queue=None)

    Return a tuple `(significands, exponents)` such that
    `arg == significand * 2**exponent`.

.. TODO: hypot

.. function:: ilogb(array, queue=None)
.. function:: ldexp(significand, exponent, queue=None)

    Return a new array of floating point values composed from the
    entries of `significand` and `exponent`, paired together as
    `result = significand * 2**exponent`.


.. function:: lgamma(array, queue=None)
.. TODO: lgamma_r

.. function:: log(array, queue=None)
.. function:: log2(array, queue=None)
.. function:: log10(array, queue=None)
.. function:: log1p(array, queue=None)
.. function:: logb(array, queue=None)

.. TODO: mad
.. TODO: maxmag
.. TODO: minmag


.. function:: modf(arg, queue=None)

    Return a tuple `(fracpart, intpart)` of arrays containing the
    integer and fractional parts of `arg`.

.. function:: nan(array, queue=None)

.. TODO: nextafter
.. TODO: remainder
.. TODO: remquo

.. function:: rint(array, queue=None)
.. TODO: rootn
.. function:: round(array, queue=None)

.. function:: sin(array, queue=None)
.. TODO: sincos
.. function:: sinh(array, queue=None)
.. function:: sinpi(array, queue=None)

.. function:: sqrt(array, queue=None)

.. function:: tan(array, queue=None)
.. function:: tanh(array, queue=None)
.. function:: tanpi(array, queue=None)
.. function:: tgamma(array, queue=None)
.. function:: trunc(array, queue=None)

Generating Arrays of Random Numbers
-----------------------------------

.. automodule:: pyopencl.clrandom
