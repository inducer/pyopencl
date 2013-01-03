Multi-dimensional arrays on the Compute Device
==============================================

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
<https://github.com/inducer/pyopencl/blob/master/src/cl/pyopencl-complex.h>`_
for a precise list of what's available.

If you need double precision support, please::

    #define PYOPENCL_DEFINE_CDOUBLE

before including the header, as DP support apparently cannot be reliably
autodetected.

Under the hood, the complex types are simply `float2` and `double2`.

.. warning::
    Note that addition (real + complex) and multiplication (complex*complex)
    are defined for e.g. `float2`, but yield wrong results, so that you need to
    use the corresponding functions.

.. versionadded:: 2012.1

The :class:`Array` Class
------------------------

.. class:: Array(cqa, shape, dtype, order="C", *, allocator=None, base=None, data=None)

    A :class:`numpy.ndarray` work-alike that stores its data and performs its
    computations on the compute device.  *shape* and *dtype* work exactly as in
    :mod:`numpy`.  Arithmetic methods in :class:`Array` support the
    broadcasting of scalars. (e.g. `array+5`)

    *cqa* must be a :class:`pyopencl.CommandQueue`. *cqa*
    specifies the queue in which the array carries out its
    computations by default. *cqa* will at some point be renamed *queue*,
    so it should be considered 'positional-only'.

    *allocator* may be `None` or a callable that, upon being called with an
    argument of the number of bytes to be allocated, returns an
    :class:`pyopencl.Buffer` object. (A :class:`pyopencl.tools.MemoryPool`
    instance is one useful example of an object to pass here.)

    .. versionchanged:: 2011.1
        Renamed *context* to *cqa*, made it general-purpose.

        All arguments beyond *order* should be considered keyword-only.

    .. attribute :: data

        The :class:`pyopencl.MemoryObject` instance created for the memory that backs
        this :class:`Array`.

    .. attribute :: shape

        The tuple of lengths of each dimension in the array.

    .. attribute :: dtype

        The :class:`numpy.dtype` of the items in the GPU array.

    .. attribute :: size

        The number of meaningful entries in the array. Can also be computed by
        multiplying up the numbers in :attr:`shape`.

    .. attribute :: nbytes

        The size of the entire array in bytes. Computed as :attr:`size` times
        ``dtype.itemsize``.

    .. attribute :: strides

        Tuple of bytes to step in each dimension when traversing an array.

    .. attribute :: flags

        Return an object with attributes `c_contiguous`, `f_contiguous` and `forc`,
        which may be used to query contiguity properties in analogy to
        :attr:`numpy.ndarray.flags`.

    .. method :: __len__()

        Returns the size of the leading dimension of *self*.

    .. method :: reshape(shape)

        Returns an array containing the same data with a new shape.

    .. method :: ravel()

        Returns flattened array containing the same data.

    .. method :: view(dtype=None)

        Returns view of array with the same data. If *dtype* is different from
        current dtype, the actual bytes of memory will be reinterpreted.

    .. method :: set(ary, queue=None, async=False)

        Transfer the contents the :class:`numpy.ndarray` object *ary*
        onto the device.

        *ary* must have the same dtype and size (not necessarily shape) as *self*.


    .. method :: get(queue=None, ary=None, async=False)

        Transfer the contents of *self* into *ary* or a newly allocated
        :mod:`numpy.ndarray`. If *ary* is given, it must have the right
        size (not necessarily shape) and dtype.

    .. method :: copy(queue=None)

        .. versionadded:: 2013.1

    .. method :: __str__()
    .. method :: __repr__()

    .. method :: mul_add(self, selffac, other, otherfac, queue=None):

        Return `selffac*self + otherfac*other`.

    .. method :: __add__(other)
    .. method :: __sub__(other)
    .. method :: __iadd__(other)
    .. method :: __isub__(other)
    .. method :: __neg__(other)
    .. method :: __mul__(other)
    .. method :: __div__(other)
    .. method :: __rdiv__(other)
    .. method :: __pow__(other)

    .. method :: __abs__()

        Return a :class:`Array` containing the absolute value of each
        element of *self*.

    .. UNDOC reverse()

    .. method :: fill(scalar, queue=None)

        Fill the array with *scalar*.

    .. method :: astype(dtype, queue=None)

        Return *self*, cast to *dtype*.

    .. attribute :: real

        .. versionadded:: 2012.1

    .. attribute :: imag

        .. versionadded:: 2012.1

    .. method :: conj()

        .. versionadded:: 2012.1

Constructing :class:`Array` Instances
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: to_device(queue, ary, allocator=None, async=False)

    Return a :class:`Array` that is an exact copy of the :class:`numpy.ndarray`
    instance *ary*.

    See :class:`Array` for the meaning of *allocator*.

    .. versionchanged:: 2011.1
        *context* argument was deprecated.

.. function:: empty(queue, shape, dtype, order="C", allocator=None, data=None)

    A synonym for the :class:`Array` constructor.

.. function:: zeros(queue, shape, dtype, order="C", allocator=None)

    Same as :func:`empty`, but the :class:`Array` is zero-initialized before
    being returned.

    .. versionchanged:: 2011.1
        *context* argument was deprecated.

.. function:: empty_like(other_ary)

    Make a new, uninitialized :class:`Array` having the same properties
    as *other_ary*.

.. function:: zeros_like(other_ary)

    Make a new, zero-initialized :class:`Array` having the same properties
    as *other_ary*.

.. function:: arange(queue, start, stop, step, dtype=None, allocator=None)

    Create a :class:`Array` filled with numbers spaced `step` apart,
    starting from `start` and ending at `stop`.

    For floating point arguments, the length of the result is
    `ceil((stop - start)/step)`.  This rule may result in the last
    element of the result being greater than `stop`.

    *dtype*, if not specified, is taken as the largest common type
    of *start*, *stop* and *step*.

    .. versionchanged:: 2011.1
        *context* argument was deprecated.

    .. versionchanged:: 2011.2
        *allocator* keyword argument was added.

.. function:: take(a, indices, out=None, queue=None)

    Return the :class:`Array` ``[a[indices[0]], ..., a[indices[n]]]``.
    For the moment, *a* must be a type that can be bound to a texture.

Conditionals
^^^^^^^^^^^^

.. function:: if_positive(criterion, then_, else_, out=None, queue=None)

    Return an array like *then_*, which, for the element at index *i*,
    contains *then_[i]* if *criterion[i]>0*, else *else_[i]*.

.. function:: maximum(a, b, out=None, queue=None)

    Return the elementwise maximum of *a* and *b*.

.. function:: minimum(a, b, out=None, queue=None)

    Return the elementwise minimum of *a* and *b*.

.. _reductions:


Reductions
^^^^^^^^^^

.. function:: sum(a, dtype=None, queue=None)

    .. versionadded: 2011.1

.. function:: dot(a, b, dtype=None, queue=None)

    .. versionadded: 2011.1

.. function:: subset_dot(subset, a, b, dtype=None, queue=None)

    .. versionadded: 2011.1

.. function:: max(a, queue=None)

    .. versionadded: 2011.1

.. function:: min(a, queue=None)

    .. versionadded: 2011.1

.. function:: subset_max(subset, a, queue=None)

    .. versionadded: 2011.1

.. function:: subset_min(subset, a, queue=None)

    .. versionadded: 2011.1

See also :ref:`custom-reductions`.

Elementwise Functions on :class:`Arrray` Instances
--------------------------------------------------

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
.. TODO: atan2
.. function:: atanh(array, queue=None)
.. function:: atanpi(array, queue=None)
.. TODO: atan2pi

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

.. module:: pyopencl.clrandom

.. class:: RanluxGenerator(self, queue, num_work_items=None,  luxury=2, seed=None, max_work_items=None)

    :param queue: :class:`pyopencl.CommandQueue`, only used for initialization
    :param luxury: the "luxury value" of the generator, and should be 0-4, where 0 is fastest
        and 4 produces the best numbers. It can also be >=24, in which case it directly
        sets the p-value of RANLUXCL.
    :param num_work_items: is the number of generators to initialize, usually corresponding
        to the number of work-items in the NDRange RANLUXCL will be used with.
        May be `None`, in which case a default value is used.
    :param max_work_items: should reflect the maximum number of work-items that will be used
        on any parallel instance of RANLUXCL. So for instance if we are launching 5120
        work-items on GPU1 and 10240 work-items on GPU2, GPU1's RANLUXCLTab would be
        generated by calling ranluxcl_intialization with numWorkitems = 5120 while
        GPU2's RANLUXCLTab would use numWorkitems = 10240. However maxWorkitems must
        be at least 10240 for both GPU1 and GPU2, and it must be set to the same value
        for both. (may be `None`)

    .. versionadded:: 2011.2

    .. versionchanged:: 2013.1
        Added default value for `num_work_items`.

    .. attribute:: state

        A :class:`pyopencl.array.Array` containing the state of the generator.

    .. attribute:: nskip

        nskip is an integer which can (optionally) be defined in the kernel code
        as RANLUXCL_NSKIP. If this is done the generator will be faster for luxury setting
        0 and 1, or when the p-value is manually set to a multiple of 24.

    .. method:: fill_uniform(ary, a=0, b=1, queue=None)

        Fill *ary* with uniformly distributed random numbers in the interval
        *(a, b)*, endpoints excluded.

    .. method:: uniform(queue, shape, dtype, order="C", allocator=None, base=None, data=None, a=0, b=1)

        Make a new empty array, apply :meth:`fill_uniform` to it.

    .. method:: fill_normal(ary, mu=0, sigma=1, queue=None):

        Fill *ary* with normally distributed numbers with mean *mu* and
        standard deviation *sigma*.

    .. method:: normal(queue, shape, dtype, order="C", allocator=None, base=None, data=None, mu=0, sigma=1)

        Make a new empty array, apply :meth:`fill_normal` to it.

    .. method:: synchronize()

        The generator gets inefficient when different work items invoke
        the generator a differing number of times. This function
        ensures efficiency.

.. function:: rand(queue, shape, dtype, a=0, b=1)

    Return an array of `shape` filled with random values of `dtype`
    in the range [a,b).

.. function:: fill_rand(result, queue=None, a=0, b=1)

    Fill *result* with random values of `dtype` in the range [0,1).

PyOpenCL now includes and uses the `RANLUXCL random number generator
<https://bitbucket.org/ivarun/ranluxcl/>`_ by Ivar Ursin Nikolaisen.  In
addition to being usable through the convenience functions above, it is
available in any piece of code compiled through PyOpenCL by::

    #include <pyopencl-ranluxcl.cl>

See the `source <https://github.com/inducer/pyopencl/blob/master/src/cl/pyopencl-ranluxcl.cl>`_
for some documentation if you're planning on using RANLUXCL directly.

The RANLUX generator is described in the following two articles. If you use the
generator for scientific purposes, please consider citing them:

* Martin Lüscher, A portable high-quality random number generator for lattice
  field theory simulations, `Computer Physics Communications 79 (1994) 100-110
  <http://dx.doi.org/10.1016/0010-4655(94)90232-1>`_

* F. James, RANLUX: A Fortran implementation of the high-quality pseudorandom
  number generator of Lüscher, `Computer Physics Communications 79 (1994) 111-114
  <http://dx.doi.org/10.1016/0010-4655(94)90233-X>`_

Fast Fourier Transforms
-----------------------

Bogdan Opanchuk's `pyfft <http://pypi.python.org/pypi/pyfft>`_ package offers a
variety of GPU-based FFT implementations.
