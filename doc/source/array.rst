Multi-dimensional arrays on the Compute Device
==============================================

.. module:: pyopencl.array

Data Types
----------

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

.. function:: pyopencl.tools.register_dtype(dtype, name)

    *dtype* is a :func:`numpy.dtype`.

    .. versionadded: 2011.2

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

.. class:: DefaultAllocator(context, flags=pyopencl.mem_flags.READ_WRITE)

    An alias for :class:`pyopencl.tools.CLAllocator`.

.. class:: Array(cqa, shape, dtype, order="C", *, allocator=None, base=None, data=None, queue=None)

    A :class:`numpy.ndarray` work-alike that stores its data and performs its
    computations on the compute device.  *shape* and *dtype* work exactly as in
    :mod:`numpy`.  Arithmetic methods in :class:`Array` support the
    broadcasting of scalars. (e.g. `array+5`)

    *cqa* can be a :class:`pyopencl.Context`, :class:`pyopencl.CommandQueue`
    or an allocator, as described below. If it is either of the latter two, the *queue*
    or *allocator* arguments may not be passed.

    *queue* (or *cqa*, as the case may be) specifies the queue in which the array
    carries out its computations by default.

    *allocator* is a callable that, upon being called with an argument of the number
    of bytes to be allocated, returns an object that can be cast to an
    :class:`int` representing the address of the newly allocated memory.
    (See :class:`DefaultAllocator`.)


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

    .. attribute :: mem_size

        The total number of entries, including padding, that are present in
        the array.

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

.. function:: empty(queue, shape, dtype, order="C", allocator=None, base=None, data=None)

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
    contains *then_[i]* if *criterion[i]>0*, else *else_[i]*. (added in 0.94)

.. function:: maximum(a, b, out=None, queue=None)

    Return the elementwise maximum of *a* and *b*. (added in 0.94)

.. function:: minimum(a, b, out=None, queue=None)

    Return the elementwise minimum of *a* and *b*. (added in 0.94)

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

.. class:: RanluxGenerator(self, queue, num_work_items, max_work_items, luxury=2, seed=None)

    :param queue: :class:`pyopencl.CommandQueue`, only used for initialization
    :param luxury: the "luxury value" of the generator, and should be 0-4, where 0 is fastest
        and 4 produces the best numbers. It can also be >=24, in which case it directly
        sets the p-value of RANLUXCL.
    :param num_work_items: is the number of generators to initialize, usually corresponding
        to the number of work-items in the NDRange RANLUXCL will be used with.
    :param max_work_items: should reflect the maximum number of work-items that will be used
        on any parallel instance of RANLUXCL. So for instance if we are launching 5120
        work-items on GPU1 and 10240 work-items on GPU2, GPU1's RANLUXCLTab would be
        generated by calling ranluxcl_intialization with numWorkitems = 5120 while
        GPU2's RANLUXCLTab would use numWorkitems = 10240. However maxWorkitems must
        be at least 10240 for both GPU1 and GPU2, and it must be set to the same value
        for both.

    .. versionadded:: 2011.2

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


Single-pass Custom Expression Evaluation
----------------------------------------

.. module:: pyopencl.elementwise

Evaluating involved expressions on :class:`pyopencl.array.Array` instances can be
somewhat inefficient, because a new temporary is created for each
intermediate result. The functionality in the module :mod:`pyopencl.elementwise`
contains tools to help generate kernels that evaluate multi-stage expressions
on one or several operands in a single pass.

.. class:: ElementwiseKernel(context, arguments, operation, name="kernel", preamble="", options=[])

    Generate a kernel that takes a number of scalar or vector *arguments*
    and performs the scalar *operation* on each entry of its arguments, if that
    argument is a vector.

    *arguments* is specified as a string formatted as a C argument list.
    *operation* is specified as a C assignment statement, without a semicolon.
    Vectors in *operation* should be indexed by the variable *i*.

    *name* specifies the name as which the kernel is compiled, 
    and *options* are passed unmodified to :meth:`pyopencl.Program.build`.

    *preamble* is a piece of C source code that gets inserted outside of the
    function context in the elementwise operation's kernel source code.

    .. method:: __call__(*args, wait_for=None)

        Invoke the generated scalar kernel. The arguments may either be scalars or
        :class:`GPUArray` instances.

Here's a usage example::

    import pyopencl as cl
    import pyopencl.array as cl_array
    import numpy

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    n = 10
    a_gpu = cl_array.to_device(
            ctx, queue, numpy.random.randn(n).astype(numpy.float32))
    b_gpu = cl_array.to_device(
            ctx, queue, numpy.random.randn(n).astype(numpy.float32))

    from pyopencl.elementwise import ElementwiseKernel
    lin_comb = ElementwiseKernel(ctx,
            "float a, float *x, "
            "float b, float *y, "
            "float *z",
            "z[i] = a*x[i] + b*y[i]",
            "linear_combination")

    c_gpu = cl_array.empty_like(a_gpu)
    lin_comb(5, a_gpu, 6, b_gpu, c_gpu)

    import numpy.linalg as la
    assert la.norm((c_gpu - (5*a_gpu+6*b_gpu)).get()) < 1e-5

(You can find this example as :file:`examples/demo_elementwise.py` in the PyOpenCL
distribution.)

.. _custom-reductions:

Custom Reductions
-----------------

.. module:: pyopencl.reduction

.. class:: ReductionKernel(ctx, dtype_out, neutral, reduce_expr, map_expr=None, arguments=None, name="reduce_kernel", options=[], preamble="")

    Generate a kernel that takes a number of scalar or vector *arguments*
    (at least one vector argument), performs the *map_expr* on each entry of
    the vector argument and then the *reduce_expr* on the outcome of that.
    *neutral* serves as an initial value. *preamble* offers the possibility
    to add preprocessor directives and other code (such as helper functions)
    to be added before the actual reduction kernel code.

    Vectors in *map_expr* should be indexed by the variable *i*. *reduce_expr*
    uses the formal values "a" and "b" to indicate two operands of a binary
    reduction operation. If you do not specify a *map_expr*, "in[i]" -- and
    therefore the presence of only one input argument -- is automatically
    assumed.

    *dtype_out* specifies the :class:`numpy.dtype` in which the reduction is
    performed and in which the result is returned. *neutral* is specified as
    float or integer formatted as string. *reduce_expr* and *map_expr* are
    specified as string formatted operations and *arguments* is specified as a
    string formatted as a C argument list. *name* specifies the name as which
    the kernel is compiled. *options* are passed unmodified to
    :meth:`pyopencl.Program.build`. *preamble* specifies a string of code that
    is inserted before the actual kernels.

    .. method:: __call__(*args, queue=None)

    .. versionadded: 2011.1

Here's a usage example::

    a = pyopencl.array.arange(queue, 400, dtype=numpy.float32)
    b = pyopencl.array.arange(queue, 400, dtype=numpy.float32)

    krnl = ReductionKernel(ctx, numpy.float32, neutral="0",
            reduce_expr="a+b", map_expr="x[i]*y[i]",
            arguments="__global float *x, __global float *y")

    my_dot_prod = krnl(a, b).get()

Parallel Scan / Prefix Sum
--------------------------

.. module:: pyopencl.scan

.. class:: ExclusiveScanKernel(ctx, dtype, scan_expr, neutral, name_prefix="scan", options=[], preamble="", devices=None)

    Generates a kernel that can compute a `prefix sum <https://secure.wikimedia.org/wikipedia/en/wiki/Prefix_sum>`_
    using any associative operation given as *scan_expr*.
    *scan_expr* uses the formal values "a" and "b" to indicate two operands of
    an associative binary operation. *neutral* is the neutral element
    of *scan_expr*, obeying *scan_expr(a, neutral) == a*.

    *dtype* specifies the type of the arrays being operated on. 
    *name_prefix* is used for kernel names to ensure recognizability
    in profiles and logs. *options* is a list of compiler options to use
    when building. *preamble* specifies a string of code that is
    inserted before the actual kernels. *devices* may be used to restrict
    the set of devices on which the kernel is meant to run. (defaults
    to all devices in the context *ctx*.

    .. method:: __call__(self, input_ary, output_ary=None, allocator=None, queue=None)

.. class:: InclusiveScanKernel(dtype, scan_expr, neutral=None, name_prefix="scan", options=[], preamble="", devices=None)

    Works like :class:`ExclusiveScanKernel`. Unlike the exclusive case,
    *neutral* is not required.

Here's a usage example::

    knl = InclusiveScanKernel(context, np.int32, "a+b")

    n = 2**20-2**18+5
    host_data = np.random.randint(0, 10, n).astype(np.int32)
    dev_data = cl_array.to_device(queue, host_data)

    knl(dev_data)
    assert (dev_data.get() == np.cumsum(host_data, axis=0)).all()



Fast Fourier Transforms
-----------------------

Bogdan Opanchuk's `pyfft <http://pypi.python.org/pypi/pyfft>`_ package offers a
variety of GPU-based FFT implementations.

