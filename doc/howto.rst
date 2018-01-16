How-tos
=======

How to use struct types with PyOpenCL
-------------------------------------

We import and initialize PyOpenCL as usual:

.. doctest::
    :options: +ELLIPSIS

    >>> import numpy as np
    >>> import pyopencl as cl
    >>> import pyopencl.tools
    >>> import pyopencl.array

    >>> ctx = cl.create_some_context(interactive=False)
    >>> queue = cl.CommandQueue(ctx)

Then, suppose we would like to declare a struct consisting of an integer and a
floating point number. We first create a :class:`numpy.dtype` along these
lines:

.. doctest::

    >>> my_struct = np.dtype([("field1", np.int32), ("field2", np.float32)])
    >>> print(my_struct)
    [('field1', '<i4'), ('field2', '<f4')]

.. note::

    Not all :mod:`numpy` dtypes are supported yet. For example strings (and
    generally things that have a shape of their own) are not supported.

Since OpenCL C may have a different opinion for :mod:`numpy` on how the struct
should be laid out, for example because of `alignment
<https://en.wikipedia.org/wiki/Data_structure_alignment>`_. So as a first step, we
match our dtype against CL's version:

.. doctest::

    >>> my_struct, my_struct_c_decl = cl.tools.match_dtype_to_c_struct(
    ...    ctx.devices[0], "my_struct", my_struct)
    >>> print(my_struct_c_decl)
    typedef struct {
      int field1;
      float field2;
    } my_struct;
    <BLANKLINE>
    <BLANKLINE>

We then tell PyOpenCL about our new type.

.. doctest::

    >>> my_struct = cl.tools.get_or_register_dtype("my_struct", my_struct)

Next, we can create some data of that type on the host and transfer it to
the device:

.. doctest::

    >>> ary_host = np.empty(20, my_struct)
    >>> ary_host["field1"].fill(217)
    >>> ary_host["field2"].fill(1000)
    >>> ary_host[13]["field2"] = 12
    >>> print(ary_host) #doctest: +NORMALIZE_WHITESPACE
    [(217,  1000.) (217,  1000.) (217,  1000.) (217,  1000.) (217,  1000.)
     (217,  1000.) (217,  1000.) (217,  1000.) (217,  1000.) (217,  1000.)
     (217,  1000.) (217,  1000.) (217,  1000.) (217,    12.) (217,  1000.)
     (217,  1000.) (217,  1000.) (217,  1000.) (217,  1000.) (217,  1000.)]

    >>> ary = cl.array.to_device(queue, ary_host)

We can then operate on the array with our own kernels:

.. doctest::

    >>> prg = cl.Program(ctx, my_struct_c_decl + """
    ...     __kernel void set_to_1(__global my_struct *a)
    ...     {
    ...         a[get_global_id(0)].field1 = 1;
    ...     }
    ...     """).build()

    >>> evt = prg.set_to_1(queue, ary.shape, None, ary.data)
    >>> print(ary) #doctest: +NORMALIZE_WHITESPACE
    [(1,  1000.) (1,  1000.) (1,  1000.) (1,  1000.) (1,  1000.) (1,  1000.)
     (1,  1000.) (1,  1000.) (1,  1000.) (1,  1000.) (1,  1000.) (1,  1000.)
     (1,  1000.) (1,    12.) (1,  1000.) (1,  1000.) (1,  1000.) (1,  1000.)
     (1,  1000.) (1,  1000.)]

as well as with PyOpenCL's built-in operations:

.. doctest::

    >>> from pyopencl.elementwise import ElementwiseKernel
    >>> elwise = ElementwiseKernel(ctx, "my_struct *a", "a[i].field1 = 2;",
    ...    preamble=my_struct_c_decl)
    >>> evt = elwise(ary)
    >>> print(ary) #doctest: +NORMALIZE_WHITESPACE
    [(2,  1000.) (2,  1000.) (2,  1000.) (2,  1000.) (2,  1000.) (2,  1000.)
     (2,  1000.) (2,  1000.) (2,  1000.) (2,  1000.) (2,  1000.) (2,  1000.)
     (2,  1000.) (2,    12.) (2,  1000.) (2,  1000.) (2,  1000.) (2,  1000.)
     (2,  1000.) (2,  1000.)]
