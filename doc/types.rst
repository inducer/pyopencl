OpenCL Type Mapping
===================

.. module:: pyopencl.cltypes

.. _type-mappings:

Scalar Types
------------

For ease of use, a the :mod:`pyopencl.cltypes` module provides convenient mapping
from OpenCL type names to their equivalent :mod:`numpy` types. This saves you
from referring back to the OpenCL spec to see that a ``cl_long`` is 64 bit
unsigned integer. Use the module as follows:

.. doctest::

    >>> import numpy as np
    >>> import pyopencl as cl
    >>> import pyopencl.cltypes
    >>> cl_uint = cl.cltypes.uint(42)   # maps to numpy.uint32
    >>> cl_long = cl.cltypes.long(1235) # maps to numpy.int64
    >>> floats = np.empty((128,), dtype=cl.cltypes.float) # array of numpy.float32

.. note::

    The OpenCL type ``bool`` does not have a corresponding :mod:`numpy` type
    defined here, because OpenCL does not specify the in-memory representation
    (or even the storage size) for this type.

.. _cltypes-vector-example:

Vector Types
------------

Use a vector dtype such as ``cltypes.float4`` for an array of OpenCL vectors.
The matching factory, ``cltypes.make_float4(x, y, z, w)``, constructs one vector.
You can read a component across the whole array by its field name:

.. doctest::

    >>> import numpy as np
    >>> import pyopencl as cl
    >>> from pyopencl import cltypes
    >>> points = np.zeros(2, dtype=cltypes.float4)
    >>> points[0] = cltypes.make_float4(1, 2, 3, 4)
    >>> points["y"].tolist()
    [2.0, 0.0]

The same factory works for a kernel argument passed by value. Here the buffer
contains two ``float4`` elements, and both receive the same offset:

.. doctest::

    >>> ctx = cl.create_some_context(interactive=False)
    >>> queue = cl.CommandQueue(ctx)
    >>> flags = cl.mem_flags
    >>> buf = cl.Buffer(ctx, flags.READ_WRITE | flags.COPY_HOST_PTR, hostbuf=points)
    >>> program = cl.Program(ctx, """
    ...     __kernel void translate(__global float4 *points, float4 offset) {
    ...         points[get_global_id(0)] += offset;
    ...     }
    ... """).build()
    >>> offset = cltypes.make_float4(0, 10, 0, 0)
    >>> _ = program.translate(queue, points.shape, None, buf, offset)
    >>> cl.enqueue_copy(queue, points, buf).wait()
    >>> points["y"].tolist()
    [12.0, 10.0]

For a buffer of ``float3`` elements, use ``dtype=cltypes.float3``:
OpenCL gives each element the storage of four floats. A packed NumPy
array of shape ``(n, 3)`` has a different stride.

