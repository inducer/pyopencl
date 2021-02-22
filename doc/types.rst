OpenCL Type Mapping
===================

.. module:: pyopencl.cltypes

.. _type-mappings:

Scalar Types
------------

For ease of use, a the cltypes module provides convenient mapping from OpenCL type names
to their equivalent numpy types. This saves you from referring back to the OpenCL spec to
see that a cl_long is 64 bit unsigned integer. Use the module as follows:

.. doctest::

    >>> import numpy as np
    >>> import pyopencl as cl
    >>> import pyopencl.cltypes
    >>> cl_uint = cl.cltypes.uint(42)   # maps to numpy.uint32
    >>> cl_long = cl.cltypes.long(1235) # maps to numpy.int64
    >>> floats = np.empty((128,), dtype=cl.cltypes.float) # array of numpy.float32

.. note::

    The OpenCL type ``bool`` does not have a correpsonding :mod:`numpy` type defined here,
    because OpenCL does not specify the in-memory representation (or even the storage
    size) for this type.

Vector Types
------------
The corresponding vector types are also made available in the same package, allowing you to easily create
numpy arrays with the appropriate memory layout.

.. doctest::

    >>> import numpy as np
    >>> array_of_float16 = np.empty((128,), dtype=cl.cltypes.float16) # array of float16

