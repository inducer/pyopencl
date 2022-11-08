Parallel Algorithms
===================

.. include:: subst.rst

Element-wise expression evaluation ("map")
------------------------------------------

.. module:: pyopencl.elementwise

Evaluating involved expressions on :class:`pyopencl.array.Array` instances by
using overloaded operators can be somewhat inefficient, because a new temporary
is created for each intermediate result. The functionality in the module
:mod:`pyopencl.elementwise` contains tools to help generate kernels that
evaluate multi-stage expressions on one or several operands in a single pass.

.. autoclass:: ElementwiseKernel

Here's a usage example:

.. literalinclude:: ../examples/demo_elementwise.py

(You can find this example as
:download:`examples/demo_elementwise.py <../examples/demo_elementwise.py>`
in the PyOpenCL distribution.)

.. _custom-reductions:

Sums and counts ("reduce")
--------------------------

.. module:: pyopencl.reduction

.. autoclass:: ReductionKernel

Here's a usage example::

    a = pyopencl.array.arange(queue, 400, dtype=numpy.float32)
    b = pyopencl.array.arange(queue, 400, dtype=numpy.float32)

    krnl = ReductionKernel(ctx, numpy.float32, neutral="0",
            reduce_expr="a+b", map_expr="x[i]*y[i]",
            arguments="__global float *x, __global float *y")

    my_dot_prod = krnl(a, b).get()

.. _custom-scan:

Prefix Sums ("scan")
--------------------

.. module:: pyopencl.scan

.. |scan_extra_args| replace:: a list of tuples *(name, value)* specifying
    extra arguments to pass to the scan procedure. For version 2013.1,
    *value* must be a of a :mod:`numpy` sized scalar type. As of version 2013.2,
    *value* may also be a :class:`pyopencl.array.Array`.
.. |preamble| replace:: A snippet of C that is inserted into the compiled kernel
    before the actual kernel function. May be used for, e.g. type definitions
    or include statements.

A prefix sum is a running sum of an array, as provided by
e.g. :func:`numpy.cumsum`::

    >>> import numpy as np
    >>> a = [1,1,1,1,1,2,2,2,2,2]
    >>> np.cumsum(a)
    array([ 1,  2,  3,  4,  5,  7,  9, 11, 13, 15])

This is a very simple example of what a scan can do. It turns out that scans
are significantly more versatile. They are a basic building block of many
non-trivial parallel algorithms. Many of the operations enabled by scans seem
difficult to parallelize because of loop-carried dependencies.

.. seealso::

    `Prefix sums and their applications <https://doi.org/10.1184/R1/6608579.v1>`__, by Guy Blelloch.
        This article gives an overview of some surprising applications of scans.

    :ref:`predefined-scans`
        These operations built into PyOpenCL are realized using
        :class:`GenericScanKernel`.

Usage Example
^^^^^^^^^^^^^

This example illustrates the implementation of a simplified version of
:func:`pyopencl.algorithm.copy_if`,
which copies integers from an array into the (variable-size) output if they are
greater than 300::

    knl = GenericScanKernel(
            ctx, np.int32,
            arguments="__global int *ary, __global int *out",
            input_expr="(ary[i] > 300) ? 1 : 0",
            scan_expr="a+b", neutral="0",
            output_statement="""
                if (prev_item != item) out[item-1] = ary[i];
                """)

    out = a.copy()
    knl(a, out)

    a_host = a.get()
    out_host = a_host[a_host > 300]

    assert (out_host == out.get()[:len(out_host)]).all()

The value being scanned over is a number of flags indicating whether each array
element is greater than 300. These flags are computed by *input_expr*. The
prefix sum over this array gives a running count of array items greater than
300. The *output_statement* the compares ``prev_item`` (the previous item's scan
result, i.e. index) to ``item`` (the current item's scan result, i.e.
index). If they differ, i.e. if the predicate was satisfied at this
position, then the item is stored in the output at the computed index.

This example does not make use of the following advanced features also available
in PyOpenCL:

* Segmented scans

* Access to the previous item in *input_expr* (e.g. for comparisons)
  See the `implementation <https://github.com/inducer/pyopencl/blob/36afe57784368e8d2505bc7cad8df964ba3c0264/pyopencl/algorithm.py#L226>`__
  of :func:`pyopencl.algorithm.unique` for an example.

Making Custom Scan Kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 2013.1

.. autoclass:: GenericScanKernel

Debugging aids
~~~~~~~~~~~~~~

.. autoclass:: GenericDebugScanKernel

.. _predefined-scans:

Simple / Legacy Interface
^^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: ExclusiveScanKernel(ctx, dtype, scan_expr, neutral, name_prefix="scan", options=[], preamble="", devices=None)

    Generates a kernel that can compute a `prefix sum
    <https://en.wikipedia.org/wiki/Prefix_sum>`__
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

.. class:: InclusiveScanKernel(ctx, dtype, scan_expr, neutral=None, name_prefix="scan", options=[], preamble="", devices=None)

    Works like :class:`ExclusiveScanKernel`.

    .. versionchanged:: 2013.1
        *neutral* is now always required.

For the array ``[1, 2, 3]``, inclusive scan results in ``[1, 3, 6]``, and exclusive
scan results in ``[0, 1, 3]``.

Here's a usage example::

    knl = InclusiveScanKernel(context, np.int32, "a+b")

    n = 2**20-2**18+5
    rng = np.random.default_rng(seed=42)
    host_data = rng.integers(0, 10, size=n, dtype=np.int32)
    dev_data = cl_array.to_device(queue, host_data)

    knl(dev_data)
    assert (dev_data.get() == np.cumsum(host_data, axis=0)).all()

Predicated copies ("partition", "unique", ...)
----------------------------------------------

.. module:: pyopencl.algorithm

.. autofunction:: copy_if

.. autofunction:: remove_if

.. autofunction:: partition

.. autofunction:: unique

Sorting (radix sort)
--------------------

.. autoclass:: RadixSort

    .. automethod:: __call__

Building many variable-size lists
---------------------------------

.. autoclass:: ListOfListsBuilder

Bitonic Sort
------------

.. module:: pyopencl.bitonic_sort

.. autoclass:: BitonicSort
