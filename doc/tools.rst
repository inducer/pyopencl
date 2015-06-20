Built-in Utilities
==================

.. module:: pyopencl.tools

.. _memory-pools:

Memory Pools
------------

The constructor :func:`pyopencl.Buffer` can consume a fairly large amount of
processing time if it is invoked very frequently. For example, code based on
:class:`pyopencl.array.Array` can easily run into this issue because a
fresh memory area is allocated for each intermediate result. Memory pools are a
remedy for this problem based on the observation that often many of the block
allocations are of the same sizes as previously used ones.

Then, instead of fully returning the memory to the system and incurring the 
associated reallocation overhead, the pool holds on to the memory and uses it
to satisfy future allocations of similarly-sized blocks. The pool reacts
appropriately to out-of-memory conditions as long as all memory allocations
are made through it. Allocations performed from outside of the pool may run
into spurious out-of-memory conditions due to the pool owning much or all of
the available memory.

Using :class:`pyopencl.array.Array` instances with a :class:`MemoryPool` is
not complicated::

    mem_pool = pyopencl.tools.MemoryPool(pyopencl.tools.ImmediateAllocator(queue))
    a_dev = cl_array.arange(queue, 2000, dtype=np.float32, allocator=mem_pool)

.. class:: PooledBuffer

    An object representing a :class:`MemoryPool`-based allocation of
    device memory.  Once this object is deleted, its associated device
    memory is returned to the pool. This supports the same interface
    as :class:`pyopencl.Buffer`.

.. class:: DeferredAllocator(context, mem_flags=pyopencl.mem_flags.READ_WRITE)

    *mem_flags* takes its values from :class:`pyopencl.mem_flags` and corresponds
    to the *flags* argument of :class:`pyopencl.Buffer`. DeferredAllocator
    has the same semantics as regular OpenCL buffer allocation, i.e. it may
    promise memory to be available that may (in any call to a buffer-using
    CL function) turn out to not exist later on. (Allocations in CL are
    bound to contexts, not devices, and memory availability depends on which
    device the buffer is used with.)

    .. versionchanged::
        In version 2013.1, :class:`CLAllocator` was deprecated and replaced
        by :class:`DeferredAllocator`.

    .. method:: __call__(size)

        Allocate a :class:`pyopencl.Buffer` of the given *size*.

.. class:: ImmediateAllocator(queue, mem_flags=pyopencl.mem_flags.READ_WRITE)

    *mem_flags* takes its values from :class:`pyopencl.mem_flags` and corresponds
    to the *flags* argument of :class:`pyopencl.Buffer`. DeferredAllocator
    has the same semantics as regular OpenCL buffer allocation, i.e. it may
    promise memory to be available that later on (in any call to a buffer-using
    CL function).

    .. versionadded:: 2013.1

    .. method:: __call__(size)

        Allocate a :class:`pyopencl.Buffer` of the given *size*.

.. class:: MemoryPool(allocator)

    A memory pool for OpenCL device memory. *allocator* must be an instance of
    one of the above classes, and should be an :class:`ImmediateAllocator`.
    The memory pool assumes that allocation failures are reported
    by the allocator immediately, and not in the OpenCL-typical
    deferred manner.

    .. attribute:: held_blocks

        The number of unused blocks being held by this pool.

    .. attribute:: active_blocks

        The number of blocks in active use that have been allocated
        through this pool.

    .. method:: allocate(size)

        Return a :class:`PooledBuffer` of the given *size*.

    .. method:: __call__(size)

        Synoynm for :meth:`allocate` to match :class:`CLAllocator` interface.

        .. versionadded: 2011.2

    .. method:: free_held

        Free all unused memory that the pool is currently holding.

    .. method:: stop_holding

        Instruct the memory to start immediately freeing memory returned
        to it, instead of holding it for future allocations.
        Implicitly calls :meth:`free_held`.
        This is useful as a cleanup action when a memory pool falls out
        of use.

CL-Object-dependent Caching
---------------------------

.. autofunction:: first_arg_dependent_memoize
.. autofunction:: clear_first_arg_caches

Testing
-------

.. function:: pytest_generate_tests_for_pyopencl(metafunc)

    Using the line::

        from pyopencl.tools import pytest_generate_tests_for_pyopencl \
                as pytest_generate_tests

    in your `pytest <http://pytest.org>`_ test scripts allows you to use the
    arguments *ctx_factory*, *device*, or *platform* in your test functions,
    and they will automatically be run for each OpenCL device/platform in the
    system, as appropriate.

    The following two environment variables are also supported to control
    device/platform choice::

        PYOPENCL_TEST=0:0,1;intel=i5,i7

Device Characterization
-----------------------

.. automodule:: pyopencl.characterize
    :members:
