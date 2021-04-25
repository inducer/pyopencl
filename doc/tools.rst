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

.. class:: AllocatorInterface

   An interface implemented by various memory allocation functions
   in :mod:`pyopencl`.

    .. method:: __call__(size)

        Allocate and return a :class:`pyopencl.Buffer` of the given *size*.

.. class:: DeferredAllocator(context, mem_flags=pyopencl.mem_flags.READ_WRITE)

    *mem_flags* takes its values from :class:`pyopencl.mem_flags` and corresponds
    to the *flags* argument of :class:`pyopencl.Buffer`. DeferredAllocator
    has the same semantics as regular OpenCL buffer allocation, i.e. it may
    promise memory to be available that may (in any call to a buffer-using
    CL function) turn out to not exist later on. (Allocations in CL are
    bound to contexts, not devices, and memory availability depends on which
    device the buffer is used with.)

    Implements :class:`AllocatorInterface`.

    .. versionchanged :: 2013.1

        ``CLAllocator`` was deprecated and replaced
        by :class:`DeferredAllocator`.

    .. method:: __call__(size)

        Allocate a :class:`pyopencl.Buffer` of the given *size*.

        .. versionchanged :: 2020.2

            The allocator will succeed even for allocations of size zero,
            returning *None*.

.. class:: ImmediateAllocator(queue, mem_flags=pyopencl.mem_flags.READ_WRITE)

    *mem_flags* takes its values from :class:`pyopencl.mem_flags` and corresponds
    to the *flags* argument of :class:`pyopencl.Buffer`.
    :class:`ImmediateAllocator` will attempt to ensure at allocation time that
    allocated memory is actually available. If no memory is available, an out-of-memory
    error is reported at allocation time.

    Implements :class:`AllocatorInterface`.

    .. versionadded:: 2013.1

    .. method:: __call__(size)

        Allocate a :class:`pyopencl.Buffer` of the given *size*.

        .. versionchanged :: 2020.2

            The allocator will succeed even for allocations of size zero,
            returning *None*.

.. class:: MemoryPool(allocator[, leading_bits_in_bin_id])

    A memory pool for OpenCL device memory. *allocator* must be an instance of
    one of the above classes, and should be an :class:`ImmediateAllocator`.
    The memory pool assumes that allocation failures are reported
    by the allocator immediately, and not in the OpenCL-typical
    deferred manner.

    Implements :class:`AllocatorInterface`.

    .. note::

        The current implementation of the memory pool will retain allocated
        memory after it is returned by the application and keep it in a bin
        identified by the leading *leading_bits_in_bin_id* bits of the
        allocation size. To ensure that allocations within each bin are
        interchangeable, allocation sizes are rounded up to the largest size
        that shares the leading bits of the requested allocation size.

        The current default value of *leading_bits_in_bin_id* is
        four, but this may change in future versions and is not
        guaranteed.

        *leading_bits_in_bin_id* must be passed by keyword,
        and its role is purely advisory. It is not guaranteed
        that future versions of the pool will use the
        same allocation scheme and/or honor *leading_bits_in_bin_id*.

    .. versionchanged:: 2019.1

        Current bin allocation behavior documented, *leading_bits_in_bin_id*
        added.

    .. attribute:: held_blocks

        The number of unused blocks being held by this pool.

    .. attribute:: active_blocks

        The number of blocks in active use that have been allocated
        through this pool.

    .. attribute:: managed_bytes

        "Managed" memory is "active" and "held" memory.

        .. versionadded: 2021.1.2

    .. attribute:: active_bytes

        "Active" bytes are bytes under the control of the application.
        This may be smaller than the actual allocated size reflected
        in :attr:`managed_bytes`.

        .. versionadded: 2021.1.2

    .. method:: allocate(size)

        Return a :class:`PooledBuffer` of the given *size*.

    .. method:: __call__(size)

        Synonym for :meth:`allocate` to match the :class:`AllocatorInterface`.

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
