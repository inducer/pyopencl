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

.. class:: PooledBuffer

    An object representing a :class:`MemoryPool`-based allocation of
    device memory.  Once this object is deleted, its associated device
    memory is returned to the pool. This supports the same interface
    as :class:`pyopencl.Buffer`.

.. class:: CLAllocator(context, mem_flags=pyopencl.mem_flags.READ_WRITE)

    *mem_flags* takes its values from :class:`pyopencl.mem_flags` and corresponds
    to the *flags* argument of :class:`pyopencl.Buffer`.

    .. method:: __call__(size)

        Allocate a :class:`pyopencl.Buffer` of the given *size*.

.. class:: MemoryPool(allocator=CLAllocator())

    A memory pool for OpenCL device memory.

    .. attribute:: held_blocks

        The number of unused blocks being held by this pool.

    .. attribute:: active_blocks

        The number of blocks in active use that have been allocated
        through this pool.

    .. method:: allocate(size)

        Return a :class:`PooledBuffer` of the given *size*.

    .. method:: free_held

        Free all unused memory that the pool is currently holding.

    .. method:: stop_holding

        Instruct the memory to start immediately freeing memory returned
        to it, instead of holding it for future allocations.
        Implicitly calls :meth:`free_held`.
        This is useful as a cleanup action when a memory pool falls out
        of use.
