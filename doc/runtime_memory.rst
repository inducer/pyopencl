.. include:: subst.rst

OpenCL Runtime: Memory
======================

.. currentmodule:: pyopencl

.. class:: MemoryObject

    .. attribute:: info

        Lower case versions of the :class:`mem_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    .. attribute:: hostbuf

    .. method:: get_info(param)

        See :class:`mem_info` for values of *param*.

    .. method:: release()

    .. method:: get_host_array(shape, dtype, order="C")

        Return the memory object's associated host memory
        area as a :class:`numpy.ndarray` of the given *shape*,
        *dtype* and *order*.

    .. automethod:: from_int_ptr
    .. autoattribute:: int_ptr

    |comparable|

Memory Migration
----------------

.. function:: enqueue_migrate_mem_objects(queue, mem_objects, flags=0, wait_for=None)

    :param flags: from :class:`mem_migration_flags`

    .. versionadded:: 2011.2

    Only available with CL 1.2.

Buffer
------

.. class:: Buffer(context, flags, size=0, hostbuf=None)

    Create a :class:`Buffer`.
    See :class:`mem_flags` for values of *flags*.
    If *hostbuf* is specified, *size* defaults to the size of
    the specified buffer if it is passed as zero.

    :class:`Buffer` inherits from :class:`MemoryObject`.

    .. note::

        Python also defines a type of `buffer object
        <https://docs.python.org/3/c-api/buffer.html>`__,
        and PyOpenCL interacts with those, too, as the host-side
        target of :func:`enqueue_copy`. Make sure to always be
        clear on whether a :class:`Buffer` or a Python buffer
        object is needed.

    Note that actual memory allocation in OpenCL may be deferred.
    Buffers are attached to a :class:`Context` and are only
    moved to a device once the buffer is used on that device.
    That is also the point when out-of-memory errors will occur.
    If you'd like to be sure that there's enough memory for
    your allocation, either use :func:`enqueue_migrate_mem_objects`
    (if available) or simply perform a small transfer to the
    buffer. See also :class:`pyopencl.tools.ImmediateAllocator`.

    .. method:: get_sub_region(origin, size, flags=0)

        Only available in OpenCL 1.1 and newer.

    .. method:: __getitem__(slc)

        *slc* is a :class:`slice` object indicating from which byte index range
        a sub-buffer is to be created. The *flags* argument of
        :meth:`get_sub_region` is set to the same flags with which *self* was
        created.


.. function:: enqueue_fill_buffer(queue, mem, pattern, offset, size, wait_for=None)

    :arg mem: the on device :class:`Buffer`
    :arg pattern: a buffer object (likely a :class:`numpy.ndarray`, eg.
        ``np.uint32(0)``). The memory associated with *pattern* can be reused or
        freed once the function completes.
    :arg size: The size in bytes of the region to be filled. Must be a multiple of the
        size of the pattern.
    :arg offset: The location in bytes of the region being filled in *mem*.
        Must be a multiple of the size of the pattern.

    Fills a buffer with the provided pattern

    |std-enqueue-blurb|

    Only available with CL 1.2.

    .. versionadded:: 2011.2

.. _svm:

Shared Virtual Memory (SVM)
---------------------------

Shared virtual memory allows the host and the compute device to share
address space, so that pointers on the host and on the device may have
the same meaning. In addition, it allows the same memory to be accessed
by both the host and the device. *Coarse-grain* SVM requires that
buffers be mapped before being accessed on the host, *fine-grain* SVM
does away with that requirement.

.. warning::

    Compared to :class:`Buffer`\ s, SVM brings with it a new concern: the
    synchronization of memory deallocation. Unlike other objects in OpenCL,
    SVM is represented by a plain (C-language) pointer and thus has no ability for
    reference counting.

    As a result, it is perfectly legal to allocate a :class:`Buffer`, enqueue an
    operation on it, and release the buffer, without worrying about whether the
    operation has completed. The OpenCL implementation will keep the buffer alive
    until the operation has completed. This is *not* the case with SVM: Unless
    otherwise specified, memory deallocation is performed immediately when
    requested, and so SVM will be deallocated whenever the Python
    garbage collector sees fit, even if the operation has not completed,
    immediately leading to undefined behavior (i.e., typically, memory corruption and,
    before too long, a crash).

    Version 2022.2 of PyOpenCL offers substantially improved tools
    for dealing with this. In particular, all means for allocating SVM
    allow specifying a :class:`CommandQueue`, so that deallocation
    is enqueued and performed after previously-enqueued operations
    have completed.

SVM requires OpenCL 2.0.

.. _opaque-svm:

Opaque and "Wrapped-:mod:`numpy`" Styles of Referencing SVM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When trying to pass SVM pointers to functionality in :mod:`pyopencl`,
two styles are supported:

- First, the opaque style. This style most closely resembles
  :class:`Buffer`-based allocation available in OpenCL 1.x.
  SVM pointers are held in opaque "handle" objects such as :class:`SVMAllocation`.

- Second, the wrapped-:mod:`numpy` style. In this case, a :class:`numpy.ndarray`
  (or another object implementing the  :c:func:`Python buffer protocol
  <PyObject_GetBuffer>`) serves as the reference to an area of SVM.
  This style permits using memory areas with :mod:`pyopencl`'s SVM
  interfaces even if they were allocated outside of :mod:`pyopencl`.

  Since passing a :class:`numpy.ndarray` (or another type of object obeying the
  buffer interface) already has existing semantics in most settings in
  :mod:`pyopencl` (such as when passing arguments to a kernel or calling
  :func:`enqueue_copy`), there exists a wrapper object, :class:`SVM`, that may
  be "wrapped around" these objects to mark them as SVM.

The commonality between the two styles is that both ultimately implement
the :class:`SVMPointer` interface, which :mod:`pyopencl` uses to obtain
the actual SVM pointer.

Note that it is easily possible to obtain a :class:`numpy.ndarray` view of SVM
areas held in the opaque style, see :attr:`SVMPointer.buf`, permitting
transitions from opaque to wrapped-:mod:`numpy` style. The opposite transition
(from wrapped-:mod:`numpy` to opaque) is not necessarily straightforward,
as it would require "fishing" the opaque SVM handle out of a chain of
:attr:`numpy.ndarray.base` attributes (or similar, depending on
the actual object serving as the main SVM reference).

See :ref:`numpy-svm-helpers` for helper functions that ease setting up the
wrapped-:mod:`numpy` structure.

Wrapped-:mod:`numpy` SVM tends to be a good fit for fine-grain SVM because of
the ease of direct host-side access, but the creation of the nested structure
that makes this possible is associated with a certain amount of cost.

By comparison, opaque SVM access tends to be a good fit for coarse-grain
SVM, because direct host access is not possible without mapping the array
anyway, and it has lower setup cost. It is of course entirely possible to use
opaque SVM access with fine-grain SVM.

.. versionchanged:: 2022.2

   This version adds the opaque style of SVM access.

Using SVM with Arrays
^^^^^^^^^^^^^^^^^^^^^

While all types of SVM can be used as the memory backing
:class:`pyopencl.array.Array` objects, ensuring that new arrays returned
by array operations (e.g. arithmetic) also use SVM is easiest to accomplish
by passing an :class:`~pyopencl.tools.SVMAllocator` (or
:class:`~pyopencl.tools.SVMPool`) as the *allocator* parameter in functions
returning new arrays.

SVM Pointers, Allocations, and Maps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: SVMPointer

.. autoclass:: SVMAllocation

.. autoclass:: SVM

.. autoclass:: SVMMap


.. _numpy-svm-helpers:

Helper functions for :mod:`numpy`-based SVM allocation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: svm_empty
.. autofunction:: svm_empty_like
.. autofunction:: csvm_empty
.. autofunction:: csvm_empty_like
.. autofunction:: fsvm_empty
.. autofunction:: fsvm_empty_like

Operations on SVM
^^^^^^^^^^^^^^^^^

(See also :ref:`mem-transfer`.)

.. autofunction:: enqueue_svm_memfill
.. autofunction:: enqueue_svm_migratemem

Image
-----

.. class:: ImageFormat([channel_order, channel_type])

    .. attribute:: channel_order

        See :class:`channel_order` for possible values.

    .. attribute:: channel_data_type

        See :class:`channel_type` for possible values.

    .. attribute:: channel_count

        .. versionadded:: 0.91.5

    .. attribute:: dtype_size

        .. versionadded:: 0.91.5

    .. attribute:: itemsize

        .. versionadded:: 0.91.5

    .. method:: __repr__

        Returns a :class:`str` representation of the image format.

        .. versionadded:: 0.91

    |comparable|

    .. versionchanged:: 0.91

        Constructor arguments added.

    .. versionchanged:: 2013.2

        :class:`ImageFormat` was made comparable and hashable

.. function:: get_supported_image_formats(context, flags, image_type)

    See :class:`mem_flags` for possible values of *flags*
    and :class:`mem_object_type` for possible values of *image_type*.

.. class:: Image(context, flags, format, shape=None, pitches=None, hostbuf=None, is_array=False, buffer=None)

    See :class:`mem_flags` for values of *flags*.
    *shape* is a 2- or 3-tuple. *format* is an instance of :class:`ImageFormat`.
    *pitches* is a 1-tuple for 2D images and a 2-tuple for 3D images, indicating
    the distance in bytes from one scan line to the next, and from one 2D image
    slice to the next.

    If *hostbuf* is given and *shape* is *None*, then *hostbuf.shape* is
    used as the *shape* parameter.

    :class:`Image` inherits from :class:`MemoryObject`.

    .. note::

        If you want to load images from :class:`numpy.ndarray` instances or read images
        back into them, be aware that OpenCL images expect the *x* dimension to vary
        fastest, whereas in the default (C) order of :mod:`numpy` arrays, the last index
        varies fastest. If your array is arranged in the wrong order in memory,
        there are two possible fixes for this:

        * Convert the array to Fortran (column-major) order using :func:`numpy.asarray`.

        * Pass *ary.T.copy()* to the image creation function.

    .. versionadded:: 0.91

    .. versionchanged:: 2011.2
        Added *is_array* and *buffer*, which are only available on CL 1.2 and newer.

    .. attribute:: info

        Lower case versions of the :class:`mem_info`
        and :class:`image_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    .. attribute:: shape

        Return the value of the *shape* constructor argument as a :class:`tuple`.

    .. method:: get_image_info(param)

        See :class:`image_info` for values of *param*.

    .. method:: release()

    |comparable|

.. function:: image_from_array(ctx, ary, num_channels=None, mode="r", norm_int=False)

    Build a 2D or 3D :class:`Image` from the :class:`numpy.ndarray` *ary*.  If
    *num_channels* is greater than one, the last dimension of *ary* must be
    identical to *num_channels*. *ary* must be in C order. If *num_channels* is
    not given, it defaults to 1 for scalar types and the number of entries
    for :ref:`vector-types`.

    The :class:`ImageFormat` is chosen as the first *num_channels* components
    of "RGBA".

    :param mode: "r" or "w" for read/write

    .. note::

        When reading from the image object, the indices passed to ``read_imagef``
        are in the reverse order from what they would be when accessing *ary* from
        Python.

    If *norm_int* is *True*, then the integer values are normalized to a floating
    point scale of 0..1 when read.

    .. versionadded:: 2011.2

.. function:: enqueue_fill_image(queue, mem, color, origin, region, wait_for=None)

    :arg color: a buffer object (likely a :class:`numpy.ndarray`)

    |std-enqueue-blurb|

    Only available with CL 1.2.

    .. versionadded:: 2011.2

.. _mem-transfer:

Transfers
---------

.. autofunction:: enqueue_copy(queue, dest, src, **kwargs)

.. autofunction:: enqueue_fill(queue, dest, src, **kwargs)

Mapping Memory into Host Address Space
--------------------------------------

.. autoclass:: MemoryMap

.. function:: enqueue_map_buffer(queue, buf, flags, offset, shape, dtype, order="C", strides=None, wait_for=None, is_blocking=True)

    |explain-waitfor|
    *shape*, *dtype*, and *order* have the same meaning
    as in :func:`numpy.empty`.
    See :class:`map_flags` for possible values of *flags*.
    *strides*, if given, overrides *order*.

    :return: a tuple *(array, event)*. *array* is a
        :class:`numpy.ndarray` representing the host side
        of the map. Its *.base* member contains a
        :class:`MemoryMap`.

    .. versionchanged:: 2011.1
        *is_blocking* now defaults to True.

    .. versionchanged:: 2013.1
        *order* now defaults to "C".

    .. versionchanged:: 2013.2
        Added *strides* argument.

    Sample usage::

        mapped_buf = cl.enqueue_map_buffer(queue, buf, ...)
        with mapped_buf.base:
            # work with mapped_buf
            ...

        # memory will be unmapped here

.. function:: enqueue_map_image(queue, buf, flags, origin, region, shape, dtype, order="C", strides=None, wait_for=None, is_blocking=True)

    |explain-waitfor|
    *shape*, *dtype*, and *order* have the same meaning
    as in :func:`numpy.empty`.
    See :class:`map_flags` for possible values of *flags*.
    *strides*, if given, overrides *order*.

    :return: a tuple *(array, event)*. *array* is a
        :class:`numpy.ndarray` representing the host side
        of the map. Its *.base* member contains a
        :class:`MemoryMap`.

    .. versionchanged:: 2011.1
        *is_blocking* now defaults to True.

    .. versionchanged:: 2013.1
        *order* now defaults to "C".

    .. versionchanged:: 2013.2
        Added *strides* argument.

Samplers
--------

.. class:: Sampler

    .. method:: __init__(context, normalized_coords, addressing_mode, filter_mode)

        *normalized_coords* is a :class:`bool` indicating whether
        to use coordinates between 0 and 1 (*True*) or the texture's
        natural pixel size (*False*).
        See :class:`addressing_mode` and :class:`filter_mode` for possible
        argument values.

        Also supports an alternate signature ``(context, properties)``.

        :arg properties: a sequence
            of keys and values from :class:`sampler_properties` as accepted
            by :c:func:`clCreateSamplerWithProperties` (see the OpenCL
            spec for details). The trailing *0* is added automatically
            and does not need to be included.

        This signature Requires OpenCL 2 or newer.

        .. versionchanged:: 2018.2

            The properties-based signature was added.

    .. attribute:: info

        Lower case versions of the :class:`sampler_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    .. method:: get_info(param)

        See :class:`sampler_info` for values of *param*.

    .. automethod:: from_int_ptr
    .. autoattribute:: int_ptr

    |comparable|

Pipes
-----

.. class:: Pipe(context, flags, packet_size, max_packets, properties=())

    See :class:`mem_flags` for values of *flags*.

    :arg properties: a sequence
        of keys and values from :class:`pipe_properties` as accepted
        by :c:func:`clCreatePipe`. The trailing *0* is added automatically
        and does not need to be included.
        (This argument must currently be empty.)

    This function requires OpenCL 2 or newer.

    .. versionadded:: 2020.3

    .. versionchanged:: 2021.1.7

        *properties* now defaults to an empty tuple.

    .. method:: get_pipe_info(param)

        See :class:`pipe_info` for values of *param*.

Type aliases
------------

.. currentmodule:: pyopencl._cl

.. class:: Buffer

   See :class:`pyopencl.Buffer`.
