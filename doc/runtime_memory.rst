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
        <https://docs.python.org/3.4/c-api/buffer.html>`_,
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
    :arg pattern: a buffer object (likely a :class:`numpy.ndarray`, eg. `np.uint32(0)`)
        The memory associated with *pattern* can be reused or freed once the function
        completes.
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

SVM requires OpenCL 2.0.

.. autoclass:: SVM

.. autoclass:: SVMMap

Allocating SVM
^^^^^^^^^^^^^^

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

SVM Allocation Holder
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: SVMAllocation

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

    If *hostbuf* is given and *shape* is `None`, then *hostbuf.shape* is
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

        When reading from the image object, the indices passed to `read_imagef` are
        in the reverse order from what they would be when accessing *ary* from
        Python.

    If *norm_int* is `True`, then the integer values are normalized to a floating
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

.. class:: Pipe(context, flags, packet_size, max_packets, properties)

    See :class:`mem_flags` for values of *flags*.

    :arg properties: a sequence
        of keys and values from :class:`pipe_properties` as accepted
        by :c:func:`clCreatePipe`. The trailing *0* is added automatically
        and does not need to be included.

    This function Requires OpenCL 2 or newer.

    .. versionadded:: 2020.3

    .. method:: get_pipe_info(param)

        See :class:`pipe_info` for values of *param*.

