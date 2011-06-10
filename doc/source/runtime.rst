.. _reference-doc:

OpenCL Platform/Runtime Documentation
=====================================

Version Queries
---------------

.. module:: pyopencl
.. moduleauthor:: Andreas Kloeckner <inform@tiker.net>

.. data:: VERSION

    Gives the numeric version of PyOpenCL as a variable-length tuple
    of integers. Enables easy version checks such as
    *VERSION >= (0, 93)*.

.. data:: VERSION_STATUS

    A text string such as `"rc4"` or `"beta"` qualifying the status
    of the release.

.. data:: VERSION_TEXT

    The full release name (such as `"0.93rc4"`) in string form.

.. function:: get_cl_header_version()

    Return a variable-length tuple of integers representing the
    version of the OpenCL header against which PyOpenCL was
    compiled.

    .. versionadded:: 0.92

.. _errors:

Error Reporting
---------------

.. class:: Error

    Base class for all PyOpenCL exceptions.

.. class:: MemoryError

.. class:: LogicError

.. class:: RuntimeError

Constants
---------

.. include:: constants.inc

Platforms, Devices and Contexts
-------------------------------

.. |comparable| replace:: Two instances of this class may be compared
    using *"=="* and *"!="*.
.. |buf-iface| replace:: must implement the Python buffer interface.
    (e.g. by being an :class:`numpy.ndarray`)
.. |explain-waitfor| replace:: *wait_for*
    may either be *None* or a list of :class:`Event` instances for
    whose completion this command waits before starting exeuction.
.. |std-enqueue-blurb| replace:: Returns a new :class:`Event`. |explain-waitfor|

.. |copy-depr| replace:: **Note:** This function is deprecated as of PyOpenCL 2011.1.
        Use :func:`enqueue_copy` instead.

.. function:: get_platforms()

    Return a list of :class:`Platform` instances.

.. class:: Platform

    .. attribute:: info

        Lower case versions of the :class:`platform_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    .. method:: get_info(param)

        See :class:`platform_info` for values of *param*.

    .. method:: get_devices(device_type=device_type.ALL)

        Return a list of devices matching *device_type*.
        See :class:`device_type` for values of *device_type*.

    |comparable|

.. class:: Device

    .. attribute:: info

        Lower case versions of the :class:`device_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    .. method:: get_info(param)

        See :class:`device_info` for values of *param*.

    Two instances of this class may be compared using *=="* and *"!="*.

.. class:: Context(devices=None, properties=None, dev_type=None)

    Create a new context. *properties* is a list of key-value
    tuples, where each key must be one of :class:`context_properties`.
    At most one of *devices* and *dev_type* may be not `None`, where
    *devices* is a list of :class:`Device` instances, and
    *dev_type* is one of the :class:`device_type` constants.
    If neither is specified, a context with a *dev_type* of
    :attr:`device_type.DEFAULT` is created.

    .. note::

        Calling the constructor with no arguments will fail for recent
        CL drivers that support the OpenCL ICD. If you want similar,
        just-give-me-a-context-already behavior, we recommend
        :func:`create_some_context`. See, e.g. this
        `explanation by AMD <http://developer.amd.com/support/KnowledgeBase/Lists/KnowledgeBase/DispForm.aspx?ID=71>`_.

    .. note::

        For
        :attr:`context_properties.CL_GL_CONTEXT_KHR`,
        :attr:`context_properties.CL_EGL_DISPLAY_KHR`,
        :attr:`context_properties.CL_GLX_DISPLAY_KHR`,
        :attr:`context_properties.CL_WGL_HDC_KHR`, and
        :attr:`context_properties.CL_CGL_SHAREGROUP_KHR`
        :attr:`context_properties.CL_CGL_SHAREGROUP_APPLE`
        the value in the key-value pair is a PyOpenGL context or display
        instance.

    .. versionchanged:: 0.91.2
        Constructor arguments *dev_type* added.

    .. attribute:: info

        Lower case versions of the :class:`context_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    .. method:: get_info(param)

        See :class:`context_info` for values of *param*.

    .. method:: create_sub_devices(properties)

        *properties* is an array of one (or more) of the forms::

            [ dppe.EQUALLY, 8]
            [ dppe.BY_COUNTS, 5, 7, 9, dppe.PARTITION_BY_COUNTS_LIST_END]
            [ dppe.BY_NAMES, 5, 7, 9, dppe.PARTITION_BY_NAMES_LIST_END]
            [ dppe.BY_AFFINITY_DOMAIN, ad.L1_CACHE]

        where `dppe` represents :class:`device_partition_property_ext`
        and `ad` represent :class:`affinity_domain_ext`.

        `PROPERTIES_LIST_END_EXT` is added automatically.

        Only available with the `cl_ext_device_fission` extension.

        .. versionadded:: 2011.1

    |comparable|

.. function:: create_some_context(interactive=True)

    Create a :class:`Context` 'somehow'.

    If multiple choices for platform and/or device exist, *interactive*
    is True, and *sys.stdin.isatty()* is also True,
    then the user is queried about which device should be chosen.
    Otherwise, a device is chosen in an implementation-defined manner.

Command Queues and Events
-------------------------

.. class:: CommandQueue(context, device=None, properties=None)

    Create a new command queue. *properties* is a bit field
    consisting of :class:`command_queue_properties` values.

    if *device* is None, one of the devices in *context* is chosen
    in an implementation-defined manner.

    .. attribute:: info

        Lower case versions of the :class:`command_queue_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    .. method:: get_info(param)

        See :class:`command_queue_info` for values of *param*.

    .. method:: set_property(prop, enable)

        See :class:`command_queue_properties` for possible values of *prop*.
        *enable* is a :class:`bool`.

        Unavailable in OpenCL 1.1 and newer.

    .. method:: flush()
    .. method:: finish()

    |comparable|

.. class:: Event

    .. attribute:: info

        Lower case versions of the :class:`event_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    .. attribute:: profile.info

        Lower case versions of the :class:`profiling_info` constants
        may be used as attributes on the attribute `profile` of this
        class to directly query profiling info.

        For example, you may use *evt.profile.end* instead of
        *evt.get_profiling_info(pyopencl.profiling_info.END)*.

    .. method:: get_info(param)

        See :class:`event_info` for values of *param*.

    .. method:: get_profiling_info(param)

        See :class:`profiling_info` for values of *param*.
        See :attr:`profile` for an easier way of obtaining
        the same information.

    .. method:: wait()

    |comparable|

.. function:: wait_for_events(events)
.. function:: enqueue_marker(queue)

    Returns an :class:`Event`.

.. function:: enqueue_wait_for_events(queue, events)

    Returns an :class:`Event`.

.. function:: enqueue_barrier(queue)

    Enqueues a barrier operation. which ensures that all queued commands in
    command_queue have finished execution. This command is a synchronization
    point.

    .. versionadded:: 0.91.5

.. class:: UserEvent(context)

    A subclass of :class:`Event`. Only available with OpenCL 1.1 and newer.

    .. versionadded:: 0.92

    .. method:: set_status(status)

        See :class:`command_execution_status` for possible values of *status*.

Memory
------

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

    |comparable|

Buffers
^^^^^^^

.. class:: Buffer(context, flags, size=0, hostbuf=None)

    Create a :class:`Buffer`.
    See :class:`mem_flags` for values of *flags*.
    If *hostbuf* is specified, *size* defaults to the size of
    the specified buffer if it is passed as zero.

    :class:`Buffer` is a subclass of :class:`MemoryObject`.

    .. method:: get_sub_region(origin, size, flags=0)

    .. method:: __getitem__(slc)

        *slc* is a :class:`slice` object indicating from which byte index range
        a sub-buffer is to be created. The *flags* argument of
        :meth:`get_sub_region` is set to the same flags with which *self* was
        created.

.. function:: enqueue_read_buffer(queue, mem, hostbuf, device_offset=0, wait_for=None, is_blocking=True)

    |std-enqueue-blurb|

    *hostbuf* |buf-iface|

    |copy-depr|

    .. versionchanged:: 2011.1
        *is_blocking* now defaults to True.

.. function:: enqueue_write_buffer(queue, mem, hostbuf, device_offset=0, wait_for=None, is_blocking=True)

    |std-enqueue-blurb|

    *hostbuf* |buf-iface|

    |copy-depr|

    .. versionchanged:: 2011.1
        *is_blocking* now defaults to True.

.. function:: enqueue_copy_buffer(queue, src, dst, byte_count=0, src_offset=0, dst_offset=0, wait_for=None)

    If *byte_count* is passed as 0 (the default), the size of the
    :class:`Buffer` *src* is used instead.

    |std-enqueue-blurb|

    |copy-depr|

    .. versionadded:: 0.91.5

.. function:: enqueue_read_buffer_rect(queue, mem, hostbuf, buffer_origin, host_origin, region, buffer_pitches=None, host_pitches=None, wait_for=None, is_blocking=True)

    The *origin* and *region* parameters are :class:`tuple` instances of length
    three or shorter. The *pitches* parameters are :class:`tuple` instances of
    length two or shorter, which may be zero to indicate 'tight packing'.

    |std-enqueue-blurb|

    *hostbuf* |buf-iface|

    |copy-depr|

    Only available in OpenCL 1.1 and newer.

    .. versionadded:: 0.92

    .. versionchanged:: 2011.1
        *is_blocking* now defaults to True.

.. function:: enqueue_write_buffer_rect(queue, mem, hostbuf, buffer_origin, host_origin, region, buffer_pitches=None, host_pitches=None, wait_for=None, is_blocking=True)

    The *origin* and *region* parameters are :class:`tuple` instances of length
    three or shorter. The *pitches* parameters are :class:`tuple` instances of
    length two or shorter, which may be zero to indicate 'tight packing'.

    |std-enqueue-blurb|

    *hostbuf* |buf-iface|

    |copy-depr|

    Only available in OpenCL 1.1 and newer.

    .. versionadded:: 0.92

    .. versionchanged:: 2011.1
        *is_blocking* now defaults to True.

.. function:: enqueue_copy_buffer_rect(queue, src, dst, src_origin, dst_origin, region, src_pitches=None, dst_pitches=None, wait_for=None)

    The *origin* and *region* parameters are :class:`tuple` instances of length
    three or shorter. The *pitches* parameters are :class:`tuple` instances of
    length two or shorter, which may be zero to indicate 'tight packing'.

    |std-enqueue-blurb|

    |copy-depr|

    Only available in OpenCL 1.1 and newer.

    .. versionadded:: 0.92

Image Formats
^^^^^^^^^^^^^

.. class:: ImageFormat([channel_order, channel_type])

    .. versionchanged:: 0.91
        Constructor arguments added.

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

.. function:: get_supported_image_formats(context, flags, image_type)

    See :class:`mem_flags` for possible values of *flags*
    and :class:`mem_object_type` for possible values of *image_type*.

Images
^^^^^^
.. class:: Image(context, flags, format, shape=None, pitches=None, hostbuf=None)

    See :class:`mem_flags` for values of *flags*.
    *shape* is a 2- or 3-tuple. *format* is an instance of :class:`ImageFormat`.
    *pitches* is a 1-tuple for 2D images and a 2-tuple for 3D images, indicating 
    the distance in bytes from one scan line to the next, and from one 2D image
    slice to the next.

    If *hostbuf* is given and *shape* is `None`, then *hostbuf.shape* is
    used as the *shape* parameter.

    :class:`Image` is a subclass of :class:`MemoryObject`.

    .. note::

        If you want to load images from :mod:`numpy.ndarray` instances or read images 
        back into them, be aware that OpenCL images expect the *x* dimension to vary 
        fastest, whereas in the default (C) order of :mod:`numpy` arrays, the last index
        varies fastest. If your array is arranged in the wrong order in memory,
        there are two possible fixes for this:

        * Convert the array to Fortran (column-major) order using :func:`numpy.asarray`.

        * Pass *ary.T.copy()* to the image creation function.

    .. versionadded:: 0.91

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

.. function:: enqueue_read_image(queue, mem, origin, region, hostbuf, row_pitch=0, slice_pitch=0, wait_for=None, is_blocking=True)

    |copy-depr|

    |std-enqueue-blurb|

    .. versionchanged:: 0.91
        *pitch* arguments defaults to zero, moved.

    .. versionchanged:: 2011.1
        *is_blocking* now defaults to True.


.. function:: enqueue_write_image(queue, mem, origin, region, hostbuf, row_pitch=0, slice_pitch=0, wait_for=None, is_blocking=True)

    |copy-depr|

    |std-enqueue-blurb|

    .. versionchanged:: 0.91
        *pitch* arguments defaults to zero, moved.

    .. versionchanged:: 2011.1
        *is_blocking* now defaults to True.


.. function:: enqueue_copy_image(queue, src, dest, src_origin, dest_origin, region, wait_for=None)

    |copy-depr|

    |std-enqueue-blurb|

.. function:: enqueue_copy_image_to_buffer(queue, src, dest, origin, region, offset, wait_for=None)

    |copy-depr|

    |std-enqueue-blurb|

.. function:: enqueue_copy_buffer_to_image(queue, src, dest, offset, origin, region, wait_for=None)

    |copy-depr|

    |std-enqueue-blurb|

Transfers
^^^^^^^^^

.. function:: enqueue_copy(queue, dest, src, **kwargs)

    Copy from :class:`Image`, :class:`Buffer` or the host to 
    :class:`Image`, :class:`Buffer` or the host. (Note: host-to-host
    copies are unsupported.)

    The following keyword arguments are available:

    :arg wait_for: (optional, default empty)
    :arg is_blocking: Wait for completion. Defaults to *True*. 
      (Available on any copy involving host memory)

    :class:`Buffer` ↔ host transfers:

    :arg device_offset: offset in bytes (optional)

    :class:`Buffer` ↔ :class:`Buffer` transfers:

    :arg byte_count: (optional)
    :arg src_offset: (optional)
    :arg dest_offset: (optional)

    Rectangular :class:`Buffer` ↔  host transfers (CL 1.1 and newer):

    :arg buffer_origin: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg host_origin: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg region: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg buffer_pitches: :class:`tuple` of :class:`int` of length
        two or shorter. (optional, "tightly-packed" if unspecified)
    :arg host_pitches: :class:`tuple` of :class:`int` of length
        two or shorter. (optional, "tightly-packed" if unspecified)

    :class:`Image` ↔ host transfers:

    :arg origin: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg region: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg pitches: :class:`tuple` of :class:`int` of length
        two or shorter. (optional)

    :class:`Buffer` ↔ :class:`Image` transfers:

    :arg offset: offset in buffer (mandatory)
    :arg origin: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg region: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)

    :class:`Image` ↔ :class:`Image` transfers:

    :arg src_origin: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg dest_origin: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg region: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)

    |std-enqueue-blurb|

    .. versionadded:: 2011.1

Mapping Memory into Host Address Space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: MemoryMap

    .. method:: release(queue=None, wait_for=None)

.. function:: enqueue_map_buffer(queue, buf, flags, offset, shape, dtype, order, wait_for=None, is_blocking=True)

    |explain-waitfor|
    *shape*, *dtype*, and *order* have the same meaning
    as in :func:`numpy.empty`.
    See :class:`map_flags` for possible values of *flags*.

    :return: a tuple *(array, event)*. *array* is a
        :class:`numpy.ndarray` representing the host side
        of the map. Its *.base* member contains a
        :class:`MemoryMap`.

    .. versionchanged:: 2011.1
        *is_blocking* now defaults to True.

.. function:: enqueue_map_image(queue, buf, flags, origin, region, shape, dtype, order, wait_for=None, is_blocking=True)

    |explain-waitfor|
    *shape*, *dtype*, and *order* have the same meaning
    as in :func:`numpy.empty`.
    See :class:`map_flags` for possible values of *flags*.

    :return: a tuple *(array, event)*. *array* is a
        :class:`numpy.ndarray` representing the host side
        of the map. Its *.base* member contains a
        :class:`MemoryMap`.

    .. versionchanged:: 2011.1
        *is_blocking* now defaults to True.


Samplers
^^^^^^^^

.. class:: Sampler(context, normalized_coords, addressing_mode, filter_mode)

    *normalized_coords* is a :class:`bool` indicating whether
    to use coordinates between 0 and 1 (*True*) or the texture's
    natural pixel size (*False*).
    See :class:`addressing_mode` and :class:`filter_mode` for possible
    argument values.

    .. attribute:: info

        Lower case versions of the :class:`sampler_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    .. method:: get_info(param)

        See :class:`sampler_info` for values of *param*.

    |comparable|

Programs and Kernels
--------------------

.. class:: Program(context, src)
           Program(context, devices, binaries)

    *binaries* must contain one binary for each entry in *devices*.

    .. attribute:: info

        Lower case versions of the :class:`program_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    .. method:: get_info(param)

        See :class:`program_info` for values of *param*.

    .. method:: get_build_info(device, param)

        See :class:`program_build_info` for values of *param*.

    .. method:: build(options=[], devices=None)

        *options* is a string of compiler flags.
        Returns *self*.

        .. versionchanged:: 2011.1
            *options* may now also be a :class:`list` of :class:`str`.

    .. attribute:: kernel_name

        :class:`Kernel` objects can be produced from a built
        (see :meth:`build`) program simply by attribute lookup.

        .. note::

            The :class:`program_info` attributes live
            in the same name space and take precedence over
            :class:`Kernel` names.

    |comparable|

    .. method:: all_kernels()

        Returns a list of all :class:`Kernel` objects in the :class:`Program`.



.. class:: Kernel(program, name)

    .. attribute:: info

        Lower case versions of the :class:`kernel_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    .. method:: get_info(param)

        See :class:`kernel_info` for values of *param*.

    .. method:: get_work_group_info(param, device)

        See :class:`kernel_work_group_info` for values of *param*.

    .. method:: set_arg(self, index, arg)

        *arg* may be

        * `None`: This may be passed for `__global` memory references
          to pass a NULL pointer to the kernel.
        * Anything that satisfies the Python buffer interface,
          in particular :class:`numpy.ndarray`, :class:`str`,
          or :mod:`numpy`'s sized scalars, such as :class:`numpy.int32`
          or :class:`numpy.float64`.

          .. note::

              Note that Python's own :class:`int` or :class:`float`
              objects will not work out of the box. See
              :meth:`Kernel.set_scalar_arg_dtypes` for a way to make
              them work. Alternatively, the standard library module
              :mod:`struct` can be used to convert Python's native
              number types to binary data in a :class:`str`.

        * An instance of :class:`MemoryObject`. (e.g. :class:`Buffer`,
          :class:`Image`, etc.)
        * An instance of :class:`LocalMemory`.
        * An instance of :class:`Sampler`.

    .. method:: set_args(self, *args)

        Invoke :meth:`set_arg` on each element of *args* in turn.

        ..versionadded:: 0.92

    .. method:: set_scalar_arg_dtypes(arg_dtypes)

        Inform the wrapper about the sized types of scalar
        :class:`Kernel` arguments. For each argument,
        *arg_dtypes* contains an entry. For non-scalars,
        this must be *None*. For scalars, it must be an
        object acceptable to the :class:`numpy.dtype` 
        constructor, indicating that the corresponding
        scalar argument is of that type.

        After invoking this function with the proper information,
        most suitable number types will automatically be
        cast to the right type for kernel invocation.

        .. note :: 

            The information set by this rountine is attached to a single kernel
            instance. A new kernel instance is created every time you use
            `program.kernel` attribute access. The following will therefore not
            work::

               prg = cl.Program(...).build()
               prg.kernel.set_scalar_arg_dtypes(...)
               prg.kernel(queue, n_globals, None, args)


    .. method:: __call__(queue, global_size, local_size, *args, global_offset=None, wait_for=None)

        Use :func:`enqueue_nd_range_kernel` to enqueue a kernel execution, after using
        :meth:`set_args` to set each argument in turn. See the documentation for
        :meth:`set_arg` to see what argument types are allowed.
        |std-enqueue-blurb|

        *None* may be passed for local_size.

        .. versionchanged:: 0.92
            *local_size* was promoted to third positional argument from being a
            keyword argument. The old keyword argument usage will continue to
            be accepted with a warning throughout the 0.92 release cycle.
            This is a backward-compatible change (just barely!) because
            *local_size* as third positional argument can only be a
            :class:`tuple` or *None*.  :class:`tuple` instances are never valid
            :class:`Kernel` arguments, and *None* is valid as an argument, but
            its treatment in the wrapper had a bug (now fixed) that prevented
            it from working.

    |comparable|

.. class:: LocalMemory(size)

    A helper class to pass `__local` memory arguments to kernels.

    .. versionadded:: 0.91.2

    .. attribute:: size

        The size of local buffer in bytes to be provided.

.. function:: enqueue_nd_range_kernel(queue, kernel, global_work_size, local_work_size, global_work_offset=None, wait_for=None)

    |std-enqueue-blurb|

.. function:: enqueue_task(queue, kernel, wait_for=None)

    |std-enqueue-blurb|

.. _gl-interop:

GL Interoperability
-------------------

Functionality in this section is only available when PyOpenCL is compiled
with GL support. See :func:`have_gl`.

.. versionadded:: 0.91

.. function:: have_gl()

    Return *True* if PyOpenCL was compiled with OpenGL interoperability, otherwise *False*.

.. function:: get_gl_sharing_context_properties()

    Return a :class:`list` of :class:`context_properties` that will
    allow a newly created context to share the currently active GL
    context.

.. function:: get_apple_cgl_share_group()

    Get share group handle for current CGL context.

    Apple OS X only.

    ..versionadded:: 2011.1

.. class:: GLBuffer(context, flags, bufobj)

    :class:`GLBuffer` is a subclass of :class:`MemoryObject`.

    .. attribute:: gl_object

.. class:: GLRenderBuffer(context, flags, bufobj)

    :class:`GLRenderBuffer` is a subclass of :class:`MemoryObject`.

    .. attribute:: gl_object

.. class:: GLTexture(context, flags, texture_target, miplevel, texture, dims)

    *dims* is either 2 or 3.
    :class:`GLTexture` is a subclass of :class:`Image`.

    .. attribute:: gl_object

    .. method:: get_gl_texture_info(param)

        See :class:`gl_texture_info` for values of *param*.  Only available when PyOpenCL is compiled with GL support. See :func:`have_gl`.

.. function:: enqueue_acquire_gl_objects(queue, mem_objects, wait_for=None)

    *mem_objects* is a list of :class:`MemoryObject` instances.
    |std-enqueue-blurb|

.. function:: enqueue_release_gl_objects(queue, mem_objects, wait_for=None)

    *mem_objects* is a list of :class:`MemoryObject` instances. |std-enqueue-blurb|

.. function:: get_gl_context_info_khr(properties, param_name)

    Get information on which CL device corresponds to a given
    GL/EGL/WGL/CGL device.

    See the :class:`Context` constructor for the meaning of
    *properties* and :class:`gl_context_info` for *param_name*.
