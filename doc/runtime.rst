.. _reference-doc:

.. include:: subst.rst

OpenCL Runtime
==============

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

        .. versionchanged:: 2013.2

            This used to raise an exception if no matching
            devices were found. Now, it will simply return
            an empty list.

    .. automethod:: from_int_ptr
    .. autoattribute:: int_ptr

    |comparable|

.. class:: Device

    .. attribute:: info

        Lower case versions of the :class:`device_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    .. method:: get_info(param)

        See :class:`device_info` for values of *param*.

    .. automethod:: from_int_ptr
    .. autoattribute:: int_ptr

    .. method:: create_sub_devices(properties)

        *properties* is an array of one (or more) of the forms::

            [ dpp.EQUALLY, 8]
            [ dpp.BY_COUNTS, 5, 7, 9, dpp.PARTITION_BY_COUNTS_LIST_END]
            [ dpp.BY_NAMES, 5, 7, 9, dpp.PARTITION_BY_NAMES_LIST_END]
            [ dpp.BY_AFFINITY_DOMAIN, dad.L1_CACHE]

        where `dpp` represents :class:`device_partition_property`
        and `dad` represent :class:`device_affinity_domain`.

        `PROPERTIES_LIST_END_EXT` is added automatically.

        Only available with CL 1.2.

        .. versionadded:: 2011.2

    Two instances of this class may be compared using *=="* and *"!="*.

.. class:: Context(devices=None, properties=None, dev_type=None, cache_dir=None)

    Create a new context. *properties* is a list of key-value
    tuples, where each key must be one of :class:`context_properties`.
    At most one of *devices* and *dev_type* may be not `None`, where
    *devices* is a list of :class:`Device` instances, and
    *dev_type* is one of the :class:`device_type` constants.
    If neither is specified, a context with a *dev_type* of
    :attr:`device_type.DEFAULT` is created.

    If *cache_dir* is not `None` - it will be used as default *cache_dir*
    for all its' :class:`Program` instances builds (see also :meth:`Program.build`).

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

    .. automethod:: from_int_ptr
    .. autoattribute:: int_ptr

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

    A :class:`CommandQueue` may be used as a context manager, like this::

        with cl.CommandQueue(self.cl_context) as queue:
            enqueue_stuff(queue, ...)

    :meth:`finish` is automatically called at the end of the context.

    .. versionadded:: 2013.1

        Context manager capability.

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

    .. automethod:: from_int_ptr
    .. autoattribute:: int_ptr

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

    .. automethod:: from_int_ptr
    .. autoattribute:: int_ptr

    .. method:: set_callback(type, cb)

        Add the callback *cb* with signature ``cb(status)`` to the callback
        queue for the event status *type* (one of the values of
        :class:`command_execution_status`, except :attr:`command_execution_status.QUEUED`).

        See the OpenCL specification for restrictions on what *cb* may and may not do.

        .. versionadded:: 2015.2

    |comparable|

.. function:: wait_for_events(events)

.. function:: enqueue_barrier(queue, wait_for=None)

    Enqueues a barrier operation. which ensures that all queued commands in
    command_queue have finished execution. This command is a synchronization
    point.

    .. versionadded:: 0.91.5
    .. versionchanged:: 2011.2
        Takes *wait_for* and returns an :class:`Event`

.. function:: enqueue_marker(queue, wait_for=None)

    Returns an :class:`Event`.

    .. versionchanged:: 2011.2
        Takes *wait_for*.

.. class:: UserEvent(context)

    A subclass of :class:`Event`. Only available with OpenCL 1.1 and newer.

    .. versionadded:: 0.92

    .. method:: set_status(status)

        See :class:`command_execution_status` for possible values of *status*.

.. class:: NannyEvent

    Transfers between host and device return events of this type. They hold
    a reference to the host-side buffer and wait for the transfer to complete
    when they are freed. Therefore, they can safely release the reference to
    the object they're guarding upon destruction.

    A subclass of :class:`Event`.

    .. versionadded:: 2011.2

    .. method:: get_ward()

    .. method:: wait()

        In addition to performing the same wait as :meth:`Event.wait()`, this
        method also releases the reference to the guarded object.

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

    .. automethod:: from_int_ptr
    .. autoattribute:: int_ptr

    |comparable|

.. function:: enqueue_migrate_mem_objects(queue, mem_objects, flags=0, wait_for=None)

    :param flags: from :class:`mem_migration_flags`

    .. versionadded:: 2011.2

    Only available with CL 1.2.

.. function:: enqueue_migrate_mem_object_ext(queue, mem_objects, flags=0, wait_for=None)

    :param flags: from :class:`migrate_mem_object_flags_ext`

    .. versionadded:: 2011.2

    Only available with the `cl_ext_migrate_memobject`
    extension.

Buffers
^^^^^^^

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

    :arg pattern: a buffer object (likely a :class:`numpy.ndarray`)

    |std-enqueue-blurb|

    Only available with CL 1.2.

    .. versionadded:: 2011.2

Image Formats
^^^^^^^^^^^^^

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

Images
^^^^^^
.. class:: Image(context, flags, format, shape=None, pitches=None, hostbuf=None, is_array=False, buffer=None):

    See :class:`mem_flags` for values of *flags*.
    *shape* is a 2- or 3-tuple. *format* is an instance of :class:`ImageFormat`.
    *pitches* is a 1-tuple for 2D images and a 2-tuple for 3D images, indicating
    the distance in bytes from one scan line to the next, and from one 2D image
    slice to the next.

    If *hostbuf* is given and *shape* is `None`, then *hostbuf.shape* is
    used as the *shape* parameter.

    :class:`Image` inherits from :class:`MemoryObject`.

    .. note::

        If you want to load images from :mod:`numpy.ndarray` instances or read images
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

Transfers
^^^^^^^^^

.. autofunction:: enqueue_copy(queue, dest, src, **kwargs)

Mapping Memory into Host Address Space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: MemoryMap

    .. method:: release(queue=None, wait_for=None)

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

    .. automethod:: from_int_ptr
    .. autoattribute:: int_ptr

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

    .. method:: build(options=[], devices=None, cache_dir=None)

        *options* is a string of compiler flags.
        Returns *self*.

        If *cache_dir* is not None - built binaries are cached in an on-disk cache
        with given path.
        If passed *cache_dir* is None, but context of this program was created with
        not-None cache_dir - it will be used as cache directory.
        If passed *cache_dir* is None and context was created with None cache_dir:
        built binaries will be cached in an on-disk cache called
        :file:`pyopencl-compiler-cache-vN-uidNAME-pyVERSION` in the directory
        returned by :func:`tempfile.gettempdir`.  By setting the environment
        variable :envvar:`PYOPENCL_NO_CACHE` to any non-empty value, this
        caching is suppressed.  Any options found in the environment variable
        :envvar:`PYOPENCL_BUILD_OPTIONS` will be appended to *options*.

        .. versionchanged:: 2011.1
            *options* may now also be a :class:`list` of :class:`str`.

        .. versionchanged:: 2013.1
            Added :envvar:`PYOPENCL_NO_CACHE`.
            Added :envvar:`PYOPENCL_BUILD_OPTIONS`.

    .. method:: compile(self, options=[], devices=None, headers=[])

        :param headers: a list of tuples *(name, program)*.

        Only available with CL 1.2.

        .. versionadded:: 2011.2

    .. attribute:: kernel_name

        You may use ``program.kernel_name`` to obtain a :class:`Kernel`
        objects from a program. Note that every lookup of this type
        produces a new kernel object, so that this **won't** work::

            prg.sum.set_args(a_g, b_g, res_g)
            ev = cl.enqueue_nd_range_kernel(queue, prg.sum, a_np.shape, None)

        Instead, either use the (recommended, stateless) calling interface::

            prg.sum(queue, prg.sum, a_np.shape, None)

        or keep the kernel in a temporary variable::

            sum_knl = prg.sum
            sum_knl.set_args(a_g, b_g, res_g)
            ev = cl.enqueue_nd_range_kernel(queue, sum_knl, a_np.shape, None)

        Note that the :class:`Program` has to be built (see :meth:`build`) in
        order for this to work simply by attribute lookup.

        .. note::

            The :class:`program_info` attributes live
            in the same name space and take precedence over
            :class:`Kernel` names.

    .. method:: all_kernels()

        Returns a list of all :class:`Kernel` objects in the :class:`Program`.

    .. automethod:: from_int_ptr
    .. autoattribute:: int_ptr

    |comparable|

.. function:: create_program_with_built_in_kernels(context, devices, kernel_names)

    Only available with CL 1.2.

    .. versionadded:: 2011.2

.. function:: link_program(context, programs, options=[], devices=None)

    Only available with CL 1.2.

    .. versionadded:: 2011.2

.. function:: unload_platform_compiler(platform)

    Only available with CL 1.2.

    .. versionadded:: 2011.2

.. class:: Kernel(program, name)

    .. attribute:: info

        Lower case versions of the :class:`kernel_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    .. method:: get_info(param)

        See :class:`kernel_info` for values of *param*.

    .. method:: get_work_group_info(param, device)

        See :class:`kernel_work_group_info` for values of *param*.

    .. method:: get_arg_info(arg_index, param)

        See :class:`kernel_arg_info` for values of *param*.

        Only available in OpenCL 1.2 and newer.

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

        .. versionadded:: 0.92

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


    .. method:: __call__(queue, global_size, local_size, *args, global_offset=None, wait_for=None, g_times_l=False)

        Use :func:`enqueue_nd_range_kernel` to enqueue a kernel execution, after using
        :meth:`set_args` to set each argument in turn. See the documentation for
        :meth:`set_arg` to see what argument types are allowed.
        |std-enqueue-blurb|

        *None* may be passed for local_size.

        If *g_times_l* is specified, the global size will be multiplied by the
        local size. (which makes the behavior more like Nvidia CUDA) In this case,
        *global_size* and *local_size* also do not have to have the same number
        of dimensions.

        .. note::

            :meth:`__call__` is *not* thread-safe. It sets the arguments using :meth:`set_args`
            and then runs :func:`enqueue_nd_range_kernel`. Another thread could race it
            in doing the same things, with undefined outcome. This issue is inherited
            from the C-level OpenCL API. The recommended solution is to make a kernel
            (i.e. access `prg.kernel_name`, which corresponds to making a new kernel)
            for every thread that may enqueue calls to the kernel.

            A solution involving implicit locks was discussed and decided against on the
            mailing list in `October 2012
            <http://lists.tiker.net/pipermail/pyopencl/2012-October/001311.html>`_.

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

        .. versionchanged:: 2011.1
            Added the *g_times_l* keyword arg.

    .. method:: capture_call(filename, queue, global_size, local_size, *args, global_offset=None, wait_for=None, g_times_l=False)

        This method supports the exact same interface as :meth:`__call__`, but
        instead of invoking the kernel, it writes a self-contained PyOpenCL program
        to *filename* that reproduces this invocation. Data and kernel source code
        will be packaged up in *filename*'s source code.

        This is mainly intended as a debugging aid. For example, it can be used
        to automate the task of creating a small, self-contained test case for
        an observed problem. It can also help separate a misbehaving kernel from
        a potentially large or time-consuming outer code.

        To use, simply change::

            evt = my_kernel(queue, gsize, lsize, arg1, arg2, ...)

        to::

            evt = my_kernel.capture_call("bug.py", queue, gsize, lsize, arg1, arg2, ...)

        .. versionadded:: 2013.1

    .. automethod:: from_int_ptr
    .. autoattribute:: int_ptr

    |comparable|

.. class:: LocalMemory(size)

    A helper class to pass `__local` memory arguments to kernels.

    .. versionadded:: 0.91.2

    .. attribute:: size

        The size of local buffer in bytes to be provided.

.. function:: enqueue_nd_range_kernel(queue, kernel, global_work_size, local_work_size, global_work_offset=None, wait_for=None, g_times_l=False)

    |std-enqueue-blurb|

    If *g_times_l* is specified, the global size will be multiplied by the
    local size. (which makes the behavior more like Nvidia CUDA) In this case,
    *global_size* and *local_size* also do not have to have the same number
    of dimensions.

    .. versionchanged:: 2011.1
        Added the *g_times_l* keyword arg.


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

    .. versionadded:: 2011.1

.. class:: GLBuffer(context, flags, bufobj)

    :class:`GLBuffer` inherits from :class:`MemoryObject`.

    .. attribute:: gl_object

.. class:: GLRenderBuffer(context, flags, bufobj)

    :class:`GLRenderBuffer` inherits from :class:`MemoryObject`.

    .. attribute:: gl_object

.. class:: GLTexture(context, flags, texture_target, miplevel, texture, dims)

    *dims* is either 2 or 3.
    :class:`GLTexture` inherits from :class:`Image`.

    .. attribute:: gl_object

    .. method:: get_gl_texture_info(param)

        See :class:`gl_texture_info` for values of *param*.  Only available when PyOpenCL is compiled with GL support. See :func:`have_gl`.

.. function:: enqueue_acquire_gl_objects(queue, mem_objects, wait_for=None)

    *mem_objects* is a list of :class:`MemoryObject` instances.
    |std-enqueue-blurb|

.. function:: enqueue_release_gl_objects(queue, mem_objects, wait_for=None)

    *mem_objects* is a list of :class:`MemoryObject` instances. |std-enqueue-blurb|

.. function:: get_gl_context_info_khr(properties, param_name, platform=None)

    Get information on which CL device corresponds to a given
    GL/EGL/WGL/CGL device.

    See the :class:`Context` constructor for the meaning of
    *properties* and :class:`gl_context_info` for *param_name*.


    .. versionchanged:: 2011.2
        Accepts the *platform* argument.  Using *platform* equal to None is
        deprecated as of PyOpenCL 2011.2.
