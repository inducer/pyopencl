.. _reference-doc:

Reference Documentation
=======================

Version Queries
---------------

.. module:: pyopencl
.. moduleauthor:: Andreas Kloeckner <inform@tiker.net>

.. data:: VERSION

    Gives the numeric version of PyCUDA as a variable-length tuple 
    of integers. Enables easy version checks such as
    *VERSION >= (0, 93)*.

.. data:: VERSION_STATUS

    A text string such as `"rc4"` or `"beta"` qualifying the status
    of the release.

.. data:: VERSION_TEXT

    The full release name (such as `"0.93rc4"`) in string form.

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

    .. versionchanged:: 0.91.2
        Constructor arguments *dev_type* added.

    .. attribute:: info

        Lower case versions of the :class:`context_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    .. method:: get_info(param)

        See :class:`context_info` for values of *param*.

    |comparable|

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

Memory
------

.. class:: MemoryObject

    .. attribute:: info

        Lower case versions of the :class:`mem_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    .. method:: get_info(param)

        See :class:`mem_info` for values of *param*.

    .. attribute:: hostbuf

        Contains the *hostbuf* parameter if the MemoryObject was constructed
        with :attr:`mem_flags.USE_HOST_PTR`.

    .. method:: release()

    |comparable|

Buffers
^^^^^^^

.. class:: Buffer(context, flags, size=0, hostbuf=None)

    Create a :class:`Buffer`.
    See :class:`mem_flags` for values of *flags*.
    If *hostbuf* is specified, *size* defaults to the size of
    the specified buffer if it is passed as zero.

    :class:`Buffer` is a subclass of :class:`MemoryObject`.

.. function:: enqueue_read_buffer(queue, mem, hostbuf, device_offset=0, wait_for=None, is_blocking=False)

    *hostbuf* |buf-iface|

.. function:: enqueue_write_buffer(queue, mem, hostbuf, device_offset=0, wait_for=None, is_blocking=False)

    *hostbuf* |buf-iface|

Image Formats
^^^^^^^^^^^^^

.. class:: ImageFormat([channel_order, channel_type])

    .. versionchanged:: 0.91
        Constructor arguments added.

    .. attribute:: channel_order

        See :class:`channel_order` for possible values.

    .. attribute:: channel_data_type

        See :class:`channel_type` for possible values.

    .. method:: __repr__

        Returns a :class:`str` representation of the image format.

        .. versionadded:: 0.91

.. function:: get_supported_image_formats(context, flags, image_type)

    See :class:`mem_flags` for possible values of *flags*
    and :class:`mem_object_type` for possible values of *image_type*.

Images
^^^^^^
.. class:: Image(context, flags, format, shape=None, pitches=None, hostbuf=None)

    *shape* is a 2- or 3-tuple.

    If *hostbuf* is given and *shape* is `None`, then *hostbuf.shape* is 
    used as the *shape* parameter.

    :class:`Image` is a subclass of :class:`MemoryObject`.

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

.. function:: enqueue_read_image(queue, mem, origin, region, hostbuf, row_pitch=0, slice_pitch=0, wait_for=None, is_blocking=False)

    |std-enqueue-blurb|

    .. versionchanged:: 0.91
        *pitch* arguments defaults to zero, moved.

.. function:: enqueue_write_image(queue, mem, origin, region, hostbuf, row_pitch=0, slice_pitch=0, wait_for=None, is_blocking=False)

    |std-enqueue-blurb|

    .. versionchanged:: 0.91
        *pitch* arguments defaults to zero, moved.

.. function:: enqueue_copy_image(queue, src, dest, src_origin, dest_origin, region, wait_for=None)

    |std-enqueue-blurb|

.. function:: enqueue_copy_image_to_buffer(queue, src, dest, origin, region, offset, wait_for=None)

    |std-enqueue-blurb|

.. function:: enqueue_copy_buffer_to_image(queue, src, dest, offset, origin, region, wait_for=None)

    |std-enqueue-blurb|

Mapping Memory into Host Address Space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: MemoryMap

    .. method:: release(queue=None, wait_for=None)

.. function:: enqueue_map_buffer(queue, buf, flags, offset, shape, dtype, order, wait_for=None, is_blocking=False)

    |explain-waitfor|
    *shape*, *dtype*, and *order* have the same meaning
    as in :func:`numpy.empty`.
    See :class:`map_flags` for possible values of *flags*.

    :return: a tuple *(array, event)*. *array* is a
        :class:`numpy.ndarray` representing the host side
        of the map. Its *.base* member contains a 
        :class:`MemoryMap`.

.. function:: enqueue_map_image(queue, buf, flags, origin, region, shape, dtype, order, wait_for=None, is_blocking=False)

    |explain-waitfor|
    *shape*, *dtype*, and *order* have the same meaning
    as in :func:`numpy.empty`.
    See :class:`map_flags` for possible values of *flags*.

    :return: a tuple *(array, event)*. *array* is a
        :class:`numpy.ndarray` representing the host side
        of the map. Its *.base* member contains a 
        :class:`MemoryMap`.


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

    .. method:: build(options="", devices=None)

        *options* is a string of compiler flags.  
        Returns *self*.

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

.. function:: unload_compiler()

.. class:: Kernel(program, name)

    .. attribute:: info

        Lower case versions of the :class:`kernel_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    .. method:: get_info(param)

        See :class:`kernel_info` for values of *param*.

    .. method:: get_work_group_info(param, device)

        See :class:`kernel_work_group_info` for values of *param*.

    .. method:: set_arg(self, arg)

        *arg* may be

        * `None`: This may be passed for `__global` memory references 
          to pass a NULL pointer to the kernel.
        * Anything that satisfies the Python buffer interface,
          in particular :class:`numpy.ndarray`, :class:`str`,
          or :mod:`numpy`'s sized scalars, such as :class:`numpy.int32`
          or :class:`numpy.float64`. 

          .. note:: 

              Note that Python's own :class:`int`
              or :class:`float` objects will not work as-is, but 
              :mod:`struct` can be used to convert them to binary 
              data in a :class:`str`, which will work.

        * An instance of :class:`MemoryObject`. (e.g. :class:`Buffer`,
          :class:`Image`, etc.)
        * An instance of :class:`LocalMemory`.
        * An instance of :class:`Sampler`.

    .. method:: __call__(queue, global_size, *args, global_offset=None, local_size=None, wait_for=None)

        Use :func:`enqueue_nd_range_kernel` to enqueue a kernel execution, after using
        :meth:`set_arg` to set each argument in turn. See the documentation for 
        :meth:`set_arg` to see what argument types are allowed.
        |std-enqueue-blurb|

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
