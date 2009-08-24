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
.. |enqueue-waitfor| replace:: Returns a new :class:`Event`.

.. function:: get_platforms()

    Return a list of :class:`Platform` instances.

.. class:: Platform

    .. method:: get_info(param)

        See :class:`platform_info` for values of *param*.

    .. attribute:: info

        Lower case versions of the :class:`platform_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    .. method:: get_devices(device_type)

        Return a list of devices matching *device_type*.
        See :class:`device_type` for values of *device_type*.

    |comparable|

.. class:: Device

    .. method:: get_info(param)

        See :class:`device_info` for values of *param*.

    .. attribute:: info

        Lower case versions of the :class:`device_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    Two instances of this class may be compared using *=="* and *"!="*.

.. class:: Context(devices, properties=[])

    Create a new context. *properties* is a list of key-value
    tuples, where each key must be one of :class:`context_properties`.

    .. method:: get_info(param)

        See :class:`context_info` for values of *param*.

    .. attribute:: info

        Lower case versions of the :class:`context_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    |comparable|

.. function:: create_context_from_type(dev_type, properties=[])

Command Queues and Events
-------------------------

.. class:: CommandQueue(context, device=None, properties=[])

    Create a new command queue. *properties* is a list of key-value
    tuples, where each key must be one of :class:`command_queue_properties`.

    if *device* is None, one of the devices in *context* is chosen
    in an implementation-defined manner.

    .. method:: get_info(param)

        See :class:`command_queue_info` for values of *param*.

    .. attribute:: info

        Lower case versions of the :class:`command_queue_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    .. method:: set_property(prop, enable)

        See :class:`command_queue_properties` for possible values of *prop*.

    .. method:: flush()
    .. method:: finish()

    |comparable|

.. class:: Event

    .. method:: get_info(param)

        See :class:`event_info` for values of *param*.

    .. attribute:: info

        Lower case versions of the :class:`event_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    .. method:: get_profiling_info(param)

        See :class:`profiling_info` for values of *param*.

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

    .. method:: get_info(param)

        See :class:`mem_info` for values of *param*.

    .. attribute:: info

        Lower case versions of the :class:`mem_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    .. method:: get_image_info(param)

        See :class:`image_info` for values of *param*.

    .. method:: release()

    |comparable|

Buffers
^^^^^^^

.. function:: create_buffer(context, flags, size)

    See :class:`mem_flags` for values of *flags*.
    Returns a new buffer-type :class:`MemoryObject`.

.. function:: create_host_buffer(context, flags, buffer)

    Create a buffer :class:`MemoryObject` using host memory. *host_buffer* |buf-iface|.

    See :class:`mem_flags` for values of *flags*.

.. function:: enqueue_read_buffer(queue, mem, host_buffer, device_offset=0, wait_for=None, is_blocking=False)

    *host_buffer* |buf-iface|

.. function:: enqueue_write_buffer(queue, mem, host_buffer, device_offset=0, wait_for=None, is_blocking=False)

    *host_buffer* |buf-iface|

Image Formats
^^^^^^^^^^^^^

.. class:: ImageFormat

    .. attribute:: channel_order

        See :class:`channel_order` for possible values.

    .. attribute:: channel_data_type

        See :class:`channel_type` for possible values.

.. function:: get_supported_image_formats(context, flags, image_type)

    See :class:`mem_flags` for possible values of *flags*
    and :class:`mem_object_type` for possible values of *image_type*.

Images
^^^^^^

.. function:: create_image_2d(context, flags, format, width, height, pitch, host_buffer=None)

    See :class:`mem_flags` for possible values of *flags*.
    Returns a new image-type :class:`MemoryObject`.

.. function:: create_image_3d(context, flags, format, width, height, depth, row_pitch, slice_pitch, host_buffer=None)

    See :class:`mem_flags` for possible values of *flags*.
    Returns a new image-type :class:`MemoryObject`.

.. function:: enqueue_read_image(queue, mem, origin, region, row_pitch, slice_pitch, host_buffer, wait_for=None, is_blocking=False)

    |enqueue-waitfor|

.. function:: enqueue_write_image(queue, mem, origin, region, row_pitch, slice_pitch, host_buffer, wait_for=None, is_blocking=False)

    |enqueue-waitfor|

.. function:: enqueue_copy_image(queue, src, dest, src_origin, dest_origin, region, wait_for=None)

    |enqueue-waitfor|

.. function:: enqueue_copy_image_to_buffer(queue, src, dest, origin, region, offset, wait_for=None)

    |enqueue-waitfor|

.. function:: enqueue_copy_buffer_to_image(queue, src, dest, offset, origin, region, wait_for=None)

    |enqueue-waitfor|

Mapping Memory into Host Address Space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. class:: MemoryMap

    .. method:: release(queue=None, wait_for=None)

.. function:: enqueue_map_buffer(queue, buf, flags, offset, shape, dtype, order, wait_for=None, is_blocking=False)

    |enqueue-waitfor| 
    *shape*, *dtype*, and *order* have the same meaning
    as in :func:`numpy.empty`.
    See :class:`map_flags` for possible values of *flags*.

.. function:: enqueue_map_image(queue, buf, flags, origin, region, shape, dtype, order, wait_for=None, is_blocking=False)

    |enqueue-waitfor|
    *shape*, *dtype*, and *order* have the same meaning
    as in :func:`numpy.empty`.
    See :class:`map_flags` for possible values of *flags*.

Samplers
^^^^^^^^

.. class:: Sampler(context, normalized_coords, addressing_mode, filter_mode)

    See :class:`addressing_mode` and :class:`filter_mode` for possible
    argument values.

    .. method:: get_info(param)

        See :class:`sampler_info` for values of *param*.

    .. attribute:: info

        Lower case versions of the :class:`sampler_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    |comparable|

Programs and Kernels
--------------------

.. class:: Program

    .. method:: get_info(param)

        See :class:`program_info` for values of *param*.

    .. attribute:: info

        Lower case versions of the :class:`program_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    .. method:: get_build_info(param, device)

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

.. function:: unload_compiler()
.. function:: create_program_with_source(context, src)
.. function:: create_program_with_binary(context, devices, binaries)

    *binaries* must contain one binary for each entry in *devices*.

.. class:: Kernel(program, name)

    .. method:: get_info(param)

        See :class:`kernel_info` for values of *param*.

    .. attribute:: info

        Lower case versions of the :class:`kernel_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.


    .. method:: get_work_group_info(param, device)

        See :class:`kernel_work_group_info` for values of *param*.

    .. method:: __call__(queue, global_size, *args, global_offset=None, local_size=None, wait_for=None)

        |enqueue-waitfor|


    |comparable|

.. function:: enqueue_nd_range_kernel(queue, kernel, global_work_size, local_work_size, global_work_offset=None, wait_for=None)

    |enqueue-waitfor|

.. function:: enqueue_task(queue, kernel, wait_for=None)

    |enqueue-waitfor|
