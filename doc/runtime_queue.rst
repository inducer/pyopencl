.. include:: subst.rst

OpenCL Runtime: Command Queues and Events
=========================================

.. currentmodule:: pyopencl

Command Queue
-------------

.. class:: CommandQueue(context, device=None, properties=None)

    Create a new command queue. *properties* is a bit field
    consisting of :class:`command_queue_properties` values.

    If *device* is None, one of the devices in *context* is chosen
    in an implementation-defined manner.

    *properties* may be a bitwise combination of values from
    :class:`queue_properties` (or *None* which is equivalent to
    passing *0*). This is compatible with both OpenCL 1.x and 2.x.

    For OpenCL 2.0 and above, *properties* may also be a sequence
    of keys and values from :class:`queue_properties` as accepted
    by :c:func:`clCreateCommandQueueWithProperties` (see the OpenCL
    spec for details). The trailing *0* is added automatically
    and does not need to be included.

    A :class:`CommandQueue` may be used as a context manager, like this::

        with cl.CommandQueue(self.cl_context) as queue:
            enqueue_stuff(queue, ...)

    :meth:`finish` is automatically called at the end of the ``with``-delimited
    context.

    .. versionadded:: 2013.1

        Context manager capability.

    .. versionchanged:: 2018.2

        Added the sequence-of-properties interface for OpenCL 2.

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

Event
-----

.. class:: Event

    .. attribute:: info

        Lower case versions of the :class:`event_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    .. attribute:: profile

        An instance of :class:`ProfilingInfoGetter`.

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

.. class:: ProfilingInfoGetter

   .. attribute:: info

        Lower case versions of the :class:`profiling_info` constants
        may be used as attributes on the attribute `profile` of this
        class to directly query profiling info.

        For example, you may use *evt.profile.end* instead of
        *evt.get_profiling_info(pyopencl.profiling_info.END)*.

Event Subclasses
----------------

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

Synchronization Functions
-------------------------

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

