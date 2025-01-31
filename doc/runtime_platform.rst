.. include:: subst.rst

OpenCL Runtime: Platforms, Devices and Contexts
===============================================

.. currentmodule:: pyopencl

Platform
--------

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

Device
------

.. class:: Device

    Two instances of this class may be compared using *=="* and *"!="*.

    .. attribute:: info

        Lower case versions of the :class:`device_info` constants
        may be used as attributes on instances of this class
        to directly query info attributes.

    .. method:: get_info(param)

        See :class:`device_info` for values of *param*.

    .. automethod:: from_int_ptr
    .. autoattribute:: int_ptr

    .. attribute :: hashable_model_and_version_identifier

        An unspecified data type that can be used to (as precisely as possible,
        given identifying information available in OpenCL) identify a given
        model and software stack version of a compute device. Note that this
        identifier does not differentiate between different instances of the
        same device installed in a single host.

        The returned data type is hashable.

        .. versionadded:: 2020.1

    .. method:: create_sub_devices(properties)

        *properties* is an array of one (or more) of the forms::

            [ dpp.EQUALLY, 8]
            [ dpp.BY_COUNTS, 5, 7, 9, dpp.PARTITION_BY_COUNTS_LIST_END]
            [ dpp.BY_NAMES, 5, 7, 9, dpp.PARTITION_BY_NAMES_LIST_END]
            [ dpp.BY_AFFINITY_DOMAIN, dad.L1_CACHE]

        where ``dpp`` represents :class:`device_partition_property`
        and ``dad`` represent :class:`device_affinity_domain`.

        ``PROPERTIES_LIST_END_EXT`` is added automatically.

        Only available with CL 1.2.

        .. versionadded:: 2011.2

    .. method:: device_and_host_timer

        :returns: a tuple ``(device_timestamp, host_timestamp)``.

        Only available with CL 2.0.

        .. versionadded:: 2020.3

    .. method:: host_timer

        Only available with CL 2.0.

        .. versionadded:: 2020.3

.. autofunction:: choose_devices

Context
-------

.. class:: Context(devices=None, properties=None, dev_type=None)

    Create a new context. *properties* is a list of key-value
    tuples, where each key must be one of :class:`context_properties`.
    At most one of *devices* and *dev_type* may be not *None*, where
    *devices* is a list of :class:`Device` instances, and
    *dev_type* is one of the :class:`device_type` constants.
    If neither is specified, a context with a *dev_type* of
    :attr:`device_type.DEFAULT` is created.

    .. note::

        Calling the constructor with no arguments may fail for
        CL drivers that support the OpenCL ICD (which applies to most modern systems).
        If you want similar, just-give-me-a-context-already behavior, we recommend
        :func:`create_some_context`.

        See e.g. this
        `explanation by AMD
        <https://web.archive.org/web/20101114195033/https://developer.amd.com/support/KnowledgeBase/Lists/KnowledgeBase/DispForm.aspx?ID=71>`__:

            **What has changed?**

            In previous beta releases functions such as clGetDeviceIDs() and clCreateContext()
            accepted a NULL value for the platform parameter. This release no longer
            allows this - the platform must be a valid one obtained by using the platform API.

    .. note::

        Because of how OpenCL changed in order to support Installable Client
        Drivers (ICDs) in OpenCL 1.1, the following will *look* reasonable
        but often actually not work::

            import pyopencl as cl
            ctx = cl.Context(dev_type=cl.device_type.ALL)

        Instead, make sure to choose a platform when choosing a device by type::

            import pyopencl as cl

            platforms = cl.get_platforms()
            ctx = cl.Context(
                    dev_type=cl.device_type.ALL,
                    properties=[(cl.context_properties.PLATFORM, platforms[0])])

    .. note::

        For
        ``context_properties.CL_GL_CONTEXT_KHR``,
        ``context_properties.CL_EGL_DISPLAY_KHR``,
        ``context_properties.CL_GLX_DISPLAY_KHR``,
        ``context_properties.CL_WGL_HDC_KHR``, and
        ``context_properties.CL_CGL_SHAREGROUP_KHR``
        ``context_properties.CL_CGL_SHAREGROUP_APPLE``
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

    .. method:: set_default_device_command_queue(dev, queue)

    |comparable|

.. autofunction:: create_some_context
