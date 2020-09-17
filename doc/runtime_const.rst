OpenCL Runtime: Constants
=========================

.. currentmodule:: pyopencl

.. include:: constants.inc

.. class:: NameVersion
    Describes the version of a specific feature.

    .. note::

        Only available with OpenCL 3.0 or newer.

    .. versionadded:: 2020.3

    .. method:: __init__(version, name)
    .. attribute:: version
    .. attribute:: name

.. class:: DeviceTopologyAmd
    .. method:: __init__(bus, device, function)
    .. attribute:: type
    .. attribute:: bus
    .. attribute:: device
    .. attribute:: function

.. vim: shiftwidth=4
