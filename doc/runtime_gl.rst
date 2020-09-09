.. include:: subst.rst

.. _gl-interop:

OpenCL Runtime: OpenGL Interoperability
=======================================

.. currentmodule:: pyopencl

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

    :class:`GLTexture` inherits from :class:`Image`. Only available in OpenCL 1.2
    and newer.

    .. attribute:: gl_object

    .. method:: get_gl_texture_info(param)

        See ``gl_texture_info`` for values of *param*.  Only available when PyOpenCL is compiled with GL support. See :func:`have_gl`.

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
