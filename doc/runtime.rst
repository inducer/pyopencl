.. _reference-doc:

.. include:: subst.rst

OpenCL Runtime: Basics
======================

Version Queries
---------------

.. module:: pyopencl
.. moduleauthor:: Andreas Kloeckner <inform@tiker.net>

.. data:: VERSION

    Gives the numeric version of PyOpenCL as a variable-length tuple
    of integers. Enables easy version checks such as
    ``VERSION >= (0, 93)``.

.. data:: VERSION_STATUS

    A text string such as ``"rc4"`` or ``"beta"`` qualifying the status
    of the release.

.. data:: VERSION_TEXT

    The full release name (such as ``"0.93rc4"``) in string form.

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

References
----------

These are only here because Sphinx, our documentation tool, struggles to
resolve them.

.. class:: WaitList

    A :class:`Sequence` of :class:`Event`\ s, or None.

.. class:: NDArray

    See :data:`numpy.typing.NDArray`.

.. class:: DTypeLike

    See :data:`numpy.typing.DTypeLike`.

.. class:: SVMInnerT

    A type variable for the object wrapped by an :class:`SVM`.

.. class:: RetT

    A generic type variable, used for a return type.

.. class:: P

    A :class:`~typing.ParamSpec`.

.. currentmodule:: pyopencl.typing

.. class:: DTypeT

    A type variable for a :class:`numpy.dtype`.

.. currentmodule:: pyopencl._cl

.. class:: svm_mem_flags

    See :class:`pyopencl.svm_mem_flags`.

.. currentmodule:: cl

.. class:: WaitList

    A :class:`Sequence` of :class:`Event`\ s, or None.

.. class:: Context

    See :class:`pyopencl.CommandQueue`.

.. class:: CommandQueue

    See :class:`pyopencl.CommandQueue`.

.. class:: Event

    See :class:`pyopencl.Event`.

.. currentmodule:: cl_tools
..
.. class:: AllocatorBase

    See :class:`pyopencl.tools.AllocatorBase`.

