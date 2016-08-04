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

