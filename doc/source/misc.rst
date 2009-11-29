Installation
============

Installation information is maintained collaboratively on the 
`PyOpenCL Wiki <http://wiki.tiker.net/PyOpenCL/Installation>`_.

Acknowledgments
===============

* James Snyder provided patches to make PyOpenCL work on OS X 10.6.
* Roger Pau Monné supplied the example :file:`examples/benchmark-all.py`.
* David Garcia contributed significantly to PyOpenCL's API design
  and reported many bugs.
* Michal Januszewski sent a patch.
* Achim Gottinger submitted a fix for an example.
* Andrew Karpushin provided a fix for a whole class of crash bugs in
  PyOpenCL.

Guidelines
==========

.. _api-compatibility:

API Stability
-------------

I consider PyOpenCL's API "stable".  That doesn't mean it can't
change. But if it does, your code will generally continue to run. It
may however start spewing warnings about things you need to change to
stay compatible with future versions.

Deprecation warnings will be around for a whole release cycle, as
identified by the second number in the release name.  (the "90" in
"0.90") Further, the stability promise applies for any code that's
part of a released version. It doesn't apply to undocumented bits of
the API, and it doesn't apply to unreleased code downloaded from git.

.. _versus-c:

Relation with OpenCL's C Bindings
---------------------------------

We've tried to follow these guidelines when binding the OpenCL's
C interface to Python:

* Remove the `cl_`, `CL_` and `cl` prefix from data types, macros and
  function names.
* Follow :pep:`8`, i.e.

  * Make function names lowercase.
  * If a data type or function name is composed of more than one word,
    separate the words with a single underscore.

* `get_info` functions become attributes.
* Object creation is done by constructors, to the extent possible.
  (i.e. minimize use of "factory functions")

* If an operation involves two or more "complex" objects (like e.g. a
  kernel enqueue involves a kernel and a queue), refuse the temptation 
  to guess which one should get a method for the operation.
  Instead, simply leave that command to be a function.

User-visible Changes
====================

Version 0.92
------------

.. note::

    Version 0.92 is currently under development. You can get snapshots from
    PyOpenCL's git version control.

Version 0.91.4
--------------

A bugfix release. No user-visible changes.

Version 0.91.3
--------------

* All parameters named *host_buffer* were renamed *hostbuf* for consistency
  with the :class:`pyopencl.Buffer` constructor introduced in 0.91.
  Compatibility code is in place.
* The :class:`pyopencl.Image` constructor does not need a *shape* parameter if the 
  given *hostbuf* has *hostbuf.shape*.
* The :class:`pyopencl.Context` constructor can now be called without parameters.

Version 0.91.2
--------------

* :meth:`pyopencl.Program.build` now captures build logs and adds them
  to the exception text.
* Deprecate :func:`pyopencl.create_context_from_type` in favor of second
  form of :class:`pyopencl.Context` constructor
* Introduce :class:`pyopencl.LocalMemory`.
* Document kernel invocation and :meth:`pyopencl.Kernel.set_arg`.

Version 0.91.1
--------------

* Fixed a number of bugs, notably involving :class:`pyopencl.Sampler`.
* :class:`pyopencl.Device`, :class:`pyopencl.Platform`,
  :class:`pyopencl.Context` now have nicer string representations.
* Add :attr:`Image.shape`. (suggested by David Garcia)

Version 0.91
------------

* Add :ref:`gl-interop`.
* Add a test suite.
* Fix numerous `get_info` bugs. (reports by David Garcia and the test suite)
* Add :meth:`pyopencl.ImageFormat.__repr__`.
* Add :meth:`pyopencl.addressing_mode.to_string` and colleagues.
* The `pitch` arguments to 
  :func:`pyopencl.create_image_2d`,
  :func:`pyopencl.create_image_3d`,
  :func:`pyopencl.enqueue_read_image`, and
  :func:`pyopencl.enqueue_write_image`
  are now defaulted to zero. The argument order of `enqueue_{read,write}_image`
  has changed for this reason.
* Deprecate
  :func:`pyopencl.create_image_2d`,
  :func:`pyopencl.create_image_3d`
  in favor of the :class:`pyopencl.Image` constructor.
* Deprecate
  :func:`pyopencl.create_program_with_source`,
  :func:`pyopencl.create_program_with_binary`
  in favor of the :class:`pyopencl.Program` constructor.
* Deprecate
  :func:`pyopencl.create_buffer`,
  :func:`pyopencl.create_host_buffer`
  in favor of the :class:`pyopencl.Buffer` constructor.
* :meth:`pyopencl.MemoryObject.get_image_info` now actually exists.
* Add :attr:`pyopencl.MemoryObject.image.info`.
* Fix API tracing.
* Add constructor arguments to :class:`pyopencl.ImageFormat`.  (suggested by David Garcia) 

Version 0.90.4
--------------

* Add build fixes for Windows and OS X.

Version 0.90.3
--------------

* Fix a GNU-ism in the C++ code of the wrapper.

Version 0.90.2
--------------

* Fix :meth:`pyopencl.Platform.get_info`.
* Fix passing properties to :class:`pyopencl.CommandQueue`.
  Also fix related documentation.

Version 0.90.1
--------------

* Fix building on the Mac.

Version 0.90
------------

* Initial release.

.. _license:

Licensing
=========

PyOpenCL is licensed to you under the MIT/X Consortium license:

Copyright (c) 2009 Andreas Klöckner and Contributors.

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

Frequently Asked Questions
==========================

The FAQ is maintained collaboratively on the 
`Wiki FAQ page <http://wiki.tiker.net/PyOpenCL/FrequentlyAskedQuestions>`_.

