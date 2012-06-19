Installation
============

Installation information is maintained collaboratively on the
`PyOpenCL Wiki <http://wiki.tiker.net/PyOpenCL/Installation>`_.

Acknowledgments
===============

Too many to list. Please see the
`commit log <https://github.com/inducer/pyopencl/commits/master>`_
for detailed acknowledgments.

Tips
====

Syntax highlighting
-------------------

You can obtain Vim syntax highlighting for OpenCL C inlined in Python by
checking `this file
<https://github.com/inducer/pyopencl/blob/master/contrib/pyopencl.vim>`_.

Note that the triple-quoted strings containing the source must start with
`"""//CL// ..."""`.

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

Version 2012.1
----------------

.. note::

    This version is currently under development. You can get snapshots from
    PyOpenCL's git version control.

* Support for complex numbers.
* Support for Bessel functions. (experimental)
* Numerous fixes.

Version 2011.2
--------------

* Add :func:`pyopencl.enqueue_migrate_mem_object`.
* Add :func:`pyopencl.image_from_array`.
* IMPORTANT BUGFIX: Kernel caching was broken for all the 2011.1.x releases, with
  severe consequences on the execution time of :class:`pyopencl.array.Array`
  operations.
  Henrik Andresen at a `PyOpenCL workshop at DTU <http://gpulab.imm.dtu.dk/courses.html>`_
  first noticed the strange timings.
* All comparable PyOpenCL objects are now also hashable.
* Add :func:`pyopencl.tools.context_dependent_memoize` to the documented
  functionality.
* Base :mod:`pyopencl.clrandom` on `RANLUXCL <https://bitbucket.org/ivarun/ranluxcl>`_,
  add functionality.
* Add :class:`pyopencl.NannyEvent` objects.
* Add :mod:`pyopencl.characterize`.
* Ensure compatibility with OS X Lion.
* Add :func:`pyopencl.tools.register_dtype` to enable scan/reduction on struct types.
* :func:`pyopencl.enqueue_migrate_mem_object` was renamed
  :func:`pyopencl.enqueue_migrate_mem_object_ext`.
  :func:`pyopencl.enqueue_migrate_mem_object` now refers to the OpenCL 1.2 function
  of this name, if available.
* :func:`pyopencl.create_sub_devices` was renamed
  :func:`pyopencl.create_sub_devices_ext`.
  :func:`pyopencl.create_sub_devices` now refers to the OpenCL 1.2 function
  of this name, if available.
* Alpha support for OpenCL 1.2.

Version 2011.1.2
----------------

* More bug fixes.

Version 2011.1.1
----------------

* Fixes for Python 3 compatibility. (with work by Christoph Gohlke)

Version 2011.1
--------------

* All *is_blocking* parameters now default to *True* to avoid
  crashy-by-default behavior. (suggested by Jan Meinke)
  In particular, this change affects
  :func:`pyopencl.enqueue_read_buffer`,
  :func:`pyopencl.enqueue_write_buffer`,
  :func:`pyopencl.enqueue_read_buffer_rect`,
  :func:`pyopencl.enqueue_write_buffer_rect`,
  :func:`pyopencl.enqueue_read_image`,
  :func:`pyopencl.enqueue_write_image`,
  :func:`pyopencl.enqueue_map_buffer`,
  :func:`pyopencl.enqueue_map_image`.
* Add :mod:`pyopencl.reduction`.
* Add :ref:`reductions`.
* Add :mod:`pyopencl.scan`.
* Add :meth:`pyopencl.MemoryObject.get_host_array`.
* Deprecate context arguments of
  :func:`pyopencl.array.to_device`,
  :func:`pyopencl.array.zeros`,
  :func:`pyopencl.array.arange`.
* Make construction of :class:`pyopencl.array.Array` more flexible (*cqa* argument.)
* Add :ref:`memory-pools`.
* Add vector types, see :class:`pyopencl.array.vec`.
* Add :attr:`pyopencl.array.Array.strides`, :attr:`pyopencl.array.Array.flags`.
  Allow the creation of arrys in C and Fortran order.
* Add :func:`pyopencl.enqueue_copy`. Deprecate all other transfer functions.
* Add support for numerous extensions, among them device fission.
* Add a compiler cache.
* Add the 'g_times_l' keyword arg to kernel execution.

Version 0.92
------------

* Add support for OpenCL 1.1.
* Add support for the
  `cl_khr_gl_sharing <ghttp://www.khronos.org/registry/cl/extensions/khr/cl_khr_gl_sharing.txt>`_
  extension, leading to working GL interoperability.
* Add :meth:`pyopencl.Kernel.set_args`.
* The call signature of :meth:`pyopencl.Kernel.__call__` changed to
  emphasize the importance of *local_size*.
* Add :meth:`pyopencl.Kernel.set_scalar_arg_dtypes`.
* Add support for the
  `cl_nv_device_attribute_query <http://www.khronos.org/registry/cl/extensions/khr/cl_nv_device_attribute_query.txt>`_
  extension.
* Add :meth:`pyopencl.array.Array` and related functionality.
* Make build not depend on Boost C++.

Version 0.91.5
--------------

* Add :attr:`pyopencl.ImageFormat.channel_count`,
  :attr:`pyopencl.ImageFormat.dtype_size`,
  :attr:`pyopencl.ImageFormat.itemsize`.
* Add missing :func:`pyopencl.enqueue_copy_buffer`.
* Add :func:`pyopencl.create_some_context`.
* Add :func:`pyopencl.enqueue_barrier`, which was previously missing.

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

Copyright (c) 2009-11 Andreas Klöckner and Contributors.

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

PyOpenCL includes derivatives of parts of the `Thrust
<https://code.google.com/p/thrust/>`_ computing package (in particular the scan
implementation). These parts are licensed as follows:

    Copyright 2008-2011 NVIDIA Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        <http://www.apache.org/licenses/LICENSE-2.0>

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

.. note::

    If you use Apache-licensed parts, be aware that these may be incompatible
    with software licensed exclusively under GPL2.  (Most software is licensed
    as GPL2 or later, in which case this is not an issue.)

PyOpenCL includes the RANLUXCL random number generator:

    Copyright (c) 2011 Ivar Ursin Nikolaisen

    Permission is hereby granted, free of charge, to any person obtaining a copy of this
    software and associated documentation files (the "Software"), to deal in the Software
    without restriction, including without limitation the rights to use, copy, modify,
    merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following
    conditions:

    The above copyright notice and this permission notice shall be included in all copies
    or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
    PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
    OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Frequently Asked Questions
==========================

The FAQ is maintained collaboratively on the
`Wiki FAQ page <http://wiki.tiker.net/PyOpenCL/FrequentlyAskedQuestions>`_.

Citing PyOpenCL
===============

We are not asking you to gratuitously cite PyOpenCL in work that is otherwise
unrelated to software. That said, if you do discuss some of the development
aspects of your code and would like to highlight a few of the ideas behind
PyOpenCL, feel free to cite this article:

    Andreas Klöckner, Nicolas Pinto, Yunsup Lee, Bryan Catanzaro, Paul Ivanov,
    Ahmed Fasih, PyCUDA and PyOpenCL: A scripting-based approach to GPU
    run-time code generation, Parallel Computing, Volume 38, Issue 3, March
    2012, Pages 157-174.

Here's a Bibtex entry for your convenience::

    @article{kloeckner_pycuda_2012,
       author = {{Kl{\"o}ckner}, Andreas·
            and {Pinto}, Nicolas·
            and {Lee}, Yunsup·
            and {Catanzaro}, B.·
            and {Ivanov}, Paul·
            and {Fasih}, Ahmed },
       title = "{PyCUDA and PyOpenCL: A Scripting-Based Approach to GPU Run-Time Code Generation}",
       journal = "Parallel Computing",
       volume = "38",
       number = "3",
       pages = "157--174",
       year = "2012",
       issn = "0167-8191",
       doi = "10.1016/j.parco.2011.09.001",
    }

