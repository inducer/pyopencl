Installation
============

Installation information is maintained collaboratively on the
`PyOpenCL Wiki <http://wiki.tiker.net/PyOpenCL/Installation>`_.

Tips
====

Syntax highlighting
-------------------

You can obtain Vim syntax highlighting for OpenCL C inlined in Python by
checking `this file
<https://github.com/pyopencl/pyopencl/blob/master/contrib/pyopencl.vim>`_.

Note that the triple-quoted strings containing the source must start with
`"""//CL// ..."""`.

IPython integration
-------------------

PyOpenCL comes with IPython integration, which lets you seamlessly integrate
PyOpenCL kernels into your IPython notebooks. Simply load the PyOpenCL 
IPython extension using::

    %load_ext pyopencl.ipython_ext

and then use the ``%%cl_kernel`` 'cell-magic' command. See `this notebook
<http://nbviewer.ipython.org/urls/raw.githubusercontent.com/pyopencl/pyopencl/master/examples/ipython-demo.ipynb>`_
(which ships with PyOpenCL) for a demonstration.

You can pass build options to be used for building the program executable by using the ``-o`` flag on the first line of the cell (next to the ``%%cl_kernel`` directive). For example: `%%cl_kernel -o "-cl-fast-relaxed-math"``.

There are also line magics: ``cl_load_edit_kernel`` which will load a file into the next cell (adding ``cl_kernel`` to the first line) and ``cl_kernel_from_file`` which will compile kernels from a file (as if you copy-and-pasted the contents of the file to a cell with ``cl_kernel``). Both of these magics take options ``-f`` to specify the file and optionally ``-o`` for build options.

.. versionadded:: 2014.1

Guidelines
==========

.. _api-compatibility:

API Stability
-------------

I consider PyOpenCL's API "stable".  That doesn't mean it can't
change. But if it does, your code will generally continue to run. It
may however start spewing warnings about things you need to change to
stay compatible with future versions.

Deprecation warnings will be around for a whole year, as identified by the
first number in the release name.  (the "2014" in "2014.1") I.e. a function
that was deprecated in 2014.n will generally be removed in 2015.n (or perhaps
later). Further, the stability promise applies for any code that's part of a
released version. It doesn't apply to undocumented bits of the API, and it
doesn't apply to unreleased code downloaded from git.

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

.. _interoperability:

Interoperability with other OpenCL software
-------------------------------------------

Just about every object in :mod:`pyopencl` supports the following
interface (here shown as an example for :class:`pyopencl.MemoryObject`,
from which :class:`pyopencl.Buffer` and :class:`pyopencl.Image` inherit):

* :meth:`pyopencl.MemoryObject.from_int_ptr`
* :attr:`pyopencl.MemoryObject.int_ptr`

This allows retrieving the C-level pointer to an OpenCL object as a Python
integer, which may then be passed to other C libraries whose interfaces expose
OpenCL objects. It also allows turning C-level OpenCL objects obtained from
other software to be turned into the corresponding :mod:`pyopencl` objects.

.. versionadded:: 2013.2

User-visible Changes
====================

Version 2016.3
--------------

.. note::

    This version is currently under development. You can get snapshots from
    PyOpenCL's `git repository <https://github.com/pyopencl/pyopencl>`_

Version 2016.2
--------------

* Deprecate RANLUXCL. It will be removed in the 2018.x series of PyOpenCL.
* Introduce Random123 random number generators. See :mod:`pyopencl.clrandom`
  for more information.
* Add support for **range** and **slice** kwargs and data-less reductions
  to :class:`pyopencl.reduction.ReductionKernel`.
* Add support for SPIR-V. (See :class:`pyopencl.Program`.)
* Add support for :ref:`svm`.
* :class:`pyopencl.MemoryMap` is usable as a context manager.

Version 2016.1
--------------

* The ``from_int_ptr`` methods now take a *retain* parameter for more convenient
  ownership management.
* Kernel build options (if passed as a list) are now properly quoted.
  (This is a potentially compatibility-breaking change.)
* Many bug fixes. (GL interop, Windows, event callbacks and more)

Version 2015.2.4
----------------

* Fix building on Windows, using mingwpy and VS 2015.

Version 2015.2.3
----------------

* Fix one more Ubuntu 14.x build issue.

Version 2015.2.2
----------------

* Fix compatibility with CL 1.1
* Fix compatibility with Ubuntu 14.x.
* Various bug fixes

Version 2015.2.1
----------------

* Fix global_offset kernel launch parameter

Version 2015.2
--------------

* **[INCOMPATIBLE]** Changed PyOpenCL's complex numbers from ``float2`` and
  ``double2`` OpenCL vector types to custom ``struct``. This was changed
  because it very easily introduced bugs where

  * complex*complex
  * real+complex

  *look* like they may do the right thing, but silently do the wrong thing.
* Rewrite of the wrapper layer to be based on CFFI
* Pypy compatibility
* Faster kernel invocation through Python launcher code generation
* POCL compatibility

Version 2015.1
--------------

* Support for new-style buffer protocol
* Numerous fixes

Version 2014.1
--------------

* :ref:`ipython-integration`
* Bug fixes

Version 2013.2
--------------

* Add :meth:`pyopencl.array.Array.map_to_host`.
* Support *strides* on :func:`pyopencl.enqueue_map_buffer` and
  :func:`pyopencl.enqueue_map_image`.
* :class:`pyopencl.ImageFormat` was made comparable and hashable.
* :mod:`pyopencl.reduction` supports slicing (contributed by Alex Nitz)
* Added :ref:`interoperability`
* Bug fixes

Version 2013.1
--------------

* Vastly improved :ref:`custom-scan`.
* Add :func:`pyopencl.tools.match_dtype_to_c_struct`,
  for better integration of the CL and :mod:`numpy` type systems.
* More/improved Bessel functions.
  See `the source <https://github.com/pyopencl/pyopencl/tree/master/src/cl>`_.
* Add :envvar:`PYOPENCL_NO_CACHE` environment variable to aid debugging.
  (e.g. with AMD's CPU implementation, see
  `their programming guide <http://developer.amd.com/sdks/AMDAPPSDK/assets/AMD_Accelerated_Parallel_Processing_OpenCL_Programming_Guide.pdf>`_)
* Deprecated :func:`pyopencl.tools.register_dtype` in favor of
  :func:`pyopencl.tools.get_or_register_dtype`.
* Clean up the :class:`pyopencl.array.Array` constructor interface.
* Deprecate :class:`pyopencl.array.DefaultAllocator`.
* Deprecate :class:`pyopencl.tools.CLAllocator`.
* Introduce :class:`pyopencl.tools.DeferredAllocator`, :class:`pyopencl.tools.ImmediateAllocator`.
* Allow arrays whose beginning does not coincide with the beginning of their
  :attr:`pyopencl.array.Array.data` :class:`pyopencl.Buffer`.
  See :attr:`pyopencl.array.Array.base_data` and :attr:`pyopencl.array.Array.offset`.
  Note that not all functions in PyOpenCL support such arrays just yet. These
  will fail with :exc:`pyopencl.array.ArrayHasOffsetError`.
* Add :meth:`pyopencl.array.Array.__getitem__` and :meth:`pyopencl.array.Array.__setitem__`,
  supporting generic slicing.

  It is *possible* to create non-contiguous arrays using this functionality.
  Most operations (elementwise etc.) will not work on such arrays.

  Note also that some operations (specifically, reductions and scans) on sliced
  arrays that start past the beginning of the original array will fail for now.
  This will be fixed in a future release.

* :class:`pyopencl.CommandQueue` may be used as a context manager (in a ``with`` statement)
* Add :func:`pyopencl.clmath.atan2`, :func:`pyopencl.clmath.atan2pi`.
* Add :func:`pyopencl.array.concatenate`.
* Add :meth:`pyopencl.Kernel.capture_call`.

.. note::

    The addition of :meth:`pyopencl.array.Array.__getitem__` has an unintended
    consequence due to `numpy bug 3375
    <https://github.com/numpy/numpy/issues/3375>`_.  For instance, this
    expression::

        numpy.float32(5) * some_pyopencl_array

    may take a very long time to execute. This is because :mod:`numpy` first
    builds an object array of (compute-device) scalars (!) before it decides that
    that's probably not such a bright idea and finally calls
    :meth:`pyopencl.array.Array.__rmul__`.

    Note that only left arithmetic operations of :class:`pyopencl.array.Array`
    by :mod:`numpy` scalars are affected. Python's number types (:class:`float` etc.)
    are unaffected, as are right multiplications.

    If a program that used to run fast suddenly runs extremely slowly, it is
    likely that this bug is to blame.

    Here's what you can do:

    * Use Python scalars instead of :mod:`numpy` scalars.
    * Switch to right multiplications if possible.
    * Use a patched :mod:`numpy`. See the bug report linked above for a pull
      request with a fix.
    * Switch to a fixed version of :mod:`numpy` when available.

Version 2012.1
--------------

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
  Allow the creation of arrays in C and Fortran order.
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

License
=======

.. include:: ../LICENSE


Frequently Asked Questions
==========================

The FAQ is maintained collaboratively on the
`Wiki FAQ page <http://wiki.tiker.net/PyOpenCL/FrequentlyAskedQuestions>`_.

Citing PyOpenCL
===============

We are not asking you to gratuitously cite PyOpenCL in work that is otherwise
unrelated to software. That said, if you do discuss some of the development
aspects of your code and would like to highlight a few of the ideas behind
PyOpenCL, feel free to cite `this article
<http://dx.doi.org/10.1016/j.parco.2011.09.001>`_:

    Andreas Klöckner, Nicolas Pinto, Yunsup Lee, Bryan Catanzaro, Paul Ivanov,
    Ahmed Fasih, PyCUDA and PyOpenCL: A scripting-based approach to GPU
    run-time code generation, Parallel Computing, Volume 38, Issue 3, March
    2012, Pages 157-174.

Here's a Bibtex entry for your convenience::

    @article{kloeckner_pycuda_2012,
       author = {{Kl{\"o}ckner}, Andreas
            and {Pinto}, Nicolas
            and {Lee}, Yunsup
            and {Catanzaro}, B.
            and {Ivanov}, Paul
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

Acknowledgments
===============

Contributors
------------

Too many to list. Please see the
`commit log <https://github.com/pyopencl/pyopencl/commits/master>`_
for detailed acknowledgments.

Funding
-------

Andreas Klöckner's work on :mod:`pyopencl` was supported in part by

* US Navy ONR grant number N00014-14-1-0117
* the US National Science Foundation under grant numbers DMS-1418961 and CCF-1524433.

AK also gratefully acknowledges a hardware gift from Nvidia Corporation.  The
views and opinions expressed herein do not necessarily reflect those of the
funding agencies.
