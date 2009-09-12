Installation
============

Installation information is maintained collaboratively on the 
`PyOpenCL Wiki <http://wiki.tiker.net/PyOpenCL/Installation>`_.

Acknowledgments
===============

* James Snyder provided patches to make PyOpenCL work on OS X 10.6.
* Roger Pau Monné supplied the example :file:`examples/benchmark-all.py`.

User-visible Changes
====================

Version 0.91
------------

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
* :meth:`pyopencl.MemoryObject.get_image_info` now actually exists.
* Add :meth:`pyopencl.MemoryObject.image`.

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

