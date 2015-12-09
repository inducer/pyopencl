PyOpenCL lets you access GPUs and other massively parallel compute
devices from Python. It tries to offer computing goodness in the
spirit of its sister project `PyCUDA <http://mathema.tician.de/software/pycuda>`_:

* Object cleanup tied to lifetime of objects. This idiom, often
  called
  `RAII <http://en.wikipedia.org/wiki/Resource_Acquisition_Is_Initialization>`_
  in C++, makes it much easier to write correct, leak- and
  crash-free code.

* Completeness. PyOpenCL puts the full power of OpenCL's API at
  your disposal, if you wish.  Every obscure `get_info()` query and 
  all CL calls are accessible.

* Automatic Error Checking. All CL errors are automatically
  translated into Python exceptions.

* Speed. PyOpenCL's base layer is written in C++, so all the niceties
  above are virtually free.

* Helpful and complete `Documentation <http://documen.tician.de/pyopencl>`_
  as well as a `Wiki <http://wiki.tiker.net/PyOpenCL>`_.

* Liberal license. PyOpenCL is open-source under the 
  `MIT license <http://en.wikipedia.org/wiki/MIT_License>`_
  and free for commercial, academic, and private use.

* Broad support. PyOpenCL was tested and works with Apple's, AMD's, and Nvidia's 
  CL implementations.

What you'll need:

*   gcc/g++ at or newer than version 4.8.2 and binutils at or newer than 2.23.52.0.1-10
    (CentOS version number).
    On Windows, use the `mingwpy <https://anaconda.org/carlkl/mingwpy>`_ compilers.
*   `numpy <http://numpy.org>`_, and
*   an OpenCL implementation. (See this `howto <http://wiki.tiker.net/OpenCLHowTo>`_ for how to get one.)

Places on the web related to PyOpenCL:

* `Python package index <http://pypi.python.org/pypi/pyopencl>`_ (download releases)

  .. image:: https://badge.fury.io/py/pyopencl.png
      :target: http://pypi.python.org/pypi/pyopencl
* `C. Gohlke's Windows binaries <http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl>`_ (download Windows binaries)
* `Github <http://github.com/pyopencl/pyopencl>`_ (get latest source code, file bugs)
* `Documentation <http://documen.tician.de/pyopencl>`_ (read how things work)
* `Wiki <http://wiki.tiker.net/PyOpenCL>`_ (read installation tips, get examples, read FAQ)
