PyOpenCL: Pythonic Access to OpenCL, with Arrays and Algorithms
---------------------------------------------------------------

.. image:: https://gitlab.tiker.net/inducer/pyopencl/badges/master/pipeline.svg
    :alt: Gitlab Build Status
    :target: https://gitlab.tiker.net/inducer/pyopencl/commits/master
.. image:: https://dev.azure.com/ak-spam/inducer/_apis/build/status/inducer.pyopencl?branchName=master
    :alt: Azure Build Status
    :target: https://dev.azure.com/ak-spam/inducer/_build/latest?definitionId=5&branchName=master
.. image:: https://badge.fury.io/py/pyopencl.png
    :alt: Python Package Index Release Page
    :target: https://pypi.org/project/pyopencl/

(Also: `Travis CI <https://travis-ci.org/inducer/pyopencl/builds>`_ to build binary wheels for releases, see `#264 <https://github.com/inducer/pyopencl/pull/264>`_)

PyOpenCL lets you access GPUs and other massively parallel compute
devices from Python. It tries to offer computing goodness in the
spirit of its sister project `PyCUDA <https://mathema.tician.de/software/pycuda>`_:

* Object cleanup tied to lifetime of objects. This idiom, often
  called
  `RAII <https://en.wikipedia.org/wiki/Resource_Acquisition_Is_Initialization>`_
  in C++, makes it much easier to write correct, leak- and
  crash-free code.

* Completeness. PyOpenCL puts the full power of OpenCL's API at
  your disposal, if you wish.  Every obscure `get_info()` query and 
  all CL calls are accessible.

* Automatic Error Checking. All CL errors are automatically
  translated into Python exceptions.

* Speed. PyOpenCL's base layer is written in C++, so all the niceties
  above are virtually free.

* Helpful and complete `Documentation <https://documen.tician.de/pyopencl>`__
  as well as a `Wiki <https://wiki.tiker.net/PyOpenCL>`_.

* Liberal license. PyOpenCL is open-source under the 
  `MIT license <https://en.wikipedia.org/wiki/MIT_License>`_
  and free for commercial, academic, and private use.

* Broad support. PyOpenCL was tested and works with Apple's, AMD's, and Nvidia's 
  CL implementations.

Simple 4-step `install instructions <https://documen.tician.de/pyopencl/misc.html#installation>`_
using Conda on Linux and macOS (that also install a working OpenCL implementation!)
can be found in the `documentation <https://documen.tician.de/pyopencl/>`__.

What you'll need if you do *not* want to use the convenient instructions above and
instead build from source:

*   gcc/g++ new enough to be compatible with pybind11
    (see their `FAQ <https://pybind11.readthedocs.io/en/stable/faq.html>`_)
*   `numpy <https://numpy.org>`_, and
*   an OpenCL implementation. (See this `howto <https://wiki.tiker.net/OpenCLHowTo>`_ for how to get one.)

Places on the web related to PyOpenCL:

* `Python package index <https://pypi.python.org/pypi/pyopencl>`_ (download releases)

* `Documentation <https://documen.tician.de/pyopencl>`__ (read how things work)
* `Conda Forge <https://anaconda.org/conda-forge/pyopencl>`_ (download binary packages for Linux, macOS, Windows)
* `C. Gohlke's Windows binaries <https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl>`_ (download Windows binaries)
* `Github <https://github.com/inducer/pyopencl>`_ (get latest source code, file bugs)
* `Wiki <https://wiki.tiker.net/PyOpenCL>`_ (read installation tips, get examples, read FAQ)
