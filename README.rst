PyOpenCL: Pythonic Access to OpenCL, with Arrays and Algorithms
===============================================================

.. image:: https://gitlab.tiker.net/inducer/pyopencl/badges/main/pipeline.svg
    :alt: Gitlab Build Status
    :target: https://gitlab.tiker.net/inducer/pyopencl/commits/main
.. image:: https://github.com/inducer/pyopencl/workflows/CI/badge.svg?branch=main&event=push
    :alt: Github Build Status
    :target: https://github.com/inducer/pyopencl/actions?query=branch%3Amain+workflow%3ACI+event%3Apush
.. image:: https://badge.fury.io/py/pyopencl.svg
    :alt: Python Package Index Release Page
    :target: https://pypi.org/project/pyopencl/
.. image:: https://zenodo.org/badge/1575307.svg
    :alt: Zenodo DOI for latest release
    :target: https://zenodo.org/badge/latestdoi/1575307

PyOpenCL lets you access GPUs and other massively parallel compute
devices from Python. It tries to offer computing goodness in the
spirit of its sister project `PyCUDA <https://mathema.tician.de/software/pycuda>`__:

* Object cleanup tied to lifetime of objects. This idiom, often
  called `RAII <https://en.wikipedia.org/wiki/Resource_Acquisition_Is_Initialization>`__
  in C++, makes it much easier to write correct, leak- and
  crash-free code.

* Completeness. PyOpenCL puts the full power of OpenCL's API at
  your disposal, if you wish.  Every obscure ``get_info()`` query and
  all CL calls are accessible.

* Automatic Error Checking. All CL errors are automatically
  translated into Python exceptions.

* Speed. PyOpenCL's base layer is written in C++, so all the niceties
  above are virtually free.

* Helpful and complete `Documentation <https://documen.tician.de/pyopencl>`__
  as well as a `Wiki <https://wiki.tiker.net/PyOpenCL>`__.

* Liberal license. PyOpenCL is open-source under the
  `MIT license <https://en.wikipedia.org/wiki/MIT_License>`__
  and free for commercial, academic, and private use.

* Broad support. PyOpenCL was tested and works with Apple's, AMD's, and Nvidia's
  CL implementations.

Simple 4-step `install instructions <https://documen.tician.de/pyopencl/misc.html#installation>`__
using Conda on Linux and macOS (that also install a working OpenCL implementation!)
can be found in the `documentation <https://documen.tician.de/pyopencl/>`__.

What you'll need if you do *not* want to use the convenient instructions above and
instead build from source:

* gcc/g++ new enough to be compatible with pybind11
  (see their `FAQ <https://pybind11.readthedocs.io/en/stable/faq.html>`__)
* `numpy <https://numpy.org>`__, and
* an OpenCL implementation. (See this `howto <https://wiki.tiker.net/OpenCLHowTo>`__
  for how to get one.)

Links
-----

* `Documentation <https://documen.tician.de/pyopencl>`__
  (read how things work)
* `Conda Forge <https://anaconda.org/conda-forge/pyopencl>`__
  (download binary packages for Linux, macOS, Windows)
* `Python package index <https://pypi.python.org/pypi/pyopencl>`__
  (download releases)
* `C. Gohlke's Windows binaries <https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl>`__
  (download Windows binaries)
* `Github <https://github.com/inducer/pyopencl>`__
  (get latest source code, file bugs)
