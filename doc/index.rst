Welcome to PyOpenCL's documentation!
====================================

PyOpenCL gives you easy, Pythonic access to the `OpenCL
<https://www.khronos.org/opencl/>`__ parallel computation API.
What makes PyOpenCL special?

* Object cleanup tied to lifetime of objects. This idiom,
  often called
  `RAII <https://en.wikipedia.org/wiki/Resource_Acquisition_Is_Initialization>`__
  in C++, makes it much easier to write correct, leak- and
  crash-free code.

* Completeness. PyOpenCL puts the full power of OpenCL's API at your
  disposal, if you wish. Every obscure ``get_info()`` query and
  all CL calls are accessible.

* Automatic Error Checking. All errors are automatically translated
  into Python exceptions.

* Speed. PyOpenCL's base layer is written in C++, so all the niceties above
  are virtually free.

* Helpful Documentation. You're looking at it. ;)

* Liberal license. PyOpenCL is open-source under the
  :ref:`MIT license <license>`
  and free for commercial, academic, and private use.

Here's an example, to give you an impression:

.. literalinclude:: ../examples/demo.py

(You can find this example as
:download:`examples/demo.py <../examples/demo.py>` in the PyOpenCL
source distribution.)

Tutorials
=========

* Gaston Hillar's `two-part article series
  <https://web.archive.org/web/20190707171427/www.drdobbs.com/open-source/easy-opencl-with-python/240162614>`__
  in Dr. Dobb's Journal provides a friendly introduction to PyOpenCL.
* `Simon McIntosh-Smith <http://people.cs.bris.ac.uk/~simonm/>`__
  and `Tom Deakin <https://www.tomdeakin.com/>`__'s course
  `Hands-on OpenCL <https://handsonopencl.github.io/>`__ contains
  both `lecture slides <https://github.com/HandsOnOpenCL/Lecture-Slides/releases>`__
  and `exercises (with solutions) <https://github.com/HandsOnOpenCL/Exercises-Solutions>`__
  (The course covers PyOpenCL as well as OpenCL's C and C++ APIs.)
* PyOpenCL course at `PASI <https://www.bu.edu/pasi>`__: Parts
  `1 <https://www.youtube.com/watch?v=X9mflbX1NL8>`__
  `2 <https://www.youtube.com/watch?v=MqvfCE_bKOg>`__
  `3 <https://www.youtube.com/watch?v=TAvKmV7CuUw>`__
  `4 <https://www.youtube.com/watch?v=SsuJ0LvZW1Q>`__
  (YouTube, 2011)
* PyOpenCL course at `DTU GPULab <http://gpulab.compute.dtu.dk/>`__ and
  `Simula <https://www.simula.no/>`__ (2011):
  `Lecture 1 <https://tiker.net/pub/simula-pyopencl-lec1.pdf>`__
  `Lecture 2 <https://tiker.net/pub/simula-pyopencl-lec2.pdf>`__
  `Problem set 1 <https://tiker.net/pub/simula-pyopencl-probset1.pdf>`__
  `Problem set 2 <https://tiker.net/pub/simula-pyopencl-probset2.pdf>`__
* Ian Johnson's `PyOpenCL tutorial <https://web.archive.org/web/20170907175053/http://enja.org:80/2011/02/22/adventures-in-pyopencl-part-1-getting-started-with-python>`__.

Software that works with or enhances PyOpenCL
=============================================

* Jon Roose's `pyclblas <https://pyclblas.readthedocs.io/en/latest/index.html>`__
  (`code <https://github.com/jroose/pyclblas>`__)
  makes BLAS in the form of `clBLAS <https://github.com/clMathLibraries/clBLAS>`__
  available from within :mod:`pyopencl` code.

  Two earlier wrappers continue to be available:
  one by `Eric Hunsberger <https://github.com/hunse/pyopencl_blas>`__ and one
  by `Lars Ericson <https://lists.tiker.net/pipermail/pyopencl/2015-June/001890.html>`__.

* Cedric Nugteren provides a wrapper for the
  `CLBlast <https://github.com/CNugteren/CLBlast>`__
  OpenCL BLAS library:
  `PyCLBlast <https://github.com/CNugteren/CLBlast/tree/master/src/pyclblast>`__.

* Gregor Thalhammer's `gpyfft <https://github.com/geggo/gpyfft>`__ provides a
  Python wrapper for the OpenCL FFT library clFFT from AMD.

* Bogdan Opanchuk's `reikna <https://pypi.org/project/reikna/>`__ offers a
  variety of GPU-based algorithms (FFT, random number generation, matrix
  multiplication) designed to work with :class:`pyopencl.array.Array` objects.

* Troels Henriksen, Ken Friis Larsen, and Cosmin Oancea's `Futhark
  <https://futhark-lang.org/>`__ programming language offers a nice way to code
  nested-parallel programs with reductions and scans on data in
  :class:`pyopencl.array.Array` instances.

* Robbert Harms and Alard Roebroeck's `MOT <https://github.com/robbert-harms/MOT>`__
  offers a variety of GPU-enabled non-linear optimization algorithms and MCMC
  sampling routines for parallel optimization and sampling of multiple problems.

* Vincent Favre-Nicolin's `pyvkfft <https://github.com/vincefn/pyvkfft/>`__
  makes `vkfft <https://github.com/DTolm/VkFFT>`__ accessible from PyOpenCL.

If you know of a piece of software you feel that should be on this list, please
let me know, or, even better, send a patch!

Contents
========

.. toctree::
    :maxdepth: 2

    runtime
    runtime_const
    runtime_platform
    runtime_queue
    runtime_memory
    runtime_program
    runtime_gl
    tools
    array
    types
    algorithm
    howto
    misc
    🚀 Github <https://github.com/inducer/pyopencl>
    💾 Download Releases <https://pypi.org/project/pyopencl>

Note that this guide does not explain OpenCL programming and technology. Please
refer to the official `Khronos OpenCL documentation <https://www.khronos.org/opencl/>`__
for that.

PyOpenCL also has its own `web site <https://mathema.tician.de/software/pyopencl>`__,
where you can find updates, new versions, documentation, and support.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
