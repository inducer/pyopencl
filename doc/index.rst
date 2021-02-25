Welcome to PyOpenCL's documentation!
====================================

PyOpenCL gives you easy, Pythonic access to the `OpenCL
<http://www.khronos.org/opencl/>`_ parallel computation API.
What makes PyOpenCL special?

* Object cleanup tied to lifetime of objects. This idiom,
  often called
  `RAII <http://en.wikipedia.org/wiki/Resource_Acquisition_Is_Initialization>`_
  in C++, makes it much easier to write correct, leak- and
  crash-free code.

* Completeness. PyOpenCL puts the full power of OpenCL's API at your
  disposal, if you wish. Every obscure `get_info()` query and
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
  <http://web.archive.org/web/20190707171427/www.drdobbs.com/open-source/easy-opencl-with-python/240162614>`_
  in Dr. Dobb's Journal provides a friendly introduction to PyOpenCL.
* `Simon McIntosh-Smith <http://www.cs.bris.ac.uk/~simonm/>`_
  and `Tom Deakin <http://www.tomdeakin.com/>`_'s course
  `Hands-on OpenCL <http://handsonopencl.github.io/>`_ contains
  both `lecture slides <https://github.com/HandsOnOpenCL/Lecture-Slides/releases>`_
  and `exercises (with solutions) <https://github.com/HandsOnOpenCL/Exercises-Solutions>`_
  (The course covers PyOpenCL as well as OpenCL's C and C++ APIs.)
* PyOpenCL course at `PASI <http://bu.edu/pasi>`_: Parts
  `1 <https://www.youtube.com/watch?v=X9mflbX1NL8>`_
  `2 <https://www.youtube.com/watch?v=MqvfCE_bKOg>`_
  `3 <https://www.youtube.com/watch?v=TAvKmV7CuUw>`_
  `4 <https://www.youtube.com/watch?v=SsuJ0LvZW1Q>`_
  (YouTube, 2011)
* PyOpenCL course at `DTU GPULab <http://gpulab.imm.dtu.dk/>`_ and
  `Simula <http://simula.no/>`_ (2011):
  `Lecture 1 <http://tiker.net/pub/simula-pyopencl-lec1.pdf>`_
  `Lecture 2 <http://tiker.net/pub/simula-pyopencl-lec2.pdf>`_
  `Problem set 1 <http://tiker.net/pub/simula-pyopencl-probset1.pdf>`_
  `Problem set 2 <http://tiker.net/pub/simula-pyopencl-probset2.pdf>`_
* Ian Johnson's `PyOpenCL tutorial <https://web.archive.org/web/20170907175053/http://enja.org:80/2011/02/22/adventures-in-pyopencl-part-1-getting-started-with-python>`_.

Software that works with or enhances PyOpenCL
=============================================

* Jon Roose's `pyclblas <https://pyclblas.readthedocs.io/en/latest/index.html>`_
  (`code <https://github.com/jroose/pyclblas>`_)
  makes BLAS in the form of `clBLAS <https://github.com/clMathLibraries/clBLAS>`_
  available from within :mod:`pyopencl` code.

  Two earlier wrappers continue to be available:
  one by `Eric Hunsberger <https://github.com/hunse/pyopencl_blas>`_ and one
  by `Lars Ericson <http://lists.tiker.net/pipermail/pyopencl/2015-June/001890.html>`_.

* Cedric Nugteren provides a wrapper for the `CLBlast <https://github.com/CNugteren/CLBlast>`_ OpenCL BLAS library: `PyCLBlast <https://github.com/CNugteren/CLBlast/tree/master/src/pyclblast>`_.

* Gregor Thalhammer's `gpyfft <https://github.com/geggo/gpyfft>`_ provides a
  Python wrapper for the OpenCL FFT library clFFT from AMD.

* Bogdan Opanchuk's `reikna <http://pypi.python.org/pypi/reikna>`_ offers a
  variety of GPU-based algorithms (FFT, random number generation, matrix
  multiplication) designed to work with :class:`pyopencl.array.Array` objects.

* Troels Henriksen, Ken Friis Larsen, and Cosmin Oancea's `Futhark
  <http://futhark-lang.org/>`_ programming language offers a nice way to code
  nested-parallel programs with reductions and scans on data in
  :class:`pyopencl.array.Array` instances.

* Robbert Harms and Alard Roebroeck's `MOT
  <https://github.com/cbclab/MOT>`_ offers a variety of GPU-enabled non-linear optimization algorithms
  and MCMC sampling routines for parallel optimization and sampling of multiple problems.

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
    array
    types
    algorithm
    howto
    tools
    misc
    🚀 Github <https://github.com/inducer/pyopencl>
    💾 Download Releases <https://pypi.org/project/pyopencl>

Note that this guide does not explain OpenCL programming and technology. Please
refer to the official `Khronos OpenCL documentation <http://khronos.org/opencl>`_
for that.

PyOpenCL also has its own `web site <http://mathema.tician.de/software/pyopencl>`_,
where you can find updates, new versions, documentation, and support.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
