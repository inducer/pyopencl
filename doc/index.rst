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
  <http://www.drdobbs.com/open-source/easy-opencl-with-python/240162614>`_
  in Dr. Dobb's Journal provides a friendly introduction to PyOpenCL.
* `Simon McIntosh-Smith <http://www.cs.bris.ac.uk/~simonm/>`_
  and `Tom Deakin <http://www.tomdeakin.com/>`_'s course
  `Hands-on OpenCL <http://handsonopencl.github.io/>`_ contains
  both `lecture slides <https://github.com/HandsOnOpenCL/Lecture-Slides/releases>`_
  and `excercises (with solutions) <https://github.com/HandsOnOpenCL/Exercises-Solutions>`_
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
* Ian Johnson's `PyOpenCL tutorial <http://enja.org/2011/02/22/adventures-in-pyopencl-part-1-getting-started-with-python/>`_.

Contents
========

.. toctree::
    :maxdepth: 2

    runtime
    array
    algorithm
    howto
    tools
    misc

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
