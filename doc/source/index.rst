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

Here's an example, to give you an impression::

    import pyopencl as cl
    import numpy
    import numpy.linalg as la

    a = numpy.random.rand(50000).astype(numpy.float32)
    b = numpy.random.rand(50000).astype(numpy.float32)

    ctx = cl.Context()
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)

    prg = cl.Program(ctx, """
        __kernel void sum(__global const float *a,
        __global const float *b, __global float *c)
        {
          int gid = get_global_id(0);
          c[gid] = a[gid] + b[gid];
        }
        """).build()

    prg.sum(queue, a.shape, a_buf, b_buf, dest_buf)

    a_plus_b = numpy.empty_like(a)
    cl.enqueue_read_buffer(queue, dest_buf, a_plus_b).wait()

    print la.norm(a_plus_b - (a+b))

(You can find this example as :file:`examples/demo.py` in the PyOpenCL
source distribution.)

Contents
========

.. toctree::
    :maxdepth: 2

    reference
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
