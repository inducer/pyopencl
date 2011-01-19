import pyopencl as cl
import pyopencl.array as cl_array
import numpy

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

n = 10
a_gpu = cl_array.to_device(
        queue, numpy.random.randn(n).astype(numpy.float32))
b_gpu = cl_array.to_device(
        queue, numpy.random.randn(n).astype(numpy.float32))

from pyopencl.elementwise import ElementwiseKernel
lin_comb = ElementwiseKernel(ctx,
        "float a, float *x, "
        "float b, float *y, "
        "float *z",
        "z[i] = a*x[i] + b*y[i]",
        "linear_combination")

c_gpu = cl_array.empty_like(a_gpu)
lin_comb(5, a_gpu, 6, b_gpu, c_gpu)

import numpy.linalg as la
assert la.norm((c_gpu - (5*a_gpu+6*b_gpu)).get()) < 1e-5
