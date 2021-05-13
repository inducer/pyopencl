import pyopencl as cl
import pyopencl.array as cl_array
import numpy
import numpy.linalg as la

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

n = 10
a_gpu = cl_array.to_device(queue,
        (numpy.random.randn(n) + 1j*numpy.random.randn(n)).astype(numpy.complex64))
b_gpu = cl_array.to_device(queue,
        (numpy.random.randn(n) + 1j*numpy.random.randn(n)).astype(numpy.complex64))

from pyopencl.elementwise import ElementwiseKernel
complex_prod = ElementwiseKernel(ctx,
        "float a, "
        "cfloat_t *x, "
        "cfloat_t *y, "
        "cfloat_t *z",
        "z[i] = cfloat_rmul(a, cfloat_mul(x[i], y[i]))",
        "complex_prod",
        preamble="#include <pyopencl-complex.h>")

complex_add = ElementwiseKernel(ctx,
        "cfloat_t *x, "
        "cfloat_t *y, "
        "cfloat_t *z",
        "z[i] = cfloat_add(x[i], y[i])",
        "complex_add",
        preamble="#include <pyopencl-complex.h>")

real_part = ElementwiseKernel(ctx,
        "cfloat_t *x, float *z",
        "z[i] = cfloat_real(x[i])",
        "real_part",
        preamble="#include <pyopencl-complex.h>")

c_gpu = cl_array.empty_like(a_gpu)
complex_prod(5, a_gpu, b_gpu, c_gpu)

c_gpu_real = cl_array.empty(queue, len(a_gpu), dtype=numpy.float32)
real_part(c_gpu, c_gpu_real)
print(c_gpu.get().real - c_gpu_real.get())

print(la.norm(c_gpu.get() - (5*a_gpu.get()*b_gpu.get())))
assert la.norm(c_gpu.get() - (5*a_gpu.get()*b_gpu.get())) < 1e-5
