import numpy as np
import numpy.linalg as la

import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

n = 10

rng = np.random.default_rng()
a_gpu = cl_array.to_device(queue,
        rng.standard_normal(n, dtype=np.float32)
        + 1j*rng.standard_normal(n, dtype=np.float32))
b_gpu = cl_array.to_device(queue,
        rng.standard_normal(n, dtype=np.float32)
        + 1j*rng.standard_normal(n, dtype=np.float32))

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

c_gpu_real = cl_array.empty(queue, len(a_gpu), dtype=np.float32)
real_part(c_gpu, c_gpu_real)
print(c_gpu.get().real - c_gpu_real.get())

print(la.norm(c_gpu.get() - (5*a_gpu.get()*b_gpu.get())))
assert la.norm(c_gpu.get() - (5*a_gpu.get()*b_gpu.get())) < 1e-5
