from __future__ import absolute_import
from __future__ import print_function
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel

n = 10
a_np = np.random.randn(n).astype(np.float32)
b_np = np.random.randn(n).astype(np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

a_g = cl.array.to_device(queue, a_np)
b_g = cl.array.to_device(queue, b_np)

lin_comb = ElementwiseKernel(ctx,
    "float k1, float *a_g, float k2, float *b_g, float *res_g",
    "res_g[i] = k1 * a_g[i] + k2 * b_g[i]",
    "lin_comb"
)

res_g = cl.array.empty_like(a_g)
lin_comb(2, a_g, 3, b_g, res_g)

# Check on GPU with PyOpenCL Array:
print((res_g - (2 * a_g + 3 * b_g)).get())

# Check on CPU with Numpy:
res_np = res_g.get()
print(res_np - (2 * a_np + 3 * b_np))
print(np.linalg.norm(res_np - (2 * a_np + 3 * b_np)))
