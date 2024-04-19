import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.tools import SVMAllocator, SVMPool


n = 50000

rng = np.random.default_rng()
a = rng.random(n, dtype=np.float32)
b = rng.random(n, dtype=np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

alloc = SVMAllocator(ctx, alignment=0, queue=queue)
alloc = SVMPool(alloc)

a_dev = cl_array.to_device(queue, a, allocator=alloc)
b_dev = cl_array.to_device(queue, b, allocator=alloc)
dest_dev = cl_array.empty_like(a_dev)

prg = cl.Program(ctx, """
    __kernel void sum(__global const float *a,
    __global const float *b, __global float *c)
    {
      int gid = get_global_id(0);
      c[gid] = a[gid] + b[gid];
    }
    """).build()

knl = prg.sum
knl(queue, a.shape, None, a_dev.data, b_dev.data, dest_dev.data)

print(np.linalg.norm((dest_dev - (a_dev + b_dev)).get()))
assert np.allclose(dest_dev.get(), (a_dev + b_dev).get())
