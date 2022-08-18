import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.tools import SVMAllocator, SVMPool
import numpy as np

n = 50000
a = np.random.rand(n).astype(np.float32)
b = np.random.rand(n).astype(np.float32)


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

np.testing.assert_allclose(dest_dev.get(), (a_dev+b_dev).get())
