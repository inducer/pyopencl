import numpy as np
import numpy.linalg as la

import pyopencl as cl
import pyopencl.array as cl_array


rng = np.random.default_rng()
a = rng.random(50000, dtype=np.float32)
b = rng.random(50000, dtype=np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

a_dev = cl_array.to_device(queue, a)
b_dev = cl_array.to_device(queue, b)
dest_dev = cl_array.empty_like(a_dev)

prg = cl.Program(ctx, """
    __kernel void sum(__global const float *a,
    __global const float *b, __global float *c)
    {
      int gid = get_global_id(0);
      c[gid] = a[gid] + b[gid];
    }
    """).build()

knl = prg.sum  # Use this Kernel object for repeated calls
knl(queue, a.shape, None, a_dev.data, b_dev.data, dest_dev.data)

print(la.norm((dest_dev - (a_dev+b_dev)).get()))
assert np.allclose(dest_dev.get(), (a_dev + b_dev).get())
