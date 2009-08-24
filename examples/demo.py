import pyopencl as cl
import numpy
import numpy.linalg as la

a = numpy.random.rand(50000).astype(numpy.float32)
b = numpy.random.rand(50000).astype(numpy.float32)

ctx = cl.create_context_from_type(cl.device_type.ALL)
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_buf = cl.create_host_buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, a)
b_buf = cl.create_host_buffer(
        ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, b)
dest_buf = cl.create_buffer(ctx, mf.WRITE_ONLY, b.nbytes)

prg = cl.create_program_with_source(ctx, """
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

print la.norm(a_plus_b - (a+b)), la.norm(a_plus_b)
