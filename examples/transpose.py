# Transposition of a matrix
# originally for PyCUDA by Hendrik Riedmann <riedmann@dam.brown.edu>

from __future__ import division

import pyopencl as cl
import numpy
import numpy.linalg as la




block_size = 16




def inner_transpose(ctx, queue, tgt, src, shape):
    knl = cl.Program(ctx, """
    #define BLOCK_SIZE %(block_size)d
    #define A_BLOCK_STRIDE (BLOCK_SIZE * a_width)
    #define A_T_BLOCK_STRIDE (BLOCK_SIZE * a_height)

    __kernel void transpose(
      __global float *a_t, __global float *a,
      unsigned a_width, unsigned a_height,
      __local float a_local[BLOCK_SIZE][BLOCK_SIZE+1])
    {
      // Base indices in A and A_t
      int base_idx_a   =
        get_group_id(0) * BLOCK_SIZE +
        get_group_id(1) * A_BLOCK_STRIDE;
      int base_idx_a_t =
        get_group_id(1) * BLOCK_SIZE +
        get_group_id(0) * A_T_BLOCK_STRIDE;

      // Global indices in A and A_t
      int glob_idx_a   = base_idx_a + get_local_id(0) + a_width * get_local_id(1);
      int glob_idx_a_t = base_idx_a_t + get_local_id(0) + a_height * get_local_id(1);

      // Store transposed submatrix to local memory
      a_local[get_local_id(1)][get_local_id(0)] = a[glob_idx_a];

      barrier(CLK_LOCAL_MEM_FENCE);

      // Write transposed submatrix to global memory
      a_t[glob_idx_a_t] = a_local[get_local_id(0)][get_local_id(1)];
    }
    """% {"block_size": block_size}).build().transpose

    w, h = shape
    assert w % block_size == 0
    assert h % block_size == 0

    knl(queue, (w, h),
        tgt, src, numpy.uint32(w), numpy.uint32(h),
        cl.LocalMemory(4*block_size*(block_size+1)),
        local_size=(block_size, block_size))




def transpose_using_cl(ctx, queue, cpu_src):
    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cpu_src)
    a_t_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=cpu_src.nbytes)
    inner_transpose(ctx, queue, a_t_buf, a_buf, cpu_src.shape)

    w, h = cpu_src.shape
    result = numpy.empty((h, w), dtype=cpu_src.dtype)
    cl.enqueue_read_buffer(queue, a_t_buf, result).wait()

    a_buf.release()
    a_t_buf.release()

    return result





def check_transpose():
    ctx = cl.Context(dev_type=cl.device_type.ALL)

    for dev in ctx.devices:
        assert dev.local_mem_size > 0

    queue = cl.CommandQueue(ctx)

    from pycuda.curandom import rand

    for i in numpy.arange(10, 13, 0.125):
        size = int(((2**i) // 32) * 32)
        print size

        source = numpy.random.rand(size, size).astype(numpy.float32)
        result = transpose_using_cl(ctx, queue, source)

        err = source.T - result
        err_norm = la.norm(err)

        assert err_norm == 0, (size, err_norm)




check_transpose()

