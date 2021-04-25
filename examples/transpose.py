# Transposition of a matrix
# originally for PyCUDA by Hendrik Riedmann <riedmann@dam.brown.edu>

import pyopencl as cl
import numpy
import numpy.linalg as la




block_size = 16




class NaiveTranspose:
    def __init__(self, ctx):
        self.kernel = cl.Program(ctx, """
        __kernel
        void transpose(
          __global float *a_t, __global float *a,
          unsigned a_width, unsigned a_height)
        {
          int read_idx = get_global_id(0) + get_global_id(1) * a_width;
          int write_idx = get_global_id(1) + get_global_id(0) * a_height;

          a_t[write_idx] = a[read_idx];
        }
        """% {"block_size": block_size}).build().transpose

    def __call__(self, queue, tgt, src, shape):
        w, h = shape
        assert w % block_size == 0
        assert h % block_size == 0

        return self.kernel(queue, (w, h), (block_size, block_size),
            tgt, src, numpy.uint32(w), numpy.uint32(h))




class SillyTranspose(NaiveTranspose):
    def __call__(self, queue, tgt, src, shape):
        w, h = shape
        assert w % block_size == 0
        assert h % block_size == 0

        return self.kernel(queue, (w, h), None,
            tgt, src, numpy.uint32(w), numpy.uint32(h))




class TransposeWithLocal:
    def __init__(self, ctx):
        self.kernel = cl.Program(ctx, """
        #define BLOCK_SIZE %(block_size)d
        #define A_BLOCK_STRIDE (BLOCK_SIZE * a_width)
        #define A_T_BLOCK_STRIDE (BLOCK_SIZE * a_height)

        __kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
        void transpose(
          __global float *a_t, __global float *a,
          unsigned a_width, unsigned a_height,
          __local float *a_local)
        {
          int base_idx_a   =
            get_group_id(0) * BLOCK_SIZE +
            get_group_id(1) * A_BLOCK_STRIDE;
          int base_idx_a_t =
            get_group_id(1) * BLOCK_SIZE +
            get_group_id(0) * A_T_BLOCK_STRIDE;

          int glob_idx_a   = base_idx_a + get_local_id(0) + a_width * get_local_id(1);
          int glob_idx_a_t = base_idx_a_t + get_local_id(0) + a_height * get_local_id(1);

          a_local[get_local_id(1)*BLOCK_SIZE+get_local_id(0)] = a[glob_idx_a];

          barrier(CLK_LOCAL_MEM_FENCE);

          a_t[glob_idx_a_t] = a_local[get_local_id(0)*BLOCK_SIZE+get_local_id(1)];
        }
        """% {"block_size": block_size}).build().transpose

    def __call__(self, queue, tgt, src, shape):
        w, h = shape
        assert w % block_size == 0
        assert h % block_size == 0

        return self.kernel(queue, (w, h), (block_size, block_size),
            tgt, src, numpy.uint32(w), numpy.uint32(h),
            cl.LocalMemory(4*block_size*(block_size+1)))




def transpose_using_cl(ctx, queue, cpu_src, cls):
    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cpu_src)
    a_t_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=cpu_src.nbytes)
    cls(ctx)(queue, a_t_buf, a_buf, cpu_src.shape)

    w, h = cpu_src.shape
    result = numpy.empty((h, w), dtype=cpu_src.dtype)
    cl.enqueue_copy(queue, result, a_t_buf).wait()

    a_buf.release()
    a_t_buf.release()

    return result





def check_transpose():
    for cls in [NaiveTranspose, SillyTranspose, TransposeWithLocal]:
        print("checking", cls.__name__)
        ctx = cl.create_some_context()

        for dev in ctx.devices:
            assert dev.local_mem_size > 0

        queue = cl.CommandQueue(ctx)

        for i in numpy.arange(10, 13, 0.125):
            size = int(((2**i) // 32) * 32)
            print(size)

            source = numpy.random.rand(size, size).astype(numpy.float32)
            result = transpose_using_cl(ctx, queue, source, NaiveTranspose)

            err = source.T - result
            err_norm = la.norm(err)

            assert err_norm == 0, (size, err_norm)




def benchmark_transpose():
    ctx = cl.create_some_context()

    for dev in ctx.devices:
        assert dev.local_mem_size > 0

    queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    sizes = [int(((2**i) // 32) * 32)
            for i in numpy.arange(10, 13, 0.125)]
            #for i in numpy.arange(10, 10.5, 0.125)]

    mem_bandwidths = {}

    methods = [SillyTranspose, NaiveTranspose, TransposeWithLocal]
    for cls in methods:
        name = cls.__name__.replace("Transpose", "")

        mem_bandwidths[cls] = meth_mem_bws = []

        for size in sizes:

            source = numpy.random.rand(size, size).astype(numpy.float32)

            mf = cl.mem_flags
            a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=source)
            a_t_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=source.nbytes)
            method = cls(ctx)

            for i in range(4):
                method(queue, a_t_buf, a_buf, source.shape)

            count = 12
            events = []
            for i in range(count):
                events.append(method(queue, a_t_buf, a_buf, source.shape))

            events[-1].wait()
            time = sum(evt.profile.end - evt.profile.start for evt in events)

            mem_bw = 2*source.nbytes*count/(time*1e-9)
            print("benchmarking", name, size, mem_bw/1e9, "GB/s")
            meth_mem_bws.append(mem_bw)

            a_buf.release()
            a_t_buf.release()

    try:
        from matplotlib.pyplot import clf, plot, title, xlabel, ylabel, \
                savefig, legend, grid
    except ModuleNotFoundError:
        pass
    else:
        for i in range(len(methods)):
            clf()
            for j in range(i+1):
                method = methods[j]
                name = method.__name__.replace("Transpose", "")
                plot(sizes, numpy.array(mem_bandwidths[method])/1e9, "o-", label=name)

            xlabel("Matrix width/height $N$")
            ylabel("Memory Bandwidth [GB/s]")
            legend(loc="best")
            grid()

            savefig("transpose-benchmark-%d.pdf" % i)


check_transpose()
benchmark_transpose()

