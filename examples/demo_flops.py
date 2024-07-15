import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt


src = """
    __kernel void sum(__global T *x, __global T *y, __global T *z) {
        const int i = get_global_id(0);

        z[i] = x[i] + y[i];
    }
"""


# allocates buffers of increasing size, for each run do a parallel sum interpreting
# the buffer as an array of i8, i16, ...
# profile the kernels to find the throughput in GFLOPS, useful to estimate the raw computational speed of the hardware
if __name__ == '__main__':
    types = [
        ('i8' , 'char'  , 1),
        ('i16', 'short' , 2),
        ('i32', 'int'   , 4),
        ('i64', 'long'  , 8),
        # ('f16', 'half'  , 2),
        ('f32', 'float' , 4),
        ('f64', 'double', 8)
    ]


    ctx   = cl.create_some_context()
    queue = cl.CommandQueue(ctx, properties = cl.command_queue_properties.PROFILING_ENABLE)


    buffer_size = [2 ** i for i in range(10, 31)]
    data = np.zeros((len(buffer_size), len(types)))

    for row, nbytes in enumerate(buffer_size):
        x = cl.Buffer(ctx, cl.mem_flags.READ_ONLY,  nbytes)
        y = cl.Buffer(ctx, cl.mem_flags.READ_ONLY,  nbytes)
        z = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, nbytes)

        for col, (label, literal, sizeof) in enumerate(types):
            sums    = nbytes // sizeof
            header  = f'#define T {literal}\n'
            kernel  = cl.Program(ctx, header + src).build().sum

            event   = kernel(queue, (sums,), None, x, y, z)
            event.wait()

            FLOPS = 1e9 * sums / (event.profile.end - event.profile.start)
            GFLOPS = FLOPS / 1e6

            data[row, col] = GFLOPS

        x.release()
        y.release()
        z.release()

    for col, (_, label, _) in enumerate(types):
        plt.semilogx(buffer_size, data[:, col], label = label)

    plt.title(f'{ctx.devices[0].name}')
    plt.legend()
    plt.xlabel('sizeof(vector)')
    plt.ylabel('GFLOPS')
    plt.show()
