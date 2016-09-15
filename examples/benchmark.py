# example provided by Roger Pau Monn'e

from __future__ import print_function
from __future__ import absolute_import
import pyopencl as cl
import numpy
import numpy.linalg as la
import datetime
from time import time

data_points = 2**23 # ~8 million data points, ~32 MB data
workers = 2**8 # 256 workers, play with this to see performance differences
               # eg: 2**0 => 1 worker will be non-parallel execution on gpu
               # data points must be a multiple of workers

a = numpy.random.rand(data_points).astype(numpy.float32)
b = numpy.random.rand(data_points).astype(numpy.float32)
c_result = numpy.empty_like(a)

# Speed in normal CPU usage
time1 = time()
c_temp = (a+b) # adds each element in a to its corresponding element in b
c_temp = c_temp * c_temp # element-wise multiplication
c_result = c_temp * (a/2.0) # element-wise half a and multiply
time2 = time()

print("Execution time of test without OpenCL: ", time2 - time1, "s")


for platform in cl.get_platforms():
    for device in platform.get_devices():
        print("===============================================================")
        print("Platform name:", platform.name)
        print("Platform profile:", platform.profile)
        print("Platform vendor:", platform.vendor)
        print("Platform version:", platform.version)
        print("---------------------------------------------------------------")
        print("Device name:", device.name)
        print("Device type:", cl.device_type.to_string(device.type))
        print("Device memory: ", device.global_mem_size//1024//1024, 'MB')
        print("Device max clock speed:", device.max_clock_frequency, 'MHz')
        print("Device compute units:", device.max_compute_units)
        print("Device max work group size:", device.max_work_group_size)
        print("Device max work item sizes:", device.max_work_item_sizes)

        # Simnple speed test
        ctx = cl.Context([device])
        queue = cl.CommandQueue(ctx, 
                properties=cl.command_queue_properties.PROFILING_ENABLE)

        mf = cl.mem_flags
        a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
        dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)

        prg = cl.Program(ctx, """
            __kernel void sum(__global const float *a,
            __global const float *b, __global float *c)
            {
                        int gid = get_global_id(0);
                        float a_temp;
                        float b_temp;
                        float c_temp;

                        a_temp = a[gid]; // my a element (by global ref)
                        b_temp = b[gid]; // my b element (by global ref)
                        
                        c_temp = a_temp+b_temp; // sum of my elements
                        c_temp = c_temp * c_temp; // product of sums
                        c_temp = c_temp * (a_temp/2.0f); // times 1/2 my a

                        c[gid] = c_temp; // store result in global memory
                }
                """).build()

        global_size=(data_points,)
        local_size=(workers,)
        preferred_multiple = cl.Kernel(prg, 'sum').get_work_group_info( \
            cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, \
            device)

        print("Data points:", data_points)
        print("Workers:", workers)
        print("Preferred work group size multiple:", preferred_multiple)

        if (workers % preferred_multiple):
            print("Number of workers not a preferred multiple (%d*N)." \
                    % (preferred_multiple))
            print("Performance may be reduced.")

        exec_evt = prg.sum(queue, global_size, local_size, a_buf, b_buf, dest_buf)
        exec_evt.wait()
        elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)

        print("Execution time of test: %g s" % elapsed)

        c = numpy.empty_like(a)
        cl.enqueue_read_buffer(queue, dest_buf, c).wait()
        equal = numpy.all( c == c_result)

        if not equal:
                print("Results doesn't match!!")
        else:
                print("Results OK")
