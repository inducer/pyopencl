# example provided by Roger Pau Monn'e

import pyopencl as cl
import numpy
import numpy.linalg as la
import datetime
from time import time

a = numpy.random.rand(1000).astype(numpy.float32)
b = numpy.random.rand(1000).astype(numpy.float32)
c_result = numpy.empty_like(a)

# Speed in normal CPU usage
time1 = time()
for i in range(1000):
        for j in range(1000):
                c_result[i] = a[i] + b[i]
                c_result[i] = c_result[i] * (a[i] + b[i])
                c_result[i] = c_result[i] * (a[i] / 2.0)
time2 = time()
print "Execution time of test without OpenCL: ", time2 - time1, "s"


for platform in cl.get_platforms():
    for device in platform.get_devices():
        print "==============================================================="
        print "Platform name:", platform.name
        print "Platform profile:", platform.profile
        print "Platform vendor:", platform.vendor
        print "Platform version:", platform.version
        print "---------------------------------------------------------------"
        print "Device name:", device.name
        print "Device type:", cl.device_type.to_string(device.type)
        print "Device memory: ", device.global_mem_size//1024//1024, 'MB'
        print "Device max clock speed:", device.max_clock_frequency, 'MHz'
        print "Device compute units:", device.max_compute_units

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
                        int loop;
                        int gid = get_global_id(0);
                        for(loop=1; loop<1000;loop++)
                        {
                                c[gid] = a[gid] + b[gid];
                                c[gid] = c[gid] * (a[gid] + b[gid]);
                                c[gid] = c[gid] * (a[gid] / 2.0);
                        }
                }
                """).build()

        exec_evt = prg.sum(queue, a.shape, a_buf, b_buf, dest_buf)
        exec_evt.wait()
        elapsed = 1e-9*(exec_evt.profile.end - exec_evt.profile.start)

        print "Execution time of test: %g s" % elapsed

        c = numpy.empty_like(a)
        cl.enqueue_read_buffer(queue, dest_buf, c).wait()
        error = 0
        for i in range(1000):
                if c[i] != c_result[i]:
                        error = 1
        if error:
                print "Results doesn't match!!"
        else:
                print "Results OK"
