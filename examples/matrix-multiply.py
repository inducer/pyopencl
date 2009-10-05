# example provided by Eilif Muller

from __future__ import division

kernel_code = """

// Thread block size
#define BLOCK_SIZE %(block_size)d

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA %(w_a)d // Matrix A width
#define HA %(h_a)d // Matrix A height
#define WB %(w_b)d // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width
#define HC HA  // Matrix C height


/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#define AS(i, j) As[i + j * BLOCK_SIZE]
#define BS(i, j) Bs[i + j * BLOCK_SIZE]

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! WA is A's width and WB is B's width
////////////////////////////////////////////////////////////////////////////////
__kernel __attribute__((reqd_work_group_size(16,16,1))) 
void
matrixMul( __global float* C, __global float* A, __global float* B,
           __local float* As, __local float* Bs)
{
    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);

    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    // Index of the first sub-matrix of A processed by the block
    int aBegin = WA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + WA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * WB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        AS(ty, tx) = A[a + WA * ty + tx];
        BS(ty, tx) = B[b + WB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += AS(ty, k) * BS(k, tx);

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = Csub;

}

"""

import pyopencl as cl
from time import time
import numpy

block_size = 16


ctx = cl.Context(dev_type=cl.device_type.ALL)

for dev in ctx.devices:
        assert dev.local_mem_size > 0

queue = cl.CommandQueue(ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE)

#queue = cl.CommandQueue(ctx)


if False:
    a_height = 4096
    #a_height = 1024
    a_width = 2048
    #a_width = 256
    #b_height == a_width
    b_width = a_height

elif False:
    # like PyCUDA
    a_height = 2516
    a_width = 1472
    b_height = a_width
    b_width = 2144

else:
    # CL SDK
    a_height = 50*block_size
    a_width = 100*block_size
    b_height = a_width
    b_width = 50*block_size


h_a = numpy.random.rand(a_height, a_width).astype(numpy.float32)
h_b = numpy.random.rand(a_width, a_height).astype(numpy.float32)
h_c = numpy.empty((a_height, a_height)).astype(numpy.float32)

mf = cl.mem_flags


kernel_params = {"block_size": block_size,
                 "w_a":a_width, "h_a":a_height, "w_b":a_height}

prg = cl.Program(ctx, kernel_code % kernel_params).build()
kernel = prg.matrixMul

#def __call__(self, queue, tgt, src, shape):
#        w, h = shape

assert a_width % block_size == 0
assert a_height % block_size == 0
assert b_width % block_size == 0

# kernel(queue, (w, h), tgt, src, numpy.uint32(w), numpy.uint32(h))

# __call__(queue, a_t_buf, a_buf, source.shape)

# args: queue, domain, *args

t1 = time()

d_a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_a)
d_b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_b)
d_c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=h_c.nbytes)

push_time = time()-t1

# warmup
event = kernel(queue, (a_height,a_height), d_c_buf, d_a_buf, d_b_buf, 
        cl.LocalMemory(4* block_size**2),
        cl.LocalMemory(4* block_size**2),
        local_size=(block_size, block_size))
event.wait()

t1 = time()
count = 2
for i in range(count):
    event = kernel(queue, (a_height,a_height), d_c_buf, d_a_buf, d_b_buf, 
            cl.LocalMemory(4* block_size**2),
            cl.LocalMemory(4* block_size**2),
            local_size=(block_size, block_size))

event.wait()

#print event.profile.end - event.profile.start

gpu_time = time()-t1

t1 = time()

cl.enqueue_read_buffer(queue, d_c_buf, h_c).wait()

pull_time = time()-t1

ans1 = h_c

gpu_total_time = gpu_time+push_time+pull_time

print "GPU (s) total:", gpu_total_time
print "PUSH ", push_time
print "PULL ", pull_time
print "COMPUTE ", gpu_time/count
print "COMPUTE2 ", (event.profile.end-event.profile.start)*1e-9

gflop = h_c.size * (a_width * 2.) / (1000**3.)
gflops = gflop / gpu_time
print "gflops:", gflops

do_cpu = False

if do_cpu:
        t1 = time()
        ans2 = numpy.dot(h_a,h_b)

        cpu_time = time()-t1

        print "CPU (s)", cpu_time

        print "GPU speedup: ", cpu_time/gpu_total_time

        print "GPU==CPU:",numpy.allclose(ans1,ans2)
