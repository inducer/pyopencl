# example by Roger Pau Monn'e
import pyopencl as cl
import numpy as np

demo_r = np.empty( (500,5), dtype=np.uint32)
ctx = cl.Context(dev_type=cl.device_type.ALL)
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
demo_buf = cl.Buffer(ctx, mf.WRITE_ONLY, demo_r.nbytes)

prg = cl.Program(ctx,
"""
__kernel void demo(__global uint *demo)
{
    int i;
    int gid = get_global_id(0);
    for(i=0; i<5;i++)
    {
        demo[gid*5+i] = (uint) 1;
    }
}""")

try:
    prg.build()
except:
    print "Error:"
    print prg.get_build_info(ctx.devices[0], cl.program_build_info.LOG)
    raise

prg.demo(queue, (500,), demo_buf)
cl.enqueue_read_buffer(queue, demo_buf, demo_r).wait()

for res in demo_r:
    print res

