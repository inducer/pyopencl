# example by Roger Pau Monn'e
import pyopencl as cl
import numpy as np

demo_r = np.empty( (500,5), dtype=np.uint32)
ctx = cl.create_some_context()
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
    print("Error:")
    print(prg.get_build_info(ctx.devices[0], cl.program_build_info.LOG))
    raise

prg.demo(queue, (500,), None, demo_buf)
cl.enqueue_copy(queue, demo_r, demo_buf).wait()

for res in demo_r:
    print(res)

