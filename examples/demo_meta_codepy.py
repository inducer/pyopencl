import pyopencl as cl
import numpy
import numpy.linalg as la

block_size = 16
local_size = 32
macroblock_count = 33
dtype = numpy.float32
total_size = block_size*local_size*macroblock_count

ctx = cl.Context()
queue = cl.CommandQueue(ctx)

a = numpy.random.randn(total_size).astype(dtype)
b = numpy.random.randn(total_size).astype(dtype)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)

from codepy.cgen import FunctionBody, \
        FunctionDeclaration, Typedef, POD, Value, \
        Pointer, Module, Block, Initializer, Assign, Const
from codepy.cgen.opencl import CLKernel, CLGlobal, \
        CLRequiredWorkGroupSize

mod = Module([
    FunctionBody(
        CLKernel(CLRequiredWorkGroupSize((
            FunctionDeclaration(
            Value("void", "add"),
            arg_decls=[CLGlobal(Pointer(Const(POD(dtype, name))))
                for name in ["tgt", "op1", "op2"]]))),
        Block([
            Initializer(
                POD(numpy.int32, "idx"), "get_global_id(0)")
            ]+[
            Assign(
                "tgt[idx+%d]" % (o*local_size),
                "op1[idx+%d] + op2[idx+%d]" % (
                    o*local_size, 
                    o*local_size))
            for o in range(block_size)]))])

knl = cl.Program(mod).build().add

knl(c_gpu, a_gpu, b_gpu, 
        local_size=(local_size,),
        global_size=(local_size*macroblock_count,1))

c = numpy.empty_like(a)
cl.enqueue_read_buffer(queue, c_buf, c).wait()

assert la.norm(c-(a+b)) == 0

