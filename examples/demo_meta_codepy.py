from __future__ import absolute_import
import pyopencl as cl
import numpy
import numpy.linalg as la
from six.moves import range

local_size = 256
thread_strides = 32
macroblock_count = 33
dtype = numpy.float32
total_size = local_size*thread_strides*macroblock_count

ctx = cl.create_some_context()
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
        CLKernel(CLRequiredWorkGroupSize((local_size,),
            FunctionDeclaration(
            Value("void", "add"),
            arg_decls=[CLGlobal(Pointer(Const(POD(dtype, name))))
                for name in ["tgt", "op1", "op2"]]))),
        Block([
            Initializer(POD(numpy.int32, "idx"), 
                "get_local_id(0) + %d * get_group_id(0)"
                % (local_size*thread_strides))
            ]+[
            Assign(
                "tgt[idx+%d]" % (o*local_size),
                "op1[idx+%d] + op2[idx+%d]" % (
                    o*local_size, 
                    o*local_size))
            for o in range(thread_strides)]))])

knl = cl.Program(ctx, str(mod)).build().add

knl(queue, (local_size*macroblock_count,), (local_size,),
        c_buf, a_buf, b_buf)

c = numpy.empty_like(a)
cl.enqueue_read_buffer(queue, c_buf, c).wait()

assert la.norm(c-(a+b)) == 0

