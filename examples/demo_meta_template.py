from __future__ import absolute_import
import pyopencl as cl
import numpy
import numpy.linalg as la

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

from mako.template import Template

tpl = Template("""
    __kernel void add(
            __global ${ type_name } *tgt, 
            __global const ${ type_name } *op1, 
            __global const ${ type_name } *op2)
    {
      int idx = get_local_id(0)
        + ${ local_size } * ${ thread_strides }
        * get_group_id(0);

      % for i in range(thread_strides):
          <% offset = i*local_size %>
          tgt[idx + ${ offset }] = 
            op1[idx + ${ offset }] 
            + op2[idx + ${ offset } ];
      % endfor
    }""")

rendered_tpl = tpl.render(type_name="float", 
    local_size=local_size, thread_strides=thread_strides)

knl = cl.Program(ctx, str(rendered_tpl)).build().add

knl(queue, (local_size*macroblock_count,), (local_size,),
        c_buf, a_buf, b_buf)

c = numpy.empty_like(a)
cl.enqueue_read_buffer(queue, c_buf, c).wait()

assert la.norm(c-(a+b)) == 0
