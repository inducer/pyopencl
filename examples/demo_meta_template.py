import numpy as np
import numpy.linalg as la
from mako.template import Template

import pyopencl as cl


local_size = 256
thread_strides = 32
macroblock_count = 33
dtype = np.float32
total_size = local_size*thread_strides*macroblock_count

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

rng = np.random.default_rng()
a = rng.standard_normal(total_size, dtype=dtype)
b = rng.standard_normal(total_size, dtype=dtype)

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)

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

c = np.empty_like(a)
cl.enqueue_copy(queue, c, c_buf).wait()

assert la.norm(c-(a+b)) == 0
