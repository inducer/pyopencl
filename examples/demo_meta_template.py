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

mf = cl.mem_flags
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)

from jinja2 import Template

tpl = Template("""
    __kernel void add(
            __global {{ type_name }} *tgt, 
            __global const {{ type_name }} *op1, 
            __global const {{ type_name }} *op2)
    {
      int idx = get_global_id(0);

      {% for i in range(block_size) %}
          {% set offset = i*local_size %}
          tgt[idx + {{ offset }}] = 
            op1[idx + {{ offset }}] 
            + op2[idx + {{ offset }}];
      {% endfor %}
    }""")

rendered_tpl = tpl.render(
    type_name="float", block_size=block_size,
    local_size=local_size)

knl = cl.Program(rendered_tpl).build().add

knl(c_gpu, a_gpu, b_gpu, 
        local_size=(local_size,),
        global_size=(local_size*macroblock_count,1))

c = numpy.empty_like(a)
cl.enqueue_read_buffer(queue, c_buf, c).wait()

assert la.norm(c-(a+b)) == 0
