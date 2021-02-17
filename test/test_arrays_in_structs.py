__copyright__ = "Copyright (C) 2020 Sotiris Niarchos"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np

import pyopencl as cl
import pyopencl.cltypes as cltypes
import pyopencl.tools as cl_tools
from pyopencl import mem_flags
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)


def test_struct_with_array_fields(ctx_factory):
    #
    # typedef struct {
    #     uint x[2];
    #     float y;
    #     uint z[3][4];
    # } my_struct;
    #
    cl_ctx = ctx_factory()
    device = cl_ctx.devices[0]
    queue = cl.CommandQueue(cl_ctx)

    my_struct = np.dtype([
        ("x", cltypes.uint, 2),
        ("y", cltypes.int),
        ("z", cltypes.uint, (3, 4))
    ])
    my_struct, cdecl = cl_tools.match_dtype_to_c_struct(
        device, "my_struct", my_struct
    )

    # a random buffer of 4 structs
    my_struct_arr = np.array([
        ([81, 24], -57, [[15, 28, 45,  7], [71, 95, 65, 84], [2, 11, 59,  9]]),
        ([5, 20],  47, [[15, 53,  7, 59], [73, 22, 27, 86], [59,  6, 39, 49]]),
        ([11, 99], -32, [[73, 83,  4, 65], [19, 21, 22, 27], [1, 55,  6, 64]]),
        ([57, 38], -54, [[74, 90, 38, 67], [77, 30, 99, 18], [91,  3, 63, 67]])
    ], dtype=my_struct)

    expected_res = []
    for x in my_struct_arr:
        expected_res.append(int(np.sum(x[0]) + x[1] + np.sum(x[2])))
    expected_res = np.array(expected_res, dtype=cltypes.int)

    kernel_src = """%s
    // this kernel sums every number contained in each struct
    __kernel void array_structs(__global my_struct *structs, __global int *res) {
        int i = get_global_id(0);
        my_struct s = structs[i];
        res[i] = s.x[0] + s.x[1] + s.y;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 4; c++)
                res[i] += s.z[r][c];
    }""" % cdecl

    mem_flags1 = mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR
    mem_flags2 = mem_flags.WRITE_ONLY

    my_struct_buf = cl.Buffer(cl_ctx, mem_flags1, hostbuf=my_struct_arr)
    res_buf = cl.Buffer(cl_ctx, mem_flags2, size=expected_res.nbytes)

    program = cl.Program(cl_ctx, kernel_src).build()
    kernel = program.array_structs
    kernel(queue, (4,), None, my_struct_buf, res_buf)

    res = np.empty_like(expected_res)
    cl.enqueue_copy(queue, res, res_buf)

    assert (res == expected_res).all()


if __name__ == "__main__":

    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
