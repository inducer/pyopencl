from __future__ import division, with_statement, absolute_import, print_function

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

import pyopencl.cltypes as cltypes
import pyopencl.tools as cl_tools
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
    uint_arr_2 = np.dtype((cltypes.uint, 2))
    uint_arr_2 = cl_tools.get_or_register_dtype('uint_arr_2', uint_arr_2)
    uint_arr_3_4 = np.dtype((cltypes.uint, (3, 4)))
    uint_arr_3_4 = cl_tools.get_or_register_dtype('uint_arr_3_4', uint_arr_3_4)
    my_struct = np.dtype([('x', uint_arr_2),('y', cltypes.float),('z', uint_arr_3_4)])
    my_struct, _ = cl_tools.match_dtype_to_c_struct(device, 'my_struct', my_struct)

if __name__ == "__main__":

    import pyopencl

    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
