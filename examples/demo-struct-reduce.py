from __future__ import absolute_import
import numpy as np
import pyopencl as cl

def make_collector_dtype(device):
    dtype = np.dtype([
        ("cur_min", np.int32),
        ("cur_max", np.int32),
        ("pad", np.int32),
        ])

    name = "minmax_collector"
    from pyopencl.tools import get_or_register_dtype, match_dtype_to_c_struct

    dtype, c_decl = match_dtype_to_c_struct(device, name, dtype)
    dtype = get_or_register_dtype(name, dtype)

    return dtype, c_decl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mmc_dtype, mmc_c_decl = make_collector_dtype(ctx.devices[0])

preamble = mmc_c_decl + r"""//CL//

    minmax_collector mmc_neutral()
    {
        // FIXME: needs infinity literal in real use, ok here
        minmax_collector result;
        result.cur_min = 1<<30;
        result.cur_max = -(1<<30);
        return result;
    }

    minmax_collector mmc_from_scalar(float x)
    {
        minmax_collector result;
        result.cur_min = x;
        result.cur_max = x;
        return result;
    }

    minmax_collector agg_mmc(minmax_collector a, minmax_collector b)
    {
        minmax_collector result = a;
        if (b.cur_min < result.cur_min)
            result.cur_min = b.cur_min;
        if (b.cur_max > result.cur_max)
            result.cur_max = b.cur_max;
        return result;
    }

    """

from pyopencl.clrandom import rand as clrand
a_gpu = clrand(queue, (20000,), dtype=np.int32, a=0, b=10**6)
a = a_gpu.get()

from pyopencl.reduction import ReductionKernel
red = ReductionKernel(ctx, mmc_dtype,
        neutral="mmc_neutral()",
        reduce_expr="agg_mmc(a, b)", map_expr="mmc_from_scalar(x[i])",
        arguments="__global int *x", preamble=preamble)

minmax = red(a_gpu).get()

assert abs(minmax["cur_min"] - np.min(a)) < 1e-5
assert abs(minmax["cur_max"] - np.max(a)) < 1e-5
