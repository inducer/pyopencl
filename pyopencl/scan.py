"""Scan primitive."""

from __future__ import division

__copyright__ = """
Copyright 2011 Andreas Kloeckner
Copyright 2008-2011 NVIDIA Corporation
"""

__license__ = """
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Derived from thrust/detail/backend/cuda/detail/fast_scan.h
within the Thrust project, https://code.google.com/p/thrust/
"""




import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.tools import dtype_to_ctype
import numpy as np
import pyopencl._mymako as mako




CLUDA_PREAMBLE = """
#define local_barrier() barrier(CLK_LOCAL_MEM_FENCE);

#define WITHIN_KERNEL /* empty */
#define KERNEL __kernel
#define GLOBAL_MEM __global
#define LOCAL_MEM __local
#define REQD_WG_SIZE(X,Y,Z) __attribute__((reqd_work_group_size(X, Y, Z)))

#define LID_0 get_local_id(0)
#define LID_1 get_local_id(1)
#define LID_2 get_local_id(2)

#define GID_0 get_group_id(0)
#define GID_1 get_group_id(1)
#define GID_2 get_group_id(2)

% if double_support:
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
% endif
"""




SHARED_PREAMBLE = CLUDA_PREAMBLE + """
#define WG_SIZE ${wg_size}
#define SCAN_EXPR(a, b) ${scan_expr}

${preamble}

typedef ${scan_type} scan_type;
"""




SCAN_INTERVALS_SOURCE = mako.template.Template(SHARED_PREAMBLE + """
#define K ${wg_seq_batches}

<%def name="make_group_scan(name, with_bounds_check)">
    WITHIN_KERNEL
    void ${name}(LOCAL_MEM scan_type *array
    % if with_bounds_check:
      , const unsigned n
    % endif
    )
    {
        scan_type val = array[LID_0];

        <% offset = 1 %>

        % while offset <= wg_size:
            if (LID_0 >= ${offset}
            % if with_bounds_check:
              && LID_0 < n
            % endif
            )
            {
                scan_type tmp = array[LID_0 - ${offset}];
                val = SCAN_EXPR(tmp, val);
            }

            local_barrier();
            array[LID_0] = val;
            local_barrier();

            <% offset *= 2 %>
        % endwhile
    }
</%def>

${make_group_scan("scan_group", False)}
${make_group_scan("scan_group_n", True)}

KERNEL
REQD_WG_SIZE(WG_SIZE, 1, 1)
void ${name_prefix}_scan_intervals(
    GLOBAL_MEM scan_type *input,
    const unsigned int N,
    const unsigned int interval_size,
    GLOBAL_MEM scan_type *output,
    GLOBAL_MEM scan_type *group_results)
{
    LOCAL_MEM scan_type sdata[K + 1][WG_SIZE + 1];  // padded to avoid bank conflicts

    local_barrier(); // TODO figure out why this seems necessary now

    const unsigned int interval_begin = interval_size * GID_0;
    const unsigned int interval_end   = min(interval_begin + interval_size, N);

    const unsigned int unit_size  = K * WG_SIZE;

    unsigned int base = interval_begin;

    // process full units
    for(; base + unit_size <= interval_end; base += unit_size)
    {
        // read data
        for(unsigned int k = 0; k < K; k++)
        {
            const unsigned int offset = k*WG_SIZE + LID_0;

            GLOBAL_MEM scan_type *temp = input + (base + offset);
            sdata[offset % K][offset / K] = *temp;
        }

        // carry in
        if (LID_0 == 0 && base != interval_begin)
            sdata[0][0] = SCAN_EXPR(sdata[K][WG_SIZE - 1], sdata[0][0]);

        local_barrier();

        // scan local values
        scan_type sum = sdata[0][LID_0];

        for(unsigned int k = 1; k < K; k++)
        {
            scan_type tmp = sdata[k][LID_0];
            sum = SCAN_EXPR(sum, tmp);
            sdata[k][LID_0] = sum;
        }

        // second level scan
        sdata[K][LID_0] = sum;
        local_barrier();
        scan_group(&sdata[K][0]);

        // update local values
        if (LID_0 > 0)
        {
            sum = sdata[K][LID_0 - 1];

            for(unsigned int k = 0; k < K; k++)
            {
                scan_type tmp = sdata[k][LID_0];
                sdata[k][LID_0] = SCAN_EXPR(sum, tmp);
            }
        }

        local_barrier();

        // write data
        for(unsigned int k = 0; k < K; k++)
        {
            const unsigned int offset = k*WG_SIZE + LID_0;

            GLOBAL_MEM scan_type *temp = output + (base + offset);
            *temp = sdata[offset % K][offset / K];
        }

        local_barrier();
    }

    // process partially full unit at end of input (if necessary)
    if (base < interval_end)
    {
        // read data
        for(unsigned int k = 0; k < K; k++)
        {
            const unsigned int offset = k*WG_SIZE + LID_0;

            if (base + offset < interval_end)
            {
                GLOBAL_MEM scan_type *temp = input + (base + offset);
                sdata[offset % K][offset / K] = *temp;
            }
        }

        // carry in
        if (LID_0 == 0 && base != interval_begin)
            sdata[0][0] = SCAN_EXPR(sdata[K][WG_SIZE - 1], sdata[0][0]);

        local_barrier();

        // scan local values
        scan_type sum = sdata[0][LID_0];

        const unsigned int offset_end = interval_end - base;

        for(unsigned int k = 1; k < K; k++)
        {
            if (K * LID_0 + k < offset_end)
            {
                scan_type tmp = sdata[k][LID_0];
                sum = SCAN_EXPR(sum, tmp);
                sdata[k][LID_0] = sum;
            }
        }

        // second level scan
        sdata[K][LID_0] = sum;
        local_barrier();
        scan_group_n(&sdata[K][0], offset_end / K);

        // update local values
        if (LID_0 > 0)
        {
            sum = sdata[K][LID_0 - 1];

            for(unsigned int k = 0; k < K; k++)
            {
                if (K * LID_0 + k < offset_end)
                {
                    scan_type tmp = sdata[k][LID_0];
                    sdata[k][LID_0] = SCAN_EXPR(sum, tmp);
                }
            }
        }

        local_barrier();

        // write data
        for(unsigned int k = 0; k < K; k++)
        {
            const unsigned int offset = k*WG_SIZE + LID_0;

            if (base + offset < interval_end)
            {
                GLOBAL_MEM scan_type *temp = output + (base + offset);
                *temp = sdata[offset % K][offset / K];
            }
        }

    }

    local_barrier();

    // write interval sum
    if (LID_0 == 0)
    {
        GLOBAL_MEM scan_type *temp = output + (interval_end - 1);
        group_results[GID_0] = *temp;
    }
}
""")




INCLUSIVE_UPDATE_SOURCE = mako.template.Template(SHARED_PREAMBLE + """
KERNEL
REQD_WG_SIZE(WG_SIZE, 1, 1)
void ${name_prefix}_inclusive_update(
    GLOBAL_MEM scan_type *output,
    const unsigned int N,
    const unsigned int interval_size,
    GLOBAL_MEM scan_type *group_results)
{
    const unsigned int interval_begin = interval_size * GID_0;
    const unsigned int interval_end   = min(interval_begin + interval_size, N);

    if (GID_0 == 0)
        return;

    // value to add to this segment
    scan_type sum = group_results[GID_0 - 1];

    // advance result pointer
    output += interval_begin + LID_0;

    for(unsigned int base = interval_begin; base < interval_end; base += WG_SIZE, output += WG_SIZE)
    {
        const unsigned int i = base + LID_0;

        if(i < interval_end)
        {
            scan_type tmp = *output;
            *output = SCAN_EXPR(sum, tmp);
        }

        local_barrier();
    }
}
""")




EXCLUSIVE_UPDATE_SOURCE = mako.template.Template(SHARED_PREAMBLE + """
KERNEL
REQD_WG_SIZE(WG_SIZE, 1, 1)
void ${name_prefix}_exclusive_update(
    GLOBAL_MEM scan_type *output,
    const unsigned int N,
    const unsigned int interval_size,
    GLOBAL_MEM scan_type *group_results,
    scan_type init)
{
    LOCAL_MEM scan_type sdata[WG_SIZE];

    local_barrier(); // TODO figure out why this seems necessary now

    const unsigned int interval_begin = interval_size * GID_0;
    const unsigned int interval_end   = min(interval_begin + interval_size, N);

    // value to add to this segment
    scan_type carry = init;
    if(GID_0 != 0)
    {
        scan_type tmp = group_results[GID_0 - 1];
        carry = SCAN_EXPR(carry, tmp);
    }

    scan_type val   = carry;

    // advance result pointer
    output += interval_begin + LID_0;

    for(unsigned int base = interval_begin; base < interval_end; base += WG_SIZE, output += WG_SIZE)
    {
        const unsigned int i = base + LID_0;

        if(i < interval_end)
        {
            scan_type tmp = *output;
            sdata[LID_0] = binary_op(carry, tmp);
        }

        local_barrier();

        if (LID_0 != 0)
            val = sdata[LID_0 - 1];

        if (i < interval_end)
            *output = val;

        if(LID_0 == 0)
            val = sdata[WG_SIZE - 1];

        local_barrier();
    }
}
""")




def _div_ceil(nr, dr):
    return (nr + dr -1) // dr


def _uniform_interval_splitting(n, granularity, max_intervals):
    grains  = _div_ceil(n, granularity)

    # one grain per interval
    if grains <= max_intervals:
        return granularity, grains

    # ensures that:
    #     num_intervals * interval_size is >= n
    #   and
    #     (num_intervals - 1) * interval_size is < n

    grains_per_interval = _div_ceil(grains, max_intervals)
    interval_size = grains_per_interval * granularity
    num_intervals = _div_ceil(n, interval_size)

    return interval_size, num_intervals




class InclusiveScanKernel:
    def __init__(self, ctx, dtype,
            neutral, scan_expr,
            name_prefix="scan", options="", preamble="", devices=None):

        self.context = ctx
        dtype = self.dtype = np.dtype(dtype)

        if devices is None:
            devices = ctx.devices
        self.devices = devices

        # Thrust says these are good for GT200
        self.scan_wg_size = 128
        self.update_wg_size = 256
        self.scan_wg_seq_batches = 6

        from pytools import all
        from pyopencl.characterize import has_double_support

        kw_values = dict(
            preamble=preamble,
            name_prefix=name_prefix,
            scan_type=dtype_to_ctype(dtype),
            scan_expr=scan_expr,
            double_support=all(
                has_double_support(dev) for dev in devices)
            )

        scan_intervals_src = str(SCAN_INTERVALS_SOURCE.render(
            wg_size=self.scan_wg_size,
            wg_seq_batches=self.scan_wg_seq_batches,
            **kw_values))
        scan_intervals_prg = cl.Program(ctx, scan_intervals_src).build(options)
        self.scan_intervals_knl = getattr(
                scan_intervals_prg,
                name_prefix+"_scan_intervals")
        self.scan_intervals_knl.set_scalar_arg_dtypes(
                (None, np.uint32, np.uint32, None, None))

        inclusive_update_src = str(INCLUSIVE_UPDATE_SOURCE.render(
            wg_size=self.update_wg_size,
            **kw_values))

        inclusive_update_prg = cl.Program(ctx, inclusive_update_src).build(options)
        self.inclusive_update_knl = getattr(
                inclusive_update_prg,
                name_prefix+"_inclusive_update")
        self.inclusive_update_knl.set_scalar_arg_dtypes(
                (None, np.uint32, np.uint32, None))

    def __call__(self, input_ary, output_ary=None, allocator=None,
            queue=None):
        allocator = allocator or input_ary.allocator
        queue = queue or input_ary.queue or output_ary.queue

        if output_ary is None:
            output_ary = input_ary

        if isinstance(output_ary, (str, unicode)) and output_ary == "new":
            output_ary = cl_array.empty_like(input_ary, allocator=allocator)

        if input_ary.shape != output_ary.shape:
            raise ValueError("input and output must have the same shape")

        n, = input_ary.shape

        if not n:
            return output_ary

        unit_size  = self.scan_wg_size * self.scan_wg_seq_batches
        max_groups = 3*max(dev.max_compute_units for dev in self.devices)

        interval_size, num_groups = _uniform_interval_splitting(
                n, unit_size, max_groups);

        block_results = allocator(self.dtype.itemsize*num_groups)
        dummy_results = allocator(self.dtype.itemsize)

        # first level scan of interval (one interval per block)
        self.scan_intervals_knl(
                queue, (num_groups*self.scan_wg_size,), (self.scan_wg_size,),
                input_ary.data,
                n, interval_size,
                output_ary.data,
                block_results)

        # second level inclusive scan of per-block results
        self.scan_intervals_knl(
                queue, (self.scan_wg_size,), (self.scan_wg_size,),
                block_results,
                num_groups, interval_size,
                block_results,
                dummy_results)

        # update intervals with result of second level scan
        self.inclusive_update_knl(
                queue, (num_groups*self.update_wg_size,), (self.update_wg_size,),
                output_ary.data,
                n, interval_size,
                block_results)

        return output_ary
