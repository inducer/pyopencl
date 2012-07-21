"""Scan primitive."""

from __future__ import division

__copyright__ = """
Copyright 2011-2012 Andreas Kloeckner
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

Derived from thrust/detail/backend/cuda/detail/fast_scan.inl
within the Thrust project, https://code.google.com/p/thrust/

Direct browse link:
https://code.google.com/p/thrust/source/browse/thrust/detail/backend/cuda/detail/fast_scan.inl
"""



import numpy as np

import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.tools import dtype_to_ctype, bitlog2
import pyopencl._mymako as mako
from pyopencl._cluda import CLUDA_PREAMBLE
from pyopencl.tools import context_dependent_memoize





# {{{ preamble

SHARED_PREAMBLE = CLUDA_PREAMBLE + """//CL//
#define WG_SIZE ${wg_size}

/* SCAN_EXPR has no right know the indices it is scanning at because
each index may occur an undetermined number of times in the scan tree,
and thus index-based side computations cannot be meaningful. */

#define SCAN_EXPR(a, b) ${scan_expr}
#define INPUT_EXPR(i) (${input_expr})
%if is_segmented:
    #define IS_SEG_START(i, a) (${is_i_segment_start_expr})
%endif

${preamble}

typedef ${scan_ctype} scan_type;
typedef ${index_ctype} index_type;
#define NO_SEG_BOUNDARY ${index_type_max}
"""

# }}}

# {{{ main scan code

# Algorithm: Each work group is responsible for one contiguous
# 'interval'. There are just enough intervals to fill all compute
# units.  Intervals are split into 'units'. A unit is what gets
# worked on in parallel by one work group.
#
# in index space:
# interval > unit > local-parallel > k-group
#
# (Note that there is also a transpose in here: The data is read
# with local ids along linear index order.)
#
# Each unit has two axes--the local-id axis and the k axis.
#
# unit 0:
# | | | | | | | | | | ----> lid
# | | | | | | | | | |
# | | | | | | | | | |
# | | | | | | | | | |
# | | | | | | | | | |
#
# |
# v k (fastest-moving in linear index)
#
# unit 1:
# | | | | | | | | | | ----> lid
# | | | | | | | | | |
# | | | | | | | | | |
# | | | | | | | | | |
# | | | | | | | | | |
#
# |
# v k (fastest-moving in linear index)
#
# ...
#
# At a device-global level, this is a three-phase algorithm, in
# which first each interval does its local scan, then a scan
# across intervals exchanges data globally, and the final update
# adds the exchanged sums to each interval.
#
# Exclusive scan is realized by performing a right-shift inside
# the final update.

SCAN_INTERVALS_SOURCE = SHARED_PREAMBLE + """//CL//

#define K ${k_group_size}

KERNEL
REQD_WG_SIZE(WG_SIZE, 1, 1)
void ${name_prefix}_scan_intervals(
    ${argument_signature},
    GLOBAL_MEM scan_type *partial_scan_buffer,
    const index_type N,
    const index_type interval_size
    %if is_first_level:
        , GLOBAL_MEM scan_type *interval_results
    %endif
    %if is_segmented and is_first_level:
        /* NO_SEG_BOUNDARY if no segment boundary in interval.
        NO_SEG_BOUNDARY is the largest representable integer in index_type.
        Otherwise, index relative to interval beginning.
        */
        , GLOBAL_MEM index_type *g_first_segment_start_in_interval
    %endif
    )
{
    // padded in WG_SIZE to avoid bank conflicts
    // index K in first dimension used for carry storage
    LOCAL_MEM scan_type ldata[K + 1][WG_SIZE + 1];

    %if is_segmented:
        index_type first_segment_start_in_interval = NO_SEG_BOUNDARY;
        LOCAL_MEM index_type l_first_segment_start_in_k_group[WG_SIZE];
        index_type first_segment_start_in_k_group;

        if (LID_0 == 0)
    %endif


    const index_type interval_begin = interval_size * GID_0;
    const index_type interval_end   = min(interval_begin + interval_size, N);

    const index_type unit_size  = K * WG_SIZE;

    index_type unit_base = interval_begin;

    %for is_tail in [False, True]:

        %if not is_tail:
            for(; unit_base + unit_size <= interval_end; unit_base += unit_size)
        %else:
            if (unit_base < interval_end)
        %endif

        {

            // {{{ read a unit's worth of data from global

            for(index_type k = 0; k < K; k++)
            {
                const index_type offset = k*WG_SIZE + LID_0;

                const index_type read_i = unit_base + offset;

                %if is_tail:
                if (read_i < interval_end)
                %endif
                {
                    ldata[offset % K][offset / K] = INPUT_EXPR(read_i);
                }
            }

            // }}}

            // {{{ carry in from previous unit, if applicable.

            %if is_segmented:
                if (LID_0 == 0 && unit_base != interval_begin)
                {
                    if (IS_SEG_START(unit_base, ldata[0][0]))
                        first_segment_start_in_k_group = unit_base;
                    else
                    {
                        ldata[0][0] = SCAN_EXPR(ldata[K][WG_SIZE - 1], ldata[0][0]);
                        first_segment_start_in_k_group = NO_SEG_BOUNDARY;
                    }
                }
                else
                    first_segment_start_in_k_group = NO_SEG_BOUNDARY;
            %else:
                if (LID_0 == 0 && unit_base != interval_begin)
                    ldata[0][0] = SCAN_EXPR(ldata[K][WG_SIZE - 1], ldata[0][0]);
            %endif

            // }}}

            local_barrier();

            // {{{ scan along k (sequentially in each work item)

            scan_type sum = ldata[0][LID_0];

            %if is_tail:
                const index_type offset_end = interval_end - unit_base;
            %endif

            for(index_type k = 1; k < K; k++)
            {
                %if is_tail:
                if (K * LID_0 + k < offset_end)
                %endif
                {
                    scan_type tmp = ldata[k][LID_0];
                    index_type seq_i = unit_base + K*LID_0 + k;

                    %if is_segmented:
                    if (IS_SEG_START(seq_i, tmp)
                    {
                        first_segment_start_in_k_group = seq_i;
                        sum = tmp;
                    }
                    else
                    %endif
                        sum = SCAN_EXPR(sum, tmp);

                    ldata[k][LID_0] = sum;
                }
            }

            // }}}

            // store carry in out-of-bounds (padding) array entry (index K) in the K direction
            ldata[K][LID_0] = sum;

            %if is_segmented:
                l_first_segment_start_in_k_group[LID_0] = first_segment_start_in_k_group;
            %endif

            local_barrier();

            // {{{ tree-based local parallel scan

            // This tree-based scan works as follows:
            // - Each work item adds the previous item to its current state
            // - barrier
            // - Each work item adds in the item from two positions to the left
            // - barrier
            // - Each work item adds in the item from four positions to the left
            // ...
            // At the end, each item has summed all prior items.

            // across k groups, along local id
            // (uses out-of-bounds k=K array entry for storage)

            scan_type val = ldata[K][LID_0];

            <% scan_offset = 1 %>

            %if is_segmented:
                index_type first_segment_start_in_subtree;
            %endif

            % while scan_offset <= wg_size:
                // {{{ reads from local allowed, writes to local not allowed

                if (
                    LID_0 >= ${scan_offset}
                % if is_tail:
                    && K*LID_0 < offset_end
                % endif
                )
                {
                    scan_type tmp = ldata[K][LID_0 - ${scan_offset}];
                    %if is_segmented:
                        if (l_first_segment_start_in_k_group[LID_0] == NO_SEG_BOUNDARY)
                            val = SCAN_EXPR(tmp, val);

                        // update l_first_segment_start_in_k_group regardless
                        segment_start_in_subtree = min(
                            l_first_segment_start_in_k_group[LID_0],
                            l_first_segment_start_in_k_group[LID_0 - ${scan_offset}]);
                    %else:
                        val = SCAN_EXPR(tmp, val);
                    %endif
                }
                %if is_segmented:
                    else
                    {
                        first_segment_start_in_subtree =
                            l_first_segment_start_in_k_group[LID_0];
                    }
                %endif

                // }}}

                local_barrier();

                // {{{ writes to local allowed, reads from local not allowed

                ldata[K][LID_0] = val;
                %if is_segmented:
                    segment_start_in_k_group[LID_0] = segment_start_in_subtree;
                %endif

                // }}}

                local_barrier();

                <% scan_offset *= 2 %>
            % endwhile

            // }}}

            // {{{ update local values

            if (LID_0 > 0)
            {
                sum = ldata[K][LID_0 - 1];

                for(index_type k = 0; k < K; k++)
                {
                    bool do_update = true;
                    %if is_tail:
                        do_update = K * LID_0 + k < offset_end;
                    %endif
                    %if is_segmented:
                        do_update = unit_base + K * LID_0 + k
                            < first_segment_start_in_k_group;
                    %endif

                    if (do_update)
                    {
                        scan_type tmp = ldata[k][LID_0];
                        ldata[k][LID_0] = SCAN_EXPR(sum, tmp);
                    }
                }
            }

            %if is_segmented:
                if (LID_0 == 0)
                {
                    // carry in from previous unit
                    first_segment_start_in_interval =
                        first_segment_start_in_interval
                        ||
                        segment_start_in_k_group[WG_SIZE-1];
                }
            %endif

            // }}}

            local_barrier();

            // {{{ write data

            for (index_type k = 0; k < K; k++)
            {
                const index_type offset = k*WG_SIZE + LID_0;

                %if is_tail:
                if (unit_base + offset < interval_end)
                %endif
                {
                    partial_scan_buffer[unit_base + offset] =
                        ldata[offset % K][offset / K];
                }
            }

            // }}}

            local_barrier();
        }

    % endfor

    // write interval sum
    if (LID_0 == 0)
    {
        %if is_first_level:
        interval_results[GID_0] = partial_scan_buffer[interval_end - 1];
        %endif
        %if is_segmented and is_first_level:
            g_first_segment_start_in_interval[GID_0] = first_segment_start_in_interval;
        %endif
    }
}
"""

# }}}

# {{{ local update

# used for inclusive scan

LOCAL_UPDATE_SOURCE = SHARED_PREAMBLE + r"""//CL//

#define OUTPUT_STMT(i, prev_item, item) { ${output_statement}; }

KERNEL
REQD_WG_SIZE(WG_SIZE, 1, 1)
void ${name_prefix}_final_update(
    ${argument_signature},
    const index_type N,
    const index_type interval_size,
    GLOBAL_MEM scan_type *interval_results,
    GLOBAL_MEM scan_type *partial_scan_buffer
    %if is_segmented:
        , GLOBAL_MEM index_type *g_first_segment_start_in_interval
    %endif
    )
{
    %if neutral is None:
        if (GID_0 == 0)
            return;
    %endif

    const index_type interval_begin = interval_size * GID_0;
    const index_type interval_end = min(interval_begin + interval_size, N);

    %if is_segmented:
        interval_end = min(interval_end, g_first_segment_start_in_interval[GID_0]);
    %endif

    // value to add to this segment
    scan_type prev_group_sum;
    if (GID_0 == 0)
        prev_group_sum = ${neutral};
    else
        prev_group_sum = interval_results[GID_0 - 1];

    for(index_type unit_base = interval_begin;
        unit_base < interval_end;
        unit_base += WG_SIZE)
    {
        const index_type update_i = unit_base + LID_0;

        if(update_i < interval_end)
        {
            scan_type partial_val = partial_scan_buffer[update_i];
            scan_type value = SCAN_EXPR(prev_group_sum, partial_val);

            // printf("i: %d pgs: %d pv: %d val: %d\n", update_i, prev_group_sum, partial_val, value);

            OUTPUT_STMT(update_i, prev_item_unavailable_with_local_update, value);
        }
    }
}
"""

# }}}

# {{{ lookbehind update

# used for exclusive scan or output_statements that request look-behind

LOOKBEHIND_UPDATE_SOURCE = SHARED_PREAMBLE + """//CL//

#define OUTPUT_STMT(i, prev_item, item) { ${output_statement}; }

KERNEL
REQD_WG_SIZE(WG_SIZE, 1, 1)
void ${name_prefix}_final_update(
    ${argument_signature},
    const index_type N,
    const index_type interval_size,
    GLOBAL_MEM scan_type *interval_results,
    GLOBAL_MEM scan_type *partial_scan_buffer
    %if is_segmented:
        , GLOBAL_MEM index_type *g_first_segment_start_in_interval
    %endif
    )
{
    LOCAL_MEM scan_type ldata[WG_SIZE];

    const index_type interval_begin = interval_size * GID_0;
    const index_type interval_end = min(interval_begin + interval_size, N);

    // value to add to this segment
    scan_type carry = ${neutral};
    if(GID_0 != 0)
        carry = interval_results[GID_0 - 1];

    scan_type prev_value = carry; // (A)

    for (index_type unit_base = interval_begin;
        unit_base < interval_end;
        unit_base += WG_SIZE)
    {
        const index_type update_i = unit_base + LID_0;

        // load a work group's worth of data
        if (update_i < interval_end)
        {
            scan_type tmp = partial_scan_buffer[update_i];
            ldata[LID_0] = SCAN_EXPR(carry, tmp);
        }

        local_barrier();

        // perform right shift
        if (LID_0 != 0)
            prev_value = ldata[LID_0 - 1];
        /*
        else
            prev_value = carry (see (A)) OR last tail (see (B));
        */

        %if is_segmented:
        {
            scan_type scan_item_at_i = INPUT_EXPR(update_i)
            bool is_seg_start = IS_SEG_START(update_i, scan_item_at_i);
            if (is_seg_start)
                prev_value = ${neutral};
        }
        %endif

        if (update_i < interval_end)
        {
            scan_type value = ldata[LID_0];

            OUTPUT_STMT(update_i, prev_value, value)
        }

        if(LID_0 == 0)
            prev_value = ldata[WG_SIZE - 1]; // (B)

        local_barrier();
    }
}
"""

# }}}

# {{{ driver

# {{{ helpers

def _round_down_to_power_of_2(val):
    result = 2**bitlog2(val)
    if result > val:
        result >>=1

    assert result <= val
    return result

def _parse_args(arguments):
    from pyopencl.tools import parse_c_arg
    return [parse_c_arg(arg) for arg in arguments.split(",")]

def _get_scalar_arg_dtypes(arg_types):
    result = []

    from pyopencl.tools import ScalarArg
    for arg_type in arg_types:
        if isinstance(arg_type, ScalarArg):
            result.append(arg_type.dtype)
        else:
            result.append(None)

    return result

_PREFIX_WORDS = set("""
        ldata partial_scan_buffer global scan_offset
        segment_start_in_k_group carry
        g_first_segment_start_in_interval IS_SEG_START tmp Z
        val l_first_segment_start_in_k_group unit_size
        index_type interval_begin interval_size offset_end K
        SCAN_EXPR do_update NO_SEG_BOUNDARY WG_SIZE
        first_segment_start_in_k_group scan_type
        segment_start_in_subtree offset interval_results interval_end
        first_segment_start_in_subtree unit_base
        first_segment_start_in_interval k INPUT_EXPR
        prev_group_sum prev pv this add value n partial_val pgs OUTPUT_STMT
        is_seg_start update_i scan_item_at_i seq_i read_i
        """.split())

_IGNORED_WORDS = set("""
        typedef for endfor if void while endwhile endfor endif else const printf
        None return bool
        LID_2 LID_1 LID_0
        LDIM_0 LDIM_1 LDIM_2
        GDIM_0 GDIM_1 GDIM_2
        GID_0 GID_1 GID_2
        GLOBAL_MEM LOCAL_MEM_ARG WITHIN_KERNEL LOCAL_MEM KERNEL REQD_WG_SIZE
        local_barrier
        CLK_LOCAL_MEM_FENCE OPENCL EXTENSION
        pragma __attribute__ __global __kernel __local
        get_local_size get_local_id cl_khr_fp64 reqd_work_group_size
        get_num_groups barrier get_group_id

        _final_update _scan_intervals

        positions all padded integer its previous write based true writes 0
        has local worth scan_expr to read cannot not X items False bank
        four beginning follows applicable sum item min each indices works side
        scanning right summed relative used id out index avoid current state
        boundary True across be This reads groups along Otherwise undetermined
        store of times prior s update first regardless Each number because
        array unit from segment conflicts two parallel 2 empty define direction
        CL padding work tree bounds values and adds
        scan is allowed thus it an as enable at in occur sequentially end no
        storage data 1 largest may representable uses entry Y meaningful
        computations interval At the left dimension know d
        A load B group perform shift tail see last OR

        is_tail is_first_level input_expr argument_signature preamble
        double_support neutral output_statement index_type_max
        k_group_size name_prefix is_segmented index_ctype scan_ctype
        wg_size is_i_segment_start_expr

        a b prev_item i prev_item_unavailable_with_local_update prev_value
        N
        """.split())

def _make_template(s):
    leftovers = set()

    def replace_id(match):
        # avoid name clashes with user code by adding 'psc_' prefix to
        # identifiers.

        word = match.group(1)
        if word in _IGNORED_WORDS:
            return word
        elif word in _PREFIX_WORDS:
            return "psc_"+word
        else:
            leftovers.add(word)
            return word

    import re
    s = re.sub(r"\b([a-zA-Z0-9_]+)\b", replace_id, s)

    if leftovers:
        from warnings import warn
        warn("leftover words in identifier prefixing:" + " ".join(leftovers))

    return mako.template.Template(s, strict_undefined=True, disable_unicode=True)

from pytools import Record
class _ScanKernelInfo(Record):
    pass

# }}}

class GenericScanKernel(object):
    def __init__(self, ctx, dtype,
            arguments, input_expr, scan_expr, neutral, output_statement,
            is_i_segment_start_expr=None,
            partial_scan_buffer_name=None,
            name_prefix="scan", options=[], preamble="", devices=None):
        """
        :arg ctx: a :class:`pyopencl.Context` within which the code
            for this scan kernel will be generated.
        :arg dtype: the :class:`numpy.dtype` of the result
        :arg scan_expr: The associative operation carrying out the scan,
            represented as a C string. Its arguments are available as `a`
            and `b` when it is evaluated.
        :arg arguments: A string of comma-separated C argument declarations.
            If *arguments* is specified, then *input_expr* must also be
            specified.
        :arg input_expr: A C expression, encoded as a string, to be applied
            to each array entry when scan first touches it. *arguments*
            must be given if *input_expr* is given.
        :arg output_statement: a C statement that writes
            the output of the scan. It has access to the scan result as *item*,
            the preceding scan result item as *prev_item*, and the current index
            as *i*. *prev_item* is unavailable when using exclusive scan.
            *prev_item* in a segmented scan will be the neutral element
            at a segment boundary, not the immediately preceding item.
        :arg is_i_segment_start_expr: If given, makes the scan a segmented
            scan. Has access to the current index `i` and the input element
            as `a` and returns a bool. If it returns true, then previous
            sums will not spill over into the item with index i.

        The first array in the argument list determines the size of the index
        space over which the scan is carried out.

        All code fragments further have access to N, the number of elements
        being processed in the scan.
        """

        if isinstance(self, ExclusiveScanKernel) and neutral is None:
            raise ValueError("neutral element is required for exclusive scan")

        self.context = ctx
        dtype = self.dtype = np.dtype(dtype)

        if neutral is None:
            from warnings import warn
            warn("not specifying 'neutral' is deprecated and will lead to "
                    "wrong results if your scan is not in-place or your "
                    "'output_statement' otherwise does something non-trivial",
                    stacklevel=2)

        self.index_dtype = np.dtype(np.uint32)

        if devices is None:
            devices = ctx.devices
        self.devices = devices
        self.options = options

        self.arguments = arguments
        self.parsed_args = _parse_args(self.arguments)
        from pyopencl.tools import VectorArg
        self.first_array_idx = [
                i for i, arg in enumerate(self.parsed_args)
                if isinstance(arg, VectorArg)][0]

        self.is_segmented = is_i_segment_start_expr is not None
        if self.is_segmented:
            is_i_segment_start_expr = is_i_segment_start_expr.replace("\n", " ")

        use_lookbehind_update = "prev_item" in output_statement

        if self.is_segmented and use_lookbehind_update:
            # The final update in segmented exclusive scan must be able to
            # reconstruct where the segment boundaries were, and therefore
            # can't overwrite any of the input.
            partial_scan_buffer_name = None

        if partial_scan_buffer_name is not None:
            self.partial_scan_buffer_idx, = [
                    i for i, arg in enumerate(self.parsed_args)
                    if arg.name == partial_scan_buffer_name]
        else:
            self.partial_scan_buffer_idx = None

        # {{{ set up shared code dict

        from pytools import all
        from pyopencl.characterize import has_double_support

        self.code_variables = dict(
            preamble=preamble,
            name_prefix=name_prefix,
            index_ctype=dtype_to_ctype(self.index_dtype),
            index_type_max=str(np.iinfo(self.index_dtype).max),
            scan_ctype=dtype_to_ctype(dtype),
            is_segmented=self.is_segmented,
            scan_expr=scan_expr.replace("\n", " "),
            neutral=neutral.replace("\n", " "),
            double_support=all(
                has_double_support(dev) for dev in devices),
            )

        # }}}

        # {{{ loop to find usable workgroup size, build first-level scan

        trip_count = 0

        max_scan_wg_size = min(dev.max_work_group_size for dev in self.devices)

        while True:
            candidate_scan_info = self.build_scan_kernel(
                    max_scan_wg_size, arguments, input_expr,
                    is_i_segment_start_expr, is_first_level=True)

            # Will this device actually let us execute this kernel
            # at the desired work group size? Building it is the
            # only way to find out.
            kernel_max_wg_size = min(
                    candidate_scan_info.kernel.get_work_group_info(
                        cl.kernel_work_group_info.WORK_GROUP_SIZE,
                        dev)
                    for dev in self.devices)

            if candidate_scan_info.wg_size <= kernel_max_wg_size:
                break
            else:
                max_scan_wg_size = kernel_max_wg_size

            trip_count += 1
            assert trip_count <= 2

        self.first_level_scan_info = candidate_scan_info
        assert (_round_down_to_power_of_2(candidate_scan_info.wg_size)
                == candidate_scan_info.wg_size)

        # }}}

        # {{{ build second-level scan

        second_level_arguments = [
                "__global %s *interval_sums" % dtype_to_ctype(dtype)]
        second_level_build_kwargs = {}
        if self.is_segmented:
            second_level_arguments.append(
                    "__global %s *g_first_segment_start_in_interval_input"
                    % dtype_to_ctype(self.index_dtype))

            # is_i_segment_start_expr answers the question "should previous sums
            # spill over into this item". And since g_first_segment_start_in_interval_input
            # answers the question if a segment boundary was found in an interval of data,
            # then if not, it's ok to spill over.
            second_level_build_kwargs["is_i_segment_start_expr"] = \
                    "g_first_segment_start_in_interval_input[i] != NO_SEG_BOUNDARY"
        else:
            second_level_build_kwargs["is_i_segment_start_expr"] = None

        self.second_level_scan_info = self.build_scan_kernel(
                max_scan_wg_size,
                arguments=", ".join(second_level_arguments),
                input_expr="interval_sums[i]",
                is_first_level=False,
                **second_level_build_kwargs)

        assert min(
                candidate_scan_info.kernel.get_work_group_info(
                    cl.kernel_work_group_info.WORK_GROUP_SIZE,
                    dev)
                for dev in self.devices) >= max_scan_wg_size

        # }}}

        # {{{ build final update kernel

        self.update_wg_size = min(max_scan_wg_size, 256)

        if use_lookbehind_update:
            update_src = LOOKBEHIND_UPDATE_SOURCE
        else:
            update_src = LOCAL_UPDATE_SOURCE

        final_update_tpl = _make_template(update_src)
        final_update_src = str(final_update_tpl.render(
            wg_size=self.update_wg_size,
            output_statement=output_statement.replace("\n", " "),
            argument_signature=arguments.replace("\n", " "),
            is_i_segment_start_expr=is_i_segment_start_expr,
            input_expr=input_expr.replace("\n", " "),
            **self.code_variables))

        final_update_prg = cl.Program(self.context, final_update_src).build(options)
        self.final_update_knl = getattr(
                final_update_prg,
                name_prefix+"_final_update")
        self.final_update_knl.set_scalar_arg_dtypes(
                _get_scalar_arg_dtypes(self.parsed_args)
                + [self.index_dtype, self.index_dtype, None, None])

        # }}}

    def build_scan_kernel(self, max_wg_size, arguments, input_expr,
            is_i_segment_start_expr, is_first_level):
        scalar_arg_dtypes = _get_scalar_arg_dtypes(_parse_args(arguments))

        # Thrust says that 128 is big enough for GT200
        wg_size = _round_down_to_power_of_2(
                min(max_wg_size, 128))

        # k_group_size should be a power of two because of in-kernel
        # division by that number.

        if wg_size < 16:
            # Hello, Apple CPU. Nice to see you.
            k_group_size = 128 # FIXME: guesswork
        else:
            k_group_size = 8

        scan_tpl = _make_template(SCAN_INTERVALS_SOURCE)
        scan_intervals_src = str(scan_tpl.render(
            wg_size=wg_size,
            input_expr=input_expr,
            k_group_size=k_group_size,
            argument_signature=arguments.replace("\n", " "),
            is_i_segment_start_expr=is_i_segment_start_expr,
            is_first_level=is_first_level,
            **self.code_variables))

        prg = cl.Program(self.context, scan_intervals_src).build(self.options)

        knl = getattr(
                prg,
                self.code_variables["name_prefix"]+"_scan_intervals")

        scalar_arg_dtypes.extend(
                (None, self.index_dtype, self. index_dtype))
        if is_first_level:
            scalar_arg_dtypes.append(None) # interval_results
        if self.is_segmented and is_first_level:
            scalar_arg_dtypes.append(None) # g_first_segment_start_in_interval
        knl.set_scalar_arg_dtypes(scalar_arg_dtypes)

        return _ScanKernelInfo(
                kernel=knl, wg_size=wg_size, knl=knl, k_group_size=k_group_size)

    def __call__(self, *args, **kwargs):
        # {{{ argument processing

        allocator = kwargs.get("allocator")
        queue = kwargs.get("queue")

        if len(args) != len(self.parsed_args):
            raise TypeError("invalid number of arguments in "
                    "custom-arguments mode")

        first_array = args[self.first_array_idx]
        allocator = allocator or first_array.allocator
        queue = queue or first_array.queue

        n, = first_array.shape

        data_args = []
        from pyopencl.tools import VectorArg
        for arg_descr, arg_val in zip(self.parsed_args, args):
            if isinstance(arg_descr, VectorArg):
                data_args.append(arg_val.data)
            else:
                data_args.append(arg_val)

        # }}}

        l1_info = self.first_level_scan_info
        l2_info = self.second_level_scan_info

        # see CL source above for terminology
        unit_size  = l1_info.wg_size * l1_info.k_group_size
        max_intervals = 3*max(dev.max_compute_units for dev in self.devices)

        from pytools import uniform_interval_splitting
        interval_size, num_intervals = uniform_interval_splitting(
                n, unit_size, max_intervals)

        #print "n:%d interval_size: %d num_intervals: %d k_group_size:%d" % (
                #n, interval_size, num_intervals, l1_info.k_group_size)

        # {{{ first level scan of interval (one interval per block)

        interval_results = allocator(self.dtype.itemsize*num_intervals)

        if self.partial_scan_buffer_idx is None:
            partial_scan_buffer = allocator(n*self.dtype.itemsize)
        else:
            partial_scan_buffer = data_args[self.partial_scan_buffer_idx]

        scan1_args = data_args + [
                partial_scan_buffer, n, interval_size, interval_results,
                ]

        if self.code_variables["is_segmented"]:
            first_segment_start_in_interval = allocator(self.index_dtype.itemsize*num_intervals)
            scan1_args = scan1_args + (first_segment_start_in_interval,)

        l1_info.kernel(
                queue, (num_intervals,), (l1_info.wg_size,),
                *scan1_args, **dict(g_times_l=True))

        if 0:
            psb_host = np.empty(n, self.dtype)
            cl.enqueue_copy(queue, psb_host, partial_scan_buffer)
            print "PSB", psb_host

        # }}}

        # {{{ second level inclusive scan of per-interval results

        # can scan at most one interval
        assert interval_size >= num_intervals

        scan2_args = (interval_results, interval_results)
        if self.is_segmented:
            scan2_args = scan2_args + [first_segment_start_in_interval]
        scan2_args = scan2_args + (num_intervals, interval_size)

        l2_info.kernel(
                queue, (1,), (l1_info.wg_size,),
                *scan2_args, **dict(g_times_l=True))

        # }}}

        # {{{ update intervals with result of interval scan

        upd_args = data_args + [n, interval_size, interval_results, partial_scan_buffer]
        if self.is_segmented:
            upd_args = upd_args.append(first_segment_start_in_interval)

        self.final_update_knl(
                queue, (num_intervals,), (self.update_wg_size,),
                *upd_args, **dict(g_times_l=True))

        # }}}

# }}}

# {{{ compatibility interface

class _ScanKernelBase(GenericScanKernel):
    def __init__(self, ctx, dtype,
            scan_expr, neutral=None,
            name_prefix="scan", options=[], preamble="", devices=None):
        scan_ctype = dtype_to_ctype(dtype)
        GenericScanKernel.__init__(self,
                ctx, dtype,
                arguments="__global %s *input_ary, __global %s *output_ary" % (
                    scan_ctype, scan_ctype),
                input_expr="input_ary[i]",
                scan_expr=scan_expr,
                neutral=neutral,
                output_statement=self.ary_output_statement,
                partial_scan_buffer_name="output_ary",
                options=options, preamble=preamble, devices=devices)

    def __call__(self, input_ary, output_ary=None, allocator=None, queue=None):
        allocator = allocator or input_ary.allocator
        queue = queue or input_ary.queue or output_ary.queue

        if output_ary is None:
            output_ary = input_ary

        if isinstance(output_ary, (str, unicode)) and output_ary == "new":
            output_ary = cl_array.empty_like(input_ary, allocator=allocator)

        if input_ary.shape != output_ary.shape:
            raise ValueError("input and output must have the same shape")

        if not input_ary.flags.forc:
            raise RuntimeError("ScanKernel cannot "
                    "deal with non-contiguous arrays")

        n, = input_ary.shape

        if not n:
            return output_ary

        GenericScanKernel.__call__(self,
                input_ary, output_ary, allocator=allocator, queue=queue)

        return output_ary

class InclusiveScanKernel(_ScanKernelBase):
    ary_output_statement = "output_ary[i] = item;"

class ExclusiveScanKernel(_ScanKernelBase):
    ary_output_statement = "output_ary[i] = prev_item;"

# }}}

# {{{ higher-level trickery

@context_dependent_memoize
def get_copy_if_kernel(ctx, dtype, predicate):
    ctype = dtype_to_ctype(dtype)
    return GenericScanKernel(
            ctx, np.uint32,
            arguments="__global %s *ary, __global %s *out, __global unsigned long *count" % (ctype, ctype),
            input_expr="(%s) ? 1 : 0" % predicate,
            scan_expr="a+b", neutral="0",
            output_statement="""
                if (prev_item != item) out[item-1] = ary[i];
                if (i+1 == N) *count = item;
                """
            )

def copy_if(ary, predicate, queue=None):
    # FIXME use 64-bit scan, eventually
    # (not relevant for 6GB GPUs)

    knl = get_copy_if_kernel(ary.context, ary.dtype, predicate)
    out = cl_array.empty_like(ary)
    count = ary._new_with_changes(data=None, shape=(), strides=(), dtype=np.uint64)
    knl(ary, out, count, queue=queue)
    return out, count

def remove_if(array, predicate, **kwargs):
    pass

def partition(array, predicate):
    pass

def unique_by_key(array, key="", **kwargs):
    pass

# }}}

# vim: filetype=pyopencl:fdm=marker
