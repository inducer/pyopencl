"""Computation of reductions on vectors."""

from __future__ import division

__copyright__ = "Copyright (C) 2010 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

Based on code/ideas by Mark Harris <mharris@nvidia.com>.
None of the original source code remains.
"""


import pyopencl as cl
from pyopencl.tools import (
        context_dependent_memoize,
        dtype_to_ctype, KernelTemplateBase,
        _process_code_for_macro)
import numpy as np


# {{{ kernel source

KERNEL = """//CL//
    #define GROUP_SIZE ${group_size}
    #define READ_AND_MAP(i) (${map_expr})
    #define REDUCE(a, b) (${reduce_expr})

    % if double_support:
        #pragma OPENCL EXTENSION cl_khr_fp64: enable
        #define PYOPENCL_DEFINE_CDOUBLE
    % endif

    #include <pyopencl-complex.h>

    ${preamble}

    typedef ${out_type} out_type;

    __kernel void ${name}(
      __global out_type *out, ${arguments},
      unsigned int seq_count, unsigned int n)
    {
       ${arg_prep}

        __local out_type ldata[GROUP_SIZE];

        unsigned int lid = get_local_id(0);

        unsigned int i = get_group_id(0)*GROUP_SIZE*seq_count + lid;

        out_type acc = ${neutral};
        for (unsigned s = 0; s < seq_count; ++s)
        {
          if (i >= n)
            break;
          acc = REDUCE(acc, READ_AND_MAP(i));

          i += GROUP_SIZE;
        }

        ldata[lid] = acc;

        <%
          cur_size = group_size
        %>

        % while cur_size > no_sync_size:
            barrier(CLK_LOCAL_MEM_FENCE);

            <%
            new_size = cur_size // 2
            assert new_size * 2 == cur_size
            %>

            if (lid < ${new_size})
            {
                ldata[lid] = REDUCE(
                  ldata[lid],
                  ldata[lid + ${new_size}]);
            }

            <% cur_size = new_size %>

        % endwhile

        % if cur_size > 1:
            ## we need to synchronize one last time for entry into the
            ## no-sync region.

            barrier(CLK_LOCAL_MEM_FENCE);

            <%
            # NB: There's an exact duplicate of this calculation in the
            # %while loop below.

            new_size = cur_size // 2
            assert new_size * 2 == cur_size
            %>

            if (lid < ${new_size})
            {
                __local volatile out_type *lvdata = ldata;
                % while cur_size > 1:
                    <%
                    new_size = cur_size // 2
                    assert new_size * 2 == cur_size
                    %>

                    lvdata[lid] = REDUCE(
                      lvdata[lid],
                      lvdata[lid + ${new_size}]);

                    <% cur_size = new_size %>

                % endwhile

            }
        % endif

        if (lid == 0) out[get_group_id(0)] = ldata[0];
    }
    """

# }}}


# {{{ internal codegen frontends

def _get_reduction_source(
        ctx, out_type, out_type_size,
        neutral, reduce_expr, map_expr, parsed_args,
        name="reduce_kernel", preamble="", arg_prep="",
        device=None, max_group_size=None):

    if device is not None:
        devices = [device]
    else:
        devices = ctx.devices

    # {{{ compute group size

    def get_dev_group_size(device):
        # dirty fix for the RV770 boards
        max_work_group_size = device.max_work_group_size
        if "RV770" in device.name:
            max_work_group_size = 64

        # compute lmem limit
        from pytools import div_ceil
        lmem_wg_size = div_ceil(max_work_group_size, out_type_size)
        result = min(max_work_group_size, lmem_wg_size)

        # round down to power of 2
        from pyopencl.tools import bitlog2
        return 2**bitlog2(result)

    group_size = min(get_dev_group_size(dev) for dev in devices)

    if max_group_size is not None:
        group_size = min(max_group_size, group_size)

    # }}}

    # {{{ compute synchronization-less group size

    def get_dev_no_sync_size(device):
        from pyopencl.characterize import get_simd_group_size
        result = get_simd_group_size(device, out_type_size)

        if result is None:
            from warnings import warn
            warn("Reduction might be unnecessarily slow: "
                    "can't query SIMD group size")
            return 1

        return result

    no_sync_size = min(get_dev_no_sync_size(dev) for dev in devices)

    # }}}

    from mako.template import Template
    from pytools import all
    from pyopencl.characterize import has_double_support
    src = str(Template(KERNEL).render(
        out_type=out_type,
        arguments=", ".join(arg.declarator() for arg in parsed_args),
        group_size=group_size,
        no_sync_size=no_sync_size,
        neutral=neutral,
        reduce_expr=_process_code_for_macro(reduce_expr),
        map_expr=_process_code_for_macro(map_expr),
        name=name,
        preamble=preamble,
        arg_prep=arg_prep,
        double_support=all(has_double_support(dev) for dev in devices),
        ))

    from pytools import Record

    class ReductionInfo(Record):
        pass

    return ReductionInfo(
            context=ctx,
            source=src,
            group_size=group_size)


def get_reduction_kernel(stage,
         ctx, dtype_out,
         neutral, reduce_expr, map_expr=None, arguments=None,
         name="reduce_kernel", preamble="",
         device=None, options=[], max_group_size=None):

    if map_expr is None:
        if stage == 2:
            map_expr = "pyopencl_reduction_inp[i]"
        else:
            map_expr = "in[i]"

    from pyopencl.tools import (
            parse_arg_list, get_arg_list_scalar_arg_dtypes,
            get_arg_offset_adjuster_code, VectorArg)

    arg_prep = ""
    if stage == 1 and arguments is not None:
        arguments = parse_arg_list(arguments, with_offset=True)
        arg_prep = get_arg_offset_adjuster_code(arguments)

    if stage == 2 and arguments is not None:
        arguments = parse_arg_list(arguments)
        arguments = (
                [VectorArg(dtype_out, "pyopencl_reduction_inp")]
                + arguments)

    inf = _get_reduction_source(
            ctx, dtype_to_ctype(dtype_out), dtype_out.itemsize,
            neutral, reduce_expr, map_expr, arguments,
            name, preamble, arg_prep, device, max_group_size)

    inf.program = cl.Program(ctx, inf.source)
    inf.program.build(options)
    inf.kernel = getattr(inf.program, name)

    inf.arg_types = arguments

    inf.kernel.set_scalar_arg_dtypes(
            [None]
            + get_arg_list_scalar_arg_dtypes(inf.arg_types)
            + [np.uint32]*2)

    return inf

# }}}


# {{{ main reduction kernel

class ReductionKernel:
    def __init__(self, ctx, dtype_out,
            neutral, reduce_expr, map_expr=None, arguments=None,
            name="reduce_kernel", options=[], preamble=""):

        dtype_out = self.dtype_out = np.dtype(dtype_out)

        max_group_size = None
        trip_count = 0

        while True:
            self.stage_1_inf = get_reduction_kernel(1, ctx,
                    dtype_out,
                    neutral, reduce_expr, map_expr, arguments,
                    name=name+"_stage1", options=options, preamble=preamble,
                    max_group_size=max_group_size)

            kernel_max_wg_size = self.stage_1_inf.kernel.get_work_group_info(
                    cl.kernel_work_group_info.WORK_GROUP_SIZE,
                    ctx.devices[0])

            if self.stage_1_inf.group_size <= kernel_max_wg_size:
                break
            else:
                max_group_size = kernel_max_wg_size

            trip_count += 1
            assert trip_count <= 2

        self.stage_2_inf = get_reduction_kernel(2, ctx,
                dtype_out,
                neutral, reduce_expr, arguments=arguments,
                name=name+"_stage2", options=options, preamble=preamble,
                max_group_size=max_group_size)

        from pytools import any
        from pyopencl.tools import VectorArg
        assert any(
                isinstance(arg_tp, VectorArg)
                for arg_tp in self.stage_1_inf.arg_types), \
                "ReductionKernel can only be used with functions " \
                "that have at least one vector argument"

    def __call__(self, *args, **kwargs):
        MAX_GROUP_COUNT = 1024
        SMALL_SEQ_COUNT = 4

        from pyopencl.array import empty

        stage_inf = self.stage_1_inf

        queue = kwargs.pop("queue", None)
        wait_for = kwargs.pop("wait_for", None)
        return_event = kwargs.pop("return_event", False)

        if kwargs:
            raise TypeError("invalid keyword argument to reduction kernel")

        stage1_args = args

        while True:
            invocation_args = []
            vectors = []

            from pyopencl.tools import VectorArg
            for arg, arg_tp in zip(args, stage_inf.arg_types):
                if isinstance(arg_tp, VectorArg):
                    if not arg.flags.forc:
                        raise RuntimeError("ReductionKernel cannot "
                                "deal with non-contiguous arrays")

                    vectors.append(arg)
                    invocation_args.append(arg.base_data)
                    if arg_tp.with_offset:
                        invocation_args.append(arg.offset)
                else:
                    invocation_args.append(arg)

            repr_vec = vectors[0]
            sz = repr_vec.size

            if queue is not None:
                use_queue = queue
            else:
                use_queue = repr_vec.queue

            if sz <= stage_inf.group_size*SMALL_SEQ_COUNT*MAX_GROUP_COUNT:
                total_group_size = SMALL_SEQ_COUNT*stage_inf.group_size
                group_count = (sz + total_group_size - 1) // total_group_size
                seq_count = SMALL_SEQ_COUNT
            else:
                group_count = MAX_GROUP_COUNT
                macrogroup_size = group_count*stage_inf.group_size
                seq_count = (sz + macrogroup_size - 1) // macrogroup_size

            if group_count == 1:
                result = empty(use_queue,
                        (), self.dtype_out,
                        allocator=repr_vec.allocator)
            else:
                result = empty(use_queue,
                        (group_count,), self.dtype_out,
                        allocator=repr_vec.allocator)

            last_evt = stage_inf.kernel(
                    use_queue,
                    (group_count*stage_inf.group_size,),
                    (stage_inf.group_size,),
                    *([result.data]+invocation_args+[seq_count, sz]),
                    **dict(wait_for=wait_for))
            wait_for = [last_evt]

            if group_count == 1:
                if return_event:
                    return result, last_evt
                else:
                    return result
            else:
                stage_inf = self.stage_2_inf
                args = (result,) + stage1_args

# }}}


# {{{ template

class ReductionTemplate(KernelTemplateBase):
    def __init__(self,
            arguments, neutral, reduce_expr, map_expr=None,
            is_segment_start_expr=None, input_fetch_exprs=[],
            name_prefix="reduce", preamble="", template_processor=None):

        KernelTemplateBase.__init__(
                self, template_processor=template_processor)
        self.arguments = arguments
        self.reduce_expr = reduce_expr
        self.neutral = neutral
        self.map_expr = map_expr
        self.name_prefix = name_prefix
        self.preamble = preamble

    def build_inner(self, context, type_aliases=(), var_values=(),
            more_preamble="", more_arguments=(), declare_types=(),
            options=(), devices=None):
        renderer = self.get_renderer(
                type_aliases, var_values, context, options)

        arg_list = renderer.render_argument_list(
                self.arguments, more_arguments)

        type_decl_preamble = renderer.get_type_decl_preamble(
                context.devices[0], declare_types, arg_list)

        return ReductionKernel(context, renderer.type_aliases["reduction_t"],
                renderer(self.neutral), renderer(self.reduce_expr),
                renderer(self.map_expr),
                renderer.render_argument_list(self.arguments, more_arguments),
                name=renderer(self.name_prefix), options=list(options),
                preamble=(
                    type_decl_preamble
                    + "\n"
                    + renderer(self.preamble + "\n" + more_preamble)))

# }}}


# {{{ array reduction kernel getters

@context_dependent_memoize
def get_any_kernel(ctx, dtype_in):
    from pyopencl.tools import VectorArg
    return ReductionKernel(ctx, np.int8, "false", "a || b",
            map_expr="(bool) (in[i])",
            arguments=[VectorArg(dtype_in, "in")])


@context_dependent_memoize
def get_all_kernel(ctx, dtype_in):
    from pyopencl.tools import VectorArg
    return ReductionKernel(ctx, np.int8, "true", "a && b",
            map_expr="(bool) (in[i])",
            arguments=[VectorArg(dtype_in, "in")])


@context_dependent_memoize
def get_sum_kernel(ctx, dtype_out, dtype_in):
    if dtype_out is None:
        dtype_out = dtype_in

    return ReductionKernel(ctx, dtype_out, "0", "a+b",
            arguments="const %(tp)s *in"
            % {"tp": dtype_to_ctype(dtype_in)})


def _get_dot_expr(dtype_out, dtype_a, dtype_b, conjugate_first,
        has_double_support, index_expr="i"):
    if dtype_b is None:
        if dtype_a is None:
            dtype_b = dtype_out
        else:
            dtype_b = dtype_a

    if dtype_out is None:
        from pyopencl.compyte.array import get_common_dtype
        dtype_out = get_common_dtype(
                dtype_a.type(0), dtype_b.type(0),
                has_double_support)

    a_real_dtype = dtype_a.type(0).real.dtype
    b_real_dtype = dtype_b.type(0).real.dtype
    out_real_dtype = dtype_out.type(0).real.dtype

    a_is_complex = dtype_a.kind == "c"
    b_is_complex = dtype_b.kind == "c"
    out_is_complex = dtype_out.kind == "c"

    from pyopencl.elementwise import complex_dtype_to_name

    if a_is_complex and b_is_complex:
        a = "a[%s]" % index_expr
        b = "b[%s]" % index_expr
        if dtype_a != dtype_out:
            a = "%s_cast(%s)" % (complex_dtype_to_name(dtype_out), a)
        if dtype_b != dtype_out:
            b = "%s_cast(%s)" % (complex_dtype_to_name(dtype_out), b)

        if conjugate_first and a_is_complex:
            a = "%s_conj(%s)" % (
                    complex_dtype_to_name(dtype_out), a)

        map_expr = "%s_mul(%s, %s)" % (
                complex_dtype_to_name(dtype_out), a, b)
    else:
        a = "a[%s]" % index_expr
        b = "b[%s]" % index_expr

        if out_is_complex:
            if a_is_complex and dtype_a != dtype_out:
                a = "%s_cast(%s)" % (complex_dtype_to_name(dtype_out), a)
            if b_is_complex and dtype_b != dtype_out:
                b = "%s_cast(%s)" % (complex_dtype_to_name(dtype_out), b)

            if not a_is_complex and a_real_dtype != out_real_dtype:
                a = "(%s) (%s)" % (dtype_to_ctype(out_real_dtype), a)
            if not b_is_complex and b_real_dtype != out_real_dtype:
                b = "(%s) (%s)" % (dtype_to_ctype(out_real_dtype), b)

        if conjugate_first and a_is_complex:
            a = "%s_conj(%s)" % (
                    complex_dtype_to_name(dtype_out), a)

        map_expr = "%s*%s" % (a, b)

    return map_expr, dtype_out, dtype_b


@context_dependent_memoize
def get_dot_kernel(ctx, dtype_out, dtype_a=None, dtype_b=None,
        conjugate_first=False):
    from pyopencl.characterize import has_double_support
    map_expr, dtype_out, dtype_b = _get_dot_expr(
            dtype_out, dtype_a, dtype_b, conjugate_first,
            has_double_support=has_double_support(ctx.devices[0]))

    return ReductionKernel(ctx, dtype_out, neutral="0",
            reduce_expr="a+b", map_expr=map_expr,
            arguments=
            "const %(tp_a)s *a, "
            "const %(tp_b)s *b" % {
                "tp_a": dtype_to_ctype(dtype_a),
                "tp_b": dtype_to_ctype(dtype_b),
                })


@context_dependent_memoize
def get_subset_dot_kernel(ctx, dtype_out, dtype_subset, dtype_a=None, dtype_b=None,
        conjugate_first=False):
    from pyopencl.characterize import has_double_support
    map_expr, dtype_out, dtype_b = _get_dot_expr(
            dtype_out, dtype_a, dtype_b, conjugate_first,
            has_double_support=has_double_support(ctx.devices[0]),
            index_expr="lookup_tbl[i]")

    # important: lookup_tbl must be first--it controls the length
    return ReductionKernel(ctx, dtype_out, neutral="0",
            reduce_expr="a+b", map_expr=map_expr,
            arguments=
            "const %(tp_lut)s *lookup_tbl, "
            "const %(tp_a)s *a, "
            "const %(tp_b)s *b" % {
            "tp_lut": dtype_to_ctype(dtype_subset),
            "tp_a": dtype_to_ctype(dtype_a),
            "tp_b": dtype_to_ctype(dtype_b),
            })


def get_minmax_neutral(what, dtype):
    dtype = np.dtype(dtype)
    if issubclass(dtype.type, np.inexact):
        if what == "min":
            return "MY_INFINITY"
        elif what == "max":
            return "-MY_INFINITY"
        else:
            raise ValueError("what is not min or max.")
    else:
        if what == "min":
            return str(np.iinfo(dtype).max)
        elif what == "max":
            return str(np.iinfo(dtype).min)
        else:
            raise ValueError("what is not min or max.")


@context_dependent_memoize
def get_minmax_kernel(ctx, what, dtype):
    if dtype.kind == "f":
        reduce_expr = "f%s(a,b)" % what
    elif dtype.kind in "iu":
        reduce_expr = "%s(a,b)" % what
    else:
        raise TypeError("unsupported dtype specified")

    return ReductionKernel(ctx, dtype,
            neutral=get_minmax_neutral(what, dtype),
            reduce_expr="%(reduce_expr)s" % {"reduce_expr": reduce_expr},
            arguments="const %(tp)s *in" % {
                "tp": dtype_to_ctype(dtype),
                }, preamble="#define MY_INFINITY (1./0)")


@context_dependent_memoize
def get_subset_minmax_kernel(ctx, what, dtype, dtype_subset):
    if dtype.kind == "f":
        reduce_expr = "f%s(a,b)" % what
    elif dtype.kind in "iu":
        reduce_expr = "%s(a,b)" % what
    else:
        raise TypeError("unsupported dtype specified")

    return ReductionKernel(ctx, dtype,
            neutral=get_minmax_neutral(what, dtype),
            reduce_expr="%(reduce_expr)s" % {"reduce_expr": reduce_expr},
            map_expr="in[lookup_tbl[i]]",
            arguments=
            "const %(tp_lut)s *lookup_tbl, "
            "const %(tp)s *in" % {
            "tp": dtype_to_ctype(dtype),
            "tp_lut": dtype_to_ctype(dtype_subset),
            }, preamble="#define MY_INFINITY (1./0)")

# }}}

# vim: filetype=pyopencl:fdm=marker
