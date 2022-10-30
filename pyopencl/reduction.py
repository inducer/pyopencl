"""Computation of reductions on vectors."""

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

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy as np

import pyopencl as cl
from pyopencl.tools import (
        DtypedArgument, KernelTemplateBase,
        context_dependent_memoize, dtype_to_ctype,
        _process_code_for_macro)


# {{{ kernel source

KERNEL = r"""//CL//
    #define PCL_GROUP_SIZE ${group_size}
    #define PCL_READ_AND_MAP(i) (${map_expr})
    #define PCL_REDUCE(a, b) (${reduce_expr})

    % if double_support:
        #if __OPENCL_C_VERSION__ < 120
        #pragma OPENCL EXTENSION cl_khr_fp64: enable
        #endif
        #define PYOPENCL_DEFINE_CDOUBLE
    % endif

    #include <pyopencl-complex.h>

    ${preamble}

    typedef ${out_type} pcl_out_type;

    __kernel void ${name}(
      __global pcl_out_type *pcl_out__base, long pcl_out__offset,
      ${arguments}
      long pcl_start, long pcl_step, long pcl_stop,
      unsigned int pcl_seq_count, long n)
    {
        __global pcl_out_type *pcl_out = (__global pcl_out_type *) (
            (__global char *) pcl_out__base + pcl_out__offset);
        ${arg_prep}

        __local pcl_out_type pcl_ldata[PCL_GROUP_SIZE];

        unsigned int pcl_lid = get_local_id(0);

        const long pcl_base_idx =
            get_group_id(0)*PCL_GROUP_SIZE*pcl_seq_count + pcl_lid;
        long i = pcl_start + pcl_base_idx * pcl_step;

        pcl_out_type pcl_acc = ${neutral};
        for (unsigned pcl_s = 0; pcl_s < pcl_seq_count; ++pcl_s)
        {
          if (i >= pcl_stop)
            break;
          pcl_acc = PCL_REDUCE(pcl_acc, PCL_READ_AND_MAP(i));

          i += PCL_GROUP_SIZE*pcl_step;
        }

        pcl_ldata[pcl_lid] = pcl_acc;

        <%
          cur_size = group_size
        %>

        % while cur_size > 1:
            barrier(CLK_LOCAL_MEM_FENCE);

            <%
            new_size = cur_size // 2
            assert new_size * 2 == cur_size
            %>

            if (pcl_lid < ${new_size})
            {
                pcl_ldata[pcl_lid] = PCL_REDUCE(
                  pcl_ldata[pcl_lid],
                  pcl_ldata[pcl_lid + ${new_size}]);
            }

            <% cur_size = new_size %>

        % endwhile

        if (pcl_lid == 0) pcl_out[get_group_id(0)] = pcl_ldata[0];
    }
    """

# }}}


# {{{ internal codegen frontends

@dataclass(frozen=True)
class _ReductionInfo:
    context: cl.Context
    source: str
    group_size: int

    program: cl.Program
    kernel: cl.Kernel
    arg_types: List[DtypedArgument]


def _get_reduction_source(
        ctx: cl.Context,
        out_type: str,
        out_type_size: int,
        neutral: str,
        reduce_expr: str,
        map_expr: str,
        parsed_args: List[DtypedArgument],
        name: str = "reduce_kernel",
        preamble: str = "",
        arg_prep: str = "",
        device: Optional[cl.Device] = None,
        max_group_size: Optional[int] = None) -> Tuple[str, int]:

    if device is not None:
        devices = [device]
    else:
        devices = ctx.devices

    # {{{ compute group size

    def get_dev_group_size(device: cl.Device) -> int:
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

    from mako.template import Template
    from pyopencl.characterize import has_double_support

    arguments = ", ".join(arg.declarator() for arg in parsed_args)
    if parsed_args:
        arguments += ", "

    src = str(Template(KERNEL).render(
        out_type=out_type,
        group_size=group_size,
        arguments=arguments,
        neutral=neutral,
        reduce_expr=_process_code_for_macro(reduce_expr),
        map_expr=_process_code_for_macro(map_expr),
        name=name,
        preamble=preamble,
        arg_prep=arg_prep,
        double_support=all(has_double_support(dev) for dev in devices),
        ))

    return src, group_size


def get_reduction_kernel(
        stage: int,
        ctx: cl.Context,
        dtype_out: Any,
        neutral: str,
        reduce_expr: str,
        map_expr: Optional[str] = None,
        arguments: Optional[List[DtypedArgument]] = None,
        name: str = "reduce_kernel",
        preamble: str = "",
        device: Optional[cl.Device] = None,
        options: Any = None,
        max_group_size: Optional[int] = None) -> _ReductionInfo:
    if stage not in (1, 2):
        raise ValueError(f"unknown stage index: '{stage}'")

    if map_expr is None:
        map_expr = "pyopencl_reduction_inp[i]" if stage == 2 else "in[i]"

    from pyopencl.tools import (
            parse_arg_list, get_arg_list_scalar_arg_dtypes,
            get_arg_offset_adjuster_code, VectorArg)

    if arguments is None:
        raise ValueError("arguments must not be None")

    arguments = parse_arg_list(arguments, with_offset=True)
    arg_prep = get_arg_offset_adjuster_code(arguments)

    if stage == 2 and arguments is not None:
        arguments = (
                [VectorArg(dtype_out, "pyopencl_reduction_inp")]
                + arguments)

    source, group_size = _get_reduction_source(
            ctx, dtype_to_ctype(dtype_out), dtype_out.itemsize,
            neutral, reduce_expr, map_expr, arguments,
            name, preamble, arg_prep, device, max_group_size)

    program = cl.Program(ctx, source)
    program.build(options)

    kernel = getattr(program, name)
    kernel.set_scalar_arg_dtypes(
            [None, np.int64]
            + get_arg_list_scalar_arg_dtypes(arguments)
            + [np.int64]*3
            + [np.uint32, np.int64]
            )

    return _ReductionInfo(
        context=ctx,
        source=source,
        group_size=group_size,
        program=program,
        kernel=kernel,
        arg_types=arguments
        )

# }}}


# {{{ main reduction kernel

_MAX_GROUP_COUNT = 1024
_SMALL_SEQ_COUNT = 4


class ReductionKernel:
    """A kernel that performs a generic reduction on arrays.

    Generate a kernel that takes a number of scalar or vector *arguments*
    (at least one vector argument), performs the *map_expr* on each entry of
    the vector argument and then the *reduce_expr* on the outcome of that.
    *neutral* serves as an initial value. *preamble* offers the possibility
    to add preprocessor directives and other code (such as helper functions)
    to be added before the actual reduction kernel code.

    Vectors in *map_expr* should be indexed by the variable *i*. *reduce_expr*
    uses the formal values "a" and "b" to indicate two operands of a binary
    reduction operation. If you do not specify a *map_expr*, ``in[i]`` is
    automatically assumed and treated as the only one input argument.

    *dtype_out* specifies the :class:`numpy.dtype` in which the reduction is
    performed and in which the result is returned. *neutral* is specified as
    float or integer formatted as string. *reduce_expr* and *map_expr* are
    specified as string formatted operations and *arguments* is specified as a
    string formatted as a C argument list. *name* specifies the name as which
    the kernel is compiled. *options* are passed unmodified to
    :meth:`pyopencl.Program.build`. *preamble* specifies a string of code that
    is inserted before the actual kernels.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(
            self,
            ctx: cl.Context,
            dtype_out: Any,
            neutral: str,
            reduce_expr: str,
            map_expr: Optional[str] = None,
            arguments: Optional[Union[str, List[DtypedArgument]]] = None,
            name: str = "reduce_kernel",
            options: Any = None,
            preamble: str = "") -> None:
        if arguments is None:
            raise ValueError("arguments must not be None")

        from pyopencl.tools import parse_arg_list
        arguments = parse_arg_list(arguments, with_offset=True)

        dtype_out = self.dtype_out = np.dtype(dtype_out)

        max_group_size = None
        trip_count = 0

        while True:
            self.stage_1_inf = get_reduction_kernel(1, ctx,
                    dtype_out,
                    neutral, reduce_expr, map_expr, arguments,
                    name=f"{name}_stage1", options=options, preamble=preamble,
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
                name=f"{name}_stage2", options=options, preamble=preamble,
                max_group_size=max_group_size)

    def __call__(self, *args: Any, **kwargs: Any) -> cl.Event:
        """Invoke the generated kernel.

        |explain-waitfor|

        With *out* the resulting single-entry :class:`pyopencl.array.Array` can
        be specified. Because offsets are supported one can store results
        anywhere (e.g. ``out=a[3]``).

        .. note::

            The returned :class:`pyopencl.Event` corresponds only to part of the
            execution of the reduction. It is not suitable for profiling.

        .. versionadded:: 2011.1

        .. versionchanged:: 2014.2

            Added *out* parameter.

        .. versionchanged:: 2016.2

            *range_* and *slice_* added.

        :arg range: A :class:`slice` object. Specifies the range of indices on which
            the kernel will be executed. May not be given at the same time
            as *slice*.
        :arg slice: A :class:`slice` object.
            Specifies the range of indices on which the kernel will be
            executed, relative to the first vector-like argument.
            May not be given at the same time as *range*.
        :arg return_event: a boolean flag used to return an event for the
            reduction.

        :return: the resulting scalar as a single-entry :class:`pyopencl.array.Array`
            if *return_event* is *False*, otherwise a tuple
            ``(scalar_array, event)``.
        """

        queue = kwargs.pop("queue", None)
        allocator = kwargs.pop("allocator", None)
        wait_for = kwargs.pop("wait_for", None)
        return_event = kwargs.pop("return_event", False)
        out = kwargs.pop("out", None)

        range_ = kwargs.pop("range", None)
        slice_ = kwargs.pop("slice", None)

        if kwargs:
            raise TypeError("invalid keyword argument to reduction kernel")

        if wait_for is None:
            wait_for = []
        else:
            # We'll be modifying it below.
            wait_for = list(wait_for)

        from pyopencl.array import empty

        stage_inf = self.stage_1_inf
        stage1_args = args

        while True:
            invocation_args = []
            vectors = []

            array_empty = empty

            from pyopencl.tools import VectorArg
            for arg, arg_tp in zip(args, stage_inf.arg_types):
                if isinstance(arg_tp, VectorArg):
                    array_empty = arg.__class__
                    if not arg.flags.forc:
                        raise RuntimeError(
                            f"{type(self).__name__} cannot deal with "
                            "non-contiguous arrays")

                    vectors.append(arg)
                    invocation_args.append(arg.base_data)
                    if arg_tp.with_offset:
                        invocation_args.append(arg.offset)
                    wait_for.extend(arg.events)
                else:
                    invocation_args.append(arg)

            if vectors:
                repr_vec = vectors[0]
            else:
                repr_vec = None

            # {{{ range/slice processing

            if range_ is not None:
                if slice_ is not None:
                    raise TypeError("may not specify both range and slice "
                            "keyword arguments")

            else:
                if slice_ is None:
                    slice_ = slice(None)

                if repr_vec is None:
                    raise TypeError(
                            "must have vector argument when range is not specified")

                range_ = slice(*slice_.indices(repr_vec.size))

            assert range_ is not None

            start = range_.start
            if start is None:
                start = 0
            if range_.step is None:
                step = 1
            else:
                step = range_.step
            sz = abs(range_.stop - start)//step

            # }}}

            if queue is not None:
                use_queue = queue
            else:
                if repr_vec is None:
                    raise TypeError(
                        "must specify queue argument when no vector argument present"
                        )

                use_queue = repr_vec.queue

            if allocator is None:
                if repr_vec is None:
                    from pyopencl.tools import DeferredAllocator
                    allocator = DeferredAllocator(queue.context)
                else:
                    allocator = repr_vec.allocator

            if sz == 0:
                result = array_empty(
                        use_queue, (), self.dtype_out, allocator=allocator)
                group_count = 1
                seq_count = 0

            elif sz <= stage_inf.group_size*_SMALL_SEQ_COUNT*_MAX_GROUP_COUNT:
                total_group_size = _SMALL_SEQ_COUNT*stage_inf.group_size
                group_count = (sz + total_group_size - 1) // total_group_size
                seq_count = _SMALL_SEQ_COUNT

            else:
                group_count = _MAX_GROUP_COUNT
                macrogroup_size = group_count*stage_inf.group_size
                seq_count = (sz + macrogroup_size - 1) // macrogroup_size

            size_args = [start, step, range_.stop, seq_count, sz]

            if group_count == 1 and out is not None:
                result = out
            elif group_count == 1:
                result = array_empty(use_queue,
                        (), self.dtype_out,
                        allocator=allocator)
            else:
                result = array_empty(use_queue,
                        (group_count,), self.dtype_out,
                        allocator=allocator)

            last_evt = stage_inf.kernel(
                    use_queue,
                    (group_count*stage_inf.group_size,),
                    (stage_inf.group_size,),
                    *([result.base_data, result.offset]
                        + invocation_args + size_args),
                    wait_for=wait_for)
            wait_for = [last_evt]

            result.add_event(last_evt)

            if group_count == 1:
                if return_event:
                    return result, last_evt
                else:
                    return result
            else:
                stage_inf = self.stage_2_inf
                args = (result,) + stage1_args

                range_ = slice_ = None

# }}}


# {{{ template

class ReductionTemplate(KernelTemplateBase):
    def __init__(
            self,
            arguments: Union[str, List[DtypedArgument]],
            neutral: str,
            reduce_expr: str,
            map_expr: Optional[str] = None,
            is_segment_start_expr: Optional[str] = None,
            input_fetch_exprs: Optional[List[Tuple[str, str, int]]] = None,
            name_prefix: str = "reduce",
            preamble: str = "",
            template_processor: Any = None) -> None:
        super().__init__(template_processor=template_processor)

        if input_fetch_exprs is None:
            input_fetch_exprs = []

        self.arguments = arguments
        self.reduce_expr = reduce_expr
        self.neutral = neutral
        self.map_expr = map_expr
        self.name_prefix = name_prefix
        self.preamble = preamble

    def build_inner(self, context, type_aliases=(), var_values=(),
            more_preamble="", more_arguments=(), declare_types=(),
            options=None, devices=None):
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
                name=renderer(self.name_prefix), options=options,
                preamble=(
                    type_decl_preamble
                    + "\n"
                    + renderer(f"{self.preamble}\n{more_preamble}")))

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

    reduce_expr = "a+b"
    neutral_expr = "0"
    if dtype_out.kind == "c":
        from pyopencl.elementwise import complex_dtype_to_name
        dtname = complex_dtype_to_name(dtype_out)
        reduce_expr = f"{dtname}_add(a, b)"
        neutral_expr = f"{dtname}_new(0, 0)"

    return ReductionKernel(
            ctx, dtype_out, neutral_expr, reduce_expr,
            arguments="const {} *in".format(dtype_to_ctype(dtype_in)),
            )


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

    a_is_complex = dtype_a.kind == "c"
    b_is_complex = dtype_b.kind == "c"

    from pyopencl.elementwise import complex_dtype_to_name

    a = f"a[{index_expr}]"
    b = f"b[{index_expr}]"

    if a_is_complex and (dtype_a != dtype_out):
        a = "{}_cast({})".format(complex_dtype_to_name(dtype_out), a)
    if b_is_complex and (dtype_b != dtype_out):
        b = "{}_cast({})".format(complex_dtype_to_name(dtype_out), b)

    if a_is_complex and conjugate_first and a_is_complex:
        a = "{}_conj({})".format(
                complex_dtype_to_name(dtype_out), a)

    if a_is_complex and not b_is_complex:
        map_expr = "{}_mulr({}, {})".format(complex_dtype_to_name(dtype_out), a, b)
    elif not a_is_complex and b_is_complex:
        map_expr = "{}_rmul({}, {})".format(complex_dtype_to_name(dtype_out), a, b)
    elif a_is_complex and b_is_complex:
        map_expr = "{}_mul({}, {})".format(complex_dtype_to_name(dtype_out), a, b)
    else:
        map_expr = f"{a}*{b}"

    return map_expr, dtype_out, dtype_b


@context_dependent_memoize
def get_dot_kernel(ctx, dtype_out, dtype_a=None, dtype_b=None,
        conjugate_first=False):
    from pyopencl.characterize import has_double_support
    map_expr, dtype_out, dtype_b = _get_dot_expr(
            dtype_out, dtype_a, dtype_b, conjugate_first,
            has_double_support=has_double_support(ctx.devices[0]))

    reduce_expr = "a+b"
    neutral_expr = "0"
    if dtype_out.kind == "c":
        from pyopencl.elementwise import complex_dtype_to_name
        dtname = complex_dtype_to_name(dtype_out)
        reduce_expr = f"{dtname}_add(a, b)"
        neutral_expr = f"{dtname}_new(0, 0)"

    return ReductionKernel(ctx, dtype_out, neutral=neutral_expr,
            reduce_expr=reduce_expr, map_expr=map_expr,
            arguments=(
                "const {tp_a} *a, const {tp_b} *b".format(
                    tp_a=dtype_to_ctype(dtype_a),
                    tp_b=dtype_to_ctype(dtype_b),
                    ))
            )


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
            arguments=(
                "const {tp_lut} *lookup_tbl, const {tp_a} *a, const {tp_b} *b"
                .format(
                    tp_lut=dtype_to_ctype(dtype_subset),
                    tp_a=dtype_to_ctype(dtype_a),
                    tp_b=dtype_to_ctype(dtype_b),
                    ))
            )


_MINMAX_PREAMBLE = """
#define MY_INFINITY (1./0)
#define fmin_nanprop(a, b) (isnan(a) || isnan(b)) ? a+b : fmin(a, b)
#define fmax_nanprop(a, b) (isnan(a) || isnan(b)) ? a+b : fmax(a, b)
"""


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
        reduce_expr = f"f{what}_nanprop(a,b)"
    elif dtype.kind in "iu":
        reduce_expr = f"{what}(a,b)"
    else:
        raise TypeError("unsupported dtype specified")

    return ReductionKernel(ctx, dtype,
            neutral=get_minmax_neutral(what, dtype),
            reduce_expr=f"{reduce_expr}",
            arguments="const {tp} *in".format(
                tp=dtype_to_ctype(dtype),
                ), preamble=_MINMAX_PREAMBLE)


@context_dependent_memoize
def get_subset_minmax_kernel(ctx, what, dtype, dtype_subset):
    if dtype.kind == "f":
        reduce_expr = f"f{what}(a, b)"
    elif dtype.kind in "iu":
        reduce_expr = f"{what}(a, b)"
    else:
        raise TypeError("unsupported dtype specified")

    return ReductionKernel(ctx, dtype,
            neutral=get_minmax_neutral(what, dtype),
            reduce_expr=f"{reduce_expr}",
            map_expr="in[lookup_tbl[i]]",
            arguments=(
                "const {tp_lut} *lookup_tbl, "
                "const {tp} *in".format(
                    tp=dtype_to_ctype(dtype),
                    tp_lut=dtype_to_ctype(dtype_subset),
                    )),
            preamble=_MINMAX_PREAMBLE)

# }}}

# vim: filetype=pyopencl:fdm=marker
