"""Elementwise functionality."""


__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

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
"""


from typing import Any, Tuple
import enum
from pyopencl.tools import context_dependent_memoize
import numpy as np
import pyopencl as cl
from pytools import memoize_method
from pyopencl.tools import (dtype_to_ctype, VectorArg, ScalarArg,
        KernelTemplateBase, dtype_to_c_struct)


# {{{ elementwise kernel code generator

def get_elwise_program(context, arguments, operation,
        name="elwise_kernel", options=None,
        preamble="", loop_prep="", after_loop="",
        use_range=False):

    if use_range:
        body = r"""//CL//
          if (step < 0)
          {
            for (i = start + (work_group_start + lid)*step;
              i > stop; i += gsize*step)
            {
              %(operation)s;
            }
          }
          else
          {
            for (i = start + (work_group_start + lid)*step;
              i < stop; i += gsize*step)
            {
              %(operation)s;
            }
          }
          """
    else:
        body = """//CL//
          for (i = work_group_start + lid; i < n; i += gsize)
          {
            %(operation)s;
          }
          """

    import re
    return_match = re.search(r"\breturn\b", operation)
    if return_match is not None:
        from warnings import warn
        warn("Using a 'return' statement in an element-wise operation will "
                "likely lead to incorrect results. Use "
                "PYOPENCL_ELWISE_CONTINUE instead.",
                stacklevel=3)

    source = ("""//CL//
        {preamble}

        #define PYOPENCL_ELWISE_CONTINUE continue

        __kernel void {name}({arguments})
        {{
          int lid = get_local_id(0);
          int gsize = get_global_size(0);
          int work_group_start = get_local_size(0)*get_group_id(0);
          long i;

          {loop_prep};
          {body}
          {after_loop};
        }}
        """.format(
            arguments=", ".join(arg.declarator() for arg in arguments),
            name=name,
            preamble=preamble,
            loop_prep=loop_prep,
            after_loop=after_loop,
            body=body % dict(operation=operation),
            ))

    from pyopencl import Program
    return Program(context, source).build(options)


def get_elwise_kernel_and_types(context, arguments, operation,
        name="elwise_kernel", options=None, preamble="", use_range=False,
        **kwargs):

    from pyopencl.tools import parse_arg_list, get_arg_offset_adjuster_code
    parsed_args = parse_arg_list(arguments, with_offset=True)

    auto_preamble = kwargs.pop("auto_preamble", True)

    pragmas = []
    includes = []
    have_double_pragma = False
    have_complex_include = False

    if auto_preamble:
        for arg in parsed_args:
            if arg.dtype in [np.float64, np.complex128]:
                if not have_double_pragma:
                    pragmas.append("""
                        #if __OPENCL_C_VERSION__ < 120
                        #pragma OPENCL EXTENSION cl_khr_fp64: enable
                        #endif
                        #define PYOPENCL_DEFINE_CDOUBLE
                        """)
                    have_double_pragma = True
            if arg.dtype.kind == "c":
                if not have_complex_include:
                    includes.append("#include <pyopencl-complex.h>\n")
                    have_complex_include = True

    if pragmas or includes:
        preamble = "\n".join(pragmas+includes) + "\n" + preamble

    if use_range:
        parsed_args.extend([
            ScalarArg(np.intp, "start"),
            ScalarArg(np.intp, "stop"),
            ScalarArg(np.intp, "step"),
            ])
    else:
        parsed_args.append(ScalarArg(np.intp, "n"))

    loop_prep = kwargs.pop("loop_prep", "")
    loop_prep = get_arg_offset_adjuster_code(parsed_args) + loop_prep
    prg = get_elwise_program(
        context, parsed_args, operation,
        name=name, options=options, preamble=preamble,
        use_range=use_range, loop_prep=loop_prep, **kwargs)

    from pyopencl.tools import get_arg_list_arg_types

    kernel = getattr(prg, name)
    kernel.set_scalar_arg_dtypes(get_arg_list_arg_types(parsed_args))

    return kernel, parsed_args


def get_elwise_kernel(context, arguments, operation,
        name="elwise_kernel", options=None, **kwargs):
    """Return a L{pyopencl.Kernel} that performs the same scalar operation
    on one or several vectors.
    """
    func, arguments = get_elwise_kernel_and_types(
        context, arguments, operation,
        name=name, options=options, **kwargs)

    return func

# }}}


# {{{ ElementwiseKernel driver

class ElementwiseKernel:
    """
    A kernel that takes a number of scalar or vector *arguments* and performs
    an *operation* specified as a snippet of C on these arguments.

    :arg arguments: a string formatted as a C argument list.
    :arg operation: a snippet of C that carries out the desired 'map'
        operation.  The current index is available as the variable *i*.
        *operation* may contain the statement ``PYOPENCL_ELWISE_CONTINUE``,
        which will terminate processing for the current element.
    :arg name: the function name as which the kernel is compiled
    :arg options: passed unmodified to :meth:`pyopencl.Program.build`.
    :arg preamble: a piece of C source code that gets inserted outside of the
        function context in the elementwise operation's kernel source code.

    .. warning :: Using a `return` statement in *operation* will lead to
        incorrect results, as some elements may never get processed. Use
        ``PYOPENCL_ELWISE_CONTINUE`` instead.

    .. versionchanged:: 2013.1
        Added ``PYOPENCL_ELWISE_CONTINUE``.
    """

    def __init__(self, context, arguments, operation,
            name="elwise_kernel", options=None, **kwargs):
        self.context = context
        self.arguments = arguments
        self.operation = operation
        self.name = name
        self.options = options
        self.kwargs = kwargs

    @memoize_method
    def get_kernel(self, use_range):
        knl, arg_descrs = get_elwise_kernel_and_types(
            self.context, self.arguments, self.operation,
            name=self.name, options=self.options,
            use_range=use_range, **self.kwargs)

        for arg in arg_descrs:
            if isinstance(arg, VectorArg) and not arg.with_offset:
                from warnings import warn
                warn("ElementwiseKernel '%s' used with VectorArgs that do not "
                        "have offset support enabled. This usage is deprecated. "
                        "Just pass with_offset=True to VectorArg, everything should "
                        "sort itself out automatically." % self.name,
                        DeprecationWarning)

        if not [i for i, arg in enumerate(arg_descrs)
                if isinstance(arg, VectorArg)]:
            raise RuntimeError(
                "ElementwiseKernel can only be used with "
                "functions that have at least one "
                "vector argument")
        return knl, arg_descrs

    def __call__(self, *args, **kwargs):
        repr_vec = None

        range_ = kwargs.pop("range", None)
        slice_ = kwargs.pop("slice", None)
        capture_as = kwargs.pop("capture_as", None)

        use_range = range_ is not None or slice_ is not None
        kernel, arg_descrs = self.get_kernel(use_range)

        queue = kwargs.pop("queue", None)
        wait_for = kwargs.pop("wait_for", None)

        if wait_for is None:
            wait_for = []
        else:
            # We'll be modifying it below.
            wait_for = list(wait_for)

        # {{{ assemble arg array

        invocation_args = []
        for arg, arg_descr in zip(args, arg_descrs):
            if isinstance(arg_descr, VectorArg):
                if repr_vec is None:
                    repr_vec = arg

                invocation_args.append(arg)
            else:
                invocation_args.append(arg)

        # }}}

        if kwargs:
            raise TypeError("unknown keyword arguments: '%s'"
                    % ", ".join(kwargs))

        if queue is None:
            queue = repr_vec.queue

        if slice_ is not None:
            if range_ is not None:
                raise TypeError("may not specify both range and slice "
                        "keyword arguments")

            range_ = slice(*slice_.indices(repr_vec.size))

        max_wg_size = kernel.get_work_group_info(
                cl.kernel_work_group_info.WORK_GROUP_SIZE,
                queue.device)

        if range_ is not None:
            start = range_.start
            if start is None:
                start = 0
            invocation_args.append(start)
            invocation_args.append(range_.stop)
            if range_.step is None:
                step = 1
            else:
                step = range_.step

            invocation_args.append(step)

            from pyopencl.array import _splay
            gs, ls = _splay(queue.device,
                    abs(range_.stop - start)//step,
                    max_wg_size)
        else:
            invocation_args.append(repr_vec.size)
            gs, ls = repr_vec._get_sizes(queue, max_wg_size)

        if capture_as is not None:
            kernel.set_args(*invocation_args)
            kernel.capture_call(
                    capture_as, queue,
                    gs, ls, *invocation_args, wait_for=wait_for)

        return kernel(queue, gs, ls, *invocation_args, wait_for=wait_for)

# }}}


# {{{ template

class ElementwiseTemplate(KernelTemplateBase):
    def __init__(self,
            arguments, operation, name="elwise", preamble="",
            template_processor=None):

        KernelTemplateBase.__init__(self,
                template_processor=template_processor)
        self.arguments = arguments
        self.operation = operation
        self.name = name
        self.preamble = preamble

    def build_inner(self, context, type_aliases=(), var_values=(),
            more_preamble="", more_arguments=(), declare_types=(),
            options=None):
        renderer = self.get_renderer(
                type_aliases, var_values, context, options)

        arg_list = renderer.render_argument_list(
                self.arguments, more_arguments, with_offset=True)
        type_decl_preamble = renderer.get_type_decl_preamble(
                context.devices[0], declare_types, arg_list)

        return ElementwiseKernel(context,
            arg_list, renderer(self.operation),
            name=renderer(self.name), options=options,
            preamble=(
                type_decl_preamble
                + "\n"
                + renderer(self.preamble + "\n" + more_preamble)),
            auto_preamble=False)

# }}}


# {{{ argument kinds

class ArgumentKind(enum.Enum):
    ARRAY = enum.auto()
    DEV_SCALAR = enum.auto()
    SCALAR = enum.auto()


def get_argument_kind(v: Any) -> ArgumentKind:
    from pyopencl.array import Array
    if isinstance(v, Array):
        if v.shape == ():
            return ArgumentKind.DEV_SCALAR
        else:
            return ArgumentKind.ARRAY
    else:
        return ArgumentKind.SCALAR


def get_decl_and_access_for_kind(name: str, kind: ArgumentKind) -> Tuple[str, str]:
    if kind == ArgumentKind.ARRAY:
        return f"*{name}", f"{name}[i]"
    elif kind == ArgumentKind.SCALAR:
        return f"{name}", name
    elif kind == ArgumentKind.DEV_SCALAR:
        return f"*{name}", f"{name}[0]"
    else:
        raise AssertionError()

# }}}


# {{{ kernels supporting array functionality

@context_dependent_memoize
def get_take_kernel(context, dtype, idx_dtype, vec_count=1):
    ctx = {
            "idx_tp": dtype_to_ctype(idx_dtype),
            "tp": dtype_to_ctype(dtype),
            }

    args = ([VectorArg(dtype, "dest" + str(i), with_offset=True)
             for i in range(vec_count)]
            + [VectorArg(dtype, "src" + str(i), with_offset=True)
               for i in range(vec_count)]
            + [VectorArg(idx_dtype, "idx", with_offset=True)])
    body = (
            ("%(idx_tp)s src_idx = idx[i];\n" % ctx)
            + "\n".join(
                "dest%d[i] = src%d[src_idx];" % (i, i)
                for i in range(vec_count)))

    return get_elwise_kernel(context, args, body,
            preamble=dtype_to_c_struct(context.devices[0], dtype),
            name="take")


@context_dependent_memoize
def get_take_put_kernel(context, dtype, idx_dtype, with_offsets, vec_count=1):
    ctx = {
            "idx_tp": dtype_to_ctype(idx_dtype),
            "tp": dtype_to_ctype(dtype),
            }

    args = [
            VectorArg(dtype, "dest%d" % i)
            for i in range(vec_count)
            ] + [
                VectorArg(idx_dtype, "gmem_dest_idx", with_offset=True),
                VectorArg(idx_dtype, "gmem_src_idx", with_offset=True),
            ] + [
                VectorArg(dtype, "src%d" % i, with_offset=True)
                for i in range(vec_count)
            ] + [
                ScalarArg(idx_dtype, "offset%d" % i)
                for i in range(vec_count) if with_offsets
            ]

    if with_offsets:
        def get_copy_insn(i):
            return ("dest%d[dest_idx] = "
                    "src%d[src_idx+offset%d];"
                    % (i, i, i))
    else:
        def get_copy_insn(i):
            return ("dest%d[dest_idx] = "
                    "src%d[src_idx];" % (i, i))

    body = (("%(idx_tp)s src_idx = gmem_src_idx[i];\n"
                "%(idx_tp)s dest_idx = gmem_dest_idx[i];\n" % ctx)
            + "\n".join(get_copy_insn(i) for i in range(vec_count)))

    return get_elwise_kernel(context, args, body,
            preamble=dtype_to_c_struct(context.devices[0], dtype),
            name="take_put")


@context_dependent_memoize
def get_put_kernel(context, dtype, idx_dtype, vec_count=1):
    ctx = {
            "idx_tp": dtype_to_ctype(idx_dtype),
            "tp": dtype_to_ctype(dtype),
            }

    args = [
            VectorArg(dtype, "dest%d" % i, with_offset=True)
            for i in range(vec_count)
            ] + [
                VectorArg(idx_dtype, "gmem_dest_idx", with_offset=True),
            ] + [
                VectorArg(dtype, "src%d" % i, with_offset=True)
                for i in range(vec_count)
            ] + [
                VectorArg(np.uint8, "use_fill", with_offset=True)
            ] + [
                VectorArg(np.int64, "val_ary_lengths", with_offset=True)
            ]

    body = (
            "%(idx_tp)s dest_idx = gmem_dest_idx[i];\n" % ctx
            + "\n".join(
                    "dest{i}[dest_idx] = (use_fill[{i}] ? src{i}[0] : "
                    "src{i}[i % val_ary_lengths[{i}]]);".format(i=i)
                    for i in range(vec_count)
                    )
            )

    return get_elwise_kernel(context, args, body,
            preamble=dtype_to_c_struct(context.devices[0], dtype),
            name="put")


@context_dependent_memoize
def get_copy_kernel(context, dtype_dest, dtype_src):
    src = "src[i]"
    if dtype_dest.kind == "c" != dtype_src.kind:
        src = "{}_fromreal({})".format(complex_dtype_to_name(dtype_dest), src)

    if dtype_dest.kind == "c" and dtype_src != dtype_dest:
        src = "{}_cast({})".format(complex_dtype_to_name(dtype_dest), src),

    if dtype_dest != dtype_src and (
            dtype_dest.kind == "V" or dtype_src.kind == "V"):
        raise TypeError("copying between non-identical struct types")

    return get_elwise_kernel(context,
            "{tp_dest} *dest, {tp_src} *src".format(
                tp_dest=dtype_to_ctype(dtype_dest),
                tp_src=dtype_to_ctype(dtype_src),
                ),
            "dest[i] = %s" % src,
            preamble=dtype_to_c_struct(context.devices[0], dtype_dest),
            name="copy")


def complex_dtype_to_name(dtype):
    if dtype == np.complex128:
        return "cdouble"
    elif dtype == np.complex64:
        return "cfloat"
    else:
        raise RuntimeError("invalid complex type")


def real_dtype(dtype):
    return dtype.type(0).real.dtype


@context_dependent_memoize
def get_axpbyz_kernel(context, dtype_x, dtype_y, dtype_z,
                      x_is_scalar=False, y_is_scalar=False):
    result_t = dtype_to_ctype(dtype_z)

    x_is_complex = dtype_x.kind == "c"
    y_is_complex = dtype_y.kind == "c"

    x = "x[0]" if x_is_scalar else "x[i]"
    y = "y[0]" if y_is_scalar else "y[i]"

    if dtype_z.kind == "c":
        # a and b will always be complex here.
        z_ct = complex_dtype_to_name(dtype_z)

        if x_is_complex:
            ax = f"{z_ct}_mul(a, {z_ct}_cast({x}))"
        else:
            ax = f"{z_ct}_mulr(a, {x})"

        if y_is_complex:
            by = f"{z_ct}_mul(b, {z_ct}_cast({y}))"
        else:
            by = f"{z_ct}_mulr(b, {y})"

        result = f"{z_ct}_add({ax}, {by})"
    else:
        # real-only

        ax = f"a*(({result_t}) {x})"
        by = f"b*(({result_t}) {y})"

        result = f"{ax} + {by}"

    return get_elwise_kernel(context,
            "{tp_z} *z, {tp_z} a, {tp_x} *x, {tp_z} b, {tp_y} *y".format(
                tp_x=dtype_to_ctype(dtype_x),
                tp_y=dtype_to_ctype(dtype_y),
                tp_z=dtype_to_ctype(dtype_z),
                ),
            "z[i] = %s" % result,
            name="axpbyz")


@context_dependent_memoize
def get_axpbz_kernel(context, dtype_a, dtype_x, dtype_b, dtype_z):
    a_is_complex = dtype_a.kind == "c"
    x_is_complex = dtype_x.kind == "c"
    b_is_complex = dtype_b.kind == "c"

    z_is_complex = dtype_z.kind == "c"

    ax = "a*x[i]"
    if x_is_complex:
        a = "a"
        x = "x[i]"

        if dtype_x != dtype_z:
            x = "{}_cast({})".format(complex_dtype_to_name(dtype_z), x)

        if a_is_complex:
            if dtype_a != dtype_z:
                a = "{}_cast({})".format(complex_dtype_to_name(dtype_z), a)

            ax = "{}_mul({}, {})".format(complex_dtype_to_name(dtype_z), a, x)
        else:
            ax = "{}_rmul({}, {})".format(complex_dtype_to_name(dtype_z), a, x)
    elif a_is_complex:
        a = "a"
        x = "x[i]"

        if dtype_a != dtype_z:
            a = "{}_cast({})".format(complex_dtype_to_name(dtype_z), a)
        ax = "{}_mulr({}, {})".format(complex_dtype_to_name(dtype_z), a, x)

    b = "b"
    if z_is_complex and not b_is_complex:
        b = "{}_fromreal({})".format(complex_dtype_to_name(dtype_z), b)

    if z_is_complex and not (a_is_complex or x_is_complex):
        ax = "{}_fromreal({})".format(complex_dtype_to_name(dtype_z), ax)

    if z_is_complex:
        ax = "{}_cast({})".format(complex_dtype_to_name(dtype_z), ax)
        b = "{}_cast({})".format(complex_dtype_to_name(dtype_z), b)

    if a_is_complex or x_is_complex or b_is_complex:
        expr = "{root}_add({ax}, {b})".format(
                ax=ax,
                b=b,
                root=complex_dtype_to_name(dtype_z))
    else:
        expr = f"{ax} + {b}"

    return get_elwise_kernel(context,
            "{tp_z} *z, {tp_a} a, {tp_x} *x,{tp_b} b".format(
                tp_a=dtype_to_ctype(dtype_a),
                tp_x=dtype_to_ctype(dtype_x),
                tp_b=dtype_to_ctype(dtype_b),
                tp_z=dtype_to_ctype(dtype_z),
                ),
            "z[i] = " + expr,
            name="axpb")


@context_dependent_memoize
def get_multiply_kernel(context, dtype_x, dtype_y, dtype_z,
                        x_is_scalar=False, y_is_scalar=False):
    x_is_complex = dtype_x.kind == "c"
    y_is_complex = dtype_y.kind == "c"

    x = "x[0]" if x_is_scalar else "x[i]"
    y = "y[0]" if y_is_scalar else "y[i]"

    if x_is_complex and dtype_x != dtype_z:
        x = "{}_cast({})".format(complex_dtype_to_name(dtype_z), x)
    if y_is_complex and dtype_y != dtype_z:
        y = "{}_cast({})".format(complex_dtype_to_name(dtype_z), y)

    if x_is_complex and y_is_complex:
        xy = "{}_mul({}, {})".format(complex_dtype_to_name(dtype_z), x, y)
    elif x_is_complex and not y_is_complex:
        xy = "{}_mulr({}, {})".format(complex_dtype_to_name(dtype_z), x, y)
    elif not x_is_complex and y_is_complex:
        xy = "{}_rmul({}, {})".format(complex_dtype_to_name(dtype_z), x, y)
    else:
        xy = f"{x} * {y}"

    return get_elwise_kernel(context,
            "{tp_z} *z, {tp_x} *x, {tp_y} *y".format(
                tp_x=dtype_to_ctype(dtype_x),
                tp_y=dtype_to_ctype(dtype_y),
                tp_z=dtype_to_ctype(dtype_z),
                ),
            "z[i] = %s" % xy,
            name="multiply")


@context_dependent_memoize
def get_divide_kernel(context, dtype_x, dtype_y, dtype_z,
                      x_is_scalar=False, y_is_scalar=False):
    x_is_complex = dtype_x.kind == "c"
    y_is_complex = dtype_y.kind == "c"
    z_is_complex = dtype_z.kind == "c"

    x = "x[0]" if x_is_scalar else "x[i]"
    y = "y[0]" if y_is_scalar else "y[i]"

    if z_is_complex and dtype_x != dtype_y:
        if x_is_complex and dtype_x != dtype_z:
            x = "{}_cast({})".format(complex_dtype_to_name(dtype_z), x)
        if y_is_complex and dtype_y != dtype_z:
            y = "{}_cast({})".format(complex_dtype_to_name(dtype_z), y)
    else:
        if dtype_x != dtype_z:
            x = f"({dtype_to_ctype(dtype_z)}) ({x})"
        if dtype_y != dtype_z:
            y = f"({dtype_to_ctype(dtype_z)}) ({y})"

    if x_is_complex and y_is_complex:
        xoy = "{}_divide({}, {})".format(complex_dtype_to_name(dtype_z), x, y)
    elif not x_is_complex and y_is_complex:
        xoy = "{}_rdivide({}, {})".format(complex_dtype_to_name(dtype_z), x, y)
    elif x_is_complex and not y_is_complex:
        xoy = "{}_divider({}, {})".format(complex_dtype_to_name(dtype_z), x, y)
    else:
        xoy = f"{x} / {y}"

    if z_is_complex:
        xoy = "{}_cast({})".format(complex_dtype_to_name(dtype_z), xoy)

    return get_elwise_kernel(context,
            "{tp_z} *z, {tp_x} *x, {tp_y} *y".format(
                tp_x=dtype_to_ctype(dtype_x),
                tp_y=dtype_to_ctype(dtype_y),
                tp_z=dtype_to_ctype(dtype_z),
                ),
            "z[i] = %s" % xoy,
            name="divide")


@context_dependent_memoize
def get_rdivide_elwise_kernel(context, dtype_x, dtype_y, dtype_z):
    # implements y / x!
    x_is_complex = dtype_x.kind == "c"
    y_is_complex = dtype_y.kind == "c"
    z_is_complex = dtype_z.kind == "c"

    x = "x[i]"
    y = "y"

    if z_is_complex and dtype_x != dtype_y:
        if x_is_complex and dtype_x != dtype_z:
            x = "{}_cast({})".format(complex_dtype_to_name(dtype_z), x)
        if y_is_complex and dtype_y != dtype_z:
            y = "{}_cast({})".format(complex_dtype_to_name(dtype_z), y)

    if x_is_complex and y_is_complex:
        yox = "{}_divide({}, {})".format(complex_dtype_to_name(dtype_z), y, x)
    elif not y_is_complex and x_is_complex:
        yox = "{}_rdivide({}, {})".format(complex_dtype_to_name(dtype_z), y, x)
    elif y_is_complex and not x_is_complex:
        yox = "{}_divider({}, {})".format(complex_dtype_to_name(dtype_z), y, x)
    else:
        yox = f"{y} / {x}"

    return get_elwise_kernel(context,
            "{tp_z} *z, {tp_x} *x, {tp_y} y".format(
                tp_x=dtype_to_ctype(dtype_x),
                tp_y=dtype_to_ctype(dtype_y),
                tp_z=dtype_to_ctype(dtype_z),
                ),
            "z[i] = %s" % yox,
            name="divide_r")


@context_dependent_memoize
def get_fill_kernel(context, dtype):
    return get_elwise_kernel(context,
            "{tp} *z, {tp} a".format(
                tp=dtype_to_ctype(dtype),
                ),
            "z[i] = a",
            preamble=dtype_to_c_struct(context.devices[0], dtype),
            name="fill")


@context_dependent_memoize
def get_reverse_kernel(context, dtype):
    return get_elwise_kernel(context,
            "{tp} *z, {tp} *y".format(
                tp=dtype_to_ctype(dtype),
                ),
            "z[i] = y[n-1-i]",
            name="reverse")


@context_dependent_memoize
def get_arange_kernel(context, dtype):
    if dtype.kind == "c":
        expr = (
                "{root}_add(start, {root}_rmul(i, step))"
                .format(root=complex_dtype_to_name(dtype)))
    else:
        expr = "start + ((%s) i)*step" % dtype_to_ctype(dtype)

    return get_elwise_kernel(context, [
        VectorArg(dtype, "z", with_offset=True),
        ScalarArg(dtype, "start"),
        ScalarArg(dtype, "step"),
        ],
        "z[i] = " + expr,
        name="arange")


@context_dependent_memoize
def get_pow_kernel(context, dtype_x, dtype_y, dtype_z,
        is_base_array, is_exp_array):
    if is_base_array:
        x = "x[i]"
        x_ctype = "%(tp_x)s *x"
    else:
        x = "x"
        x_ctype = "%(tp_x)s x"

    if is_exp_array:
        y = "y[i]"
        y_ctype = "%(tp_y)s *y"
    else:
        y = "y"
        y_ctype = "%(tp_y)s y"

    x_is_complex = dtype_x.kind == "c"
    y_is_complex = dtype_y.kind == "c"
    z_is_complex = dtype_z.kind == "c"

    if z_is_complex and dtype_x != dtype_y:
        if x_is_complex and dtype_x != dtype_z:
            x = "{}_cast({})".format(complex_dtype_to_name(dtype_z), x)
        if y_is_complex and dtype_y != dtype_z:
            y = "{}_cast({})".format(complex_dtype_to_name(dtype_z), y)
    elif dtype_x != dtype_y:
        if dtype_x != dtype_z:
            x = "({}) ({})".format(dtype_to_ctype(dtype_z), x)
        if dtype_y != dtype_z:
            y = "({}) ({})".format(dtype_to_ctype(dtype_z), y)

    if x_is_complex and y_is_complex:
        result = "{}_pow({}, {})".format(complex_dtype_to_name(dtype_z), x, y)
    elif x_is_complex and not y_is_complex:
        result = "{}_powr({}, {})".format(complex_dtype_to_name(dtype_z), x, y)
    elif not x_is_complex and y_is_complex:
        result = "{}_rpow({}, {})".format(complex_dtype_to_name(dtype_z), x, y)
    else:
        result = f"pow({x}, {y})"

    return get_elwise_kernel(context,
            ("%(tp_z)s *z, " + x_ctype + ", "+y_ctype) % {
                "tp_x": dtype_to_ctype(dtype_x),
                "tp_y": dtype_to_ctype(dtype_y),
                "tp_z": dtype_to_ctype(dtype_z),
                },
            "z[i] = %s" % result,
            name="pow_method")


@context_dependent_memoize
def get_unop_kernel(context, operator, res_dtype, in_dtype):
    return get_elwise_kernel(context, [
        VectorArg(res_dtype, "z", with_offset=True),
        VectorArg(in_dtype, "y", with_offset=True),
        ],
        "z[i] = %s y[i]" % operator,
        name="unary_op_kernel")


@context_dependent_memoize
def get_array_scalar_binop_kernel(context, operator, dtype_res, dtype_a, dtype_b):
    return get_elwise_kernel(context, [
        VectorArg(dtype_res, "out", with_offset=True),
        VectorArg(dtype_a, "a", with_offset=True),
        ScalarArg(dtype_b, "b"),
        ],
        "out[i] = a[i] %s b" % operator,
        name="scalar_binop_kernel")


@context_dependent_memoize
def get_array_binop_kernel(context, operator, dtype_res, dtype_a, dtype_b,
                           a_is_scalar=False, b_is_scalar=False):
    a = "a[0]" if a_is_scalar else "a[i]"
    b = "b[0]" if b_is_scalar else "b[i]"
    return get_elwise_kernel(context, [
        VectorArg(dtype_res, "out", with_offset=True),
        VectorArg(dtype_a, "a", with_offset=True),
        VectorArg(dtype_b, "b", with_offset=True),
        ],
        f"out[i] = {a} {operator} {b}",
        name="binop_kernel")


@context_dependent_memoize
def get_array_scalar_comparison_kernel(context, operator, dtype_a):
    return get_elwise_kernel(context, [
        VectorArg(np.int8, "out", with_offset=True),
        VectorArg(dtype_a, "a", with_offset=True),
        ScalarArg(dtype_a, "b"),
        ],
        "out[i] = a[i] %s b" % operator,
        name="scalar_comparison_kernel")


@context_dependent_memoize
def get_array_comparison_kernel(context, operator, dtype_a, dtype_b):
    return get_elwise_kernel(context, [
        VectorArg(np.int8, "out", with_offset=True),
        VectorArg(dtype_a, "a", with_offset=True),
        VectorArg(dtype_b, "b", with_offset=True),
        ],
        "out[i] = a[i] %s b[i]" % operator,
        name="comparison_kernel")


@context_dependent_memoize
def get_unary_func_kernel(context, func_name, in_dtype, out_dtype=None):
    if out_dtype is None:
        out_dtype = in_dtype

    return get_elwise_kernel(context, [
        VectorArg(out_dtype, "z", with_offset=True),
        VectorArg(in_dtype, "y", with_offset=True),
        ],
        "z[i] = %s(y[i])" % func_name,
        name="%s_kernel" % func_name)


@context_dependent_memoize
def get_binary_func_kernel(context, func_name, x_dtype, y_dtype, out_dtype,
                           preamble="", name=None):
    return get_elwise_kernel(context, [
        VectorArg(out_dtype, "z", with_offset=True),
        VectorArg(x_dtype, "x", with_offset=True),
        VectorArg(y_dtype, "y", with_offset=True),
        ],
        "z[i] = %s(x[i], y[i])" % func_name,
        name="%s_kernel" % func_name if name is None else name,
        preamble=preamble)


@context_dependent_memoize
def get_float_binary_func_kernel(context, func_name, x_dtype, y_dtype,
                                 out_dtype, preamble="", name=None):
    if (np.array(0, x_dtype) * np.array(0, y_dtype)).itemsize > 4:
        arg_type = "double"
        preamble = """
        #if __OPENCL_C_VERSION__ < 120
        #pragma OPENCL EXTENSION cl_khr_fp64: enable
        #endif
        #define PYOPENCL_DEFINE_CDOUBLE
        """ + preamble
    else:
        arg_type = "float"
    return get_elwise_kernel(context, [
        VectorArg(out_dtype, "z", with_offset=True),
        VectorArg(x_dtype, "x", with_offset=True),
        VectorArg(y_dtype, "y", with_offset=True),
        ],
        f"z[i] = {func_name}(({arg_type})x[i], ({arg_type})y[i])",
        name="%s_kernel" % func_name if name is None else name,
        preamble=preamble)


@context_dependent_memoize
def get_fmod_kernel(context, out_dtype=np.float32, arg_dtype=np.float32,
                    mod_dtype=np.float32):
    return get_float_binary_func_kernel(context, "fmod", arg_dtype,
                                        mod_dtype, out_dtype)


@context_dependent_memoize
def get_modf_kernel(context, int_dtype=np.float32,
                    frac_dtype=np.float32, x_dtype=np.float32):
    return get_elwise_kernel(context, [
        VectorArg(int_dtype, "intpart", with_offset=True),
        VectorArg(frac_dtype, "fracpart", with_offset=True),
        VectorArg(x_dtype, "x", with_offset=True),
        ],
        """
        fracpart[i] = modf(x[i], &intpart[i])
        """,
        name="modf_kernel")


@context_dependent_memoize
def get_frexp_kernel(context, sign_dtype=np.float32, exp_dtype=np.float32,
                     x_dtype=np.float32):
    return get_elwise_kernel(context, [
        VectorArg(sign_dtype, "significand", with_offset=True),
        VectorArg(exp_dtype, "exponent", with_offset=True),
        VectorArg(x_dtype, "x", with_offset=True),
        ],
        """
        int expt = 0;
        significand[i] = frexp(x[i], &expt);
        exponent[i] = expt;
        """,
        name="frexp_kernel")


@context_dependent_memoize
def get_ldexp_kernel(context, out_dtype=np.float32, sig_dtype=np.float32,
                     expt_dtype=np.float32):
    return get_binary_func_kernel(
        context, "_PYOCL_LDEXP", sig_dtype, expt_dtype, out_dtype,
        preamble="#define _PYOCL_LDEXP(x, y) ldexp(x, (int)(y))",
        name="ldexp_kernel")


@context_dependent_memoize
def get_minmaximum_kernel(context, minmax, dtype_z, dtype_x, dtype_y,
        kind_x: ArgumentKind, kind_y: ArgumentKind):
    if dtype_z.kind == "f":
        reduce_func = f"f{minmax}_nanprop"
    elif dtype_z.kind in "iu":
        reduce_func = minmax
    else:
        raise TypeError("unsupported dtype specified")

    tp_x = dtype_to_ctype(dtype_x)
    tp_y = dtype_to_ctype(dtype_y)
    tp_z = dtype_to_ctype(dtype_z)
    decl_x, acc_x = get_decl_and_access_for_kind("x", kind_x)
    decl_y, acc_y = get_decl_and_access_for_kind("y", kind_y)

    return get_elwise_kernel(context,
            f"{tp_z} *z, {tp_x} {decl_x}, {tp_y} {decl_y}",
            f"z[i] = {reduce_func}({acc_x}, {acc_y})",
            name=f"{minmax}imum",
            preamble="""
                #define fmin_nanprop(a, b) (isnan(a) || isnan(b)) ? a+b : fmin(a, b)
                #define fmax_nanprop(a, b) (isnan(a) || isnan(b)) ? a+b : fmax(a, b)
                """)


@context_dependent_memoize
def get_bessel_kernel(context, which_func, out_dtype=np.float64,
                      order_dtype=np.int32, x_dtype=np.float64):
    if x_dtype.kind != "c":
        return get_elwise_kernel(context, [
            VectorArg(out_dtype, "z", with_offset=True),
            ScalarArg(order_dtype, "ord_n"),
            VectorArg(x_dtype, "x", with_offset=True),
            ],
            "z[i] = bessel_%sn(ord_n, x[i])" % which_func,
            name="bessel_%sn_kernel" % which_func,
            preamble="""
            #if __OPENCL_C_VERSION__ < 120
            #pragma OPENCL EXTENSION cl_khr_fp64: enable
            #endif
            #define PYOPENCL_DEFINE_CDOUBLE
            #include <pyopencl-bessel-%s.cl>
            """ % which_func)
    else:
        if which_func != "j":
            raise NotImplementedError("complex arguments for Bessel Y")

        if x_dtype != np.complex128:
            raise NotImplementedError("non-complex double dtype")
        if x_dtype != out_dtype:
            raise NotImplementedError("different input/output types")

        return get_elwise_kernel(context, [
            VectorArg(out_dtype, "z", with_offset=True),
            ScalarArg(order_dtype, "ord_n"),
            VectorArg(x_dtype, "x", with_offset=True),
            ],
            """
            cdouble_t jv_loc;
            cdouble_t jvp1_loc;
            bessel_j_complex(ord_n, x[i], &jv_loc, &jvp1_loc);
            z[i] = jv_loc;
            """,
            name="bessel_j_complex_kernel",
            preamble="""
            #if __OPENCL_C_VERSION__ < 120
            #pragma OPENCL EXTENSION cl_khr_fp64: enable
            #endif
            #define PYOPENCL_DEFINE_CDOUBLE
            #include <pyopencl-complex.h>
            #include <pyopencl-bessel-j-complex.cl>
            """)


@context_dependent_memoize
def get_hankel_01_kernel(context, out_dtype, x_dtype):
    if x_dtype != np.complex128:
        raise NotImplementedError("non-complex double dtype")
    if x_dtype != out_dtype:
        raise NotImplementedError("different input/output types")

    return get_elwise_kernel(context, [
        VectorArg(out_dtype, "h0", with_offset=True),
        VectorArg(out_dtype, "h1", with_offset=True),
        VectorArg(x_dtype, "x", with_offset=True),
        ],
        """
        cdouble_t h0_loc;
        cdouble_t h1_loc;
        hankel_01_complex(x[i], &h0_loc, &h1_loc, 1);
        h0[i] = h0_loc;
        h1[i] = h1_loc;
        """,
        name="hankel_complex_kernel",
        preamble="""
        #if __OPENCL_C_VERSION__ < 120
        #pragma OPENCL EXTENSION cl_khr_fp64: enable
        #endif
        #define PYOPENCL_DEFINE_CDOUBLE
        #include <pyopencl-complex.h>
        #include <pyopencl-hankel-complex.cl>
        """)


@context_dependent_memoize
def get_diff_kernel(context, dtype):
    return get_elwise_kernel(context, [
            VectorArg(dtype, "result", with_offset=True),
            VectorArg(dtype, "array", with_offset=True),
            ],
            "result[i] = array[i+1] - array[i]",
            name="diff")


@context_dependent_memoize
def get_if_positive_kernel(
        context, crit_dtype, then_else_dtype,
        is_then_array, is_else_array,
        is_then_scalar, is_else_scalar):
    if is_then_array:
        then_ = "then_[0]" if is_then_scalar else "then_[i]"
        then_arg = VectorArg(then_else_dtype, "then_", with_offset=True)
    else:
        assert is_then_scalar
        then_ = "then_"
        then_arg = ScalarArg(then_else_dtype, "then_")

    if is_else_array:
        else_ = "else_[0]" if is_else_scalar else "else_[i]"
        else_arg = VectorArg(then_else_dtype, "else_", with_offset=True)
    else:
        assert is_else_scalar
        else_ = "else_"
        else_arg = ScalarArg(then_else_dtype, "else_")

    return get_elwise_kernel(context, [
            VectorArg(then_else_dtype, "result", with_offset=True),
            VectorArg(crit_dtype, "crit", with_offset=True),
            then_arg, else_arg,
            ],
            f"result[i] = crit[i] > 0 ? {then_} : {else_}",
            name="if_positive")

# }}}

# vim: fdm=marker
