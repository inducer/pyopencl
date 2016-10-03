"""Elementwise functionality."""

from __future__ import division
from __future__ import absolute_import
from six.moves import range
from six.moves import zip

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


from pyopencl.tools import context_dependent_memoize
import numpy as np
import pyopencl as cl
from pytools import memoize_method
from pyopencl.tools import (dtype_to_ctype, VectorArg, ScalarArg,
        KernelTemplateBase, dtype_to_c_struct)


# {{{ elementwise kernel code generator

def get_elwise_program(context, arguments, operation,
        name="elwise_kernel", options=[],
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
        %(preamble)s

        #define PYOPENCL_ELWISE_CONTINUE continue

        __kernel void %(name)s(%(arguments)s)
        {
          int lid = get_local_id(0);
          int gsize = get_global_size(0);
          int work_group_start = get_local_size(0)*get_group_id(0);
          long i;

          %(loop_prep)s;
          %(body)s
          %(after_loop)s;
        }
        """ % {
            "arguments": ", ".join(arg.declarator() for arg in arguments),
            "name": name,
            "preamble": preamble,
            "loop_prep": loop_prep,
            "after_loop": after_loop,
            "body": body % dict(operation=operation),
            })

    from pyopencl import Program
    return Program(context, source).build(options)


def get_elwise_kernel_and_types(context, arguments, operation,
        name="elwise_kernel", options=[], preamble="", use_range=False,
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
            if arg.dtype.kind == 'c':
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

    from pyopencl.tools import get_arg_list_scalar_arg_dtypes

    kernel = getattr(prg, name)
    kernel.set_scalar_arg_dtypes(get_arg_list_scalar_arg_dtypes(parsed_args))

    return kernel, parsed_args


def get_elwise_kernel(context, arguments, operation,
        name="elwise_kernel", options=[], **kwargs):
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
            name="elwise_kernel", options=[], **kwargs):
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

        # {{{ assemble arg array

        invocation_args = []
        for arg, arg_descr in zip(args, arg_descrs):
            if isinstance(arg_descr, VectorArg):
                if not arg.flags.forc:
                    raise RuntimeError("ElementwiseKernel cannot "
                            "deal with non-contiguous arrays")

                if repr_vec is None:
                    repr_vec = arg

                invocation_args.append(arg.base_data)
                if arg_descr.with_offset:
                    invocation_args.append(arg.offset)
            else:
                invocation_args.append(arg)

        # }}}

        queue = kwargs.pop("queue", None)
        wait_for = kwargs.pop("wait_for", None)
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

            from pyopencl.array import splay
            gs, ls = splay(queue,
                    abs(range_.stop - start)//step,
                    max_wg_size)
        else:
            invocation_args.append(repr_vec.size)
            gs, ls = repr_vec.get_sizes(queue, max_wg_size)

        if capture_as is not None:
            kernel.set_args(*invocation_args)
            kernel.capture_call(
                    capture_as, queue,
                    gs, ls, *invocation_args, wait_for=wait_for)

        kernel.set_args(*invocation_args)
        return cl.enqueue_nd_range_kernel(queue, kernel,
                gs, ls, wait_for=wait_for)

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
            options=()):
        renderer = self.get_renderer(
                type_aliases, var_values, context, options)

        arg_list = renderer.render_argument_list(
                self.arguments, more_arguments, with_offset=True)
        type_decl_preamble = renderer.get_type_decl_preamble(
                context.devices[0], declare_types, arg_list)

        return ElementwiseKernel(context,
            arg_list, renderer(self.operation),
            name=renderer(self.name), options=list(options),
            preamble=(
                type_decl_preamble
                + "\n"
                + renderer(self.preamble + "\n" + more_preamble)),
            auto_preamble=False)

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
            ]

    body = (
            "%(idx_tp)s dest_idx = gmem_dest_idx[i];\n" % ctx
            + "\n".join("dest%d[dest_idx] = src%d[i];" % (i, i)
                for i in range(vec_count)))

    return get_elwise_kernel(context, args, body,
            preamble=dtype_to_c_struct(context.devices[0], dtype),
            name="put")


@context_dependent_memoize
def get_copy_kernel(context, dtype_dest, dtype_src):
    src = "src[i]"
    if dtype_dest.kind == "c" != dtype_src.kind:
        src = "%s_fromreal(%s)" % (complex_dtype_to_name(dtype_dest), src)

    if dtype_dest.kind == "c" and dtype_src != dtype_dest:
        src = "%s_cast(%s)" % (complex_dtype_to_name(dtype_dest), src),

    if dtype_dest != dtype_src and (
            dtype_dest.kind == "V" or dtype_src.kind == "V"):
        raise TypeError("copying between non-identical struct types")

    return get_elwise_kernel(context,
            "%(tp_dest)s *dest, %(tp_src)s *src" % {
                "tp_dest": dtype_to_ctype(dtype_dest),
                "tp_src": dtype_to_ctype(dtype_src),
                },
            "dest[i] = %s" % src,
            preamble=dtype_to_c_struct(context.devices[0], dtype_dest),
            name="copy")


@context_dependent_memoize
def get_linear_combination_kernel(summand_descriptors,
        dtype_z):
    # TODO: Port this!
    raise NotImplementedError

    from pyopencl.tools import dtype_to_ctype
    from pyopencl.elementwise import \
            VectorArg, ScalarArg, get_elwise_module

    args = []
    preamble = []
    loop_prep = []
    summands = []
    tex_names = []

    for i, (is_gpu_scalar, scalar_dtype, vector_dtype) in \
            enumerate(summand_descriptors):
        if is_gpu_scalar:
            preamble.append(
                    "texture <%s, 1, cudaReadModeElementType> tex_a%d;"
                    % (dtype_to_ctype(scalar_dtype, with_fp_tex_hack=True), i))
            args.append(VectorArg(vector_dtype, "x%d" % i, with_offset=True))
            tex_names.append("tex_a%d" % i)
            loop_prep.append(
                    "%s a%d = fp_tex1Dfetch(tex_a%d, 0)"
                    % (dtype_to_ctype(scalar_dtype), i, i))
        else:
            args.append(ScalarArg(scalar_dtype, "a%d" % i))
            args.append(VectorArg(vector_dtype, "x%d" % i, with_offset=True))

        summands.append("a%d*x%d[i]" % (i, i))

    args.append(VectorArg(dtype_z, "z", with_offset=True))
    args.append(ScalarArg(np.uintp, "n"))

    mod = get_elwise_module(args,
            "z[i] = " + " + ".join(summands),
            "linear_combination",
            preamble="\n".join(preamble),
            loop_prep=";\n".join(loop_prep))

    func = mod.get_function("linear_combination")
    tex_src = [mod.get_texref(tn) for tn in tex_names]
    func.prepare("".join(arg.struct_char for arg in args),
            (1, 1, 1), texrefs=tex_src)

    return func, tex_src


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
def get_axpbyz_kernel(context, dtype_x, dtype_y, dtype_z):
    ax = "a*x[i]"
    by = "b*y[i]"

    x_is_complex = dtype_x.kind == "c"
    y_is_complex = dtype_y.kind == "c"

    if x_is_complex:
        ax = "%s_mul(a, x[i])" % complex_dtype_to_name(dtype_x)

    if y_is_complex:
        by = "%s_mul(b, y[i])" % complex_dtype_to_name(dtype_y)

    if x_is_complex and not y_is_complex:
        by = "%s_fromreal(%s)" % (complex_dtype_to_name(dtype_x), by)

    if not x_is_complex and y_is_complex:
        ax = "%s_fromreal(%s)" % (complex_dtype_to_name(dtype_y), ax)

    if x_is_complex or y_is_complex:
        result = (
                "{root}_add({root}_cast({ax}), {root}_cast({by}))"
                .format(
                    ax=ax,
                    by=by,
                    root=complex_dtype_to_name(dtype_z)))
    else:
        result = "%s + %s" % (ax, by)

    return get_elwise_kernel(context,
            "%(tp_z)s *z, %(tp_x)s a, %(tp_x)s *x, %(tp_y)s b, %(tp_y)s *y" % {
                "tp_x": dtype_to_ctype(dtype_x),
                "tp_y": dtype_to_ctype(dtype_y),
                "tp_z": dtype_to_ctype(dtype_z),
                },
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
            x = "%s_cast(%s)" % (complex_dtype_to_name(dtype_z), x)

        if a_is_complex:
            if dtype_a != dtype_z:
                a = "%s_cast(%s)" % (complex_dtype_to_name(dtype_z), a)

            ax = "%s_mul(%s, %s)" % (complex_dtype_to_name(dtype_z), a, x)
        else:
            ax = "%s_rmul(%s, %s)" % (complex_dtype_to_name(dtype_z), a, x)
    elif a_is_complex:
        a = "a"
        x = "x[i]"

        if dtype_a != dtype_z:
            a = "%s_cast(%s)" % (complex_dtype_to_name(dtype_z), a)
        ax = "%s_mulr(%s, %s)" % (complex_dtype_to_name(dtype_z), a, x)

    b = "b"
    if z_is_complex and not b_is_complex:
        b = "%s_fromreal(%s)" % (complex_dtype_to_name(dtype_z), b)

    if z_is_complex and not (a_is_complex or x_is_complex):
        ax = "%s_fromreal(%s)" % (complex_dtype_to_name(dtype_z), ax)

    if z_is_complex:
        ax = "%s_cast(%s)" % (complex_dtype_to_name(dtype_z), ax)
        b = "%s_cast(%s)" % (complex_dtype_to_name(dtype_z), b)

    if a_is_complex or x_is_complex or b_is_complex:
        expr = "{root}_add({ax}, {b})".format(
                ax=ax,
                b=b,
                root=complex_dtype_to_name(dtype_z))
    else:
        expr = "%s + %s" % (ax, b)

    return get_elwise_kernel(context,
            "%(tp_z)s *z, %(tp_a)s a, %(tp_x)s *x,%(tp_b)s b" % {
                "tp_a": dtype_to_ctype(dtype_a),
                "tp_x": dtype_to_ctype(dtype_x),
                "tp_b": dtype_to_ctype(dtype_b),
                "tp_z": dtype_to_ctype(dtype_z),
                },
            "z[i] = " + expr,
            name="axpb")


@context_dependent_memoize
def get_multiply_kernel(context, dtype_x, dtype_y, dtype_z):
    x_is_complex = dtype_x.kind == "c"
    y_is_complex = dtype_y.kind == "c"

    x = "x[i]"
    y = "y[i]"

    if x_is_complex and dtype_x != dtype_z:
        x = "%s_cast(%s)" % (complex_dtype_to_name(dtype_z), x)
    if y_is_complex and dtype_y != dtype_z:
        y = "%s_cast(%s)" % (complex_dtype_to_name(dtype_z), y)

    if x_is_complex and y_is_complex:
        xy = "%s_mul(%s, %s)" % (complex_dtype_to_name(dtype_z), x, y)
    elif x_is_complex and not y_is_complex:
        xy = "%s_mulr(%s, %s)" % (complex_dtype_to_name(dtype_z), x, y)
    elif not x_is_complex and y_is_complex:
        xy = "%s_rmul(%s, %s)" % (complex_dtype_to_name(dtype_z), x, y)
    else:
        xy = "%s * %s" % (x, y)

    return get_elwise_kernel(context,
            "%(tp_z)s *z, %(tp_x)s *x, %(tp_y)s *y" % {
                "tp_x": dtype_to_ctype(dtype_x),
                "tp_y": dtype_to_ctype(dtype_y),
                "tp_z": dtype_to_ctype(dtype_z),
                },
            "z[i] = %s" % xy,
            name="multiply")


@context_dependent_memoize
def get_divide_kernel(context, dtype_x, dtype_y, dtype_z):
    x_is_complex = dtype_x.kind == "c"
    y_is_complex = dtype_y.kind == "c"
    z_is_complex = dtype_z.kind == "c"

    x = "x[i]"
    y = "y[i]"

    if z_is_complex and dtype_x != dtype_y:
        if x_is_complex and dtype_x != dtype_z:
            x = "%s_cast(%s)" % (complex_dtype_to_name(dtype_z), x)
        if y_is_complex and dtype_y != dtype_z:
            y = "%s_cast(%s)" % (complex_dtype_to_name(dtype_z), y)

    if x_is_complex and y_is_complex:
        xoy = "%s_divide(%s, %s)" % (complex_dtype_to_name(dtype_z), x, y)
    elif not x_is_complex and y_is_complex:
        xoy = "%s_rdivide(%s, %s)" % (complex_dtype_to_name(dtype_z), x, y)
    elif x_is_complex and not y_is_complex:
        xoy = "%s_divider(%s, %s)" % (complex_dtype_to_name(dtype_z), x, y)
    else:
        xoy = "%s / %s" % (x, y)

    if z_is_complex:
        xoy = "%s_cast(%s)" % (complex_dtype_to_name(dtype_z), xoy)

    return get_elwise_kernel(context,
            "%(tp_z)s *z, %(tp_x)s *x, %(tp_y)s *y" % {
                "tp_x": dtype_to_ctype(dtype_x),
                "tp_y": dtype_to_ctype(dtype_y),
                "tp_z": dtype_to_ctype(dtype_z),
                },
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
            x = "%s_cast(%s)" % (complex_dtype_to_name(dtype_z), x)
        if y_is_complex and dtype_y != dtype_z:
            y = "%s_cast(%s)" % (complex_dtype_to_name(dtype_z), y)

    if x_is_complex and y_is_complex:
        yox = "%s_divide(%s, %s)" % (complex_dtype_to_name(dtype_z), y, x)
    elif not y_is_complex and x_is_complex:
        yox = "%s_rdivide(%s, %s)" % (complex_dtype_to_name(dtype_z), y, x)
    elif y_is_complex and not x_is_complex:
        yox = "%s_divider(%s, %s)" % (complex_dtype_to_name(dtype_z), y, x)
    else:
        yox = "%s / %s" % (y, x)

    return get_elwise_kernel(context,
            "%(tp_z)s *z, %(tp_x)s *x, %(tp_y)s y" % {
                "tp_x": dtype_to_ctype(dtype_x),
                "tp_y": dtype_to_ctype(dtype_y),
                "tp_z": dtype_to_ctype(dtype_z),
                },
            "z[i] = %s" % yox,
            name="divide_r")


@context_dependent_memoize
def get_fill_kernel(context, dtype):
    return get_elwise_kernel(context,
            "%(tp)s *z, %(tp)s a" % {
                "tp": dtype_to_ctype(dtype),
                },
            "z[i] = a",
            preamble=dtype_to_c_struct(context.devices[0], dtype),
            name="fill")


@context_dependent_memoize
def get_reverse_kernel(context, dtype):
    return get_elwise_kernel(context,
            "%(tp)s *z, %(tp)s *y" % {
                "tp": dtype_to_ctype(dtype),
                },
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
            x = "%s_cast(%s)" % (complex_dtype_to_name(dtype_z), x)
        if y_is_complex and dtype_y != dtype_z:
            y = "%s_cast(%s)" % (complex_dtype_to_name(dtype_z), y)
    elif dtype_x != dtype_y:
        if dtype_x != dtype_z:
            x = "(%s) (%s)" % (dtype_to_ctype(dtype_z), x)
        if dtype_y != dtype_z:
            y = "(%s) (%s)" % (dtype_to_ctype(dtype_z), y)

    if x_is_complex and y_is_complex:
        result = "%s_pow(%s, %s)" % (complex_dtype_to_name(dtype_z), x, y)
    elif x_is_complex and not y_is_complex:
        result = "%s_powr(%s, %s)" % (complex_dtype_to_name(dtype_z), x, y)
    elif not x_is_complex and y_is_complex:
        result = "%s_rpow(%s, %s)" % (complex_dtype_to_name(dtype_z), x, y)
    else:
        result = "pow(%s, %s)" % (x, y)

    return get_elwise_kernel(context,
            ("%(tp_z)s *z, " + x_ctype + ", "+y_ctype) % {
                "tp_x": dtype_to_ctype(dtype_x),
                "tp_y": dtype_to_ctype(dtype_y),
                "tp_z": dtype_to_ctype(dtype_z),
                },
            "z[i] = %s" % result,
            name="pow_method")


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
        arg_type = 'double'
        preamble = """
        #if __OPENCL_C_VERSION__ < 120
        #pragma OPENCL EXTENSION cl_khr_fp64: enable
        #endif
        #define PYOPENCL_DEFINE_CDOUBLE
        """ + preamble
    else:
        arg_type = 'float'
    return get_elwise_kernel(context, [
        VectorArg(out_dtype, "z", with_offset=True),
        VectorArg(x_dtype, "x", with_offset=True),
        VectorArg(y_dtype, "y", with_offset=True),
        ],
        "z[i] = %s((%s)x[i], (%s)y[i])" % (func_name, arg_type, arg_type),
        name="%s_kernel" % func_name if name is None else name,
        preamble=preamble)


@context_dependent_memoize
def get_fmod_kernel(context, out_dtype=np.float32, arg_dtype=np.float32,
                    mod_dtype=np.float32):
    return get_float_binary_func_kernel(context, 'fmod', arg_dtype,
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
        context, '_PYOCL_LDEXP', sig_dtype, expt_dtype, out_dtype,
        preamble="#define _PYOCL_LDEXP(x, y) ldexp(x, (int)(y))",
        name="ldexp_kernel")


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
def get_if_positive_kernel(context, crit_dtype, dtype):
    return get_elwise_kernel(context, [
            VectorArg(dtype, "result", with_offset=True),
            VectorArg(crit_dtype, "crit", with_offset=True),
            VectorArg(dtype, "then_", with_offset=True),
            VectorArg(dtype, "else_", with_offset=True),
            ],
            "result[i] = crit[i] > 0 ? then_[i] : else_[i]",
            name="if_positive")

# }}}

# vim: fdm=marker:filetype=pyopencl
