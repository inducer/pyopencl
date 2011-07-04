"""Elementwise functionality."""

from __future__ import division

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
from pyopencl.tools import dtype_to_ctype, VectorArg, ScalarArg


def get_elwise_program(context, arguments, operation,
        name="elwise_kernel", options=[],
        preamble="", loop_prep="", after_loop=""):
    from pyopencl import Program
    source = ("""
        %(preamble)s

        __kernel void %(name)s(%(arguments)s)
        {
          unsigned lid = get_local_id(0);
          unsigned gsize = get_global_size(0);
          unsigned work_item_start = get_local_size(0)*get_group_id(0);
          unsigned i;

          %(loop_prep)s;

          for (i = work_item_start + lid; i < n; i += gsize)
          {
            %(operation)s;
          }

          %(after_loop)s;
        }
        """ % {
            "arguments": ", ".join(arg.declarator() for arg in arguments),
            "operation": operation,
            "name": name,
            "preamble": preamble,
            "loop_prep": loop_prep,
            "after_loop": after_loop,
            })

    return Program(context, source).build(options)


def get_elwise_kernel_and_types(context, arguments, operation,
        name="elwise_kernel", options=[], preamble="", **kwargs):
    if isinstance(arguments, str):
        from pyopencl.tools import parse_c_arg
        parsed_args = [parse_c_arg(arg) for arg in arguments.split(",")]
    else:
        parsed_args = arguments

    for arg in parsed_args:
        if np.float64 == arg.dtype:
            preamble = (
                    "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n\n\n"
                    + preamble)
            break

    parsed_args.append(ScalarArg(np.uintp, "n"))

    prg = get_elwise_program(
        context, parsed_args, operation,
        name=name, options=options, preamble=preamble, **kwargs)

    scalar_arg_dtypes = []
    for arg in parsed_args:
        if isinstance(arg, ScalarArg):
            scalar_arg_dtypes.append(arg.dtype)
        else:
            scalar_arg_dtypes.append(None)

    kernel = getattr(prg, name)
    kernel.set_scalar_arg_dtypes(scalar_arg_dtypes)

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


class ElementwiseKernel:

    def __init__(self, context, arguments, operation,
            name="elwise_kernel", options=[], **kwargs):

        self.kernel, self.arguments = get_elwise_kernel_and_types(
            context, arguments, operation,
            name=name, options=options,
            **kwargs)

        if not [i for i, arg in enumerate(self.arguments)
                if isinstance(arg, VectorArg)]:
            raise RuntimeError(
                "ElementwiseKernel can only be used with "
                "functions that have at least one "
                "vector argument")

    def __call__(self, *args, **kwargs):
        vectors = []

        invocation_args = []
        for arg, arg_descr in zip(args, self.arguments):
            if isinstance(arg_descr, VectorArg):
                if not arg.flags.forc:
                    raise RuntimeError("ElementwiseKernel cannot "
                            "deal with non-contiguous arrays")

                vectors.append(arg)
                invocation_args.append(arg.data)
            else:
                invocation_args.append(arg)

        queue = kwargs.pop("queue", None)
        if kwargs:
            raise TypeError("too many/unknown keyword arguments")

        repr_vec = vectors[0]
        if queue is None:
            queue = repr_vec.queue
        invocation_args.append(repr_vec.mem_size)

        gs, ls = repr_vec.get_sizes(queue)
        self.kernel.set_args(*invocation_args)
        return cl.enqueue_nd_range_kernel(queue, self.kernel, gs, ls)


@context_dependent_memoize
def get_take_kernel(context, dtype, idx_dtype, vec_count=1):
    ctx = {
            "idx_tp": dtype_to_ctype(idx_dtype),
            "tp": dtype_to_ctype(dtype),
            }

    args = ([VectorArg(dtype, "dest" + str(i))
             for i in range(vec_count)]
            + [VectorArg(dtype, "src" + str(i))
               for i in range(vec_count)]
            + [VectorArg(idx_dtype, "idx")])
    body = (
            ("%(idx_tp)s src_idx = idx[i];\n" % ctx)
            + "\n".join(
            "dest%d[i] = src%d[src_idx];" % (i, i)
            for i in range(vec_count)))

    return get_elwise_kernel(context, args, body, name="take")


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
            VectorArg(idx_dtype, "gmem_dest_idx"),
            VectorArg(idx_dtype, "gmem_src_idx"),
            ] + [
            VectorArg(dtype, "src%d" % i)
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

    return get_elwise_kernel(context, args, body, name="take_put")


@context_dependent_memoize
def get_put_kernel(context, dtype, idx_dtype, vec_count=1):
    ctx = {
            "idx_tp": dtype_to_ctype(idx_dtype),
            "tp": dtype_to_ctype(dtype),
            }

    args = [
            VectorArg(dtype, "dest%d" % i)
                for i in range(vec_count)
            ] + [
            VectorArg(idx_dtype, "gmem_dest_idx"),
            ] + [
            VectorArg(dtype, "src%d" % i)
                for i in range(vec_count)
            ]

    body = (
            "%(idx_tp)s dest_idx = gmem_dest_idx[i];\n" % ctx
            + "\n".join("dest%d[dest_idx] = src%d[i];" % (i, i)
                for i in range(vec_count)))

    return get_elwise_kernel(args, body, name="put")


@context_dependent_memoize
def get_copy_kernel(context, dtype_dest, dtype_src):
    return get_elwise_kernel(context,
            "%(tp_dest)s *dest, %(tp_src)s *src" % {
                "tp_dest": dtype_to_ctype(dtype_dest),
                "tp_src": dtype_to_ctype(dtype_src),
                },
            "dest[i] = src[i]",
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
            args.append(VectorArg(vector_dtype, "x%d" % i))
            tex_names.append("tex_a%d" % i)
            loop_prep.append(
                    "%s a%d = fp_tex1Dfetch(tex_a%d, 0)"
                    % (dtype_to_ctype(scalar_dtype), i, i))
        else:
            args.append(ScalarArg(scalar_dtype, "a%d" % i))
            args.append(VectorArg(vector_dtype, "x%d" % i))

        summands.append("a%d*x%d[i]" % (i, i))

    args.append(VectorArg(dtype_z, "z"))
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


@context_dependent_memoize
def get_axpbyz_kernel(context, dtype_x, dtype_y, dtype_z):
    return get_elwise_kernel(context,
            "%(tp_z)s *z, %(tp_x)s a, %(tp_x)s *x, %(tp_y)s b, %(tp_y)s *y" % {
                "tp_x": dtype_to_ctype(dtype_x),
                "tp_y": dtype_to_ctype(dtype_y),
                "tp_z": dtype_to_ctype(dtype_z),
                },
            "z[i] = a*x[i] + b*y[i]",
            name="axpbyz")


@context_dependent_memoize
def get_axpbz_kernel(context, dtype):
    return get_elwise_kernel(context,
            "%(tp)s *z, %(tp)s a, %(tp)s *x,%(tp)s b" % {
                "tp": dtype_to_ctype(dtype)},
            "z[i] = a * x[i] + b",
            name="axpb")


@context_dependent_memoize
def get_multiply_kernel(context, dtype_x, dtype_y, dtype_z):
    return get_elwise_kernel(context,
            "%(tp_z)s *z, %(tp_x)s *x, %(tp_y)s *y" % {
                "tp_x": dtype_to_ctype(dtype_x),
                "tp_y": dtype_to_ctype(dtype_y),
                "tp_z": dtype_to_ctype(dtype_z),
                },
            "z[i] = x[i] * y[i]",
            name="multiply")


@context_dependent_memoize
def get_divide_kernel(context, dtype_x, dtype_y, dtype_z):
    return get_elwise_kernel(context,
            "%(tp_z)s *z, %(tp_x)s *x, %(tp_y)s *y" % {
                "tp_x": dtype_to_ctype(dtype_x),
                "tp_y": dtype_to_ctype(dtype_y),
                "tp_z": dtype_to_ctype(dtype_z),
                },
            "z[i] = x[i] / y[i]",
            name="divide")


@context_dependent_memoize
def get_rdivide_elwise_kernel(context, dtype):
    return get_elwise_kernel(context,
            "%(tp)s *z, %(tp)s *x, %(tp)s y" % {
                "tp": dtype_to_ctype(dtype),
                },
            "z[i] = y / x[i]",
            name="divide_r")


@context_dependent_memoize
def get_fill_kernel(context, dtype):
    return get_elwise_kernel(context,
            "%(tp)s *z, %(tp)s a" % {
                "tp": dtype_to_ctype(dtype),
                },
            "z[i] = a",
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
    return get_elwise_kernel(context,
            "%(tp)s *z, %(tp)s start, %(tp)s step" % {
                "tp": dtype_to_ctype(dtype),
                },
            "z[i] = start + i*step",
            name="arange")


@context_dependent_memoize
def get_pow_kernel(context, dtype):
    return get_elwise_kernel(context,
            "%(tp)s *z, %(tp)s *y, %(tp)s value" % {
                "tp": dtype_to_ctype(dtype),
                },
            "z[i] = pow(y[i], value)",
            name="pow_method")


@context_dependent_memoize
def get_pow_array_kernel(context, dtype_x, dtype_y, dtype_z):
    return get_elwise_kernel(context,
            "%(tp_z)s *z, %(tp_x)s *x, %(tp_y)s *y" % {
                "tp_x": dtype_to_ctype(dtype_x),
                "tp_y": dtype_to_ctype(dtype_y),
                "tp_z": dtype_to_ctype(dtype_z),
                },
            "z[i] = pow(x[i], y[i])",
            name="pow_method")


@context_dependent_memoize
def get_fmod_kernel(context):
    return get_elwise_kernel(context,
            "float *z, float *arg, float *mod",
            "z[i] = fmod(arg[i], mod[i])",
            name="fmod_kernel")


@context_dependent_memoize
def get_modf_kernel(context):
    return get_elwise_kernel(context,
            "float *intpart ,float *fracpart, float *x",
            "fracpart[i] = modf(x[i], &intpart[i])",
            name="modf_kernel")


@context_dependent_memoize
def get_frexp_kernel(context):
    return get_elwise_kernel(context,
            "float *significand, float *exponent, float *x",
            """
                int expt = 0;
                significand[i] = frexp(x[i], &expt);
                exponent[i] = expt;
            """,
            name="frexp_kernel")


@context_dependent_memoize
def get_ldexp_kernel(context):
    return get_elwise_kernel(context,
            "float *z, float *sig, float *expt",
            "z[i] = ldexp(sig[i], (int) expt[i])",
            name="ldexp_kernel")


@context_dependent_memoize
def get_unary_func_kernel(context, func_name, in_dtype, out_dtype=None):
    if out_dtype is None:
        out_dtype = in_dtype

    return get_elwise_kernel(context,
            "%(tp_out)s *z, %(tp_in)s *y" % {
                "tp_in": dtype_to_ctype(in_dtype),
                "tp_out": dtype_to_ctype(out_dtype),
                },
            "z[i] = %s(y[i])" % func_name,
            name="%s_kernel" % func_name)


@context_dependent_memoize
def get_if_positive_kernel(context, crit_dtype, dtype):
    return get_elwise_kernel(context, [
            VectorArg(dtype, "result"),
            VectorArg(crit_dtype, "crit"),
            VectorArg(dtype, "then_"),
            VectorArg(dtype, "else_"),
            ],
            "result[i] = crit[i] > 0 ? then_[i] : else_[i]",
            name="if_positive")
