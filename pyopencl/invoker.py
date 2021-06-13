__copyright__ = """
Copyright (C) 2017 Andreas Kloeckner
"""

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

from warnings import warn
import pyopencl._cl as _cl
from pytools.persistent_dict import WriteOncePersistentDict
from pytools.py_codegen import Indentation, PythonCodeGenerator
from pyopencl.tools import _NumpyTypesKeyBuilder, VectorArg


# {{{ arg packing helpers

_size_t_char = ({
    8: "Q",
    4: "L",
    2: "H",
    1: "B",
})[_cl._sizeof_size_t()]
_type_char_map = {
    "n": _size_t_char.lower(),
    "N": _size_t_char
}
del _size_t_char

# }}}


# {{{ generic arg handling body

def generate_generic_arg_handling_body(num_args):
    gen = PythonCodeGenerator()

    if num_args == 0:
        gen("pass")
    else:
        gen_indices_and_args = []
        for i in range(num_args):
            gen_indices_and_args.append(i)
            gen_indices_and_args.append(f"arg{i}")

        gen(f"self._set_arg_multi("
                f"({', '.join(str(i) for i in gen_indices_and_args)},), "
                ")")

    return gen

# }}}


# {{{ specific arg handling body

BUF_PACK_TYPECHARS = ["c", "b", "B", "h", "H", "i", "I", "l", "L", "f", "d"]


def generate_specific_arg_handling_body(function_name, num_cl_args, arg_types, *,
        work_around_arg_count_bug, warn_about_arg_count_bug,
        in_enqueue, include_debug_code):

    assert work_around_arg_count_bug is not None
    assert warn_about_arg_count_bug is not None

    fp_arg_count = 0
    cl_arg_idx = 0

    gen = PythonCodeGenerator()

    if not arg_types:
        gen("pass")

    gen_indices_and_args = []
    buf_indices_and_args = []
    buf_pack_indices_and_args = []

    def add_buf_arg(arg_idx, typechar, expr_str):
        if typechar in BUF_PACK_TYPECHARS:
            buf_pack_indices_and_args.append(arg_idx)
            buf_pack_indices_and_args.append(repr(typechar.encode()))
            buf_pack_indices_and_args.append(expr_str)
        else:
            buf_indices_and_args.append(arg_idx)
            buf_indices_and_args.append(f"pack('{typechar}', {expr_str})")

    wait_for_parts = []

    for arg_idx, arg_type in enumerate(arg_types):
        arg_var = "arg%d" % arg_idx

        if arg_type is None:
            gen_indices_and_args.append(cl_arg_idx)
            gen_indices_and_args.append(arg_var)
            cl_arg_idx += 1
            gen("")
            continue

        elif isinstance(arg_type, VectorArg):
            if include_debug_code:
                gen(f"if not {arg_var}.flags.forc:")
                with Indentation(gen):
                    gen("raise RuntimeError('only contiguous arrays may '")
                    gen("   'be used as arguments to this operation')")
                    gen("")

            if in_enqueue and include_debug_code:
                gen(f"assert {arg_var}.queue is None or {arg_var}.queue == queue, "
                    "'queues for all arrays must match the queue supplied "
                    "to enqueue'")

            gen_indices_and_args.append(cl_arg_idx)
            gen_indices_and_args.append(f"{arg_var}.base_data")
            cl_arg_idx += 1

            if arg_type.with_offset:
                add_buf_arg(cl_arg_idx, np.dtype(np.int64).char, f"{arg_var}.offset")
                cl_arg_idx += 1

            if in_enqueue:
                wait_for_parts .append(f"{arg_var}.events")

            continue

        arg_dtype = np.dtype(arg_type)

        if arg_dtype.char == "V":
            buf_indices_and_args.append(cl_arg_idx)
            buf_indices_and_args.append(arg_var)
            cl_arg_idx += 1

        elif arg_dtype.kind == "c":
            if warn_about_arg_count_bug:
                warn("{knl_name}: arguments include complex numbers, and "
                        "some (but not all) of the target devices mishandle "
                        "struct kernel arguments (hence the workaround is "
                        "disabled".format(
                            knl_name=function_name), stacklevel=2)

            if arg_dtype == np.complex64:
                arg_char = "f"
            elif arg_dtype == np.complex128:
                arg_char = "d"
            else:
                raise TypeError("unexpected complex type: %s" % arg_dtype)

            if (work_around_arg_count_bug == "pocl"
                    and arg_dtype == np.complex128
                    and fp_arg_count + 2 <= 8):
                add_buf_arg(cl_arg_idx, arg_char, f"{arg_var}.real")
                cl_arg_idx += 1
                add_buf_arg(cl_arg_idx, arg_char, f"{arg_var}.imag")
                cl_arg_idx += 1

            elif (work_around_arg_count_bug == "apple"
                    and arg_dtype == np.complex128
                    and fp_arg_count + 2 <= 8):
                raise NotImplementedError("No work-around to "
                        "Apple's broken structs-as-kernel arg "
                        "handling has been found. "
                        "Cannot pass complex numbers to kernels.")

            else:
                buf_indices_and_args.append(cl_arg_idx)
                buf_indices_and_args.append(
                    f"pack('{arg_char}{arg_char}', {arg_var}.real, {arg_var}.imag)")
                cl_arg_idx += 1

            fp_arg_count += 2

        else:
            if arg_dtype.kind == "f":
                fp_arg_count += 1

            arg_char = arg_dtype.char
            arg_char = _type_char_map.get(arg_char, arg_char)
            add_buf_arg(cl_arg_idx, arg_char, arg_var)
            cl_arg_idx += 1

        gen("")

    for arg_kind, args_and_indices, entry_length in [
            ("", gen_indices_and_args, 2),
            ("_buf", buf_indices_and_args, 2),
            ("_buf_pack", buf_pack_indices_and_args, 3),
            ]:
        assert len(args_and_indices) % entry_length == 0
        if args_and_indices:
            gen(f"self._set_arg{arg_kind}_multi("
                    f"({', '.join(str(i) for i in args_and_indices)},), "
                    ")")

    if cl_arg_idx != num_cl_args:
        raise TypeError(
            "length of argument list (%d) and "
            "CL-generated number of arguments (%d) do not agree"
            % (cl_arg_idx, num_cl_args))

    if in_enqueue:
        return gen, wait_for_parts
    else:
        return gen

# }}}


def _generate_enqueue_and_set_args_module(function_name,
        num_passed_args, num_cl_args,
        arg_types, include_debug_code,
        work_around_arg_count_bug, warn_about_arg_count_bug):

    arg_names = ["arg%d" % i for i in range(num_passed_args)]

    def gen_arg_setting(in_enqueue):
        if arg_types is None:
            result = generate_generic_arg_handling_body(num_passed_args)
            if in_enqueue:
                return result, []
            else:
                return result

        else:
            return generate_specific_arg_handling_body(
                    function_name, num_cl_args, arg_types,
                    warn_about_arg_count_bug=warn_about_arg_count_bug,
                    work_around_arg_count_bug=work_around_arg_count_bug,
                    in_enqueue=in_enqueue, include_debug_code=include_debug_code)

    gen = PythonCodeGenerator()

    gen("from struct import pack")
    gen("from pyopencl import status_code")
    gen("import numpy as np")
    gen("import pyopencl._cl as _cl")
    gen("")

    # {{{ generate _enqueue

    enqueue_name = "enqueue_knl_%s" % function_name
    gen("def %s(%s):"
            % (enqueue_name,
                ", ".join(
                    ["self", "queue", "global_size", "local_size"]
                    + arg_names
                    + ["global_offset=None",
                        "g_times_l=None",
                        "allow_empty_ndrange=False",
                        "wait_for=None"])))

    with Indentation(gen):
        subgen, wait_for_parts = gen_arg_setting(in_enqueue=True)
        gen.extend(subgen)

        if wait_for_parts:
            wait_for_expr = (
                    "[*(() if wait_for is None else wait_for), "
                    + ", ".join("*"+wfp for wfp in wait_for_parts)
                    + "]")
        else:
            wait_for_expr = "wait_for"

        # Using positional args here because pybind is slow with keyword args
        gen(f"""
            return _cl.enqueue_nd_range_kernel(queue, self,
                    global_size, local_size, global_offset,
                    {wait_for_expr},
                    g_times_l, allow_empty_ndrange)
            """)

    # }}}

    # {{{ generate set_args

    gen("")
    gen("def set_args(%s):"
            % (", ".join(["self"] + arg_names)))

    with Indentation(gen):
        gen.extend(gen_arg_setting(in_enqueue=False))

    # }}}

    return (
            gen.get_picklable_module(
                name=f"<pyopencl invoker for '{function_name}'>"),
            enqueue_name)


invoker_cache = WriteOncePersistentDict(
        "pyopencl-invoker-cache-v41",
        key_builder=_NumpyTypesKeyBuilder())


def generate_enqueue_and_set_args(function_name,
        num_passed_args, num_cl_args,
        arg_types,
        work_around_arg_count_bug, warn_about_arg_count_bug):

    cache_key = (function_name, num_passed_args, num_cl_args,
            arg_types, __debug__,
            work_around_arg_count_bug, warn_about_arg_count_bug)

    from_cache = False

    try:
        result = invoker_cache[cache_key]
        from_cache = True
    except KeyError:
        pass

    if not from_cache:
        result = _generate_enqueue_and_set_args_module(*cache_key)
        invoker_cache.store_if_not_present(cache_key, result)

    pmod, enqueue_name = result

    return (
            pmod.mod_globals[enqueue_name],
            pmod.mod_globals["set_args"])

# }}}


# vim: foldmethod=marker
