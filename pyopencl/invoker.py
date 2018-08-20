from __future__ import division, absolute_import

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

import sys
import numpy as np

from warnings import warn
import pyopencl._cl as _cl
from pytools.persistent_dict import WriteOncePersistentDict
from pyopencl.tools import _NumpyTypesKeyBuilder

_PYPY = '__pypy__' in sys.builtin_module_names
_CPY2 = not _PYPY and sys.version_info < (3,)
_CPY26 = _CPY2 and sys.version_info < (2, 7)


# {{{ arg packing helpers

_size_t_char = ({
    8: 'Q',
    4: 'L',
    2: 'H',
    1: 'B',
})[_cl._sizeof_size_t()]
_type_char_map = {
    'n': _size_t_char.lower(),
    'N': _size_t_char
}
del _size_t_char

# }}}


# {{{ individual arg handling

def generate_buffer_arg_setter(gen, arg_idx, buf_var):
    from pytools.py_codegen import Indentation

    if _CPY2 or _PYPY:
        # https://github.com/numpy/numpy/issues/5381
        gen("if isinstance({buf_var}, np.generic):".format(buf_var=buf_var))
        with Indentation(gen):
            if _PYPY:
                gen("{buf_var} = np.asarray({buf_var})".format(buf_var=buf_var))
            else:
                gen("{buf_var} = np.getbuffer({buf_var})".format(buf_var=buf_var))

    gen("""
        self._set_arg_buf({arg_idx}, {buf_var})
        """
        .format(arg_idx=arg_idx, buf_var=buf_var))


def generate_bytes_arg_setter(gen, arg_idx, buf_var):
    gen("""
        self._set_arg_buf({arg_idx}, {buf_var})
        """
        .format(arg_idx=arg_idx, buf_var=buf_var))


def generate_generic_arg_handler(gen, arg_idx, arg_var):
    from pytools.py_codegen import Indentation

    gen("""
        if {arg_var} is None:
            self._set_arg_null({arg_idx})
        elif isinstance({arg_var}, _KERNEL_ARG_CLASSES):
            self.set_arg({arg_idx}, {arg_var})
        """
        .format(arg_idx=arg_idx, arg_var=arg_var))

    gen("else:")
    with Indentation(gen):
        generate_buffer_arg_setter(gen, arg_idx, arg_var)

# }}}


# {{{ generic arg handling body

def generate_generic_arg_handling_body(num_args):
    from pytools.py_codegen import PythonCodeGenerator
    gen = PythonCodeGenerator()

    if num_args == 0:
        gen("pass")

    for i in range(num_args):
        gen("# process argument {arg_idx}".format(arg_idx=i))
        gen("")
        gen("current_arg = {arg_idx}".format(arg_idx=i))
        generate_generic_arg_handler(gen, i, "arg%d" % i)
        gen("")

    return gen

# }}}


# {{{ specific arg handling body

def generate_specific_arg_handling_body(function_name,
        num_cl_args, scalar_arg_dtypes,
        work_around_arg_count_bug, warn_about_arg_count_bug):

    assert work_around_arg_count_bug is not None
    assert warn_about_arg_count_bug is not None

    fp_arg_count = 0
    cl_arg_idx = 0

    from pytools.py_codegen import PythonCodeGenerator
    gen = PythonCodeGenerator()

    if not scalar_arg_dtypes:
        gen("pass")

    for arg_idx, arg_dtype in enumerate(scalar_arg_dtypes):
        gen("# process argument {arg_idx}".format(arg_idx=arg_idx))
        gen("")
        gen("current_arg = {arg_idx}".format(arg_idx=arg_idx))
        arg_var = "arg%d" % arg_idx

        if arg_dtype is None:
            generate_generic_arg_handler(gen, cl_arg_idx, arg_var)
            cl_arg_idx += 1
            gen("")
            continue

        arg_dtype = np.dtype(arg_dtype)

        if arg_dtype.char == "V":
            generate_generic_arg_handler(gen, cl_arg_idx, arg_var)
            cl_arg_idx += 1

        elif arg_dtype.kind == "c":
            if warn_about_arg_count_bug:
                warn("{knl_name}: arguments include complex numbers, and "
                        "some (but not all) of the target devices mishandle "
                        "struct kernel arguments (hence the workaround is "
                        "disabled".format(
                            knl_name=function_name, stacklevel=2))

            if arg_dtype == np.complex64:
                arg_char = "f"
            elif arg_dtype == np.complex128:
                arg_char = "d"
            else:
                raise TypeError("unexpected complex type: %s" % arg_dtype)

            if (work_around_arg_count_bug == "pocl"
                    and arg_dtype == np.complex128
                    and fp_arg_count + 2 <= 8):
                gen(
                        "buf = pack('{arg_char}', {arg_var}.real)"
                        .format(arg_char=arg_char, arg_var=arg_var))
                generate_bytes_arg_setter(gen, cl_arg_idx, "buf")
                cl_arg_idx += 1
                gen("current_arg = current_arg + 1000")
                gen(
                        "buf = pack('{arg_char}', {arg_var}.imag)"
                        .format(arg_char=arg_char, arg_var=arg_var))
                generate_bytes_arg_setter(gen, cl_arg_idx, "buf")
                cl_arg_idx += 1

            elif (work_around_arg_count_bug == "apple"
                    and arg_dtype == np.complex128
                    and fp_arg_count + 2 <= 8):
                raise NotImplementedError("No work-around to "
                        "Apple's broken structs-as-kernel arg "
                        "handling has been found. "
                        "Cannot pass complex numbers to kernels.")

            else:
                gen(
                        "buf = pack('{arg_char}{arg_char}', "
                        "{arg_var}.real, {arg_var}.imag)"
                        .format(arg_char=arg_char, arg_var=arg_var))
                generate_bytes_arg_setter(gen, cl_arg_idx, "buf")
                cl_arg_idx += 1

            fp_arg_count += 2

        elif arg_dtype.char in "IL" and _CPY26:
            # Prevent SystemError: ../Objects/longobject.c:336: bad
            # argument to internal function

            gen(
                    "buf = pack('{arg_char}', long({arg_var}))"
                    .format(arg_char=arg_dtype.char, arg_var=arg_var))
            generate_bytes_arg_setter(gen, cl_arg_idx, "buf")
            cl_arg_idx += 1

        else:
            if arg_dtype.kind == "f":
                fp_arg_count += 1

            arg_char = arg_dtype.char
            arg_char = _type_char_map.get(arg_char, arg_char)
            gen(
                    "buf = pack('{arg_char}', {arg_var})"
                    .format(
                        arg_char=arg_char,
                        arg_var=arg_var))
            generate_bytes_arg_setter(gen, cl_arg_idx, "buf")
            cl_arg_idx += 1

        gen("")

    if cl_arg_idx != num_cl_args:
        raise TypeError(
            "length of argument list (%d) and "
            "CL-generated number of arguments (%d) do not agree"
            % (cl_arg_idx, num_cl_args))

    return gen

# }}}


# {{{ error handler

def wrap_in_error_handler(body, arg_names):
    from pytools.py_codegen import PythonCodeGenerator, Indentation

    err_gen = PythonCodeGenerator()

    def gen_error_handler():
        err_gen("""
            if current_arg is not None:
                args = [{args}]
                advice = ""
                from pyopencl.array import Array
                if isinstance(args[current_arg], Array):
                    advice = " (perhaps you meant to pass 'array.data' " \
                        "instead of the array itself?)"

                raise _cl.LogicError(
                        "when processing argument #%d (1-based): %s%s"
                        % (current_arg+1, str(e), advice))
            else:
                raise
            """
            .format(args=", ".join(arg_names)))
        err_gen("")

    err_gen("try:")
    with Indentation(err_gen):
        err_gen.extend(body)
    err_gen("except TypeError as e:")
    with Indentation(err_gen):
        gen_error_handler()
    err_gen("except _cl.LogicError as e:")
    with Indentation(err_gen):
        gen_error_handler()

    return err_gen

# }}}


def add_local_imports(gen):
    gen("import numpy as np")
    gen("import pyopencl._cl as _cl")
    gen("from pyopencl import _KERNEL_ARG_CLASSES")
    gen("")


def _generate_enqueue_and_set_args_module(function_name,
        num_passed_args, num_cl_args,
        scalar_arg_dtypes,
        work_around_arg_count_bug, warn_about_arg_count_bug):

    from pytools.py_codegen import PythonCodeGenerator, Indentation

    arg_names = ["arg%d" % i for i in range(num_passed_args)]

    if scalar_arg_dtypes is None:
        body = generate_generic_arg_handling_body(num_passed_args)
    else:
        body = generate_specific_arg_handling_body(
                function_name, num_cl_args, scalar_arg_dtypes,
                warn_about_arg_count_bug=warn_about_arg_count_bug,
                work_around_arg_count_bug=work_around_arg_count_bug)

    err_handler = wrap_in_error_handler(body, arg_names)

    gen = PythonCodeGenerator()

    gen("from struct import pack")
    gen("from pyopencl import status_code")
    gen("")

    # {{{ generate _enqueue

    enqueue_name = "enqueue_knl_%s" % function_name
    gen("def %s(%s):"
            % (enqueue_name,
                ", ".join(
                    ["self", "queue", "global_size", "local_size"]
                    + arg_names
                    + ["global_offset=None", "g_times_l=None",
                        "wait_for=None"])))

    with Indentation(gen):
        add_local_imports(gen)
        gen.extend(err_handler)

        gen("""
            return _cl.enqueue_nd_range_kernel(queue, self, global_size, local_size,
                    global_offset, wait_for, g_times_l=g_times_l)
            """)

    # }}}

    # {{{ generate set_args

    gen("")
    gen("def set_args(%s):"
            % (", ".join(["self"] + arg_names)))

    with Indentation(gen):
        add_local_imports(gen)
        gen.extend(err_handler)

    # }}}

    return gen.get_picklable_module(), enqueue_name


invoker_cache = WriteOncePersistentDict(
        "pyopencl-invoker-cache-v6",
        key_builder=_NumpyTypesKeyBuilder())


def generate_enqueue_and_set_args(function_name,
        num_passed_args, num_cl_args,
        scalar_arg_dtypes,
        work_around_arg_count_bug, warn_about_arg_count_bug):

    cache_key = (function_name, num_passed_args, num_cl_args,
            scalar_arg_dtypes,
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
