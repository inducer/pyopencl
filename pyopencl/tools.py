"""H."""

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
"""



import numpy
from decorator import decorator
import pyopencl as cl




@decorator
def context_dependent_memoize(func, context, *args):
    """Provides memoization for things that get created inside
    a context, i.e. mainly programs and kernels. Assumes that
    the first argument of the decorated function is the context.
    """
    dicname = "_ctx_memoize_dic_%s_%x" % (
            func.__name__, hash(func))

    try:
        return getattr(context, dicname)[args]
    except AttributeError:
        result = func(context, *args)
        setattr(context, dicname, {args: result})
        return result
    except KeyError:
        result = func(context, *args)
        getattr(context,dicname)[args] = result
        return result




def pytest_generate_tests_for_pyopencl(metafunc):
    class ContextGetter:
        def __init__(self, device):
            self.device = device

        def __call__(self):
            return cl.Context([device])

        def __str__(self):
            return "<context getter for %s>" % self.device
    if ("device" in metafunc.funcargnames
            or "ctx_getter" in metafunc.funcargnames):
        arg_dict = {}

        for platform in cl.get_platforms():
            if "platform" in metafunc.funcargnames:
                arg_dict["platform"] = platform

            for device in platform.get_devices():
                if "device" in metafunc.funcargnames:
                    arg_dict["device"] = device

                if "ctx_getter" in metafunc.funcargnames:
                    arg_dict["ctx_getter"] = ContextGetter(device)

                metafunc.addcall(funcargs=arg_dict.copy(),
                        id=", ".join("%s=%s" % (arg, value)
                                for arg, value in arg_dict.iteritems()))

    elif "platform" in metafunc.funcargnames:
        for platform in cl.get_platforms():
            metafunc.addcall(
                    funcargs=dict(platform=platform),
                    id=str(platform))




# {{{ C code generation helpers -----------------------------------------------
def dtype_to_ctype(dtype):
    if dtype is None:
        raise ValueError("dtype may not be None")

    dtype = numpy.dtype(dtype)
    if dtype == numpy.int64:
        return "long"
    elif dtype == numpy.uint64:
        return "unsigned long"
    elif dtype == numpy.int32:
        return "int"
    elif dtype == numpy.uint32:
        return "unsigned int"
    elif dtype == numpy.int16:
        return "short int"
    elif dtype == numpy.uint16:
        return "short unsigned int"
    elif dtype == numpy.int8:
        return "signed char"
    elif dtype == numpy.uint8:
        return "unsigned char"
    elif dtype == numpy.bool:
        return "bool"
    elif dtype == numpy.float32:
        return "float"
    elif dtype == numpy.float64:
        return "double"
    elif dtype == numpy.complex64:
        return "complex float"
    elif dtype == numpy.complex128:
        return "complex double"
    else:
        raise ValueError, "unable to map dtype '%s'" % dtype

# }}}




# {{{ C argument lists --------------------------------------------------------
class Argument:
    def __init__(self, dtype, name, vector_len=1):
        self.dtype = numpy.dtype(dtype)
        self.name = name
        self.vector_len = vector_len

    def __repr__(self):
        return "%s(%r, %s, %d)" % (
                self.__class__.__name__,
                self.name,
                self.dtype,
                self.vector_len)

class VectorArg(Argument):
    def declarator(self):
        if self.vector_len == 1:
            vlen_str = ""
        else:
            vlen_str = str(self.vector_len)

        return "__global %s%s *%s" % (dtype_to_ctype(self.dtype), 
                vlen_str, self.name)

class ScalarArg(Argument):
    def declarator(self):
        if self.vector_len == 1:
            vlen_str = ""
        else:
            vlen_str = str(self.vector_len)

        return "%s%s %s" % (dtype_to_ctype(self.dtype), vlen_str, self.name)





def parse_c_arg(c_arg):
    c_arg = (c_arg
            .replace("const", "")
            .replace("volatile", "")
            .replace("__global", "")
            .replace("__local", "")
            .replace("__constant", ""))

    # process and remove declarator
    import re
    decl_re = re.compile(r"(\**)\s*([_a-zA-Z0-9]+)(\s*\[[ 0-9]*\])*\s*$")
    decl_match = decl_re.search(c_arg)

    if decl_match is None:
        raise ValueError("couldn't parse C declarator '%s'" % c_arg)

    name = decl_match.group(2)

    if decl_match.group(1) or decl_match.group(3) is not None:
        arg_class = VectorArg
    else:
        arg_class = ScalarArg

    tp = c_arg[:decl_match.start()]
    tp = " ".join(tp.split())

    type_re = re.compile(r"^([a-z ]+)([0-9]*)$")
    type_match = type_re.match(tp)
    if not type_match:
        raise RuntimeError("type '%s' did not match expected shape of type"
                % tp)

    tp = type_match.group(1)
    if type_match.group(2):
        vector_len = int(type_match.group(2))
    else:
        vector_len = 1

    if tp == "float": dtype = numpy.float32
    elif tp == "double": dtype = numpy.float64
    elif tp in ["int", "signed int"]: dtype = numpy.int32
    elif tp in ["unsigned", "unsigned int"]: dtype = numpy.uint32
    elif tp in ["long", "long int"]: dtype = numpy.int64
    elif tp in ["unsigned long", "unsigned long int"]:
        dtype = numpy.uint64
    elif tp in ["short", "short int"]: dtype = numpy.int16
    elif tp in ["unsigned short", "unsigned short int"]: dtype = numpy.uint16
    elif tp in ["char"]: dtype = numpy.int8
    elif tp in ["unsigned char"]: dtype = numpy.uint8
    elif tp in ["bool"]: dtype = numpy.bool
    else: raise ValueError, "unknown type '%s'" % tp

    return arg_class(dtype, name, vector_len)

# }}}




def has_double_support(dev):
    for ext in dev.extensions.split(" "):
        if ext == "cl_khr_fp64":
            return True
    return False




# vim: foldmethod=marker
