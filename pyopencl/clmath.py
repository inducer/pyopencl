# pylint:disable=unexpected-keyword-arg  # for @elwise_kernel_runner

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

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

import pyopencl.array as cl_array
import pyopencl.elementwise as elementwise
from pyopencl.array import _get_common_dtype
import numpy as np


def _make_unary_array_func(name):
    @cl_array.elwise_kernel_runner
    def knl_runner(result, arg):
        if arg.dtype.kind == "c":
            from pyopencl.elementwise import complex_dtype_to_name
            fname = "{}_{}".format(complex_dtype_to_name(arg.dtype), name)
        else:
            fname = name

        return elementwise.get_unary_func_kernel(
                result.context, fname, arg.dtype)

    def f(array, queue=None):
        result = array._new_like_me(queue=queue)
        event1 = knl_runner(result, array, queue=queue)
        result.add_event(event1)
        return result

    return f


# See table 6.8 in the CL 1.1 spec
acos = _make_unary_array_func("acos")
acosh = _make_unary_array_func("acosh")
acospi = _make_unary_array_func("acospi")

asin = _make_unary_array_func("asin")
asinh = _make_unary_array_func("asinh")
asinpi = _make_unary_array_func("asinpi")


@cl_array.elwise_kernel_runner
def _atan2(result, arg1, arg2):
    return elementwise.get_float_binary_func_kernel(
        result.context, "atan2", arg1.dtype, arg2.dtype, result.dtype)


@cl_array.elwise_kernel_runner
def _atan2pi(result, arg1, arg2):
    return elementwise.get_float_binary_func_kernel(
        result.context, "atan2pi", arg1.dtype, arg2.dtype, result.dtype)


atan = _make_unary_array_func("atan")


def atan2(y, x, queue=None):
    """
    .. versionadded:: 2013.1
    """
    queue = queue or y.queue
    result = y._new_like_me(_get_common_dtype(y, x, queue))
    result.add_event(_atan2(result, y, x, queue=queue))
    return result


atanh = _make_unary_array_func("atanh")
atanpi = _make_unary_array_func("atanpi")


def atan2pi(y, x, queue=None):
    """
    .. versionadded:: 2013.1
    """
    queue = queue or y.queue
    result = y._new_like_me(_get_common_dtype(y, x, queue))
    result.add_event(_atan2pi(result, y, x, queue=queue))
    return result


cbrt = _make_unary_array_func("cbrt")
ceil = _make_unary_array_func("ceil")
# TODO: copysign

cos = _make_unary_array_func("cos")
cosh = _make_unary_array_func("cosh")
cospi = _make_unary_array_func("cospi")

erfc = _make_unary_array_func("erfc")
erf = _make_unary_array_func("erf")
exp = _make_unary_array_func("exp")
exp2 = _make_unary_array_func("exp2")
exp10 = _make_unary_array_func("exp10")
expm1 = _make_unary_array_func("expm1")

fabs = _make_unary_array_func("fabs")
# TODO: fdim
floor = _make_unary_array_func("floor")
# TODO: fma
# TODO: fmax
# TODO: fmin


@cl_array.elwise_kernel_runner
def _fmod(result, arg, mod):
    return elementwise.get_fmod_kernel(result.context, result.dtype,
                                       arg.dtype, mod.dtype)


def fmod(arg, mod, queue=None):
    """Return the floating point remainder of the division `arg/mod`,
    for each element in `arg` and `mod`."""
    queue = (queue or arg.queue) or mod.queue
    result = arg._new_like_me(_get_common_dtype(arg, mod, queue))
    result.add_event(_fmod(result, arg, mod, queue=queue))
    return result

# TODO: fract


@cl_array.elwise_kernel_runner
def _frexp(sig, expt, arg):
    return elementwise.get_frexp_kernel(sig.context, sig.dtype,
                                        expt.dtype, arg.dtype)


def frexp(arg, queue=None):
    """Return a tuple `(significands, exponents)` such that
    `arg == significand * 2**exponent`.
    """
    sig = arg._new_like_me(queue=queue)
    expt = arg._new_like_me(queue=queue, dtype=np.int32)
    event1 = _frexp(sig, expt, arg, queue=queue)
    sig.add_event(event1)
    expt.add_event(event1)
    return sig, expt

# TODO: hypot


ilogb = _make_unary_array_func("ilogb")


@cl_array.elwise_kernel_runner
def _ldexp(result, sig, exp):
    return elementwise.get_ldexp_kernel(result.context, result.dtype,
                                        sig.dtype, exp.dtype)


def ldexp(significand, exponent, queue=None):
    """Return a new array of floating point values composed from the
    entries of `significand` and `exponent`, paired together as
    `result = significand * 2**exponent`.
    """
    result = significand._new_like_me(queue=queue)
    result.add_event(_ldexp(result, significand, exponent))
    return result


lgamma = _make_unary_array_func("lgamma")
# TODO: lgamma_r

log = _make_unary_array_func("log")
log2 = _make_unary_array_func("log2")
log10 = _make_unary_array_func("log10")
log1p = _make_unary_array_func("log1p")
logb = _make_unary_array_func("logb")

# TODO: mad
# TODO: maxmag
# TODO: minmag


@cl_array.elwise_kernel_runner
def _modf(intpart, fracpart, arg):
    return elementwise.get_modf_kernel(intpart.context, intpart.dtype,
                                       fracpart.dtype, arg.dtype)


def modf(arg, queue=None):
    """Return a tuple `(fracpart, intpart)` of arrays containing the
    integer and fractional parts of `arg`.
    """
    intpart = arg._new_like_me(queue=queue)
    fracpart = arg._new_like_me(queue=queue)
    event1 = _modf(intpart, fracpart, arg, queue=queue)
    fracpart.add_event(event1)
    intpart.add_event(event1)
    return fracpart, intpart


nan = _make_unary_array_func("nan")

# TODO: nextafter
# TODO: remainder
# TODO: remquo

rint = _make_unary_array_func("rint")
# TODO: rootn
round = _make_unary_array_func("round")

sin = _make_unary_array_func("sin")
# TODO: sincos
sinh = _make_unary_array_func("sinh")
sinpi = _make_unary_array_func("sinpi")

sqrt = _make_unary_array_func("sqrt")

tan = _make_unary_array_func("tan")
tanh = _make_unary_array_func("tanh")
tanpi = _make_unary_array_func("tanpi")
tgamma = _make_unary_array_func("tgamma")
trunc = _make_unary_array_func("trunc")


# no point wrapping half_ or native_

# TODO: table 6.10, integer functions
# TODO: table 6.12, clamp et al

@cl_array.elwise_kernel_runner
def _bessel_jn(result, n, x):
    return elementwise.get_bessel_kernel(result.context, "j", result.dtype,
                                         np.dtype(type(n)), x.dtype)


@cl_array.elwise_kernel_runner
def _bessel_yn(result, n, x):
    return elementwise.get_bessel_kernel(result.context, "y", result.dtype,
                                         np.dtype(type(n)), x.dtype)


@cl_array.elwise_kernel_runner
def _hankel_01(h0, h1, x):
    if h0.dtype != h1.dtype:
        raise TypeError("types of h0 and h1 must match")
    return elementwise.get_hankel_01_kernel(
            h0.context, h0.dtype, x.dtype)


def bessel_jn(n, x, queue=None):
    result = x._new_like_me(queue=queue)
    result.add_event(_bessel_jn(result, n, x, queue=queue))
    return result


def bessel_yn(n, x, queue=None):
    result = x._new_like_me(queue=queue)
    result.add_event(_bessel_yn(result, n, x, queue=queue))
    return result


def hankel_01(x, queue=None):
    h0 = x._new_like_me(queue=queue)
    h1 = x._new_like_me(queue=queue)
    event1 = _hankel_01(h0, h1, x, queue=queue)
    h0.add_event(event1)
    h1.add_event(event1)
    return h0, h1
