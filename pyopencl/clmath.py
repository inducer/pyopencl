import pyopencl.array as cl_array
import pyopencl.elementwise as elementwise

def _make_unary_array_func(name):
    @cl_array.elwise_kernel_runner
    def knl_runner(result, arg):
        if arg.dtype.kind == "c":
            from pyopencl.elementwise import complex_dtype_to_name
            fname = "%s_%s" % (complex_dtype_to_name(arg.dtype), name)
        else:
            fname = name

        return elementwise.get_unary_func_kernel(
                result.context, fname, arg.dtype)

    def f(array, queue=None):
        result = array._new_like_me(queue=queue)
        knl_runner(result, array, queue=queue)
        return result

    return f

# See table 6.8 in the CL 1.1 spec
acos = _make_unary_array_func("acos")
acosh = _make_unary_array_func("acosh")
acospi = _make_unary_array_func("acospi")

asin = _make_unary_array_func("asin")
asinh = _make_unary_array_func("asinh")
asinpi = _make_unary_array_func("asinpi")

atan = _make_unary_array_func("atan")
# TODO: atan2
atanh = _make_unary_array_func("atanh")
atanpi = _make_unary_array_func("atanpi")
# TODO: atan2pi

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
    return elementwise.get_fmod_kernel(result.context)

def fmod(arg, mod, queue=None):
    """Return the floating point remainder of the division `arg/mod`,
    for each element in `arg` and `mod`."""
    result = arg._new_like_me(queue=queue)
    _fmod(result, arg, mod, queue=queue)
    return result

# TODO: fract

@cl_array.elwise_kernel_runner
def _frexp(sig, expt, arg):
    return elementwise.get_frexp_kernel(sig.context)

def frexp(arg, queue=None):
    """Return a tuple `(significands, exponents)` such that
    `arg == significand * 2**exponent`.
    """
    sig = arg._new_like_me(queue=queue)
    expt = arg._new_like_me(queue=queue)
    _frexp(sig, expt, arg, queue=queue)
    return sig, expt

# TODO: hypot

ilogb = _make_unary_array_func("ilogb")

@cl_array.elwise_kernel_runner
def _ldexp(result, sig, exp):
    return elementwise.get_ldexp_kernel(result.context)

def ldexp(significand, exponent, queue=None):
    """Return a new array of floating point values composed from the
    entries of `significand` and `exponent`, paired together as
    `result = significand * 2**exponent`.
    """
    result = significand._new_like_me(queue=queue)
    _ldexp(result, significand, exponent)
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
    return elementwise.get_modf_kernel(intpart.context)

def modf(arg, queue=None):
    """Return a tuple `(fracpart, intpart)` of arrays containing the
    integer and fractional parts of `arg`.
    """
    intpart = arg._new_like_me(queue=queue)
    fracpart = arg._new_like_me(queue=queue)
    _modf(intpart, fracpart, arg, queue=queue)
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
def _bessel_jn(result, sig, exp):
    return elementwise.get_bessel_jn_kernel(result.context)

def bessel_jn(n, x, queue=None):
    """Return a new array of floating point values composed from the
    entries of `significand` and `exponent`, paired together as
    `result = significand * 2**exponent`.
    """
    result = x._new_like_me(queue=queue)
    _bessel_jn(result, n, x)
    return result
