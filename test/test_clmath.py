from __future__ import division

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
import math
import numpy as np


def have_cl():
    try:
        import pyopencl  # noqa
        return True
    except:
        return False

if have_cl():
    import pyopencl.array as cl_array
    import pyopencl as cl
    import pyopencl.clmath as clmath
    from pyopencl.tools import (  # noqa
            pytest_generate_tests_for_pyopencl
            as pytest_generate_tests)
    from pyopencl.characterize import has_double_support

try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()


sizes = [10, 128, 1 << 10, 1 << 11, 1 << 13]


numpy_func_names = {
        "asin": "arcsin",
        "acos": "arccos",
        "atan": "arctan",
        }


def make_unary_function_test(name, limits=(0, 1), threshold=0, use_complex=False):
    (a, b) = limits
    a = float(a)
    b = float(b)

    def test(ctx_factory):
        context = ctx_factory()
        queue = cl.CommandQueue(context)

        gpu_func = getattr(clmath, name)
        cpu_func = getattr(np, numpy_func_names.get(name, name))

        if has_double_support(context.devices[0]):
            if use_complex:
                dtypes = [np.float32, np.float64, np.complex64, np.complex128]
            else:
                dtypes = [np.float32, np.float64]
        else:
            if use_complex:
                dtypes = [np.float32, np.complex64]
            else:
                dtypes = [np.float32]

        for s in sizes:
            for dtype in dtypes:
                dtype = np.dtype(dtype)

                args = cl_array.arange(queue, a, b, (b-a)/s, dtype=dtype)
                if dtype.kind == "c":
                    # args = args + dtype.type(1j) * args
                    args = args + args * dtype.type(1j)

                gpu_results = gpu_func(args).get()
                cpu_results = cpu_func(args.get())

                my_threshold = threshold
                if dtype.kind == "c" and isinstance(use_complex, float):
                    my_threshold = use_complex

                max_err = np.max(np.abs(cpu_results - gpu_results))
                assert (max_err <= my_threshold).all(), \
                        (max_err, name, dtype)

    return test


if have_cl():
    test_ceil = make_unary_function_test("ceil", (-10, 10))
    test_floor = make_unary_function_test("ceil", (-10, 10))
    test_fabs = make_unary_function_test("fabs", (-10, 10))
    test_exp = make_unary_function_test("exp", (-3, 3), 1e-5, use_complex=True)
    test_log = make_unary_function_test("log", (1e-5, 1), 1e-6, use_complex=True)
    test_log10 = make_unary_function_test("log10", (1e-5, 1), 5e-7)
    test_sqrt = make_unary_function_test("sqrt", (1e-5, 1), 3e-7, use_complex=True)

    test_sin = make_unary_function_test("sin", (-10, 10), 2e-7, use_complex=2e-2)
    test_cos = make_unary_function_test("cos", (-10, 10), 2e-7, use_complex=2e-2)
    test_asin = make_unary_function_test("asin", (-0.9, 0.9), 5e-7)
    test_acos = make_unary_function_test("acos", (-0.9, 0.9), 5e-7)
    test_tan = make_unary_function_test("tan",
            (-math.pi/2 + 0.1, math.pi/2 - 0.1), 4e-5, use_complex=True)
    test_atan = make_unary_function_test("atan", (-10, 10), 2e-7)

    test_sinh = make_unary_function_test("sinh", (-3, 3), 2e-6, use_complex=2e-3)
    test_cosh = make_unary_function_test("cosh", (-3, 3), 2e-6, use_complex=2e-3)
    test_tanh = make_unary_function_test("tanh", (-3, 3), 2e-6, use_complex=True)


def test_atan2(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    for s in sizes:
        a = (cl_array.arange(queue, s, dtype=np.float32) - np.float32(s / 2)) / 100
        a2 = (s / 2 - 1 - cl_array.arange(queue, s, dtype=np.float32)) / 100
        b = clmath.atan2(a, a2)

        a = a.get()
        a2 = a2.get()
        b = b.get()

        for i in range(s):
            assert abs(math.atan2(a[i], a2[i]) - b[i]) < 1e-6


def test_atan2pi(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    for s in sizes:
        a = (cl_array.arange(queue, s, dtype=np.float32) - np.float32(s / 2)) / 100
        a2 = (s / 2 - 1 - cl_array.arange(queue, s, dtype=np.float32)) / 100
        b = clmath.atan2pi(a, a2)

        a = a.get()
        a2 = a2.get()
        b = b.get()

        for i in range(s):
            assert abs(math.atan2(a[i], a2[i]) / math.pi - b[i]) < 1e-6


def test_fmod(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    for s in sizes:
        a = cl_array.arange(queue, s, dtype=np.float32)/10
        a2 = cl_array.arange(queue, s, dtype=np.float32)/45.2 + 0.1
        b = clmath.fmod(a, a2)

        a = a.get()
        a2 = a2.get()
        b = b.get()

        for i in range(s):
            assert math.fmod(a[i], a2[i]) == b[i]


def test_ldexp(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    for s in sizes:
        a = cl_array.arange(queue, s, dtype=np.float32)
        a2 = cl_array.arange(queue, s, dtype=np.float32)*1e-3
        b = clmath.ldexp(a, a2)

        a = a.get()
        a2 = a2.get()
        b = b.get()

        for i in range(s):
            assert math.ldexp(a[i], int(a2[i])) == b[i]


def test_modf(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    for s in sizes:
        a = cl_array.arange(queue, s, dtype=np.float32)/10
        fracpart, intpart = clmath.modf(a)

        a = a.get()
        intpart = intpart.get()
        fracpart = fracpart.get()

        for i in range(s):
            fracpart_true, intpart_true = math.modf(a[i])

            assert intpart_true == intpart[i]
            assert abs(fracpart_true - fracpart[i]) < 1e-4


def test_frexp(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    for s in sizes:
        a = cl_array.arange(queue, s, dtype=np.float32)/10
        significands, exponents = clmath.frexp(a)

        a = a.get()
        significands = significands.get()
        exponents = exponents.get()

        for i in range(s):
            sig_true, ex_true = math.frexp(a[i])

            assert sig_true == significands[i]
            assert ex_true == exponents[i]


def test_bessel(ctx_factory):
    try:
        import scipy.special as spec
    except ImportError:
        from pytest import skip
        skip("scipy not present--cannot test Bessel function")

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    if not has_double_support(ctx.devices[0]):
        from pytest import skip
        skip("no double precision support--cannot test bessel function")

    nterms = 30

    try:
        from pyfmmlib import jfuns2d, hank103_vec
    except ImportError:
        use_pyfmmlib = False
    else:
        use_pyfmmlib = True

    print("PYFMMLIB", use_pyfmmlib)

    if use_pyfmmlib:
        a = np.logspace(-3, 3, 10**6)
    else:
        a = np.logspace(-5, 5, 10**6)

    for which_func, cl_func, scipy_func, is_rel in [
            ("j", clmath.bessel_jn, spec.jn, False),
            ("y", clmath.bessel_yn, spec.yn, True)
            ]:
        if is_rel:
            def get_err(check, ref):
                return np.max(np.abs(check-ref)) / np.max(np.abs(ref))
        else:
            def get_err(check, ref):
                return np.max(np.abs(check-ref))

        if use_pyfmmlib:
            pfymm_result = np.empty((len(a), nterms), dtype=np.complex128)
            if which_func == "j":
                for i, a_i in enumerate(a):
                    if i % 100000 == 0:
                        print("%.1f %%" % (100 * i/len(a)))
                    ier, fjs, _, _ = jfuns2d(nterms, a_i, 1, 0, 10000)
                    pfymm_result[i] = fjs[:nterms]
                assert ier == 0
            elif which_func == "y":
                h0, h1 = hank103_vec(a, ifexpon=1)
                pfymm_result[:, 0] = h0.imag
                pfymm_result[:, 1] = h1.imag

        a_dev = cl_array.to_device(queue, a)

        for n in range(0, nterms):
            cl_bessel = cl_func(n, a_dev).get()
            scipy_bessel = scipy_func(n, a)

            error_scipy = get_err(cl_bessel, scipy_bessel)
            assert error_scipy < 1e-10, error_scipy

            if use_pyfmmlib and (
                    which_func == "j"
                    or
                    (which_func == "y" and n in [0, 1])):
                pyfmm_bessel = pfymm_result[:, n]
                error_pyfmm = get_err(cl_bessel, pyfmm_bessel)
                assert error_pyfmm < 1e-10, error_pyfmm
                error_pyfmm_scipy = get_err(scipy_bessel, pyfmm_bessel)
                print(which_func, n, error_scipy, error_pyfmm, error_pyfmm_scipy)
            else:
                print(which_func, n, error_scipy)

            assert not np.isnan(cl_bessel).any()

            if 0 and n == 15:
                import matplotlib.pyplot as pt
                #pt.plot(scipy_bessel)
                #pt.plot(cl_bessel)

                pt.loglog(a, np.abs(cl_bessel-scipy_bessel), label="vs scipy")
                if use_pyfmmlib:
                    pt.loglog(a, np.abs(cl_bessel-pyfmm_bessel), label="vs pyfmmlib")
                pt.legend()
                pt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
