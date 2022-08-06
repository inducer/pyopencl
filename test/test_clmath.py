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

# avoid spurious: pytest.mark.parametrize is not callable
# avoid spurious: Module 'scipy.special' has no 'jn' member; maybe 'jv'
# pylint: disable=not-callable,no-member


import math
import numpy as np

import pytest

import pyopencl.array as cl_array
import pyopencl as cl
import pyopencl.clmath as clmath
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl
        as pytest_generate_tests)
from pyopencl.characterize import has_double_support, has_struct_arg_count_bug

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

        dev = context.devices[0]

        if has_double_support(dev):
            if use_complex and has_struct_arg_count_bug(dev) == "apple":
                dtypes = [np.float32, np.float64, np.complex64]
            elif use_complex:
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

test_sinh = make_unary_function_test("sinh", (-3, 3), 3e-6, use_complex=2e-3)
test_cosh = make_unary_function_test("cosh", (-3, 3), 3e-6, use_complex=2e-3)
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
                    or (which_func == "y" and n in [0, 1])):
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


@pytest.mark.parametrize("ref_src", ["pyfmmlib", "scipy"])
def test_complex_bessel(ctx_factory, ref_src):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    if not has_double_support(ctx.devices[0]):
        from pytest import skip
        skip("no double precision support--cannot test complex bessel function")

    v = 40
    n = 10**5

    rng = np.random.default_rng(seed=13)
    z = (
        np.logspace(-5, 2, n)
        * np.exp(1j * 2 * np.pi * rng.random(n)))

    if ref_src == "pyfmmlib":
        pyfmmlib = pytest.importorskip("pyfmmlib")

        jv_ref = np.zeros(len(z), "complex")

        vin = v+1

        for i in range(len(z)):
            ier, fjs, _, _ = pyfmmlib.jfuns2d(vin, z[i], 1, 0, 10000)
            assert ier == 0
            jv_ref[i] = fjs[v]

    elif ref_src == "scipy":
        spec = pytest.importorskip("scipy.special")
        jv_ref = spec.jv(v, z)

    else:
        raise ValueError("ref_src")

    z_dev = cl_array.to_device(queue, z)

    jv_dev = clmath.bessel_jn(v, z_dev)

    abs_err_jv = np.abs(jv_dev.get() - jv_ref)
    abs_jv_ref = np.abs(jv_ref)
    rel_err_jv = abs_err_jv/abs_jv_ref

    # use absolute error instead if the function value itself is too small
    tiny = 1e-300
    ind = abs_jv_ref < tiny
    rel_err_jv[ind] = abs_err_jv[ind]

    # if the reference value is inf or nan, set the error to zero
    ind1 = np.isinf(abs_jv_ref)
    ind2 = np.isnan(abs_jv_ref)

    rel_err_jv[ind1] = 0
    rel_err_jv[ind2] = 0

    if 0:
        print(abs(z))
        print(np.abs(jv_ref))
        print(np.abs(jv_dev.get()))
        print(rel_err_jv)

    max_err = np.max(rel_err_jv)
    assert max_err <= 2e-13, max_err

    print("Jv", np.max(rel_err_jv))

    if 0:
        import matplotlib.pyplot as pt
        pt.loglog(np.abs(z), rel_err_jv)
        pt.show()


@pytest.mark.parametrize("ref_src", ["pyfmmlib", "scipy"])
def test_hankel_01_complex(ctx_factory, ref_src):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    if not has_double_support(ctx.devices[0]):
        from pytest import skip
        skip("no double precision support--cannot test complex bessel function")

    rng = np.random.default_rng(seed=11)
    n = 10**6
    z = (
        np.logspace(-5, 2, n)
        * np.exp(1j * 2 * np.pi * rng.random(n)))

    def get_err(check, ref):
        return np.max(np.abs(check-ref)) / np.max(np.abs(ref))

    if ref_src == "pyfmmlib":
        pyfmmlib = pytest.importorskip("pyfmmlib")
        h0_ref, h1_ref = pyfmmlib.hank103_vec(z, ifexpon=1)
    elif ref_src == "scipy":
        spec = pytest.importorskip("scipy.special")
        h0_ref = spec.hankel1(0, z)
        h1_ref = spec.hankel1(1, z)

    else:
        raise ValueError("ref_src")

    z_dev = cl_array.to_device(queue, z)

    h0_dev, h1_dev = clmath.hankel_01(z_dev)

    rel_err_h0 = np.abs(h0_dev.get() - h0_ref)/np.abs(h0_ref)
    rel_err_h1 = np.abs(h1_dev.get() - h1_ref)/np.abs(h1_ref)

    max_rel_err_h0 = np.max(rel_err_h0)
    max_rel_err_h1 = np.max(rel_err_h1)

    print("H0", max_rel_err_h0)
    print("H1", max_rel_err_h1)

    assert max_rel_err_h0 < 4e-13
    assert max_rel_err_h1 < 2e-13

    if 0:
        import matplotlib.pyplot as pt
        pt.loglog(np.abs(z), rel_err_h0)
        pt.loglog(np.abs(z), rel_err_h1)
        pt.show()


def test_outoforderqueue_clmath(ctx_factory):
    context = ctx_factory()
    try:
        queue = cl.CommandQueue(context,
               properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
    except Exception:
        pytest.skip("out-of-order queue not available")

    rng = np.random.default_rng(seed=42)
    a = rng.random(10**6, dtype=np.float32)
    a_gpu = cl_array.to_device(queue, a)
    # testing that clmath functions wait for and create events
    b_gpu = clmath.fabs(clmath.sin(a_gpu * 5))
    queue.finish()
    b1 = b_gpu.get()
    b = np.abs(np.sin(a * 5))
    assert np.abs(b1 - b).mean() < 1e-5


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
