from __future__ import division
import math
import numpy as np
import pytools.test

def have_cl():
    try:
        import pyopencl
        return True
    except:
        return False

if have_cl():
    import pyopencl.array as cl_array
    import pyopencl as cl
    import pyopencl.clmath as clmath
    from pyopencl.tools import pytest_generate_tests_for_pyopencl \
            as pytest_generate_tests
    from pyopencl.characterize import has_double_support





sizes = [10, 128, 1<<10, 1<<11, 1<<13]




numpy_func_names = {
        "asin": "arcsin",
        "acos": "arccos",
        "atan": "arctan",
        }




def make_unary_function_test(name, limits=(0, 1), threshold=0):
    (a, b) = limits
    a = float(a)
    b = float(b)

    def test(ctx_getter):
        context = ctx_getter()
        queue = cl.CommandQueue(context)

        gpu_func = getattr(clmath, name)
        cpu_func = getattr(np, numpy_func_names.get(name, name))

        if has_double_support(context.devices[0]):
            dtypes = [np.float32, np.float64]
        else:
            dtypes = [np.float32]

        for s in sizes:
            for dtype in dtypes:
                args = cl_array.arange(queue, a, b, (b-a)/s, 
                        dtype=np.float32)
                gpu_results = gpu_func(args).get()
                cpu_results = cpu_func(args.get())

                max_err = np.max(np.abs(cpu_results - gpu_results))
                assert (max_err <= threshold).all(), \
                        (max_err, name, dtype)

    return pytools.test.mark_test.opencl(test)




if have_cl():
    test_ceil = make_unary_function_test("ceil", (-10, 10))
    test_floor = make_unary_function_test("ceil", (-10, 10))
    test_fabs = make_unary_function_test("fabs", (-10, 10))
    test_exp = make_unary_function_test("exp", (-3, 3), 1e-5)
    test_log = make_unary_function_test("log", (1e-5, 1), 1e-6)
    test_log10 = make_unary_function_test("log10", (1e-5, 1), 5e-7)
    test_sqrt = make_unary_function_test("sqrt", (1e-5, 1), 2e-7)

    test_sin = make_unary_function_test("sin", (-10, 10), 2e-7)
    test_cos = make_unary_function_test("cos", (-10, 10), 2e-7)
    test_asin = make_unary_function_test("asin", (-0.9, 0.9), 5e-7)
    test_acos = make_unary_function_test("acos", (-0.9, 0.9), 5e-7)
    test_tan = make_unary_function_test("tan", 
            (-math.pi/2 + 0.1, math.pi/2 - 0.1), 1e-5)
    test_atan = make_unary_function_test("atan", (-10, 10), 2e-7)

    test_sinh = make_unary_function_test("sinh", (-3, 3), 1e-6)
    test_cosh = make_unary_function_test("cosh", (-3, 3), 1e-6)
    test_tanh = make_unary_function_test("tanh", (-3, 3), 2e-6)




@pytools.test.mark_test.opencl
def test_fmod(ctx_getter):
    context = ctx_getter()
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

@pytools.test.mark_test.opencl
def test_ldexp(ctx_getter):
    context = ctx_getter()
    queue = cl.CommandQueue(context)

    for s in sizes:
        a = cl_array.arange(queue, s, dtype=np.float32)
        a2 = cl_array.arange(queue, s, dtype=np.float32)*1e-3
        b = clmath.ldexp(a,a2)

        a = a.get()
        a2 = a2.get()
        b = b.get()

        for i in range(s):
            assert math.ldexp(a[i], int(a2[i])) == b[i]

@pytools.test.mark_test.opencl
def test_modf(ctx_getter):
    context = ctx_getter()
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

@pytools.test.mark_test.opencl
def test_frexp(ctx_getter):
    context = ctx_getter()
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




if __name__ == "__main__":
    # make sure that import failures get reported, instead of skipping the tests.
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])
