#! /usr/bin/env python

import numpy as np
import numpy.linalg as la
import sys
import pytools.test
from pytools import memoize


def have_cl():
    try:
        import pyopencl
        return True
    except:
        return False

if have_cl():
    import pyopencl as cl
    import pyopencl.array as cl_array
    import pyopencl.tools as cl_tools
    from pyopencl.tools import pytest_generate_tests_for_pyopencl \
            as pytest_generate_tests
    from pyopencl.characterize import has_double_support




# {{{ helpers

TO_REAL = {
        np.dtype(np.complex64): np.float32,
        np.dtype(np.complex128): np.float64
        }

def general_clrand(queue, shape, dtype):
    from pyopencl.clrandom import rand as clrand

    dtype = np.dtype(dtype)
    if dtype.kind == "c":
        real_dtype = dtype.type(0).real.dtype
        return clrand(queue, shape, real_dtype) + 1j*clrand(queue, shape, real_dtype)
    else:
        return clrand(queue, shape, dtype)


def make_random_array(queue, dtype, size):
    from pyopencl.clrandom import rand

    dtype = np.dtype(dtype)
    if dtype.kind == "c":
        real_dtype = TO_REAL[dtype]
        return (rand(queue, shape=(size,), dtype=real_dtype).astype(dtype)
                + dtype.type(1j)
                * rand(queue, shape=(size,), dtype=real_dtype).astype(dtype))
    else:
        return rand(queue, shape=(size,), dtype=dtype)

# }}}

# {{{ dtype-related

@pytools.test.mark_test.opencl
def test_basic_complex(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand

    size = 500

    ary =  (rand(queue, shape=(size,), dtype=np.float32).astype(np.complex64)
            + 1j* rand(queue, shape=(size,), dtype=np.float32).astype(np.complex64))
    c = np.complex64(5+7j)

    host_ary = ary.get()
    assert la.norm((c*ary).get() - c*host_ary) < 1e-5 * la.norm(host_ary)

@pytools.test.mark_test.opencl
def test_mix_complex(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    size = 10

    dtypes = [
            (np.float32, np.complex64),
            #(np.int32, np.complex64),
            ]

    if has_double_support(context.devices[0]):
        dtypes.extend([
            (np.float32, np.float64),
            (np.float32, np.complex128),
            (np.float64, np.complex64),
            (np.float64, np.complex128),
            ])

    from operator import add, mul, sub, truediv
    for op in [add, sub, mul, truediv, pow]:
        for dtype_a0, dtype_b0 in dtypes:
            for dtype_a, dtype_b in [
                    (dtype_a0, dtype_b0),
                    (dtype_b0, dtype_a0),
                    ]:
                for is_scalar_a, is_scalar_b in [
                        (False, False),
                        (False, True),
                        (True, False),
                        ]:
                    if is_scalar_a:
                        ary_a = make_random_array(queue, dtype_a, 1).get()[0]
                        host_ary_a = ary_a
                    else:
                        ary_a = make_random_array(queue, dtype_a, size)
                        host_ary_a = ary_a.get()

                    if is_scalar_b:
                        ary_b = make_random_array(queue, dtype_b, 1).get()[0]
                        host_ary_b = ary_b
                    else:
                        ary_b = make_random_array(queue, dtype_b, size)
                        host_ary_b = ary_b.get()

                    print(op, dtype_a, dtype_b, is_scalar_a, is_scalar_b)
                    dev_result = op(ary_a, ary_b).get()
                    host_result = op(host_ary_a, host_ary_b)

                    if host_result.dtype != dev_result.dtype:
                        # This appears to be a numpy bug, where we get
                        # served a Python complex that is really a
                        # smaller numpy complex.

                        print("HOST_DTYPE: %s DEV_DTYPE: %s" % (
                                host_result.dtype, dev_result.dtype))

                        dev_result = dev_result.astype(host_result.dtype)

                    err = la.norm(host_result-dev_result)/la.norm(host_result)
                    print(err)
                    correct = err < 1e-5
                    if not correct:
                        print(host_result)
                        print(dev_result)
                        print(host_result - dev_result)

                    assert correct

@pytools.test.mark_test.opencl
def test_pow_neg1_vs_inv(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    device = ctx.devices[0]
    if not has_double_support(device):
        from py.test import skip
        skip("double precision not supported on %s" % device)

    a_dev = make_random_array(queue, np.complex128, 20000)

    res1 = (a_dev ** (-1)).get()
    res2 = (1/a_dev).get()
    ref = 1/a_dev.get()

    assert la.norm(res1-ref, np.inf) / la.norm(ref) < 1e-13
    assert la.norm(res2-ref, np.inf) / la.norm(ref) < 1e-13


@pytools.test.mark_test.opencl
def test_vector_fill(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a_gpu = cl_array.Array(queue, 100, dtype=cl_array.vec.float4)
    a_gpu.fill(cl_array.vec.make_float4(0.0, 0.0, 1.0, 0.0))
    a = a_gpu.get()
    assert a.dtype is cl_array.vec.float4

    a_gpu = cl_array.zeros(queue, 100, dtype=cl_array.vec.float4)

@pytools.test.mark_test.opencl
def test_absrealimag(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    def real(x): return x.real
    def imag(x): return x.imag
    def conj(x): return x.conj()

    n = 111
    for func in [abs, real, imag, conj]:
        for dtype in [np.int32, np.float32, np.complex64]:
            print(func, dtype)
            a = -make_random_array(queue, dtype, n)

            host_res = func(a.get())
            dev_res = func(a).get()

            correct = np.allclose(dev_res, host_res)
            if not correct:
                print(dev_res)
                print(host_res)
                print(dev_res-host_res)
            assert correct

# }}}

# {{{ operands

@pytools.test.mark_test.opencl
def test_pow_array(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    a_gpu = cl_array.to_device(queue, a)

    result = pow(a_gpu, a_gpu).get()
    assert (np.abs(a ** a - result) < 1e-3).all()

    result = (a_gpu ** a_gpu).get()
    assert (np.abs(pow(a, a) - result) < 1e-3).all()


@pytools.test.mark_test.opencl
def test_pow_number(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    a_gpu = cl_array.to_device(queue, a)

    result = pow(a_gpu, 2).get()
    assert (np.abs(a ** 2 - result) < 1e-3).all()


@pytools.test.mark_test.opencl
def test_multiply(ctx_factory):
    """Test the muliplication of an array with a scalar. """

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    for sz in [10, 50000]:
        for dtype, scalars in [
            (np.float32, [2]),
            (np.complex64, [2j]),
            ]:
            for scalar in scalars:
                a_gpu = make_random_array(queue, dtype, sz)
                a = a_gpu.get()
                a_mult = (scalar * a_gpu).get()

                assert (a * scalar == a_mult).all()


@pytools.test.mark_test.opencl
def test_multiply_array(ctx_factory):
    """Test the multiplication of two arrays."""

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)

    a_gpu = cl_array.to_device(queue, a)
    b_gpu = cl_array.to_device(queue, a)

    a_squared = (b_gpu * a_gpu).get()

    assert (a * a == a_squared).all()


@pytools.test.mark_test.opencl
def test_addition_array(ctx_factory):
    """Test the addition of two arrays."""

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    a_gpu = cl_array.to_device(queue, a)
    a_added = (a_gpu + a_gpu).get()

    assert (a + a == a_added).all()


@pytools.test.mark_test.opencl
def test_addition_scalar(ctx_factory):
    """Test the addition of an array and a scalar."""

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    a_gpu = cl_array.to_device(queue, a)
    a_added = (7 + a_gpu).get()

    assert (7 + a == a_added).all()


@pytools.test.mark_test.opencl
def test_substract_array(ctx_factory):
    """Test the substraction of two arrays."""
    #test data
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    b = np.array([10, 20, 30, 40, 50,
                  60, 70, 80, 90, 100]).astype(np.float32)

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a_gpu = cl_array.to_device(queue, a)
    b_gpu = cl_array.to_device(queue, b)

    result = (a_gpu - b_gpu).get()
    assert (a - b == result).all()

    result = (b_gpu - a_gpu).get()
    assert (b - a == result).all()


@pytools.test.mark_test.opencl
def test_substract_scalar(ctx_factory):
    """Test the substraction of an array and a scalar."""

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    #test data
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)

    #convert a to a gpu object
    a_gpu = cl_array.to_device(queue, a)

    result = (a_gpu - 7).get()
    assert (a - 7 == result).all()

    result = (7 - a_gpu).get()
    assert (7 - a == result).all()


@pytools.test.mark_test.opencl
def test_divide_scalar(ctx_factory):
    """Test the division of an array and a scalar."""

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    a_gpu = cl_array.to_device(queue, a)

    result = (a_gpu / 2).get()
    assert (a / 2 == result).all()

    result = (2 / a_gpu).get()
    assert (np.abs(2 / a - result) < 1e-5).all()


@pytools.test.mark_test.opencl
def test_divide_array(ctx_factory):
    """Test the division of an array and a scalar. """

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    #test data
    a = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).astype(np.float32)
    b = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]).astype(np.float32)

    a_gpu = cl_array.to_device(queue, a)
    b_gpu = cl_array.to_device(queue, b)

    a_divide = (a_gpu / b_gpu).get()
    assert (np.abs(a / b - a_divide) < 1e-3).all()

    a_divide = (b_gpu / a_gpu).get()
    assert (np.abs(b / a - a_divide) < 1e-3).all()

# }}}

# {{{ RNG

@pytools.test.mark_test.opencl
def test_random(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import RanluxGenerator

    if has_double_support(context.devices[0]):
        dtypes = [np.float32, np.float64]
    else:
        dtypes = [np.float32]

    gen = RanluxGenerator(queue, 5120)

    for ary_size in [300, 301, 302, 303, 10007]:
        for dtype in dtypes:
            ran = cl_array.zeros(queue, ary_size, dtype)
            gen.fill_uniform(ran)
            assert (0 < ran.get()).all()
            assert (ran.get() < 1).all()

            gen.synchronize(queue)

            ran = cl_array.zeros(queue, ary_size, dtype)
            gen.fill_uniform(ran, a=4, b=7)
            assert (4 < ran.get()).all()
            assert (ran.get() < 7).all()

            ran = gen.normal(queue, (10007,), dtype, mu=4, sigma=3)

    dtypes = [np.int32]
    for dtype in dtypes:
        ran = gen.uniform(queue, (10000007,), dtype, a=200, b=300)
        assert (200 <= ran.get()).all()
        assert (ran.get() < 300).all()
        #from matplotlib import pyplot as pt
        #pt.hist(ran.get())
        #pt.show()

# }}}

# {{{ elementwise

@pytools.test.mark_test.opencl
def test_elwise_kernel(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand

    a_gpu = clrand(queue, (50,), np.float32)
    b_gpu = clrand(queue, (50,), np.float32)

    from pyopencl.elementwise import ElementwiseKernel
    lin_comb = ElementwiseKernel(context,
            "float a, float *x, float b, float *y, float *z",
            "z[i] = a*x[i] + b*y[i]",
            "linear_combination")

    c_gpu = cl_array.empty_like(a_gpu)
    lin_comb(5, a_gpu, 6, b_gpu, c_gpu)

    assert la.norm((c_gpu - (5 * a_gpu + 6 * b_gpu)).get()) < 1e-5


@pytools.test.mark_test.opencl
def test_elwise_kernel_with_options(ctx_factory):
    from pyopencl.clrandom import rand as clrand
    from pyopencl.elementwise import ElementwiseKernel

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    in_gpu = clrand(queue, (50,), np.float32)

    options = ['-D', 'ADD_ONE']
    add_one = ElementwiseKernel(
        context,
        "float* out, const float *in",
        """
        out[i] = in[i]
        #ifdef ADD_ONE
            +1
        #endif
        ;
        """,
        options=options,
        )

    out_gpu = cl_array.empty_like(in_gpu)
    add_one(out_gpu, in_gpu)

    gt = in_gpu.get() + 1
    gv = out_gpu.get()
    assert la.norm(gv - gt) < 1e-5


@pytools.test.mark_test.opencl
def test_ranged_elwise_kernel(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.elementwise import ElementwiseKernel
    set_to_seven = ElementwiseKernel(context,
            "float *z", "z[i] = 7", "set_to_seven")

    for i, slc in enumerate([
            slice(5, 20000),
            slice(5, 20000, 17),
            slice(3000, 5, -1),
            slice(1000, -1),
            ]):

        a_gpu = cl_array.zeros(queue, (50000,), dtype=np.float32)
        a_cpu = np.zeros(a_gpu.shape, a_gpu.dtype)

        a_cpu[slc] = 7
        set_to_seven(a_gpu, slice=slc)

        assert (a_cpu == a_gpu.get()).all()

@pytools.test.mark_test.opencl
def test_take(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    idx = cl_array.arange(queue, 0, 200000, 2, dtype=np.uint32)
    a = cl_array.arange(queue, 0, 600000, 3, dtype=np.float32)
    result = cl_array.take(a, idx)
    assert ((3 * idx).get() == result.get()).all()


@pytools.test.mark_test.opencl
def test_arange(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    n = 5000
    a = cl_array.arange(queue, n, dtype=np.float32)
    assert (np.arange(n, dtype=np.float32) == a.get()).all()


@pytools.test.mark_test.opencl
def test_reverse(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    n = 5000
    a = np.arange(n).astype(np.float32)
    a_gpu = cl_array.to_device(queue, a)

    a_gpu = a_gpu.reverse()

    assert (a[::-1] == a_gpu.get()).all()

@pytools.test.mark_test.opencl
def test_if_positive(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand

    l = 20000
    a_gpu = clrand(queue, (l,), np.float32)
    b_gpu = clrand(queue, (l,), np.float32)
    a = a_gpu.get()
    b = b_gpu.get()

    max_a_b_gpu = cl_array.maximum(a_gpu, b_gpu)
    min_a_b_gpu = cl_array.minimum(a_gpu, b_gpu)

    print(max_a_b_gpu)
    print(np.maximum(a, b))

    assert la.norm(max_a_b_gpu.get() - np.maximum(a, b)) == 0
    assert la.norm(min_a_b_gpu.get() - np.minimum(a, b)) == 0


@pytools.test.mark_test.opencl
def test_take_put(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    for n in [5, 17, 333]:
        one_field_size = 8
        buf_gpu = cl_array.zeros(queue,
                n * one_field_size, dtype=np.float32)
        dest_indices = cl_array.to_device(queue,
                np.array([0, 1, 2,  3, 32, 33, 34, 35], dtype=np.uint32))
        read_map = cl_array.to_device(queue,
                np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=np.uint32))

        cl_array.multi_take_put(
                arrays=[buf_gpu for i in range(n)],
                dest_indices=dest_indices,
                src_indices=read_map,
                src_offsets=[i * one_field_size for i in range(n)],
                dest_shape=(96,))


@pytools.test.mark_test.opencl
def test_astype(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand

    if not has_double_support(context.devices[0]):
        from py.test import skip
        skip("double precision not supported on %s" % device)

    a_gpu = clrand(queue, (2000,), dtype=np.float32)

    a = a_gpu.get().astype(np.float64)
    a2 = a_gpu.astype(np.float64).get()

    assert a2.dtype == np.float64
    assert la.norm(a - a2) == 0, (a, a2)

    a_gpu = clrand(queue, (2000,), dtype=np.float64)

    a = a_gpu.get().astype(np.float32)
    a2 = a_gpu.astype(np.float32).get()

    assert a2.dtype == np.float32
    assert la.norm(a - a2) / la.norm(a) < 1e-7

# }}}

# {{{ reduction

@pytools.test.mark_test.opencl
def test_sum(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    n = 200000
    for dtype in [np.float32, np.complex64]:
        a_gpu = general_clrand(queue, (n,), dtype)

        a = a_gpu.get()

        sum_a = np.sum(a)
        sum_a_gpu = cl_array.sum(a_gpu).get()

        assert abs(sum_a_gpu - sum_a) / abs(sum_a) < 1e-4


@pytools.test.mark_test.opencl
def test_minmax(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand

    if has_double_support(context.devices[0]):
        dtypes = [np.float64, np.float32, np.int32]
    else:
        dtypes = [np.float32, np.int32]

    for what in ["min", "max"]:
        for dtype in dtypes:
            a_gpu = clrand(queue, (200000,), dtype)
            a = a_gpu.get()

            op_a = getattr(np, what)(a)
            op_a_gpu = getattr(cl_array, what)(a_gpu).get()

            assert op_a_gpu == op_a, (op_a_gpu, op_a, dtype, what)


@pytools.test.mark_test.opencl
def test_subset_minmax(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand

    l_a = 200000
    gran = 5
    l_m = l_a - l_a // gran + 1

    if has_double_support(context.devices[0]):
        dtypes = [np.float64, np.float32, np.int32]
    else:
        dtypes = [np.float32, np.int32]

    for dtype in dtypes:
        a_gpu = clrand(queue, (l_a,), dtype)
        a = a_gpu.get()

        meaningful_indices_gpu = cl_array.zeros(
                queue, l_m, dtype=np.int32)
        meaningful_indices = meaningful_indices_gpu.get()
        j = 0
        for i in range(len(meaningful_indices)):
            meaningful_indices[i] = j
            j = j + 1
            if j % gran == 0:
                j = j + 1

        meaningful_indices_gpu = cl_array.to_device(
                queue, meaningful_indices)
        b = a[meaningful_indices]

        min_a = np.min(b)
        min_a_gpu = cl_array.subset_min(meaningful_indices_gpu, a_gpu).get()

        assert min_a_gpu == min_a


@pytools.test.mark_test.opencl
def test_dot(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    dtypes = [np.float32, np.complex64]
    if has_double_support(context.devices[0]):
        dtypes.extend([np.float64, np.complex128])

    for a_dtype in dtypes:
        for b_dtype in dtypes:
            print(a_dtype, b_dtype)
            a_gpu = general_clrand(queue, (200000,), a_dtype)
            a = a_gpu.get()
            b_gpu = general_clrand(queue, (200000,), b_dtype)
            b = b_gpu.get()

            dot_ab = np.dot(a, b)

            dot_ab_gpu = cl_array.dot(a_gpu, b_gpu).get()

            assert abs(dot_ab_gpu - dot_ab) / abs(dot_ab) < 1e-4

@memoize
def make_mmc_dtype(device):
    dtype = np.dtype([
        ("cur_min", np.int32),
        ("cur_max", np.int32),
        ("pad", np.int32),
        ])

    name = "minmax_collector"
    from pyopencl.tools import register_dtype, match_dtype_to_c_struct

    dtype, c_decl = match_dtype_to_c_struct(device, name, dtype)
    register_dtype(dtype, name)

    return dtype, c_decl

@pytools.test.mark_test.opencl
def test_struct_reduce(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    mmc_dtype, mmc_c_decl = make_mmc_dtype(context.devices[0])

    preamble = mmc_c_decl + r"""//CL//

    minmax_collector mmc_neutral()
    {
        // FIXME: needs infinity literal in real use, ok here
        minmax_collector result;
        result.cur_min = 1<<30;
        result.cur_max = -(1<<30);
        return result;
    }

    minmax_collector mmc_from_scalar(float x)
    {
        minmax_collector result;
        result.cur_min = x;
        result.cur_max = x;
        return result;
    }

    minmax_collector agg_mmc(minmax_collector a, minmax_collector b)
    {
        minmax_collector result = a;
        if (b.cur_min < result.cur_min)
            result.cur_min = b.cur_min;
        if (b.cur_max > result.cur_max)
            result.cur_max = b.cur_max;
        return result;
    }

    """

    from pyopencl.clrandom import rand as clrand
    a_gpu = clrand(queue, (20000,), dtype=np.int32, a=0, b=10**6)
    a = a_gpu.get()

    from pyopencl.reduction import ReductionKernel
    red = ReductionKernel(context, mmc_dtype,
            neutral="mmc_neutral()",
            reduce_expr="agg_mmc(a, b)", map_expr="mmc_from_scalar(x[i])",
            arguments="__global int *x", preamble=preamble)

    minmax = red(a_gpu).get()
    #print minmax["cur_min"], minmax["cur_max"]
    #print np.min(a), np.max(a)

    assert abs(minmax["cur_min"] - np.min(a)) < 1e-5
    assert abs(minmax["cur_max"] - np.max(a)) < 1e-5

# }}}

# {{{ scan-related

def summarize_error(obtained, desired, orig, thresh=1e-5):
    err = obtained - desired
    ok_count = 0
    bad_count = 0

    bad_limit = 200

    def summarize_counts():
        if ok_count:
            entries.append("<%d ok>" % ok_count)
        if bad_count >= bad_limit:
            entries.append("<%d more bad>" % (bad_count-bad_limit))

    entries = []
    for i, val in enumerate(err):
        if abs(val) > thresh:
            if ok_count:
                summarize_counts()
                ok_count = 0

            bad_count += 1

            if bad_count < bad_limit:
                entries.append("%r (want: %r, got: %r, orig: %r)" % (obtained[i], desired[i],
                    obtained[i], orig[i]))
        else:
            if bad_count:
                summarize_counts()
                bad_count = 0

            ok_count += 1


    summarize_counts()

    return " ".join(entries)

scan_test_counts = [
    10,
    2 ** 8 - 1,
    2 ** 8,
    2 ** 8 + 1,
    2 ** 10 - 5,
    2 ** 10,
    2 ** 10 + 5,
    2 ** 12 - 5,
    2 ** 12,
    2 ** 12 + 5,
    2 ** 20 - 2 ** 18,
    2 ** 20 - 2 ** 18 + 5,
    2 ** 20 + 1,
    2 ** 20,
    2 ** 23 + 3,
    2 ** 24 + 5
    ]

@pytools.test.mark_test.opencl
def test_scan(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.scan import InclusiveScanKernel, ExclusiveScanKernel

    dtype = np.int32
    for cls in [
            InclusiveScanKernel,
            ExclusiveScanKernel
            ]:
        knl = cls(context, dtype, "a+b", "0")

        for n in scan_test_counts:
            host_data = np.random.randint(0, 10, n).astype(dtype)
            dev_data = cl_array.to_device(queue, host_data)

            assert (host_data == dev_data.get()).all() # /!\ fails on Nv GT2?? for some drivers

            knl(dev_data)

            desired_result = np.cumsum(host_data, axis=0)
            if cls is ExclusiveScanKernel:
                desired_result -= host_data

            is_ok = (dev_data.get() == desired_result).all()
            if 1 and not is_ok:
                print("something went wrong, summarizing error...")
                print(summarize_error(dev_data.get(), desired_result, host_data))

            print("n:%d %s worked:%s" % (n, cls, is_ok))
            assert is_ok
            from gc import collect
            collect()

@pytools.test.mark_test.opencl
def test_copy_if(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand
    for n in scan_test_counts:
        a_dev = clrand(queue, (n,), dtype=np.int32, a=0, b=1000)
        a = a_dev.get()

        from pyopencl.algorithm import copy_if

        crit = a_dev.dtype.type(300)
        selected = a[a>crit]
        selected_dev, count_dev = copy_if(a_dev, "ary[i] > myval", [("myval", crit)])

        assert (selected_dev.get()[:count_dev.get()] == selected).all()

@pytools.test.mark_test.opencl
def test_partition(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand
    for n in scan_test_counts:
        a_dev = clrand(queue, (n,), dtype=np.int32, a=0, b=1000)
        a = a_dev.get()

        crit = a_dev.dtype.type(300)
        true_host = a[a>crit]
        false_host = a[a<=crit]

        from pyopencl.algorithm import partition
        true_dev, false_dev, count_true_dev = partition(a_dev, "ary[i] > myval", [("myval", crit)])

        count_true_dev = count_true_dev.get()

        assert (true_dev.get()[:count_true_dev] == true_host).all()
        assert (false_dev.get()[:n-count_true_dev] == false_host).all()

@pytools.test.mark_test.opencl
def test_unique(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand
    for n in scan_test_counts:
        a_dev = clrand(queue, (n,), dtype=np.int32, a=0, b=1000)
        a = a_dev.get()
        a = np.sort(a)
        a_dev = cl_array.to_device(queue, a)

        a_unique_host = np.unique(a)

        from pyopencl.algorithm import unique
        a_unique_dev, count_unique_dev = unique(a_dev)

        count_unique_dev = count_unique_dev.get()

        assert (a_unique_dev.get()[:count_unique_dev] == a_unique_host).all()

@pytools.test.mark_test.opencl
def test_index_preservation(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.scan import GenericScanKernel, GenericDebugScanKernel
    classes = [GenericScanKernel]

    dev = context.devices[0]
    if dev.type == cl.device_type.CPU:
        classes.append(GenericDebugScanKernel)

    for cls in classes:
        for n in scan_test_counts:
            knl = cls(
                    context, np.int32,
                    arguments="__global int *out",
                    input_expr="i",
                    scan_expr="b", neutral="0",
                    output_statement="""
                        out[i] = item;
                        """)

            out = cl_array.empty(queue, n, dtype=np.int32)
            knl(out)

            assert (out.get() == np.arange(n)).all()

@pytools.test.mark_test.opencl
def test_segmented_scan(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.tools import dtype_to_ctype
    dtype = np.int32
    ctype = dtype_to_ctype(dtype)

    #for is_exclusive in [False, True]:
    for is_exclusive in [True, False]:
        if is_exclusive:
            output_statement = "out[i] = prev_item"
        else:
            output_statement = "out[i] = item"

        from pyopencl.scan import GenericScanKernel
        knl = GenericScanKernel(context, dtype,
                arguments="__global %s *ary, __global char *segflags, __global %s *out"
                    % (ctype, ctype),
                input_expr="ary[i]",
                scan_expr="across_seg_boundary ? b : (a+b)", neutral="0",
                is_segment_start_expr="segflags[i]",
                output_statement=output_statement,
                options=[])

        np.set_printoptions(threshold=2000)
        from random import randrange
        from pyopencl.clrandom import rand as clrand
        for n in scan_test_counts:
            a_dev = clrand(queue, (n,), dtype=dtype, a=0, b=10)
            a = a_dev.get()

            if 10 <= n < 20:
                seg_boundaries_values = [
                        [0, 9],
                        [0, 3],
                        [4, 6],
                        ]
            else:
                seg_boundaries_values = []
                for i in range(10):
                    seg_boundary_count = max(2, min(100, randrange(0, int(0.4*n))))
                    seg_boundaries = [randrange(n) for i in range(seg_boundary_count)]
                    if n >= 1029:
                        seg_boundaries.insert(0, 1028)
                    seg_boundaries.sort()
                    seg_boundaries_values.append(seg_boundaries)

            for seg_boundaries in seg_boundaries_values:
                #print "BOUNDARIES", seg_boundaries
                #print a

                seg_boundary_flags = np.zeros(n, dtype=np.uint8)
                seg_boundary_flags[seg_boundaries] = 1
                seg_boundary_flags_dev = cl_array.to_device(queue, seg_boundary_flags)

                seg_boundaries.insert(0, 0)

                result_host = a.copy()
                for i, seg_start in enumerate(seg_boundaries):
                    if i+1 < len(seg_boundaries):
                        seg_end = seg_boundaries[i+1]
                    else:
                        seg_end = None

                    if is_exclusive:
                        result_host[seg_start+1:seg_end] = np.cumsum(
                                a[seg_start:seg_end][:-1])
                        result_host[seg_start] = 0
                    else:
                        result_host[seg_start:seg_end] = np.cumsum(
                                a[seg_start:seg_end])

                #print "REF", result_host

                result_dev = cl_array.empty_like(a_dev)
                knl(a_dev, seg_boundary_flags_dev, result_dev)

                #print "RES", result_dev
                is_correct = (result_dev.get() == result_host).all()
                if not is_correct:
                    diff = result_dev.get() - result_host
                    print("RES-REF", diff)
                    print("ERRWHERE", np.where(diff))
                    print(n, list(seg_boundaries))

                assert is_correct

            print("%d excl:%s done" % (n, is_exclusive))


@pytools.test.mark_test.opencl
def test_sort(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    dtype = np.int32

    from pyopencl.algorithm import RadixSort
    sort = RadixSort(context, "int *ary", key_expr="ary[i]",
            sort_arg_names=["ary"])

    from pyopencl.clrandom import RanluxGenerator
    rng = RanluxGenerator(queue, seed=15)

    from time import time

    for n in scan_test_counts:
        print(n)

        print("  rng")
        a_dev = rng.uniform(queue, (n,), dtype=dtype, a=0, b=2**16)
        a = a_dev.get()

        dev_start = time()
        print("  device")
        a_dev_sorted, = sort(a_dev, key_bits=16)
        queue.finish()
        dev_end = time()
        print("  numpy")
        a_sorted = np.sort(a)
        numpy_end = time()

        numpy_elapsed = numpy_end-dev_end
        dev_elapsed = dev_end-dev_start
        print ("  dev: %.2f MKeys/s numpy: %.2f MKeys/s ratio: %.2fx" % (
                1e-6*n/dev_elapsed, 1e-6*n/numpy_elapsed, numpy_elapsed/dev_elapsed))
        assert (a_dev_sorted.get() == a_sorted).all()


# }}}

# {{{ misc

@pytools.test.mark_test.opencl
def test_len(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    a_cpu = cl_array.to_device(queue, a)
    assert len(a_cpu) == 10

@pytools.test.mark_test.opencl
def test_stride_preservation(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    A = np.random.rand(3, 3)
    AT = A.T
    print(AT.flags.f_contiguous, AT.flags.c_contiguous)
    AT_GPU = cl_array.to_device(queue, AT)
    print(AT_GPU.flags.f_contiguous, AT_GPU.flags.c_contiguous)
    assert np.allclose(AT_GPU.get(), AT)

@pytools.test.mark_test.opencl
def test_nan_arithmetic(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    def make_nan_contaminated_vector(size):
        shape = (size,)
        a = np.random.randn(*shape).astype(np.float32)
        from random import randrange
        for i in range(size // 10):
            a[randrange(0, size)] = float('nan')
        return a

    size = 1 << 20

    a = make_nan_contaminated_vector(size)
    a_gpu = cl_array.to_device(queue, a)
    b = make_nan_contaminated_vector(size)
    b_gpu = cl_array.to_device(queue, b)

    ab = a * b
    ab_gpu = (a_gpu * b_gpu).get()

    assert (np.isnan(ab) == np.isnan(ab_gpu)).all()

@pytools.test.mark_test.opencl
def test_mem_pool_with_arrays(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)
    mem_pool = cl_tools.MemoryPool(cl_tools.CLAllocator(context))

    a_dev = cl_array.arange(queue, 2000, dtype=np.float32, allocator=mem_pool)
    b_dev = cl_array.to_device(queue, np.arange(2000), allocator=mem_pool) + 4000

    result = cl_array.dot(a_dev, b_dev)
    assert a_dev.allocator is mem_pool
    assert b_dev.allocator is mem_pool
    assert result.allocator is mem_pool

@pytools.test.mark_test.opencl
def test_view(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.arange(128).reshape(8, 16).astype(np.float32)
    a_dev = cl_array.to_device(queue, a)

    # same dtype
    view = a_dev.view()
    assert view.shape == a_dev.shape and view.dtype == a_dev.dtype

    # larger dtype
    view = a_dev.view(np.complex64)
    assert view.shape == (8, 8) and view.dtype == np.complex64

    # smaller dtype
    view = a_dev.view(np.int16)
    assert view.shape == (8, 32) and view.dtype == np.int16

# }}}

@pytools.test.mark_test.opencl
def no_test_slice(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand

    l = 20000
    a_gpu = clrand(queue, (l,))
    a = a_gpu.get()

    from random import randrange
    for i in range(200):
        start = randrange(l)
        end = randrange(start, l)

        a_gpu_slice = a_gpu[start:end]
        a_slice = a[start:end]

        assert la.norm(a_gpu_slice.get() - a_slice) == 0





if __name__ == "__main__":
    # make sure that import failures get reported, instead of skipping the
    # tests.
    import pyopencl as cl

    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: filetype=pyopencl:fdm=marker
