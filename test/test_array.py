#! /usr/bin/env python
from __future__ import division, with_statement, absolute_import, print_function

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

import numpy as np
import numpy.linalg as la
import sys

from six.moves import range
import pytest

import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.cltypes as cltypes
import pyopencl.tools as cl_tools
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)
from pyopencl.characterize import has_double_support, has_struct_arg_count_bug

from pyopencl.clrandom import RanluxGenerator, PhiloxGenerator, ThreefryGenerator

_PYPY = cl._PYPY


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
                + rand(queue, shape=(size,), dtype=real_dtype).astype(dtype)
                * dtype.type(1j))
    else:
        return rand(queue, shape=(size,), dtype=dtype)

# }}}


# {{{ dtype-related

def test_basic_complex(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand

    size = 500

    ary = (rand(queue, shape=(size,), dtype=np.float32).astype(np.complex64)
            + rand(queue, shape=(size,), dtype=np.float32).astype(np.complex64) * 1j)
    c = np.complex64(5+7j)

    host_ary = ary.get()
    assert la.norm((ary*c).get() - c*host_ary) < 1e-5 * la.norm(host_ary)


def test_mix_complex(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    size = 10

    dtypes = [
            (np.float32, np.complex64),
            #(np.int32, np.complex64),
            ]

    dev = context.devices[0]
    if has_double_support(dev) and has_struct_arg_count_bug(dev) == "apple":
        dtypes.extend([
            (np.float32, np.float64),
            ])
    elif has_double_support(dev):
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
                    correct = err < 1e-4
                    if not correct:
                        print(host_result)
                        print(dev_result)
                        print(host_result - dev_result)

                    assert correct


def test_pow_neg1_vs_inv(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    device = ctx.devices[0]
    if not has_double_support(device):
        from pytest import skip
        skip("double precision not supported on %s" % device)
    if has_struct_arg_count_bug(device) == "apple":
        from pytest import xfail
        xfail("apple struct arg counting broken")

    a_dev = make_random_array(queue, np.complex128, 20000)

    res1 = (a_dev ** (-1)).get()
    res2 = (1/a_dev).get()
    ref = 1/a_dev.get()

    assert la.norm(res1-ref, np.inf) / la.norm(ref) < 1e-13
    assert la.norm(res2-ref, np.inf) / la.norm(ref) < 1e-13


def test_vector_fill(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a_gpu = cl_array.Array(queue, 100, dtype=cltypes.float4)
    a_gpu.fill(cltypes.make_float4(0.0, 0.0, 1.0, 0.0))
    a = a_gpu.get()
    assert a.dtype == cltypes.float4

    a_gpu = cl_array.zeros(queue, 100, dtype=cltypes.float4)


def test_absrealimag(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    def real(x):
        return x.real

    def imag(x):
        return x.imag

    def conj(x):
        return x.conj()

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


def test_custom_type_zeros(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    if not (
            queue._get_cl_version() >= (1, 2)
            and cl.get_cl_header_version() >= (1, 2)):
        pytest.skip("CL1.2 not available")

    dtype = np.dtype([
        ("cur_min", np.int32),
        ("cur_max", np.int32),
        ("pad", np.int32),
        ])

    from pyopencl.tools import get_or_register_dtype, match_dtype_to_c_struct

    name = "mmc_type"
    dtype, c_decl = match_dtype_to_c_struct(queue.device, name, dtype)
    dtype = get_or_register_dtype(name, dtype)

    n = 1000
    z_dev = cl.array.zeros(queue, n, dtype=dtype)

    z = z_dev.get()

    assert np.array_equal(np.zeros(n, dtype), z)


def test_custom_type_fill(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.characterize import has_struct_arg_count_bug
    if has_struct_arg_count_bug(queue.device):
        pytest.skip("device has LLVM arg counting bug")

    dtype = np.dtype([
        ("cur_min", np.int32),
        ("cur_max", np.int32),
        ("pad", np.int32),
        ])

    from pyopencl.tools import get_or_register_dtype, match_dtype_to_c_struct

    name = "mmc_type"
    dtype, c_decl = match_dtype_to_c_struct(queue.device, name, dtype)
    dtype = get_or_register_dtype(name, dtype)

    n = 1000
    z_dev = cl.array.empty(queue, n, dtype=dtype)
    z_dev.fill(np.zeros((), dtype))

    z = z_dev.get()

    assert np.array_equal(np.zeros(n, dtype), z)


def test_custom_type_take_put(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    dtype = np.dtype([
        ("cur_min", np.int32),
        ("cur_max", np.int32),
        ])

    from pyopencl.tools import get_or_register_dtype, match_dtype_to_c_struct

    name = "tp_type"
    dtype, c_decl = match_dtype_to_c_struct(queue.device, name, dtype)
    dtype = get_or_register_dtype(name, dtype)

    n = 100
    z = np.empty(100, dtype)
    z["cur_min"] = np.arange(n)
    z["cur_max"] = np.arange(n)**2

    z_dev = cl.array.to_device(queue, z)
    ind = cl.array.arange(queue, n, step=3, dtype=np.int32)

    z_ind_ref = z[ind.get()]
    z_ind = z_dev[ind]

    assert np.array_equal(z_ind.get(), z_ind_ref)

# }}}


# {{{ operators

def test_rmul_yields_right_type(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    a_gpu = cl_array.to_device(queue, a)

    two_a = 2*a_gpu
    assert isinstance(two_a, cl_array.Array)

    two_a = np.float32(2)*a_gpu
    assert isinstance(two_a, cl_array.Array)


def test_pow_array(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    a_gpu = cl_array.to_device(queue, a)

    result = pow(a_gpu, a_gpu).get()
    assert (np.abs(a ** a - result) < 3e-3).all()

    result = (a_gpu ** a_gpu).get()
    assert (np.abs(pow(a, a) - result) < 3e-3).all()


def test_pow_number(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    a_gpu = cl_array.to_device(queue, a)

    result = pow(a_gpu, 2).get()
    assert (np.abs(a ** 2 - result) < 1e-3).all()


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


def test_multiply_array(ctx_factory):
    """Test the multiplication of two arrays."""

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)

    a_gpu = cl_array.to_device(queue, a)
    b_gpu = cl_array.to_device(queue, a)

    a_squared = (b_gpu * a_gpu).get()

    assert (a * a == a_squared).all()


def test_addition_array(ctx_factory):
    """Test the addition of two arrays."""

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    a_gpu = cl_array.to_device(queue, a)
    a_added = (a_gpu + a_gpu).get()

    assert (a + a == a_added).all()


def test_addition_scalar(ctx_factory):
    """Test the addition of an array and a scalar."""

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    a_gpu = cl_array.to_device(queue, a)
    a_added = (7 + a_gpu).get()

    assert (7 + a == a_added).all()


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


def test_bitwise(ctx_factory):
    if _PYPY:
        pytest.xfail("numpypy: missing bitwise ops")

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from itertools import product

    dtypes = [np.dtype(t) for t in (np.int64, np.int32, np.int16, np.int8)]

    from pyopencl.clrandom import rand as clrand

    for a_dtype, b_dtype in product(dtypes, dtypes):
        ary_len = 16

        np.random.seed(10)

        int32_min = np.iinfo(np.int32).min
        int32_max = np.iinfo(np.int32).max

        a_dev = clrand(
            queue, (ary_len,), a=int32_min, b=1+int32_max, dtype=np.int64
            ).astype(a_dtype)
        b_dev = clrand(
            queue, (ary_len,), a=int32_min, b=1+int32_max, dtype=np.int64
            ).astype(b_dtype)

        a = a_dev.get()
        b = b_dev.get()
        s = int((clrand(queue, (), a=int32_min, b=1+int32_max, dtype=np.int64)
                 .astype(b_dtype).get()))

        import operator as o

        for op in [o.and_, o.or_, o.xor]:
            res_dev = op(a_dev, b_dev)
            res = op(a, b)

            assert (res_dev.get() == res).all()

            res_dev = op(a_dev, s)
            res = op(a, s)

            assert (res_dev.get() == res).all()

            res_dev = op(s, b_dev)
            res = op(s, b)

            assert (res_dev.get() == res).all()

        for op in [o.iand, o.ior, o.ixor]:
            res_dev = a_dev.copy()
            op_res = op(res_dev, b_dev)
            assert op_res is res_dev

            res = a.copy()
            op(res, b)

            assert (res_dev.get() == res).all()

            res_dev = a_dev.copy()
            op_res = op(res_dev, s)
            assert op_res is res_dev
            res = a.copy()
            op(res, s)

            assert (res_dev.get() == res).all()

        # Test unary ~
        res_dev = ~a_dev
        res = ~a  # pylint:disable=invalid-unary-operand-type
        assert (res_dev.get() == res).all()

# }}}


# {{{ RNG

@pytest.mark.parametrize("rng_class",
        [RanluxGenerator, PhiloxGenerator, ThreefryGenerator])
@pytest.mark.parametrize("ary_size", [300, 301, 302, 303, 10007, 1000000])
def test_random_float_in_range(ctx_factory, rng_class, ary_size, plot_hist=False):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    if has_double_support(context.devices[0]):
        dtypes = [np.float32, np.float64]
    else:
        dtypes = [np.float32]

    if rng_class is RanluxGenerator:
        gen = rng_class(queue, 5120)
    else:
        gen = rng_class(context)

    for dtype in dtypes:
        print(dtype)
        ran = cl_array.zeros(queue, ary_size, dtype)
        gen.fill_uniform(ran)

        if plot_hist:
            import matplotlib.pyplot as pt
            pt.hist(ran.get(), 30)
            pt.show()

        assert (0 <= ran.get()).all()
        assert (ran.get() <= 1).all()

        if rng_class is RanluxGenerator:
            gen.synchronize(queue)

        ran = cl_array.zeros(queue, ary_size, dtype)
        gen.fill_uniform(ran, a=4, b=7)
        ran_host = ran.get()

        for cond in [4 <= ran_host,  ran_host <= 7]:
            good = cond.all()
            if not good:
                print(np.where(~cond))
                print(ran_host[~cond])
            assert good

        ran = gen.normal(queue, ary_size, dtype, mu=10, sigma=3)

        if plot_hist:
            import matplotlib.pyplot as pt
            pt.hist(ran.get(), 30)
            pt.show()


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("rng_class",
        [RanluxGenerator, PhiloxGenerator, ThreefryGenerator])
def test_random_int_in_range(ctx_factory, rng_class, dtype, plot_hist=False):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    if rng_class is RanluxGenerator:
        gen = rng_class(queue, 5120)
    else:
        gen = rng_class(context)

    # if (dtype == np.int64
    #         and context.devices[0].platform.vendor.startswith("Advanced Micro")):
    #     pytest.xfail("AMD miscompiles 64-bit RNG math")

    ran = gen.uniform(queue, (10000007,), dtype, a=200, b=300).get()
    assert (200 <= ran).all()
    assert (ran < 300).all()

    print(np.min(ran), np.max(ran))
    assert np.max(ran) > 295

    if plot_hist:
        from matplotlib import pyplot as pt
        pt.hist(ran)
        pt.show()

# }}}


# {{{ misc

def test_numpy_integer_shape(ctx_factory):
    try:
        list(np.int32(17))
    except Exception:
        pass
    else:
        from pytest import skip
        skip("numpy implementation does not handle scalar correctly.")
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    cl_array.empty(queue, np.int32(17), np.float32)
    cl_array.empty(queue, (np.int32(17), np.int32(17)), np.float32)


def test_len(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    a_cpu = cl_array.to_device(queue, a)
    assert len(a_cpu) == 10


def test_stride_preservation(ctx_factory):
    if _PYPY:
        pytest.xfail("numpypy: no array creation from __array_interface__")

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.random.rand(3, 3)
    at = a.T
    print(at.flags.f_contiguous, at.flags.c_contiguous)
    at_gpu = cl_array.to_device(queue, at)
    print(at_gpu.flags.f_contiguous, at_gpu.flags.c_contiguous)
    assert np.allclose(at_gpu.get(), at)


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


def test_mem_pool_with_arrays(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)
    mem_pool = cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue))

    a_dev = cl_array.arange(queue, 2000, dtype=np.float32, allocator=mem_pool)
    b_dev = cl_array.to_device(queue, np.arange(2000), allocator=mem_pool) + 4000

    assert a_dev.allocator is mem_pool
    assert b_dev.allocator is mem_pool


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


def test_diff(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand

    ary_len = 20000
    a_dev = clrand(queue, (ary_len,), dtype=np.float32)
    a = a_dev.get()

    err = la.norm(
            (cl.array.diff(a_dev).get() - np.diff(a)))
    assert err < 1e-4


def test_copy(ctx_factory):
    context = ctx_factory()
    queue1 = cl.CommandQueue(context)
    queue2 = cl.CommandQueue(context)

    # Test copy

    arr = cl.array.zeros(queue1, 100, np.int32)
    arr_copy = arr.copy()

    assert (arr == arr_copy).all().get()
    assert arr.data != arr_copy.data
    assert arr_copy.queue is queue1

    # Test queue association

    arr_copy = arr.copy(queue=queue2)
    assert arr_copy.queue is queue2

    arr_copy = arr.copy(queue=None)
    assert arr_copy.queue is None

    arr_copy = arr.with_queue(None).copy(queue=queue1)
    assert arr_copy.queue is queue1

# }}}


# {{{ slices, concatenation

def test_slice(ctx_factory):
    if _PYPY:
        pytest.xfail("numpypy: spurious as_strided failure")

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand

    tp = np.float32

    ary_len = 20000
    a_gpu = clrand(queue, (ary_len,), dtype=tp)
    b_gpu = clrand(queue, (ary_len,), dtype=tp)
    a = a_gpu.get()
    b = b_gpu.get()

    from random import randrange
    for i in range(20):
        start = randrange(ary_len)
        end = randrange(start, ary_len)

        a_gpu_slice = tp(2)*a_gpu[start:end]
        a_slice = tp(2)*a[start:end]

        assert la.norm(a_gpu_slice.get() - a_slice) == 0

    for i in range(20):
        start = randrange(ary_len)
        end = randrange(start, ary_len)

        a_gpu[start:end] = tp(2)*b[start:end]
        a[start:end] = tp(2)*b[start:end]

        assert la.norm(a_gpu.get() - a) == 0

    for i in range(20):
        start = randrange(ary_len)
        end = randrange(start, ary_len)

        a_gpu[start:end] = tp(2)*b_gpu[start:end]
        a[start:end] = tp(2)*b[start:end]

        assert la.norm(a_gpu.get() - a) == 0


def test_concatenate(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand

    a_dev = clrand(queue, (5, 15, 20), dtype=np.float32)
    b_dev = clrand(queue, (4, 15, 20), dtype=np.float32)
    c_dev = clrand(queue, (3, 15, 20), dtype=np.float32)
    a = a_dev.get()
    b = b_dev.get()
    c = c_dev.get()

    cat_dev = cl.array.concatenate((a_dev, b_dev, c_dev))
    cat = np.concatenate((a, b, c))

    assert la.norm(cat - cat_dev.get()) == 0

# }}}


# {{{ conditionals, any, all

def test_comparisons(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand

    ary_len = 20000
    a_dev = clrand(queue, (ary_len,), dtype=np.float32)
    b_dev = clrand(queue, (ary_len,), dtype=np.float32)

    a = a_dev.get()
    b = b_dev.get()

    import operator as o
    for op in [o.eq, o.ne, o.le, o.lt, o.ge, o.gt]:
        res_dev = op(a_dev, b_dev)
        res = op(a, b)

        assert (res_dev.get() == res).all()

        res_dev = op(a_dev, 0)
        res = op(a, 0)

        assert (res_dev.get() == res).all()

        res_dev = op(0, b_dev)
        res = op(0, b)

        assert (res_dev.get() == res).all()


def test_any_all(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    ary_len = 20000
    a_dev = cl_array.zeros(queue, (ary_len,), dtype=np.int8)

    assert not a_dev.all().get()
    assert not a_dev.any().get()

    a_dev[15213] = 1

    assert not a_dev.all().get()
    assert a_dev.any().get()

    a_dev.fill(1)

    assert a_dev.all().get()
    assert a_dev.any().get()

# }}}


def test_map_to_host(ctx_factory):
    if _PYPY:
        pytest.skip("numpypy: no array creation from __array_interface__")

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    if context.devices[0].type & cl.device_type.GPU:
        mf = cl.mem_flags
        allocator = cl_tools.DeferredAllocator(
                context, mf.READ_WRITE | mf.ALLOC_HOST_PTR)
    else:
        allocator = None

    a_dev = cl_array.zeros(queue, (5, 6, 7,), dtype=np.float32, allocator=allocator)
    a_dev[3, 2, 1] = 10
    a_host = a_dev.map_to_host()
    a_host[1, 2, 3] = 10

    a_host_saved = a_host.copy()
    a_host.base.release(queue)

    a_dev.finish()

    print("DEV[HOST_WRITE]", a_dev.get()[1, 2, 3])
    print("HOST[DEV_WRITE]", a_host_saved[3, 2, 1])

    assert (a_host_saved == a_dev.get()).all()


def test_view_and_strides(ctx_factory):
    if _PYPY:
        pytest.xfail("numpypy: no array creation from __array_interface__")
    return

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand

    x = clrand(queue, (5, 10), dtype=np.float32)
    y = x[:3, :5]
    yv = y.view()

    assert yv.shape == y.shape
    assert yv.strides == y.strides

    with pytest.raises(AssertionError):
        assert (yv.get() == x.get()[:3, :5]).all()


def test_meshmode_view(ctx_factory):
    if _PYPY:
        # https://bitbucket.org/pypy/numpy/issue/28/indexerror-on-ellipsis-slice
        pytest.xfail("numpypy bug #28")

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    n = 2
    result = cl.array.empty(queue, (2, n*6), np.float32)

    def view(z):
        return z[..., n*3:n*6].reshape(z.shape[:-1] + (n, 3))

    result = result.with_queue(queue)
    result.fill(0)
    view(result)[0].fill(1)
    view(result)[1].fill(1)
    x = result.get()
    assert (view(x) == 1).all()


def test_event_management(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand

    x = clrand(queue, (5, 10), dtype=np.float32)
    assert len(x.events) == 1, len(x.events)

    x.finish()

    assert len(x.events) == 0

    y = x+x
    assert len(y.events) == 1
    y = x*x
    assert len(y.events) == 1
    y = 2*x
    assert len(y.events) == 1
    y = 2/x
    assert len(y.events) == 1
    y = x/2
    assert len(y.events) == 1
    y = x**2
    assert len(y.events) == 1
    y = 2**x
    assert len(y.events) == 1

    for i in range(10):
        x.fill(0)

    assert len(x.events) == 10

    for i in range(1000):
        x.fill(0)

    assert len(x.events) < 100


def test_reshape(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.arange(128).reshape(8, 16).astype(np.float32)
    a_dev = cl_array.to_device(queue, a)

    # different ways to specify the shape
    a_dev.reshape(4, 32)
    a_dev.reshape((4, 32))
    a_dev.reshape([4, 32])

    # using -1 as unknown dimension
    assert a_dev.reshape(-1, 32).shape == (4, 32)
    assert a_dev.reshape((32, -1)).shape == (32, 4)
    assert a_dev.reshape(((8, -1, 4))).shape == (8, 4, 4)

    import pytest
    with pytest.raises(ValueError):
        a_dev.reshape(-1, -1, 4)


def test_skip_slicing(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a_host = np.arange(16).reshape((4, 4))
    b_host = a_host[::3]

    a = cl_array.to_device(queue, a_host)
    b = a[::3]
    assert b.shape == b_host.shape
    assert np.array_equal(b[1].get(), b_host[1])  # noqa pylint:disable=unsubscriptable-object


def test_transpose(ctx_factory):
    if _PYPY:
        pytest.xfail("numpypy: no array creation from __array_interface__")

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand

    a_gpu = clrand(queue, (10, 20, 30), dtype=np.float32)
    a = a_gpu.get()

    # FIXME: not contiguous
    #assert np.allclose(a_gpu.transpose((1,2,0)).get(), a.transpose((1,2,0)))
    assert np.array_equal(a_gpu.T.get(), a.T)


def test_newaxis(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand

    a_gpu = clrand(queue, (10, 20, 30), dtype=np.float32)
    a = a_gpu.get()

    b_gpu = a_gpu[:, np.newaxis]
    b = a[:, np.newaxis]

    assert b_gpu.shape == b.shape
    for i in range(b.ndim):
        if b.shape[i] > 1:
            assert b_gpu.strides[i] == b.strides[i]


def test_squeeze(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    shape = (40, 2, 5, 100)
    a_cpu = np.random.random(size=shape)
    a_gpu = cl_array.to_device(queue, a_cpu)

    # Slice with length 1 on dimensions 0 and 1
    a_gpu_slice = a_gpu[0:1, 1:2, :, :]
    assert a_gpu_slice.shape == (1, 1, shape[2], shape[3])
    assert a_gpu_slice.flags.c_contiguous

    # Squeeze it and obtain contiguity
    a_gpu_squeezed_slice = a_gpu[0:1, 1:2, :, :].squeeze()
    assert a_gpu_squeezed_slice.shape == (shape[2], shape[3])
    assert a_gpu_squeezed_slice.flags.c_contiguous

    # Check that we get the original values out
    #assert np.all(a_gpu_slice.get().ravel() == a_gpu_squeezed_slice.get().ravel())

    # Slice with length 1 on dimensions 2
    a_gpu_slice = a_gpu[:, :, 2:3, :]
    assert a_gpu_slice.shape == (shape[0], shape[1], 1, shape[3])
    assert not a_gpu_slice.flags.c_contiguous

    # Squeeze it, but no contiguity here
    a_gpu_squeezed_slice = a_gpu[:, :, 2:3, :].squeeze()
    assert a_gpu_squeezed_slice.shape == (shape[0], shape[1], shape[3])
    assert not a_gpu_squeezed_slice.flags.c_contiguous

    # Check that we get the original values out
    #assert np.all(a_gpu_slice.get().ravel() == a_gpu_squeezed_slice.get().ravel())


def test_fancy_fill(ctx_factory):
    if _PYPY:
        pytest.xfail("numpypy: multi value setting is not supported")
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    numpy_dest = np.zeros((4,), np.int32)
    numpy_idx = np.arange(3, dtype=np.int32)
    numpy_src = np.arange(8, 9, dtype=np.int32)
    numpy_dest[numpy_idx] = numpy_src

    cl_dest = cl_array.zeros(queue, (4,), np.int32)
    cl_idx = cl_array.arange(queue, 3, dtype=np.int32)
    cl_src = cl_array.arange(queue, 8, 9, dtype=np.int32)
    cl_dest[cl_idx] = cl_src

    assert np.all(numpy_dest == cl_dest.get())


def test_fancy_indexing(ctx_factory):
    if _PYPY:
        pytest.xfail("numpypy: multi value setting is not supported")
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    n = 2 ** 20 + 2**18 + 22
    numpy_dest = np.zeros(n, dtype=np.int32)
    numpy_idx = np.arange(n, dtype=np.int32)
    np.random.shuffle(numpy_idx)
    numpy_src = 20000+np.arange(n, dtype=np.int32)

    cl_dest = cl_array.to_device(queue, numpy_dest)
    cl_idx = cl_array.to_device(queue, numpy_idx)
    cl_src = cl_array.to_device(queue, numpy_src)

    numpy_dest[numpy_idx] = numpy_src
    cl_dest[cl_idx] = cl_src

    assert np.array_equal(numpy_dest, cl_dest.get())

    numpy_dest = numpy_src[numpy_idx]
    cl_dest = cl_src[cl_idx]

    assert np.array_equal(numpy_dest, cl_dest.get())


def test_multi_put(ctx_factory):
    if _PYPY:
        pytest.xfail("numpypy: multi value setting is not supported")

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    cl_arrays = [
        cl_array.arange(queue, 0, 3, dtype=np.float32)
        for i in range(1, 10)
    ]
    idx = cl_array.arange(queue, 0, 6, dtype=np.int32)
    out_arrays = [
        cl_array.zeros(queue, (10,), np.float32)
        for i in range(9)
    ]

    out_compare = [np.zeros((10,), np.float32) for i in range(9)]
    for i, ary in enumerate(out_compare):
        ary[idx.get()] = np.arange(0, 6, dtype=np.float32)

    cl_array.multi_put(cl_arrays, idx, out=out_arrays)

    assert np.all(np.all(out_compare[i] == out_arrays[i].get()) for i in range(9))


def test_get_async(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.random.rand(10**6).astype(np.dtype('float32'))
    a_gpu = cl_array.to_device(queue, a)
    b = a + a**5 + 1
    b_gpu = a_gpu + a_gpu**5 + 1

    # deprecated, but still test
    b1 = b_gpu.get(async_=True)  # testing that this waits for events
    b_gpu.finish()
    assert np.abs(b1 - b).mean() < 1e-5

    b1 = b_gpu.get_async()  # testing that this waits for events
    b_gpu.finish()
    assert np.abs(b1 - b).mean() < 1e-5

    wait_event = cl.UserEvent(context)
    b_gpu.add_event(wait_event)
    b = b_gpu.get_async()  # testing that this doesn't hang
    wait_event.set_status(cl.command_execution_status.COMPLETE)
    b_gpu.finish()
    assert np.abs(b1 - b).mean() < 1e-5


def test_outoforderqueue_get(ctx_factory):
    context = ctx_factory()
    try:
        queue = cl.CommandQueue(context,
               properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
    except Exception:
        pytest.skip("out-of-order queue not available")
    a = np.random.rand(10**6).astype(np.dtype('float32'))
    a_gpu = cl_array.to_device(queue, a)
    b_gpu = a_gpu + a_gpu**5 + 1
    b1 = b_gpu.get()  # testing that this waits for events
    b = a + a**5 + 1
    assert np.abs(b1 - b).mean() < 1e-5


def test_outoforderqueue_copy(ctx_factory):
    context = ctx_factory()
    try:
        queue = cl.CommandQueue(context,
               properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
    except Exception:
        pytest.skip("out-of-order queue not available")
    a = np.random.rand(10**6).astype(np.dtype('float32'))
    a_gpu = cl_array.to_device(queue, a)
    c_gpu = a_gpu**2 - 7
    b_gpu = c_gpu.copy()  # testing that this waits for and creates events
    b_gpu *= 10
    queue.finish()
    b1 = b_gpu.get()
    b = 10 * (a**2 - 7)
    assert np.abs(b1 - b).mean() < 1e-5


def test_outoforderqueue_indexing(ctx_factory):
    context = ctx_factory()
    try:
        queue = cl.CommandQueue(context,
               properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
    except Exception:
        pytest.skip("out-of-order queue not available")
    a = np.random.rand(10**6).astype(np.dtype('float32'))
    i = (8e5 + 1e5 * np.random.rand(10**5)).astype(np.dtype('int32'))
    a_gpu = cl_array.to_device(queue, a)
    i_gpu = cl_array.to_device(queue, i)
    c_gpu = (a_gpu**2)[i_gpu - 10000]
    b_gpu = 10 - a_gpu
    b_gpu[:] = 8 * a_gpu
    b_gpu[i_gpu + 10000] = c_gpu - 10
    queue.finish()
    b1 = b_gpu.get()
    c = (a**2)[i - 10000]
    b = 8 * a
    b[i + 10000] = c - 10
    assert np.abs(b1 - b).mean() < 1e-5


def test_outoforderqueue_reductions(ctx_factory):
    context = ctx_factory()
    try:
        queue = cl.CommandQueue(context,
               properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
    except Exception:
        pytest.skip("out-of-order queue not available")
    # 0/1 values to avoid accumulated rounding error
    a = (np.random.rand(10**6) > 0.5).astype(np.dtype('float32'))
    a[800000] = 10  # all<5 looks true until near the end
    a_gpu = cl_array.to_device(queue, a)
    b1 = cl_array.sum(a_gpu).get()
    b2 = cl_array.dot(a_gpu, 3 - a_gpu).get()
    b3 = (a_gpu < 5).all().get()
    assert b1 == a.sum() and b2 == a.dot(3 - a) and b3 == 0


if __name__ == "__main__":
    # make sure that import failures get reported, instead of skipping the
    # tests.
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: filetype=pyopencl:fdm=marker
