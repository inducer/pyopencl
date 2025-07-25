#! /usr/bin/env python

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

import operator
import platform
import sys
from itertools import product

import numpy as np
import numpy.linalg as la
import pytest

import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.characterize as cl_characterize
import pyopencl.cltypes as cltypes
import pyopencl.tools as cl_tools
from pyopencl.characterize import has_double_support, has_struct_arg_count_bug
from pyopencl.clrandom import PhiloxGenerator, ThreefryGenerator
from pyopencl.tools import (
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,  # noqa: F401
)


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

# {{{ test_basic_complex

def test_basic_complex(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand

    size = 500

    ary = (rand(queue, shape=(size,), dtype=np.float32).astype(np.complex64)
            + rand(queue, shape=(size,), dtype=np.float32).astype(np.complex64) * 1j)
    assert ary.dtype != np.dtype(np.complex128)
    c = np.complex64(5+7j)

    host_ary = ary.get()
    assert la.norm((ary*c).get() - c*host_ary) < 1e-5 * la.norm(host_ary)

# }}}


# {{{ test_mix_complex

def test_mix_complex(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    size = 10

    dtypes = [
            (np.float32, np.complex64),
            # (np.int32, np.complex64),
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

                        print("HOST_DTYPE: {} DEV_DTYPE: {}".format(
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

# }}}


# {{{ test_pow_neg1_vs_inv

def test_pow_neg1_vs_inv(ctx_factory: cl.CtxFactory):
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

# }}}


# {{{ test_vector_fill

def test_vector_fill(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a_gpu = cl_array.Array(queue, 100, dtype=cltypes.float4)
    a_gpu.fill(cltypes.make_float4(0.0, 0.0, 1.0, 0.0))
    a = a_gpu.get()
    assert a.dtype == cltypes.float4

    a_gpu = cl_array.zeros(queue, 100, dtype=cltypes.float4)

# }}}


# {{{ test_zeros_large_array

def test_zeros_large_array(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)
    dev = queue.device

    if dev.platform.vendor == "Intel(R) Corporation" \
            and platform.system() == "Windows":
        pytest.xfail("large array fail with out-of-host memory with"
                "Intel CPU runtime as of 2022-10-05")
    if (dev.platform.name == "Portable Computing Language"
            and cl.get_cl_header_version() < (1, 2)):
        pytest.xfail("does not work on pocl with CL pre 1.2")

    size = 2**28 + 1
    if dev.address_bits == 64 and dev.max_mem_alloc_size >= 8 * size:
        # this shouldn't hang/cause errors
        # see https://github.com/inducer/pyopencl/issues/395
        a_gpu = cl_array.zeros(queue, (size,), dtype="float64")
        # run a couple kernels to ensure no propagated runtime errors
        a_gpu[...] = 1.
        a_gpu = 2 * a_gpu - 3

# }}}


# {{{ test_absrealimag

def test_absrealimag(ctx_factory: cl.CtxFactory):
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

# }}}


# {{{ test_custom_type_zeros

def test_custom_type_zeros(ctx_factory: cl.CtxFactory):
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
    dtype, _c_decl = match_dtype_to_c_struct(queue.device, name, dtype)
    dtype = get_or_register_dtype(name, dtype)

    n = 1000
    z_dev = cl_array.zeros(queue, n, dtype=dtype)

    z = z_dev.get()

    assert np.array_equal(np.zeros(n, dtype), z)

# }}}


# {{{ test_custom_type_fill

def test_custom_type_fill(ctx_factory: cl.CtxFactory):
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
    dtype, _c_decl = match_dtype_to_c_struct(queue.device, name, dtype)
    dtype = get_or_register_dtype(name, dtype)

    n = 1000
    z_dev = cl_array.empty(queue, n, dtype=dtype)
    z_dev.fill(np.zeros((), dtype))

    z = z_dev.get()

    assert np.array_equal(np.zeros(n, dtype), z)

# }}}


# {{{ test_custom_type_take_put

def test_custom_type_take_put(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    dtype = np.dtype([
        ("cur_min", np.int32),
        ("cur_max", np.int32),
        ])

    from pyopencl.tools import get_or_register_dtype, match_dtype_to_c_struct

    name = "tp_type"
    dtype, _c_decl = match_dtype_to_c_struct(queue.device, name, dtype)
    dtype = get_or_register_dtype(name, dtype)

    n = 100
    z = np.empty(100, dtype)
    z["cur_min"] = np.arange(n)
    z["cur_max"] = np.arange(n)**2

    z_dev = cl_array.to_device(queue, z)
    ind = cl_array.arange(queue, n, step=3, dtype=np.int32)

    z_ind_ref = z[ind.get()]
    z_ind = z_dev[ind]

    assert np.array_equal(z_ind.get(), z_ind_ref)

# }}}

# }}}


# {{{ operators

# {{{ test_div_type_matches_numpy

@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64, np.float32])
# FIXME Implement floordiv
# @pytest.mark.parametrize("op", [operator.truediv, operator.floordiv])
@pytest.mark.parametrize("op", [operator.truediv])
def test_div_type_matches_numpy(ctx_factory: cl.CtxFactory, dtype, op):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = cl_array.arange(queue, 10, dtype=dtype) + 1
    res = op(4*a, 3*a)
    a_np = a.get()
    res_np = op(4*a_np, 3*a_np)
    assert res_np.dtype == res.dtype
    assert np.allclose(res_np, res.get())

# }}}


# {{{ test_rmul_yields_right_type

def test_rmul_yields_right_type(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    a_gpu = cl_array.to_device(queue, a)

    two_a = 2*a_gpu
    assert isinstance(two_a, cl_array.Array)

    two_a = np.float32(2)*a_gpu
    assert isinstance(two_a, cl_array.Array)

# }}}


# {{{ test_pow_array

def test_pow_array(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    a_gpu = cl_array.to_device(queue, a)

    result = pow(a_gpu, a_gpu).get()
    assert (np.abs(a ** a - result) < 3e-3).all()

    result = (a_gpu ** a_gpu).get()
    assert (np.abs(pow(a, a) - result) < 3e-3).all()

# }}}


# {{{ test_pow_number

def test_pow_number(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    a_gpu = cl_array.to_device(queue, a)

    result = pow(a_gpu, 2).get()
    assert (np.abs(a ** 2 - result) < 1e-3).all()

# }}}


# {{{ test_multiply

def test_multiply(ctx_factory: cl.CtxFactory):
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

# }}}


# {{{ test_multiply_array

def test_multiply_array(ctx_factory: cl.CtxFactory):
    """Test the multiplication of two arrays."""

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)

    a_gpu = cl_array.to_device(queue, a)
    b_gpu = cl_array.to_device(queue, a)

    a_squared = (b_gpu * a_gpu).get()

    assert (a * a == a_squared).all()


# }}}


# {{{ test_addition_array

def test_addition_array(ctx_factory: cl.CtxFactory):
    """Test the addition of two arrays."""

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    a_gpu = cl_array.to_device(queue, a)
    a_added = (a_gpu + a_gpu).get()

    assert (a + a == a_added).all()

# }}}


# {{{ test_addition_scalar

def test_addition_scalar(ctx_factory: cl.CtxFactory):
    """Test the addition of an array and a scalar."""

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    a_gpu = cl_array.to_device(queue, a)
    a_added = (7 + a_gpu).get()

    assert (7 + a == a_added).all()

# }}}


# {{{ test_subtract_array

@pytest.mark.parametrize(("dtype_a", "dtype_b"),
        [
            (np.float32, np.float32),
            (np.float32, np.int32),
            (np.int32, np.int32),
            (np.int64, np.int32),
            (np.int64, np.uint32),
            ])
def test_subtract_array(ctx_factory: cl.CtxFactory, dtype_a, dtype_b):
    """Test the subtraction of two arrays."""
    # test data
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(dtype_a)
    b = np.array([10, 20, 30, 40, 50,
                  60, 70, 80, 90, 100]).astype(dtype_b)

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a_gpu = cl_array.to_device(queue, a)
    b_gpu = cl_array.to_device(queue, b)

    result = (a_gpu - b_gpu).get()
    assert (a - b == result).all()

    result = (b_gpu - a_gpu).get()
    assert (b - a == result).all()

# }}}


# {{{ test_subtract_scalar

def test_subtract_scalar(ctx_factory: cl.CtxFactory):
    """Test the subtraction of an array and a scalar."""

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    # test data
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)

    # convert a to a gpu object
    a_gpu = cl_array.to_device(queue, a)

    result = (a_gpu - 7).get()
    assert (a - 7 == result).all()

    result = (7 - a_gpu).get()
    assert (7 - a == result).all()

# }}}


# {{{ test_divide_scalar

def test_divide_scalar(ctx_factory: cl.CtxFactory):
    """Test the division of an array and a scalar."""

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    if queue.device.platform.name == "Apple":
        pytest.xfail("Apple CL compiler crashes on this.")

    dtypes = (np.uint8, np.uint16, np.uint32,
                  np.int8, np.int16, np.int32,
                  np.float32, np.complex64)
    from pyopencl.characterize import has_double_support
    if has_double_support(queue.device):
        dtypes = (*dtypes, np.float64, np.complex128)

    from itertools import product

    for dtype_a, dtype_s in product(dtypes, repeat=2):
        a = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).astype(dtype_a)
        s = dtype_s(40)
        a_gpu = cl_array.to_device(queue, a)

        b = a / s
        b_gpu = a_gpu / s
        assert (np.abs(b_gpu.get() - b) < 1e-3).all()
        assert b_gpu.dtype is b.dtype

        c = s / a
        c_gpu = s / a_gpu
        assert (np.abs(c_gpu.get() - c) < 1e-3).all()
        assert c_gpu.dtype is c.dtype

# }}}


# {{{ test_divide_array

def test_divide_array(ctx_factory: cl.CtxFactory):
    """Test the division of an array and a scalar. """

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    dtypes = (np.float32, np.complex64)
    from pyopencl.characterize import has_double_support
    if has_double_support(queue.device):
        dtypes = (*dtypes, np.float64, np.complex128)

    from itertools import product

    for dtype_a, dtype_b in product(dtypes, repeat=2):

        a = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).astype(dtype_a)
        b = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]).astype(dtype_b)

        a_gpu = cl_array.to_device(queue, a)
        b_gpu = cl_array.to_device(queue, b)
        c = a / b
        c_gpu = (a_gpu / b_gpu)
        assert (np.abs(c_gpu.get() - c) < 1e-3).all()
        assert c_gpu.dtype is c.dtype

        d = b / a
        d_gpu = (b_gpu / a_gpu)
        assert (np.abs(d_gpu.get() - d) < 1e-3).all()
        assert d_gpu.dtype is d.dtype

# }}}


# {{{ test_divide_inplace_scalar

def test_divide_inplace_scalar(ctx_factory: cl.CtxFactory):
    """Test inplace division of arrays and a scalar."""

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    if queue.device.platform.name == "Apple":
        pytest.xfail("Apple CL compiler crashes on this.")

    dtypes = (np.uint8, np.uint16, np.uint32,
                  np.int8, np.int16, np.int32,
                  np.float32, np.complex64)
    from pyopencl.characterize import has_double_support
    if has_double_support(queue.device):
        dtypes = (*dtypes, np.float64, np.complex128)

    from itertools import product

    for dtype_a, dtype_s in product(dtypes, repeat=2):
        a = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).astype(dtype_a)
        s = dtype_s(40)
        a_gpu = cl_array.to_device(queue, a)

        # ensure the same behavior as inplace numpy.ndarray division
        try:
            a /= s
        except TypeError:
            with np.testing.assert_raises(TypeError):
                a_gpu /= s
        else:
            a_gpu /= s
            assert (np.abs(a_gpu.get() - a) < 1e-3).all()
            assert a_gpu.dtype is a.dtype

# }}}


# {{{ test_divide_inplace_array

def test_divide_inplace_array(ctx_factory: cl.CtxFactory):
    """Test inplace division of arrays."""

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    dtypes = (np.uint8, np.uint16, np.uint32,
                  np.int8, np.int16, np.int32,
                  np.float32, np.complex64)
    from pyopencl.characterize import has_double_support
    if has_double_support(queue.device):
        dtypes = (*dtypes, np.float64, np.complex128)

    from itertools import product

    for dtype_a, dtype_b in product(dtypes, repeat=2):
        print(dtype_a, dtype_b)
        a = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).astype(dtype_a)
        b = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]).astype(dtype_b)

        a_gpu = cl_array.to_device(queue, a)
        b_gpu = cl_array.to_device(queue, b)

        # ensure the same behavior as inplace numpy.ndarray division
        try:
            a_gpu /= b_gpu
        except TypeError:
            # pass for now, as numpy casts differently for in-place and out-place
            # true_divide
            pass
            # with np.testing.assert_raises(TypeError):
            #     a /= b
        else:
            a /= b
            assert (np.abs(a_gpu.get() - a) < 1e-3).all()
            assert a_gpu.dtype is a.dtype

# }}}


# {{{ test_bitwise

def test_bitwise(ctx_factory: cl.CtxFactory):
    if _PYPY:
        pytest.xfail("numpypy: missing bitwise ops")

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from itertools import product

    dtypes = [np.dtype(t) for t in (np.int64, np.int32, np.int16, np.int8)]

    from pyopencl.clrandom import rand as clrand

    for a_dtype, b_dtype in product(dtypes, dtypes):
        ary_len = 16

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
        s = int(clrand(queue, (), a=int32_min, b=1+int32_max, dtype=np.int64)
                 .astype(b_dtype).get())

        import operator as o

        for op in [o.and_, o.or_, o.xor]:
            res_dev = op(a_dev, b_dev)
            res = op(a, b)

            assert (res_dev.get() == res).all()

            try:
                res = op(a, s)
            except OverflowError:
                pass
            else:
                res_dev = op(a_dev, s)

                assert (res_dev.get() == res).all()

            try:
                res = op(s, b)
            except OverflowError:
                pass
            else:
                res_dev = op(s, b_dev)

                assert (res_dev.get() == res).all()

        for op in [o.iand, o.ior, o.ixor]:
            res_dev = a_dev.copy()
            op_res = op(res_dev, b_dev)
            assert op_res is res_dev

            res = a.copy()
            try:
                op(res, b)
            except OverflowError:
                pass
            else:

                assert (res_dev.get() == res).all()

            res = a.copy()
            try:
                op(res, s)
            except OverflowError:
                pass
            else:
                res_dev = a_dev.copy()
                op_res = op(res_dev, s)
                assert op_res is res_dev

                assert (res_dev.get() == res).all()

        # Test unary ~
        res_dev = ~a_dev
        res = ~a
        assert (res_dev.get() == res).all()

# }}}

# }}}


# {{{ RNG

# {{{ test_random_float_in_range

@pytest.mark.parametrize("rng_class",
        [PhiloxGenerator, ThreefryGenerator])
@pytest.mark.parametrize("ary_size", [300, 301, 302, 303, 10007, 1000000])
def test_random_float_in_range(ctx_factory: cl.CtxFactory,
        rng_class, ary_size, plot_hist=False):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    if has_double_support(context.devices[0]):
        dtypes = [np.float32, np.float64]
    else:
        dtypes = [np.float32]

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

# }}}


# {{{ test_random_int_in_range

@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("rng_class",
        [PhiloxGenerator, ThreefryGenerator])
def test_random_int_in_range(ctx_factory: cl.CtxFactory,
        rng_class, dtype, plot_hist=False):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

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

# }}}


# {{{ misc

# {{{ test_numpy_integer_shape

def test_numpy_integer_shape(ctx_factory: cl.CtxFactory):
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

# }}}


# {{{ test_allocation_with_various_shape_scalar_types

def test_allocation_with_various_shape_scalar_types(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    dims_ok = (2, np.int32(7), np.uint64(1))
    dims_not_ok = (-1, 5.70, np.float32(7))

    shapes_ok_1d = list(product(dims_ok))
    shapes_ok_2d = list(product(dims_ok, dims_ok))
    shapes_ok_3d = list(product(dims_ok, dims_ok, dims_ok))

    shapes_not_ok_1d = list(product(dims_not_ok))
    shapes_not_ok_2d = list(product(dims_ok, dims_not_ok))
    shapes_not_ok_3d = list(product(dims_not_ok, dims_not_ok, dims_not_ok))

    for shape in shapes_ok_1d + shapes_ok_2d + shapes_ok_3d:
        cl_array.empty(queue, shape, np.float32)

    for shape in shapes_not_ok_1d + shapes_not_ok_2d + shapes_not_ok_3d:
        with pytest.raises(ValueError):
            cl_array.empty(queue, shape, np.float32)

# }}}


# {{{ test_len

def test_len(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    a_cpu = cl_array.to_device(queue, a)
    assert len(a_cpu) == 10

# }}}


# {{{ test_stride_preservation

def test_stride_preservation(ctx_factory: cl.CtxFactory):
    if _PYPY:
        pytest.xfail("numpypy: no array creation from __array_interface__")

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    rng = np.random.default_rng(seed=42)
    a = rng.random(size=(3, 3))

    at = a.T
    print(at.flags.f_contiguous, at.flags.c_contiguous)
    at_gpu = cl_array.to_device(queue, at)
    print(at_gpu.flags.f_contiguous, at_gpu.flags.c_contiguous)
    assert np.allclose(at_gpu.get(), at)

# }}}


# {{{ test_nan_arithmetic

def test_nan_arithmetic(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)
    rng = np.random.default_rng(seed=42)

    def make_nan_contaminated_vector(size):
        a = rng.standard_normal(size=(size,), dtype=np.float32)

        from random import randrange
        for _i in range(size // 10):
            a[randrange(0, size)] = float("nan")
        return a

    size = 1 << 20

    a = make_nan_contaminated_vector(size)
    a_gpu = cl_array.to_device(queue, a)
    b = make_nan_contaminated_vector(size)
    b_gpu = cl_array.to_device(queue, b)

    ab = a * b
    ab_gpu = (a_gpu * b_gpu).get()

    assert (np.isnan(ab) == np.isnan(ab_gpu)).all()

# }}}


# {{{ test_mem_pool_with_arrays

def test_mem_pool_with_arrays(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)
    mem_pool = cl_tools.MemoryPool(cl_tools.ImmediateAllocator(queue))

    a_dev = cl_array.arange(queue, 2000, dtype=np.float32, allocator=mem_pool)
    b_dev = cl_array.to_device(queue, np.arange(2000), allocator=mem_pool) + 4000

    assert a_dev.allocator is mem_pool
    assert b_dev.allocator is mem_pool

# }}}


# {{{ test_view

def test_view(ctx_factory: cl.CtxFactory):
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


# {{{ test_diff

def test_diff(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand

    ary_len = 20000
    a_dev = clrand(queue, (ary_len,), dtype=np.float32)
    a = a_dev.get()

    err = la.norm(
            cl_array.diff(a_dev).get() - np.diff(a))
    assert err < 1e-4

# }}}


# {{{ test_copy

def test_copy(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    queue1 = cl.CommandQueue(context)
    queue2 = cl.CommandQueue(context)

    # Test copy

    arr = cl_array.zeros(queue1, 100, np.int32)
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

# }}}


# {{{ slices, concatenation

# {{{ test_slice

def test_slice(ctx_factory: cl.CtxFactory):
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

    start_offset = 0

    if queue.device.platform.name == "Intel(R) OpenCL":
        pytest.skip("Intel CL regularly crashes on this test case "
            "-- https://github.com/conda-forge/"
            "intel-compiler-repack-feedstock/issues/7")

    from random import randrange
    for _i in range(20):
        start = randrange(ary_len - start_offset)
        end = randrange(start+start_offset, ary_len)

        a_gpu_slice = tp(2)*a_gpu[start:end]
        a_slice = tp(2)*a[start:end]

        assert la.norm(a_gpu_slice.get() - a_slice) == 0

    for _i in range(20):
        start = randrange(ary_len-start_offset)
        # end = randrange(start+start_offset, ary_len)
        end = start

        a_gpu[start:end] = tp(2)*b[start:end]
        a[start:end] = tp(2)*b[start:end]

        assert la.norm(a_gpu.get() - a) == 0

    for _i in range(20):
        start = randrange(ary_len-start_offset)
        end = randrange(start+start_offset, ary_len)

        a_gpu[start:end] = tp(2)*b_gpu[start:end]
        a[start:end] = tp(2)*b[start:end]

        assert la.norm(a_gpu.get() - a) == 0

# }}}


# {{{ test_concatenate

def test_concatenate(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand

    a_dev = clrand(queue, (5, 15, 20), dtype=np.float32)
    b_dev = clrand(queue, (4, 15, 20), dtype=np.float32)
    c_dev = clrand(queue, (3, 15, 20), dtype=np.float32)
    a = a_dev.get()
    b = b_dev.get()
    c = c_dev.get()

    cat_dev = cl_array.concatenate((a_dev, b_dev, c_dev))
    cat = np.concatenate((a, b, c))

    assert la.norm(cat - cat_dev.get()) == 0

# }}}

# }}}


# {{{ conditionals, any, all

# {{{ test_comparisons

def test_comparisons(ctx_factory: cl.CtxFactory):
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

        res2_dev = op(0, res_dev)
        res2 = op(0, res)
        assert (res2_dev.get() == res2).all()


# }}}


# {{{ test_any_all

def test_any_all(ctx_factory: cl.CtxFactory):
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

# }}}


# {{{ test_map_to_host

def test_map_to_host(ctx_factory: cl.CtxFactory):
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

# }}}


# {{{ test_view_and_strides

def test_view_and_strides(ctx_factory: cl.CtxFactory):
    if _PYPY:
        pytest.xfail("numpypy: no array creation from __array_interface__")

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

# }}}


# {{{ test_meshmode_view

def test_meshmode_view(ctx_factory: cl.CtxFactory):
    if _PYPY:
        # https://bitbucket.org/pypy/numpy/issue/28/indexerror-on-ellipsis-slice
        pytest.xfail("numpypy bug #28")

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    n = 2
    result = cl_array.empty(queue, (2, n*6), np.float32)

    def view(z):
        return z[..., n*3:n*6].reshape((*z.shape[:-1], n, 3))

    result = result.with_queue(queue)
    result.fill(0)
    view(result)[0].fill(1)
    view(result)[1].fill(1)
    x = result.get()
    assert (view(x) == 1).all()

# }}}


# {{{ test_event_management

def test_event_management(ctx_factory: cl.CtxFactory):
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

    for _i in range(10):
        x.fill(0)

    assert len(x.events) == 10

    for _i in range(1000):
        x.fill(0)

    assert len(x.events) < 100

# }}}


# {{{ test_reshape

def test_reshape(ctx_factory: cl.CtxFactory):
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
    assert a_dev.reshape((8, -1, 4)).shape == (8, 4, 4)

    import pytest
    with pytest.raises(ValueError):
        a_dev.reshape(-1, -1, 4)

# }}}


# {{{ test_skip_slicing

def test_skip_slicing(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a_host = np.arange(16).reshape((4, 4))
    b_host = a_host[::3]

    a = cl_array.to_device(queue, a_host)
    b = a[::3]
    assert b.shape == b_host.shape
    assert np.array_equal(b[1].get(), b_host[1])

# }}}


# {{{ test_transpose

def test_transpose(ctx_factory: cl.CtxFactory):
    if _PYPY:
        pytest.xfail("numpypy: no array creation from __array_interface__")

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand

    a_gpu = clrand(queue, (10, 20, 30), dtype=np.float32)
    a = a_gpu.get()

    # FIXME: not contiguous
    # assert np.allclose(a_gpu.transpose((1,2,0)).get(), a.transpose((1,2,0)))
    assert np.array_equal(a_gpu.T.get(), a.T)

# }}}


# {{{ test_newaxis

def test_newaxis(ctx_factory: cl.CtxFactory):
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

# }}}


# {{{ test_squeeze

def test_squeeze(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)
    rng = np.random.default_rng(seed=42)

    shape = (40, 2, 5, 100)
    a_cpu = rng.random(size=shape)
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
    # assert np.all(a_gpu_slice.get().ravel() == a_gpu_squeezed_slice.get().ravel())

    # Slice with length 1 on dimensions 2
    a_gpu_slice = a_gpu[:, :, 2:3, :]
    assert a_gpu_slice.shape == (shape[0], shape[1], 1, shape[3])
    assert not a_gpu_slice.flags.c_contiguous

    # Squeeze it, but no contiguity here
    a_gpu_squeezed_slice = a_gpu[:, :, 2:3, :].squeeze()
    assert a_gpu_squeezed_slice.shape == (shape[0], shape[1], shape[3])
    assert not a_gpu_squeezed_slice.flags.c_contiguous

    # Check that we get the original values out
    # assert np.all(a_gpu_slice.get().ravel() == a_gpu_squeezed_slice.get().ravel())

# }}}


# {{{ test_fancy_fill

def test_fancy_fill(ctx_factory: cl.CtxFactory):
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

# }}}


# {{{ test_fancy_indexing

def test_fancy_indexing(ctx_factory: cl.CtxFactory):
    if _PYPY:
        pytest.xfail("numpypy: multi value setting is not supported")
    context = ctx_factory()
    queue = cl.CommandQueue(context)
    rng = np.random.default_rng(seed=42)

    n = 2 ** 20 + 2**18 + 22
    numpy_dest = np.zeros(n, dtype=np.int32)
    numpy_idx = np.arange(n, dtype=np.int32)
    rng.shuffle(numpy_idx)
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

# }}}


# {{{ test_multi_put

def test_multi_put(ctx_factory: cl.CtxFactory):
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
    for _i, ary in enumerate(out_compare):
        ary[idx.get()] = np.arange(0, 6, dtype=np.float32)

    cl_array.multi_put(cl_arrays, idx, out=out_arrays)

    assert np.all(np.all(out_compare[i] == out_arrays[i].get()) for i in range(9))

# }}}


# {{{ test_get_async

def test_get_async(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    device = queue.device
    if device.platform.vendor == "The pocl project" \
            and device.type & cl.device_type.GPU:
        pytest.xfail("the async get test fails on PoCL + Nvidia,"
                "at least the K40, as of PoCL 1.6, 2021-01-20")

    rng = np.random.default_rng(seed=42)
    a = rng.random(10**6, dtype=np.float32)
    a_gpu = cl_array.to_device(queue, a)
    b = a + a**5 + 1
    b_gpu = a_gpu + a_gpu**5 + 1

    b1, evt = b_gpu.get_async()  # testing that this waits for events
    evt.wait()
    assert np.abs(b1 - b).mean() < 1e-5

    wait_event = cl.UserEvent(context)
    b_gpu.add_event(wait_event)
    b, evt = b_gpu.get_async()  # testing that this doesn't hang
    wait_event.set_status(cl.command_execution_status.COMPLETE)
    evt.wait()
    assert np.abs(b1 - b).mean() < 1e-5

# }}}


# {{{ test_outoforderqueue_get

def test_outoforderqueue_get(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    try:
        queue = cl.CommandQueue(context,
               properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
    except Exception:
        pytest.skip("out-of-order queue not available")

    rng = np.random.default_rng(seed=42)
    a = rng.random(10**6, dtype=np.float32)
    a_gpu = cl_array.to_device(queue, a)
    b_gpu = a_gpu + a_gpu**5 + 1
    b1 = b_gpu.get()  # testing that this waits for events
    b = a + a**5 + 1
    assert np.abs(b1 - b).mean() < 1e-5

# }}}


# {{{ test_outoforderqueue_copy

def test_outoforderqueue_copy(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    try:
        queue = cl.CommandQueue(context,
               properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
    except Exception:
        pytest.skip("out-of-order queue not available")

    rng = np.random.default_rng(seed=42)
    a = rng.random(10**6, dtype=np.float32)
    a_gpu = cl_array.to_device(queue, a)
    c_gpu = a_gpu**2 - 7
    b_gpu = c_gpu.copy()  # testing that this waits for and creates events
    b_gpu *= 10
    queue.finish()
    b1 = b_gpu.get()
    b = 10 * (a**2 - 7)
    assert np.abs(b1 - b).mean() < 1e-5


# }}}


# {{{ test_outoforderqueue_indexing

def test_outoforderqueue_indexing(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    try:
        queue = cl.CommandQueue(context,
               properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
    except Exception:
        pytest.skip("out-of-order queue not available")

    rng = np.random.default_rng(seed=42)
    a = rng.random(10**6, dtype=np.float32)
    i = (8e5 + 1e5 * rng.random(10**5)).astype(np.int32)

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

# }}}


# {{{ test_outoforderqueue_reductions

def test_outoforderqueue_reductions(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    try:
        queue = cl.CommandQueue(context,
               properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
    except Exception:
        pytest.skip("out-of-order queue not available")
    # 0/1 values to avoid accumulated rounding error
    rng = np.random.default_rng(seed=42)
    a = (rng.random(10**6) > 0.5).astype(np.float32)

    a[800000] = 10  # all<5 looks true until near the end
    a_gpu = cl_array.to_device(queue, a)
    b1 = cl_array.sum(a_gpu).get()
    b2 = cl_array.dot(a_gpu, 3 - a_gpu).get()
    b3 = (a_gpu < 5).all().get()
    assert b1 == a.sum() and b2 == a.dot(3 - a) and b3 == 0

# }}}


# {{{ test_negative_dim_rejection

def test_negative_dim_rejection(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    with pytest.raises(ValueError):
        cl_array.Array(queue, shape=-10, dtype=np.float64)

    with pytest.raises(ValueError):
        cl_array.Array(queue, shape=(-10,), dtype=np.float64)

    for left_dim in (-1, 0, 1):
        with pytest.raises(ValueError):
            cl_array.Array(queue, shape=(left_dim, -1), dtype=np.float64)

    for right_dim in (-1, 0, 1):
        with pytest.raises(ValueError):
            cl_array.Array(queue, shape=(-1, right_dim), dtype=np.float64)

# }}}


# {{{ test_zero_size_array

@pytest.mark.parametrize("empty_shape", [0, (), (3, 0, 2), (0, 5), (5, 0)])
def test_zero_size_array(ctx_factory: cl.CtxFactory, empty_shape):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    if queue.device.platform.name == "Intel(R) OpenCL":
        pytest.xfail("size-0 arrays fail on Intel CL")
    if (queue.device.platform.name == "Portable Computing Language"
            and cl.get_cl_header_version() < (1, 2)):
        pytest.xfail("does not work on pocl with CL pre 1.2")

    a = cl_array.zeros(queue, empty_shape, dtype=np.float32)
    b = cl_array.zeros(queue, empty_shape, dtype=np.float32)
    b.fill(1)
    c = a + b
    c_host = c.get()
    cl_array.to_device(queue, c_host)

    assert c.flags.c_contiguous == c_host.flags.c_contiguous
    assert c.flags.f_contiguous == c_host.flags.f_contiguous

    for order in "CF":
        c_flat = c.reshape(-1, order=order)
        c_host_flat = c_host.reshape(-1, order=order)
        assert c_flat.shape == c_host_flat.shape
        assert c_flat.strides == c_host_flat.strides
        assert c_flat.flags.c_contiguous == c_host_flat.flags.c_contiguous
        assert c_flat.flags.f_contiguous == c_host_flat.flags.f_contiguous

# }}}


# {{{ test_str_without_queue

def test_str_without_queue(ctx_factory: cl.CtxFactory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = cl_array.zeros(queue, 10, dtype=np.float32).with_queue(None)
    print(str(a))
    print(repr(a))

# }}}


# {{{ test_stack

@pytest.mark.parametrize("order", ("F", "C"))
@pytest.mark.parametrize("input_dims", (1, 2, 3))
def test_stack(ctx_factory: cl.CtxFactory, input_dims, order):
    # Replicates pytato/test/test_codegen.py::test_stack
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    shape = (2, 2, 2)[:input_dims]
    axis = -1 if order == "F" else 0

    rng = np.random.default_rng(seed=42)
    x_in = rng.random(size=shape)
    y_in = rng.random(size=shape)
    x_in = x_in if order == "C" else np.asfortranarray(x_in)
    y_in = y_in if order == "C" else np.asfortranarray(y_in)

    x = cl_array.to_device(queue, x_in)
    y = cl_array.to_device(queue, y_in)

    np.testing.assert_allclose(cl_array.stack((x, y), axis=axis).get(),
                                np.stack((x_in, y_in), axis=axis))

# }}}


# {{{ test_assign_different_strides

def test_assign_different_strides(ctx_factory: cl.CtxFactory):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    from pyopencl.clrandom import rand as clrand

    a = clrand(queue, (20, 30), dtype=np.float32)
    b = cl_array.empty(queue, (20, 30), dtype=np.float32, order="F")
    with pytest.raises(NotImplementedError):
        b[:] = a

# }}}


# {{{ test_branch_operations_on_pure_scalars

def test_branch_operations_on_pure_scalars():
    rng = np.random.default_rng(seed=42)
    x = rng.random()
    y = rng.random()
    cond = rng.choice([False, True])

    np.testing.assert_allclose(np.maximum(x, y),
                               cl_array.maximum(x, y))
    np.testing.assert_allclose(np.minimum(x, y),
                               cl_array.minimum(x, y))
    np.testing.assert_allclose(np.where(cond, x, y),
                               cl_array.if_positive(cond, x, y))

# }}}


# {{{ test_branch_operations_on_nans

@pytest.mark.parametrize("op", [
    cl_array.maximum,
    cl_array.minimum,
])
@pytest.mark.parametrize("special_a", [
    np.nan,
    np.inf,
    -np.inf,
])
@pytest.mark.parametrize("special_b", [
    np.nan,
    np.inf,
    -np.inf,
    None
])
def test_branch_operations_on_nans(ctx_factory: cl.CtxFactory,
        op, special_a, special_b):
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    def sb_or(x):
        if special_b is None:
            return x
        else:
            return special_b

    x_np = np.array([special_a, sb_or(1.), special_a, sb_or(2.), sb_or(3.)],
            dtype=np.float64)
    y_np = np.array([special_a, special_a, sb_or(1.), sb_or(3.), sb_or(2.)],
            dtype=np.float64)

    x_cl = cl_array.to_device(cq, x_np)
    y_cl = cl_array.to_device(cq, y_np)

    ref = getattr(np, op.__name__)(x_np, y_np)
    result = op(x_cl, y_cl)
    if isinstance(result, cl_array.Array):
        result = result.get()

    np.testing.assert_allclose(result, ref)

# }}}


# {{{ test_slice_copy

def test_slice_copy(ctx_factory: cl.CtxFactory):
    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    rng = np.random.default_rng(seed=42)
    x = cl_array.to_device(queue, rng.random(size=(96, 27)))
    y = x[::8, ::3]
    with pytest.raises(RuntimeError):
        y.copy()

# }}}


# {{{{ test_ravel

@pytest.mark.parametrize("order", ("C", "F"))
def test_ravel(ctx_factory: cl.CtxFactory, order):
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    rng = np.random.default_rng(seed=42)
    x = rng.standard_normal(size=(10, 4))

    if order == "F":
        x = np.asfortranarray(x)
    elif order == "C":
        pass
    else:
        raise AssertionError

    x_cl = cl_array.to_device(cq, x)

    np.testing.assert_allclose(x_cl.ravel(order=order).get(),
                               x.ravel(order=order))

# }}}


# {{{ test_arithmetic_on_non_scalars

def test_arithmetic_on_non_scalars(ctx_factory: cl.CtxFactory):
    pytest.importorskip("dataclasses")

    from dataclasses import dataclass
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    @dataclass
    class ArrayContainer:
        _data: np.ndarray

        def __eq__(self, other):
            return ArrayContainer(self._data == other)

    with pytest.raises(TypeError):
        ArrayContainer(np.ones(100)) + cl_array.zeros(cq, (10,), dtype=np.float64)

# }}}


# {{{ test_arithmetic_with_device_scalars

@pytest.mark.parametrize("which", ("add", "sub", "mul", "truediv"))
def test_arithmetic_with_device_scalars(ctx_factory: cl.CtxFactory, which):
    import operator

    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    rng = np.random.default_rng(seed=42)
    ndim = rng.integers(1, 5)

    shape = tuple(rng.integers(2, 7) for i in range(ndim))

    x_in = rng.random(shape)
    x_cl = cl_array.to_device(cq, x_in)
    idx = tuple(rng.integers(0, dim) for dim in shape)

    op = getattr(operator, which)
    res_cl = op(x_cl, x_cl[idx])
    res_np = op(x_in, x_in[idx])

    np.testing.assert_allclose(res_cl.get(), res_np)

# }}}


# {{{ test_if_positive_with_scalars

@pytest.mark.parametrize("then_type", ["array", "host_scalar", "device_scalar"])
@pytest.mark.parametrize("else_type", ["array", "host_scalar", "device_scalar"])
def test_if_positive_with_scalars(ctx_factory: cl.CtxFactory, then_type, else_type):
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    rng = np.random.default_rng(seed=42)
    shape = (512,)

    criterion_np = rng.random(shape)
    criterion_cl = cl_array.to_device(cq, criterion_np)

    def _get_array_or_scalar(rtype, value):
        if rtype == "array":
            ary_np = value + np.zeros(shape, dtype=criterion_cl.dtype)
            ary_cl = value + cl_array.zeros_like(criterion_cl)
        elif rtype == "host_scalar":
            ary_np = ary_cl = value
        elif rtype == "device_scalar":
            ary_np = value
            ary_cl = cl_array.to_device(cq, np.array(value))
        else:
            raise ValueError(rtype)

        return ary_np, ary_cl

    then_np, then_cl = _get_array_or_scalar(then_type, 0.0)
    else_np, else_cl = _get_array_or_scalar(else_type, 1.0)

    result_cl = cl_array.if_positive(criterion_cl < 0.5, then_cl, else_cl)
    result_np = np.where(criterion_np < 0.5, then_np, else_np)

    np.testing.assert_allclose(result_cl.get(), result_np)

# }}}


# {{{ test_maximum_minimum_with_scalars

def test_maximum_minimum_with_scalars(ctx_factory: cl.CtxFactory):
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    a_np = np.float64(4.0)
    a_cl = cl_array.to_device(cq, np.array(a_np)).with_queue(None)

    b_np = np.float64(-3.0)
    b_cl = cl_array.to_device(cq, np.array(b_np)).with_queue(None)

    result = cl_array.maximum(a_np, b_cl, queue=cq)
    np.testing.assert_allclose(result.get(), a_np)
    result = cl_array.maximum(a_cl, b_np, queue=cq)
    np.testing.assert_allclose(result.get(), a_np)
    result = cl_array.maximum(a_cl, b_cl, queue=cq)
    np.testing.assert_allclose(result.get(), a_np)

    result = cl_array.minimum(a_np, b_cl, queue=cq)
    np.testing.assert_allclose(result.get(), b_np)
    result = cl_array.minimum(a_cl, b_np, queue=cq)
    np.testing.assert_allclose(result.get(), b_np)
    result = cl_array.minimum(a_cl, b_cl, queue=cq)
    np.testing.assert_allclose(result.get(), b_np)

    # Test 'untyped' scalars
    # FIXME: these don't work with unsized ints
    result = cl_array.minimum(4.0, b_cl, queue=cq)
    np.testing.assert_allclose(result.get(), b_np)
    result = cl_array.maximum(4.0, b_cl, queue=cq)
    np.testing.assert_allclose(result.get(), a_np)

    result = cl_array.minimum(b_cl, 4.0, queue=cq)
    np.testing.assert_allclose(result.get(), b_np)
    result = cl_array.maximum(b_cl, 4.0, queue=cq)
    np.testing.assert_allclose(result.get(), a_np)

    result = cl_array.minimum(-3.0, 4.0, queue=cq)
    np.testing.assert_allclose(result, b_np)
    result = cl_array.maximum(-3.0, 4.0, queue=cq)
    np.testing.assert_allclose(result, a_np)

# }}}


# {{{ test_empty_reductions_vs_numpy

@pytest.mark.parametrize(("reduction", "supports_initial"), [
    (cl_array.any, False),
    (cl_array.all, False),
    (cl_array.sum, True),
    (cl_array.max, True),
    (cl_array.min, True),
    ])
def test_empty_reductions_vs_numpy(ctx_factory: cl.CtxFactory,
        reduction, supports_initial):
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    # {{{ empty

    x_np = np.array([], dtype=np.float64)
    x_cl = cl_array.to_device(cq, x_np)

    try:
        ref = getattr(np, reduction.__name__)(x_np)
    except ValueError:
        ref = None

    if ref is None:
        with pytest.raises(ValueError):
            reduction(x_cl)
    else:
        result = reduction(x_cl)
        if isinstance(result, cl_array.Array):
            result = result.get()

        np.testing.assert_allclose(result, ref)

    # }}}

    # {{{ empty with initial

    if supports_initial:
        ref = getattr(np, reduction.__name__)(x_np, initial=5.0)
        result = reduction(x_cl, initial=5.0)
        if isinstance(result, cl_array.Array):
            result = result.get()

        np.testing.assert_allclose(result, ref)

    # }}}

    # {{{ non-empty with initial

    if supports_initial:
        x_np = np.linspace(-1, 1, 10)
        x_cl = cl_array.to_device(cq, x_np)

        ref = getattr(np, reduction.__name__)(x_np, initial=5.0)
        result = reduction(x_cl, initial=5.0).get()
        np.testing.assert_allclose(result, ref)

        ref = getattr(np, reduction.__name__)(x_np, initial=-5.0)
        result = reduction(x_cl, initial=-5.0).get()
        np.testing.assert_allclose(result, ref)

    # }}}

# }}}


# {{{ test_reduction_nan_handling

@pytest.mark.parametrize("with_initial", [False, True])
@pytest.mark.parametrize("input_case", ["only nans", "mixed"])
@pytest.mark.parametrize("reduction", [
    cl_array.sum,
    cl_array.max,
    cl_array.min,
    ])
def test_reduction_nan_handling(ctx_factory: cl.CtxFactory,
        reduction, input_case, with_initial):
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    if input_case == "only nans":
        x_np = np.array([np.nan, np.nan], dtype=np.float64)
    elif input_case == "mixed":
        x_np = np.array([np.nan, 1.], dtype=np.float64)
    else:
        raise ValueError("invalid input case")

    x_cl = cl_array.to_device(cq, x_np)

    if with_initial:
        ref = getattr(np, reduction.__name__)(x_np, initial=5.0)
        result = reduction(x_cl, initial=5.0)
    else:
        ref = getattr(np, reduction.__name__)(x_np)
        result = reduction(x_cl)

    if isinstance(result, cl_array.Array):
        result = result.get()

    np.testing.assert_allclose(result, ref)

# }}}


# {{{ test_reductions_dtype

def test_dtype_conversions(ctx_factory: cl.CtxFactory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    ary = cl_array.to_device(queue, np.linspace(0, 1, 32))

    for func, nargs, arg_name in [
            (cl_array.sum, 1, "dtype"),
            (cl_array.dot, 2, "dtype"),
            (cl_array.vdot, 2, "dtype"),
            (cl_array.cumsum, 1, "output_dtype"),
            ]:
        for dtype in [np.float32, np.float64]:
            result = func(*((ary,) * nargs), **{arg_name: dtype})
            assert result.dtype == dtype, result.dtype

# }}}


# {{{ test_svm_mem_pool_with_arrays

@pytest.mark.parametrize("use_mempool", [False, True])
def test_arrays_with_svm_allocators(ctx_factory: cl.CtxFactory, use_mempool):
    context = ctx_factory()
    queue = cl.CommandQueue(context)
    queue2 = cl.CommandQueue(context)

    from pyopencl.characterize import has_coarse_grain_buffer_svm
    has_cg_svm = has_coarse_grain_buffer_svm(queue.device)

    if not has_cg_svm:
        pytest.skip("Need coarse-grained SVM support for this test.")

    alloc = cl_tools.SVMAllocator(context, queue=queue)
    if use_mempool:
        alloc = cl_tools.SVMPool(alloc)

    def alloc2(size):
        allocation = alloc(size)
        allocation.bind_to_queue(queue2)
        return allocation

    a_dev = cl_array.arange(queue, 2000, dtype=np.float32, allocator=alloc)
    b_dev = cl_array.to_device(queue, np.arange(2000), allocator=alloc) + 4000

    assert a_dev.allocator is alloc
    assert b_dev.allocator is alloc

    assert a_dev.data._queue == queue
    assert b_dev.data._queue == queue

    a_dev2 = cl_array.arange(queue2, 2000, dtype=np.float32, allocator=alloc2)
    b_dev2 = cl_array.to_device(queue2, np.arange(2000), allocator=alloc2) + 4000

    assert a_dev2.allocator is alloc2
    assert b_dev2.allocator is alloc2

    assert a_dev2.data._queue == queue2
    assert b_dev2.data._queue == queue2

    np.testing.assert_allclose((a_dev+b_dev).get(), (a_dev2+b_dev2).get())

    with pytest.warns(cl_array.InconsistentOpenCLQueueWarning):
        a_dev2.with_queue(queue)

        # safe to let this proceed to deallocation, since we're not
        # operating on the memory

    with pytest.warns(cl_array.InconsistentOpenCLQueueWarning):
        cl_array.empty(queue2, 2000, np.float32, allocator=alloc)

        # safe to let this proceed to deallocation, since we're not
        # operating on the memory

# }}}


def test_logical_and_or(ctx_factory: cl.CtxFactory):
    # NOTE: Copied over from pycuda/test/test_gpuarray.py
    rng = np.random.default_rng(seed=0)
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    for op in ["logical_and", "logical_or"]:
        x_np = rng.random((10, 4))
        y_np = rng.random((10, 4))
        zeros_np = np.zeros((10, 4))
        ones_np = np.ones((10, 4))

        x_cl = cl_array.to_device(cq, x_np)
        y_cl = cl_array.to_device(cq, y_np)
        zeros_cl = cl_array.zeros(cq, (10, 4), np.float64)
        ones_cl = cl_array.zeros(cq, (10, 4), np.float64) + 1

        np.testing.assert_array_equal(
            getattr(cl_array, op)(x_cl, y_cl).get(),
            getattr(np, op)(x_np, y_np))
        np.testing.assert_array_equal(
            getattr(cl_array, op)(x_cl, ones_cl).get(),
            getattr(np, op)(x_np, ones_np))
        np.testing.assert_array_equal(
            getattr(cl_array, op)(x_cl, zeros_cl).get(),
            getattr(np, op)(x_np, zeros_np))
        np.testing.assert_array_equal(
            getattr(cl_array, op)(x_cl, 1.0).get(),
            getattr(np, op)(x_np, ones_np))
        np.testing.assert_array_equal(
            getattr(cl_array, op)(x_cl, 0.0).get(),
            getattr(np, op)(x_np, 0.0))


def test_logical_not(ctx_factory: cl.CtxFactory):
    # NOTE: Copied over from pycuda/test/test_gpuarray.py
    ctx = ctx_factory()
    cq = cl.CommandQueue(ctx)

    rng = np.random.default_rng(seed=0)
    x_np = rng.random((10, 4))
    x_cl = cl_array.to_device(cq, x_np)

    np.testing.assert_array_equal(
        cl_array.logical_not(x_cl).get(),
        np.logical_not(x_np))
    np.testing.assert_array_equal(
        cl_array.logical_not(cl_array.zeros(cq, 10, np.float64)).get(),
        np.logical_not(np.zeros(10)))
    np.testing.assert_array_equal(
        cl_array.logical_not(cl_array.zeros(cq, 10, np.float64) + 1).get(),
        np.logical_not(np.ones(10)))


# {{{ test XDG_CACHE_HOME handling

@pytest.mark.skipif(sys.platform == "win32",
                    reason="XDG_CACHE_HOME is not used on Windows")
def test_xdg_cache_home(ctx_factory: cl.CtxFactory):
    import os
    import shutil
    from os.path import join

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    a_gpu = cl_array.to_device(queue, a)

    xdg_dir = "tmpdir_pyopencl_xdg_test"

    # PyOpenCL uses pytools.PersistentDict for invoker caches,
    # which is why xdg_dir will always exist. Therefore, check
    # whether xdg_pyopencl_dir exists.
    xdg_pyopencl_dir = join(xdg_dir, "pyopencl")
    assert not os.path.exists(xdg_dir)

    old_xdg_cache_home = None
    old_characterize_has_src_build_cache = None

    try:
        # Make sure that the source build cache is enabled
        old_characterize_has_src_build_cache = \
            cl_characterize.has_src_build_cache
        cl_characterize.has_src_build_cache = lambda dev: False

        old_xdg_cache_home = os.getenv("XDG_CACHE_HOME")
        os.environ["XDG_CACHE_HOME"] = xdg_dir

        result = pow(a_gpu, a_gpu).get()
        assert (np.abs(a ** a - result) < 3e-3).all()

        assert os.path.exists(xdg_pyopencl_dir)
    finally:
        cl_characterize.has_src_build_cache = \
            old_characterize_has_src_build_cache

        if old_xdg_cache_home is not None:
            os.environ["XDG_CACHE_HOME"] = old_xdg_cache_home
        else:
            del os.environ["XDG_CACHE_HOME"]

        shutil.rmtree(xdg_dir)

# }}}


def test_numpy_type_promotion_with_cl_arrays(ctx_factory: cl.CtxFactory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    class NotReallyAnArray:
        @property
        def dtype(self):
            return np.dtype("float64")

    # Make sure that np.result_type accesses only the dtype attribute of the
    # class, not (e.g.) its data.
    assert np.result_type(42, NotReallyAnArray()) == np.float64

    from pyopencl.array import _get_common_dtype
    assert _get_common_dtype(42, NotReallyAnArray(), queue) == np.float64
    assert _get_common_dtype(42.0, NotReallyAnArray(), queue) == np.float64


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
