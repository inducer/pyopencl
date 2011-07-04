#! /usr/bin/env python
import numpy as np
import numpy.linalg as la
import sys
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
    from pyopencl.tools import pytest_generate_tests_for_pyopencl \
            as pytest_generate_tests
    from pyopencl.characterize import has_double_support


@pytools.test.mark_test.opencl
def test_pow_array(ctx_getter):
    context = ctx_getter()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    a_gpu = cl_array.to_device(queue, a)

    result = pow(a_gpu, a_gpu).get()
    assert (np.abs(a ** a - result) < 1e-3).all()

    result = (a_gpu ** a_gpu).get()
    assert (np.abs(pow(a, a) - result) < 1e-3).all()


@pytools.test.mark_test.opencl
def test_pow_number(ctx_getter):
    context = ctx_getter()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    a_gpu = cl_array.to_device(queue, a)

    result = pow(a_gpu, 2).get()
    assert (np.abs(a ** 2 - result) < 1e-3).all()


@pytools.test.mark_test.opencl
def test_abs(ctx_getter):
    context = ctx_getter()
    queue = cl.CommandQueue(context)

    a = -cl_array.arange(queue, 111, dtype=np.float32)
    res = a.get()

    for i in range(111):
        assert res[i] <= 0

    a = abs(a)

    res = a.get()

    for i in range(111):
        assert abs(res[i]) >= 0
        assert res[i] == i


@pytools.test.mark_test.opencl
def test_len(ctx_getter):
    context = ctx_getter()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    a_cpu = cl_array.to_device(queue, a)
    assert len(a_cpu) == 10


@pytools.test.mark_test.opencl
def test_multiply(ctx_getter):
    """Test the muliplication of an array with a scalar. """

    context = ctx_getter()
    queue = cl.CommandQueue(context)

    for sz in [10, 50000]:
        for dtype, scalars in [
            (np.float32, [2]),
            ]:
            for scalar in scalars:
                a = np.arange(sz).astype(dtype)
                a_gpu = cl_array.to_device(queue, a)
                a_doubled = (scalar * a_gpu).get()

                assert (a * scalar == a_doubled).all()


@pytools.test.mark_test.opencl
def test_multiply_array(ctx_getter):
    """Test the multiplication of two arrays."""

    context = ctx_getter()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)

    a_gpu = cl_array.to_device(queue, a)
    b_gpu = cl_array.to_device(queue, a)

    a_squared = (b_gpu * a_gpu).get()

    assert (a * a == a_squared).all()


@pytools.test.mark_test.opencl
def test_addition_array(ctx_getter):
    """Test the addition of two arrays."""

    context = ctx_getter()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    a_gpu = cl_array.to_device(queue, a)
    a_added = (a_gpu + a_gpu).get()

    assert (a + a == a_added).all()


@pytools.test.mark_test.opencl
def test_addition_scalar(ctx_getter):
    """Test the addition of an array and a scalar."""

    context = ctx_getter()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    a_gpu = cl_array.to_device(queue, a)
    a_added = (7 + a_gpu).get()

    assert (7 + a == a_added).all()


@pytools.test.mark_test.opencl
def test_substract_array(ctx_getter):
    """Test the substraction of two arrays."""
    #test data
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    b = np.array([10, 20, 30, 40, 50,
                  60, 70, 80, 90, 100]).astype(np.float32)

    context = ctx_getter()
    queue = cl.CommandQueue(context)

    a_gpu = cl_array.to_device(queue, a)
    b_gpu = cl_array.to_device(queue, b)

    result = (a_gpu - b_gpu).get()
    assert (a - b == result).all()

    result = (b_gpu - a_gpu).get()
    assert (b - a == result).all()


@pytools.test.mark_test.opencl
def test_substract_scalar(ctx_getter):
    """Test the substraction of an array and a scalar."""

    context = ctx_getter()
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
def test_divide_scalar(ctx_getter):
    """Test the division of an array and a scalar."""

    context = ctx_getter()
    queue = cl.CommandQueue(context)

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
    a_gpu = cl_array.to_device(queue, a)

    result = (a_gpu / 2).get()
    assert (a / 2 == result).all()

    result = (2 / a_gpu).get()
    assert (np.abs(2 / a - result) < 1e-5).all()


@pytools.test.mark_test.opencl
def test_divide_array(ctx_getter):
    """Test the division of an array and a scalar. """

    context = ctx_getter()
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


@pytools.test.mark_test.opencl
def test_random(ctx_getter):
    context = ctx_getter()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand

    if has_double_support(context.devices[0]):
        dtypes = [np.float32, np.float64]
    else:
        dtypes = [np.float32]

    for dtype in dtypes:
        a = clrand(context, queue, (10, 100), dtype=dtype).get()

        assert (0 <= a).all()
        assert (a < 1).all()


@pytools.test.mark_test.opencl
def test_nan_arithmetic(ctx_getter):
    context = ctx_getter()
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
def test_elwise_kernel(ctx_getter):
    context = ctx_getter()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand

    a_gpu = clrand(context, queue, (50,), np.float32)
    b_gpu = clrand(context, queue, (50,), np.float32)

    from pyopencl.elementwise import ElementwiseKernel
    lin_comb = ElementwiseKernel(context,
            "float a, float *x, float b, float *y, float *z",
            "z[i] = a*x[i] + b*y[i]",
            "linear_combination")

    c_gpu = cl_array.empty_like(a_gpu)
    lin_comb(5, a_gpu, 6, b_gpu, c_gpu)

    assert la.norm((c_gpu - (5 * a_gpu + 6 * b_gpu)).get()) < 1e-5


@pytools.test.mark_test.opencl
def test_elwise_kernel_with_options(ctx_getter):
    from pyopencl.clrandom import rand as clrand
    from pyopencl.elementwise import ElementwiseKernel

    context = ctx_getter()
    queue = cl.CommandQueue(context)

    in_gpu = clrand(context, queue, (50,), np.float32)

    options = ['-DADD_ONE']
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
def test_take(ctx_getter):
    context = ctx_getter()
    queue = cl.CommandQueue(context)

    idx = cl_array.arange(queue, 0, 200000, 2, dtype=np.uint32)
    a = cl_array.arange(queue, 0, 600000, 3, dtype=np.float32)
    result = cl_array.take(a, idx)
    assert ((3 * idx).get() == result.get()).all()


@pytools.test.mark_test.opencl
def test_arange(ctx_getter):
    context = ctx_getter()
    queue = cl.CommandQueue(context)

    n = 5000
    a = cl_array.arange(queue, n, dtype=np.float32)
    assert (np.arange(n, dtype=np.float32) == a.get()).all()


@pytools.test.mark_test.opencl
def test_reverse(ctx_getter):
    context = ctx_getter()
    queue = cl.CommandQueue(context)

    n = 5000
    a = np.arange(n).astype(np.float32)
    a_gpu = cl_array.to_device(queue, a)

    a_gpu = a_gpu.reverse()

    assert (a[::-1] == a_gpu.get()).all()


@pytools.test.mark_test.opencl
def test_sum(ctx_getter):
    context = ctx_getter()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand

    a_gpu = clrand(context, queue, (200000,), np.float32)
    a = a_gpu.get()

    sum_a = np.sum(a)
    sum_a_gpu = cl_array.sum(a_gpu).get()

    assert abs(sum_a_gpu - sum_a) / abs(sum_a) < 1e-4


@pytools.test.mark_test.opencl
def test_minmax(ctx_getter):
    context = ctx_getter()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand

    if has_double_support(context.devices[0]):
        dtypes = [np.float64, np.float32, np.int32]
    else:
        dtypes = [np.float32, np.int32]

    for what in ["min", "max"]:
        for dtype in dtypes:
            a_gpu = clrand(context, queue, (200000,), dtype)
            a = a_gpu.get()

            op_a = getattr(np, what)(a)
            op_a_gpu = getattr(cl_array, what)(a_gpu).get()

            assert op_a_gpu == op_a, (op_a_gpu, op_a, dtype, what)


@pytools.test.mark_test.opencl
def test_subset_minmax(ctx_getter):
    context = ctx_getter()
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
        a_gpu = clrand(context, queue, (l_a,), dtype)
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
def test_dot(ctx_getter):
    context = ctx_getter()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand
    a_gpu = clrand(context, queue, (200000,), np.float32)
    a = a_gpu.get()
    b_gpu = clrand(context, queue, (200000,), np.float32)
    b = b_gpu.get()

    dot_ab = np.dot(a, b)

    dot_ab_gpu = cl_array.dot(a_gpu, b_gpu).get()

    assert abs(dot_ab_gpu - dot_ab) / abs(dot_ab) < 1e-4


if False:
    @pytools.test.mark_test.opencl
    def test_slice(ctx_getter):
        from pyopencl.clrandom import rand as clrand

        l = 20000
        a_gpu = clrand(context, queue, (l,))
        a = a_gpu.get()

        from random import randrange
        for i in range(200):
            start = randrange(l)
            end = randrange(start, l)

            a_gpu_slice = a_gpu[start:end]
            a_slice = a[start:end]

            assert la.norm(a_gpu_slice.get() - a_slice) == 0


@pytools.test.mark_test.opencl
def test_if_positive(ctx_getter):
    context = ctx_getter()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand

    l = 20000
    a_gpu = clrand(context, queue, (l,), np.float32)
    b_gpu = clrand(context, queue, (l,), np.float32)
    a = a_gpu.get()
    b = b_gpu.get()

    max_a_b_gpu = cl_array.maximum(a_gpu, b_gpu)
    min_a_b_gpu = cl_array.minimum(a_gpu, b_gpu)

    print(max_a_b_gpu)
    print(np.maximum(a, b))

    assert la.norm(max_a_b_gpu.get() - np.maximum(a, b)) == 0
    assert la.norm(min_a_b_gpu.get() - np.minimum(a, b)) == 0


@pytools.test.mark_test.opencl
def test_take_put(ctx_getter):
    context = ctx_getter()
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
def test_astype(ctx_getter):
    context = ctx_getter()
    queue = cl.CommandQueue(context)

    from pyopencl.clrandom import rand as clrand

    if not has_double_support(context.devices[0]):
        return

    a_gpu = clrand(context, queue, (2000,), dtype=np.float32)

    a = a_gpu.get().astype(np.float64)
    a2 = a_gpu.astype(np.float64).get()

    assert a2.dtype == np.float64
    assert la.norm(a - a2) == 0, (a, a2)

    a_gpu = clrand(context, queue, (2000,), dtype=np.float64)

    a = a_gpu.get().astype(np.float32)
    a2 = a_gpu.astype(np.float32).get()

    assert a2.dtype == np.float32
    assert la.norm(a - a2) / la.norm(a) < 1e-7


@pytools.test.mark_test.opencl
def test_scan(ctx_getter):
    context = ctx_getter()
    queue = cl.CommandQueue(context)

    from pyopencl.scan import InclusiveScanKernel, ExclusiveScanKernel

    dtype = np.int32
    for cls in [InclusiveScanKernel, ExclusiveScanKernel]:
        knl = cls(context, dtype, "a+b", "0")

        for n in [
            10, 2 ** 10 - 5, 2 ** 10,
            2 ** 20 - 2 ** 18,
            2 ** 20 - 2 ** 18 + 5,
            2 ** 10 + 5,
            2 ** 20 + 5,
            2 ** 20, 2 ** 24]:
            host_data = np.random.randint(0, 10, n).astype(dtype)
            dev_data = cl_array.to_device(queue, host_data)

            knl(dev_data)

            desired_result = np.cumsum(host_data, axis=0)
            if cls is ExclusiveScanKernel:
                desired_result -= host_data

            assert (dev_data.get() == desired_result).all()
            from gc import collect
            collect()


@pytools.test.mark_test.opencl
def test_stride_preservation(ctx_getter):
    context = ctx_getter()
    queue = cl.CommandQueue(context)

    A = np.random.rand(3, 3)
    AT = A.T
    print(AT.flags.f_contiguous, AT.flags.c_contiguous)
    AT_GPU = cl_array.to_device(queue, AT)
    print(AT_GPU.flags.f_contiguous, AT_GPU.flags.c_contiguous)
    assert np.allclose(AT_GPU.get(), AT)


@pytools.test.mark_test.opencl
def test_vector_fill(ctx_getter):
    context = ctx_getter()
    queue = cl.CommandQueue(context)

    a_gpu = cl_array.Array(queue, 100, dtype=cl_array.vec.float4)
    a_gpu.fill(cl_array.vec.make_float4(0.0, 0.0, 1.0, 0.0))
    a = a_gpu.get()
    assert a.dtype is cl_array.vec.float4

    a_gpu = cl_array.zeros(queue, 100, dtype=cl_array.vec.float4)


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
