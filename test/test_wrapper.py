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
# pylint: disable=not-callable


import numpy as np
import numpy.linalg as la
import pytest

import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.cltypes as cltypes
import pyopencl.clrandom
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests,
        ImmediateAllocator, DeferredAllocator)
from pyopencl.characterize import get_pocl_version

# Are CL implementations crashy? You be the judge. :)
try:
    import faulthandler  # noqa
except ImportError:
    pass
else:
    faulthandler.enable()


def _xfail_if_pocl(plat, up_to_version, msg="unsupported by pocl"):
    if plat.vendor == "The pocl project":
        if up_to_version is None or get_pocl_version(plat) <= up_to_version:
            pytest.xfail(msg)


def _xfail_if_pocl_gpu(device, what):
    if device.platform.vendor == "The pocl project" \
            and device.type & cl.device_type.GPU:
        pytest.xfail(f"POCL's {what} support don't work right on Nvidia GPUs, "
                "at least the Titan V, as of pocl 1.6, 2021-01-20")


def test_get_info(ctx_factory):
    ctx = ctx_factory()
    device, = ctx.devices
    platform = device.platform

    device.persistent_unique_id
    device.hashable_model_and_version_identifier

    failure_count = [0]

    pocl_quirks = [
        (cl.Buffer, cl.mem_info.OFFSET),
        (cl.Program, cl.program_info.BINARIES),
        (cl.Program, cl.program_info.BINARY_SIZES),
    ]
    if ctx._get_cl_version() >= (1, 2) and cl.get_cl_header_version() >= (1, 2):
        pocl_quirks.extend([
            (cl.Program, cl.program_info.KERNEL_NAMES),
            (cl.Program, cl.program_info.NUM_KERNELS),
        ])
    CRASH_QUIRKS = [  # noqa
            (("NVIDIA Corporation", "NVIDIA CUDA",
                "OpenCL 1.0 CUDA 3.0.1"),
                [
                    (cl.Event, cl.event_info.COMMAND_QUEUE),
                    ]),
            (("NVIDIA Corporation", "NVIDIA CUDA",
                "OpenCL 1.2 CUDA 7.5"),
                [
                    (cl.Buffer, getattr(cl.mem_info, "USES_SVM_POINTER", None)),
                    ]),
            (("The pocl project", "Portable Computing Language",
                "OpenCL 1.2 pocl 0.8-pre"),
                    pocl_quirks),
            (("The pocl project", "Portable Computing Language",
                "OpenCL 1.2 pocl 0.8"),
                pocl_quirks),
            (("The pocl project", "Portable Computing Language",
                "OpenCL 1.2 pocl 0.9-pre"),
                pocl_quirks),
            (("The pocl project", "Portable Computing Language",
                "OpenCL 1.2 pocl 0.9"),
                pocl_quirks),
            (("The pocl project", "Portable Computing Language",
                "OpenCL 1.2 pocl 0.10-pre"),
                pocl_quirks),
            (("The pocl project", "Portable Computing Language",
                "OpenCL 1.2 pocl 0.10"),
                pocl_quirks),
            (("Apple", "Apple",
                "OpenCL 1.2"),
                [
                    (cl.Program, cl.program_info.SOURCE),
                    ]),
            ]
    QUIRKS = []  # noqa

    def find_quirk(quirk_list, cl_obj, info):
        for (vendor, name, version), quirks in quirk_list:
            if (
                    vendor == platform.vendor
                    and name == platform.name
                    and platform.version.startswith(version)):
                for quirk_cls, quirk_info in quirks:
                    if (isinstance(cl_obj, quirk_cls)
                            and quirk_info == info):
                        return True

        return False

    def do_test(cl_obj, info_cls, func=None, try_attr_form=True):
        if func is None:
            func = cl_obj.get_info

        for info_name in dir(info_cls):
            if not info_name.startswith("_") and info_name != "to_string":
                print(info_cls, info_name)
                info = getattr(info_cls, info_name)

                if find_quirk(CRASH_QUIRKS, cl_obj, info):
                    print("not executing get_info", type(cl_obj), info_name)
                    print("(known crash quirk for %s)" % platform.name)
                    continue

                try:
                    func(info)
                except Exception:
                    msg = "failed get_info", type(cl_obj), info_name

                    if find_quirk(QUIRKS, cl_obj, info):
                        msg += ("(known quirk for %s)" % platform.name)
                    else:
                        failure_count[0] += 1

                if try_attr_form:
                    try:
                        getattr(cl_obj, info_name.lower())
                    except Exception:
                        print("failed attr-based get_info", type(cl_obj), info_name)

                        if find_quirk(QUIRKS, cl_obj, info):
                            print("(known quirk for %s)" % platform.name)
                        else:
                            failure_count[0] += 1

    do_test(platform, cl.platform_info)
    do_test(device, cl.device_info)
    do_test(ctx, cl.context_info)

    props = 0
    if (device.queue_properties
            & cl.command_queue_properties.PROFILING_ENABLE):
        profiling = True
        props = cl.command_queue_properties.PROFILING_ENABLE
    queue = cl.CommandQueue(ctx,
            properties=props)
    do_test(queue, cl.command_queue_info)

    prg = cl.Program(ctx, """
        __kernel void sum(__global float *a)
        { a[get_global_id(0)] *= 2; }
        """).build()
    do_test(prg, cl.program_info)
    do_test(prg, cl.program_build_info,
            lambda info: prg.get_build_info(device, info),
            try_attr_form=False)

    n = 2000
    a_buf = cl.Buffer(ctx, 0, n*4)

    do_test(a_buf, cl.mem_info)

    kernel = prg.all_kernels()[0]
    do_test(kernel, cl.kernel_info)

    for i in range(2):  # exercise cache
        for info_name in dir(cl.kernel_work_group_info):
            if not info_name.startswith("_") and info_name != "to_string":
                try:
                    print("kernel_wg_info: %s" % info_name)
                    kernel.get_work_group_info(
                            getattr(cl.kernel_work_group_info, info_name),
                            device)
                except cl.LogicError as err:
                    print("<error: %s>" % err)

    evt = kernel(queue, (n,), None, a_buf)
    do_test(evt, cl.event_info)

    if profiling:
        evt.wait()
        do_test(evt, cl.profiling_info,
                lambda info: evt.get_profiling_info(info),
                try_attr_form=False)

    # crashes on intel...
    # and pocl does not support CL_ADDRESS_CLAMP
    if device.image_support and platform.vendor not in [
            "Intel(R) Corporation",
            "The pocl project",
            ]:
        smp = cl.Sampler(ctx, False,
                cl.addressing_mode.CLAMP,
                cl.filter_mode.NEAREST)
        do_test(smp, cl.sampler_info)

        img_format = cl.get_supported_image_formats(
                ctx, cl.mem_flags.READ_ONLY, cl.mem_object_type.IMAGE2D)[0]

        img = cl.Image(ctx, cl.mem_flags.READ_ONLY, img_format, (128, 256))
        assert img.shape == (128, 256)

        img.depth
        img.image.depth
        do_test(img, cl.image_info,
                lambda info: img.get_image_info(info))


def test_int_ptr(ctx_factory):
    def do_test(obj):
        new_obj = type(obj).from_int_ptr(obj.int_ptr)
        assert obj == new_obj
        assert type(obj) is type(new_obj)

    ctx = ctx_factory()
    device, = ctx.devices
    platform = device.platform
    do_test(device)
    do_test(platform)
    do_test(ctx)

    queue = cl.CommandQueue(ctx)
    do_test(queue)

    evt = cl.enqueue_marker(queue)
    do_test(evt)

    prg = cl.Program(ctx, """
        __kernel void sum(__global float *a)
        { a[get_global_id(0)] *= 2; }
        """).build()

    do_test(prg)
    do_test(prg.sum)

    n = 2000
    a_buf = cl.Buffer(ctx, 0, n*4)
    do_test(a_buf)

    # crashes on intel...
    # and pocl does not support CL_ADDRESS_CLAMP
    if device.image_support and platform.vendor not in [
            "Intel(R) Corporation",
            "The pocl project",
            ]:
        smp = cl.Sampler(ctx, False,
                cl.addressing_mode.CLAMP,
                cl.filter_mode.NEAREST)
        do_test(smp)

        img_format = cl.get_supported_image_formats(
                ctx, cl.mem_flags.READ_ONLY, cl.mem_object_type.IMAGE2D)[0]

        img = cl.Image(ctx, cl.mem_flags.READ_ONLY, img_format, (128, 256))
        do_test(img)


def test_invalid_kernel_names_cause_failures(ctx_factory):
    ctx = ctx_factory()
    device = ctx.devices[0]
    prg = cl.Program(ctx, """
        __kernel void sum(__global float *a)
        { a[get_global_id(0)] *= 2; }
        """).build()

    try:
        prg.sam
        raise RuntimeError("invalid kernel name did not cause error")
    except AttributeError:
        pass
    except RuntimeError:
        if "Intel" in device.platform.vendor:
            from pytest import xfail
            xfail("weird exception from OpenCL implementation "
                    "on invalid kernel name--are you using "
                    "Intel's implementation? (if so, known bug in Intel CL)")
        else:
            raise


def test_image_format_constructor():
    # doesn't need image support to succeed
    iform = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)

    assert iform.channel_order == cl.channel_order.RGBA
    assert iform.channel_data_type == cl.channel_type.FLOAT

    if not cl._PYPY:
        assert not hasattr(iform, "__dict__")


def test_device_topology_amd_constructor():
    # doesn't need cl_amd_device_attribute_query support to succeed
    topol = cl.DeviceTopologyAmd(3, 4, 5)

    assert topol.bus == 3
    assert topol.device == 4
    assert topol.function == 5

    if not cl._PYPY:
        assert not hasattr(topol, "__dict__")


def test_nonempty_supported_image_formats(ctx_factory):
    context = ctx_factory()

    device = context.devices[0]

    if device.image_support:
        assert len(cl.get_supported_image_formats(
                context, cl.mem_flags.READ_ONLY, cl.mem_object_type.IMAGE2D)) > 0
    else:
        from pytest import skip
        skip("images not supported on %s" % device.name)


def test_that_python_args_fail(ctx_factory):
    context = ctx_factory()

    prg = cl.Program(context, """
        __kernel void mult(__global float *a, float b, int c)
        { a[get_global_id(0)] *= (b+c); }
        """).build()

    a = np.random.rand(50000)
    queue = cl.CommandQueue(context)
    mf = cl.mem_flags
    a_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)

    knl = cl.Kernel(prg, "mult")
    try:
        knl(queue, a.shape, None, a_buf, 2, 3)
        assert False, "PyOpenCL should not accept bare Python types as arguments"
    except cl.LogicError:
        pass

    try:
        prg.mult(queue, a.shape, None, a_buf, float(2), 3)
        assert False, "PyOpenCL should not accept bare Python types as arguments"
    except cl.LogicError:
        pass

    prg.mult(queue, a.shape, None, a_buf, np.float32(2), np.int32(3))

    a_result = np.empty_like(a)
    cl.enqueue_copy(queue, a_buf, a_result).wait()


def test_image_2d(ctx_factory):
    context = ctx_factory()

    device, = context.devices

    if not device.image_support:
        from pytest import skip
        skip("images not supported on %s" % device)

    if "Intel" in device.vendor and "31360.31426" in device.version:
        from pytest import skip
        skip("images crashy on %s" % device)
    _xfail_if_pocl(device.platform, None, "pocl does not support CL_ADDRESS_CLAMP")

    prg = cl.Program(context, """
        __kernel void copy_image(
          __global float *dest,
          __read_only image2d_t src,
          sampler_t samp,
          int stride0)
        {
          int d0 = get_global_id(0);
          int d1 = get_global_id(1);
          /*
          const sampler_t samp =
            CLK_NORMALIZED_COORDS_FALSE
            | CLK_ADDRESS_CLAMP
            | CLK_FILTER_NEAREST;
            */
          dest[d0*stride0 + d1] = read_imagef(src, samp, (float2)(d1, d0)).x;
        }
        """).build()

    num_channels = 1
    a = np.random.rand(1024, 512, num_channels).astype(np.float32)
    if num_channels == 1:
        a = a[:, :, 0]

    queue = cl.CommandQueue(context)
    try:
        a_img = cl.image_from_array(context, a, num_channels)
    except cl.RuntimeError:
        import sys
        exc = sys.exc_info()[1]
        if exc.code == cl.status_code.IMAGE_FORMAT_NOT_SUPPORTED:
            from pytest import skip
            skip("required image format not supported on %s" % device.name)
        else:
            raise

    a_dest = cl.Buffer(context, cl.mem_flags.READ_WRITE, a.nbytes)

    samp = cl.Sampler(context, False,
            cl.addressing_mode.CLAMP,
            cl.filter_mode.NEAREST)
    prg.copy_image(queue, a.shape, None, a_dest, a_img, samp,
            np.int32(a.strides[0]/a.dtype.itemsize))

    a_result = np.empty_like(a)
    cl.enqueue_copy(queue, a_result, a_dest)

    good = la.norm(a_result - a) == 0
    if not good:
        if queue.device.type & cl.device_type.CPU:
            assert good, ("The image implementation on your CPU CL platform '%s' "
                    "returned bad values. This is bad, but common."
                    % queue.device.platform)
        else:
            assert good


def test_image_3d(ctx_factory):
    #test for image_from_array for 3d image of float2
    context = ctx_factory()

    device, = context.devices

    if not device.image_support:
        from pytest import skip
        skip("images not supported on %s" % device)

    if device.platform.vendor == "Intel(R) Corporation":
        from pytest import skip
        skip("images crashy on %s" % device)
    _xfail_if_pocl(device.platform, None, "pocl does not support CL_ADDRESS_CLAMP")

    prg = cl.Program(context, """
        __kernel void copy_image_plane(
          __global float2 *dest,
          __read_only image3d_t src,
          sampler_t samp,
          int stride0,
          int stride1)
        {
          int d0 = get_global_id(0);
          int d1 = get_global_id(1);
          int d2 = get_global_id(2);
          /*
          const sampler_t samp =
            CLK_NORMALIZED_COORDS_FALSE
            | CLK_ADDRESS_CLAMP
            | CLK_FILTER_NEAREST;
            */
          dest[d0*stride0 + d1*stride1 + d2] = read_imagef(
                src, samp, (float4)(d2, d1, d0, 0)).xy;
        }
        """).build()

    num_channels = 2
    shape = (3, 4, 2)
    a = np.random.random(shape + (num_channels,)).astype(np.float32)

    queue = cl.CommandQueue(context)
    try:
        a_img = cl.image_from_array(context, a, num_channels)
    except cl.RuntimeError:
        import sys
        exc = sys.exc_info()[1]
        if exc.code == cl.status_code.IMAGE_FORMAT_NOT_SUPPORTED:
            from pytest import skip
            skip("required image format not supported on %s" % device.name)
        else:
            raise

    a_dest = cl.Buffer(context, cl.mem_flags.READ_WRITE, a.nbytes)

    samp = cl.Sampler(context, False,
            cl.addressing_mode.CLAMP,
            cl.filter_mode.NEAREST)
    prg.copy_image_plane(queue, shape, None, a_dest, a_img, samp,
                         np.int32(a.strides[0]/a.itemsize/num_channels),
                         np.int32(a.strides[1]/a.itemsize/num_channels),
                         )

    a_result = np.empty_like(a)
    cl.enqueue_copy(queue, a_result, a_dest)

    good = la.norm(a_result - a) == 0
    if not good:
        if queue.device.type & cl.device_type.CPU:
            assert good, ("The image implementation on your CPU CL platform '%s' "
                    "returned bad values. This is bad, but common."
                    % queue.device.platform)
        else:
            assert good


def test_copy_buffer(ctx_factory):
    context = ctx_factory()

    queue = cl.CommandQueue(context)
    mf = cl.mem_flags

    a = np.random.rand(50000).astype(np.float32)
    b = np.empty_like(a)

    buf1 = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    buf2 = cl.Buffer(context, mf.WRITE_ONLY, b.nbytes)

    cl.enqueue_copy(queue, buf2, buf1).wait()
    cl.enqueue_copy(queue, b, buf2).wait()

    assert la.norm(a - b) == 0


def test_mempool(ctx_factory):
    from pyopencl.tools import MemoryPool, ImmediateAllocator

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    pool = MemoryPool(ImmediateAllocator(queue))
    alloc_queue = []

    e0 = 12

    for e in range(e0-6, e0-4):
        for i in range(100):
            alloc_queue.append(pool.allocate(1 << e))
            if len(alloc_queue) > 10:
                alloc_queue.pop(0)
    del alloc_queue
    pool.stop_holding()


def test_mempool_2(ctx_factory):
    from pyopencl.tools import MemoryPool, ImmediateAllocator
    from random import randrange

    context = ctx_factory()
    queue = cl.CommandQueue(context)

    pool = MemoryPool(ImmediateAllocator(queue))

    for s in [randrange(1 << 31) >> randrange(32) for _ in range(2000)] + [2**30]:
        bin_nr = pool.bin_number(s)
        asize = pool.alloc_size(bin_nr)

        assert asize >= s, s
        assert pool.bin_number(asize) == bin_nr, s
        assert asize < asize*(1+1/8)


def test_mempool_32bit_issues():
    # https://github.com/inducer/pycuda/issues/282
    from pyopencl._cl import _TestMemoryPool
    pool = _TestMemoryPool()

    for i in [30, 31, 32, 33, 34]:
        for offs in range(-5, 5):
            pool.allocate(2**i + offs)


@pytest.mark.parametrize("allocator_cls", [ImmediateAllocator, DeferredAllocator])
def test_allocator(ctx_factory, allocator_cls):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    if allocator_cls is DeferredAllocator:
        allocator = allocator_cls(context)
    else:
        allocator = allocator_cls(queue)

    mem = allocator(15)
    mem2 = allocator(0)

    assert mem is not None
    assert mem2 is None


def test_vector_args(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    prg = cl.Program(context, """
        __kernel void set_vec(float4 x, __global float4 *dest)
        { dest[get_global_id(0)] = x; }
        """).build()

    x = cltypes.make_float4(1, 2, 3, 4)
    dest = np.empty(50000, cltypes.float4)
    mf = cl.mem_flags
    dest_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=dest)

    prg.set_vec(queue, dest.shape, None, x, dest_buf)

    cl.enqueue_copy(queue, dest, dest_buf).wait()

    assert (dest == x).all()


def test_header_dep_handling(ctx_factory):
    context = ctx_factory()

    from os.path import exists
    assert exists("empty-header.h")  # if this fails, change dir to pyopencl/test

    kernel_src = """
    #include <empty-header.h>
    kernel void zonk(global int *a)
    {
      *a = 5;
    }
    """

    import os

    cl.Program(context, kernel_src).build(["-I", os.getcwd()])
    cl.Program(context, kernel_src).build(["-I", os.getcwd()])


def test_context_dep_memoize(ctx_factory):
    context = ctx_factory()

    from pyopencl.tools import context_dependent_memoize

    counter = [0]

    @context_dependent_memoize
    def do_something(ctx):
        counter[0] += 1

    do_something(context)
    do_something(context)

    assert counter[0] == 1


def test_can_build_and_run_binary(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    device = queue.device

    program = cl.Program(ctx, """
    __kernel void simple(__global float *in, __global float *out)
    {
        out[get_global_id(0)] = in[get_global_id(0)];
    }""")
    program.build()
    binary = program.get_info(cl.program_info.BINARIES)[0]

    foo = cl.Program(ctx, [device], [binary])
    foo.build()

    n = 256
    a_dev = cl.clrandom.rand(queue, n, np.float32)
    dest_dev = cl_array.empty_like(a_dev)

    foo.simple(queue, (n,), (16,), a_dev.data, dest_dev.data)


def test_enqueue_barrier_marker(ctx_factory):
    ctx = ctx_factory()
    # Still relevant on pocl 1.0RC1.
    _xfail_if_pocl(
            ctx.devices[0].platform, (1, 0), "pocl crashes on enqueue_barrier")

    queue = cl.CommandQueue(ctx)

    if queue._get_cl_version() >= (1, 2) and cl.get_cl_header_version() <= (1, 1):
        pytest.skip("CL impl version >= 1.2, header version <= 1.1--cannot be sure "
                "that clEnqueueWaitForEvents is implemented")

    cl.enqueue_barrier(queue)
    evt1 = cl.enqueue_marker(queue)
    evt2 = cl.enqueue_marker(queue, wait_for=[evt1])
    cl.enqueue_barrier(queue, wait_for=[evt1, evt2])


def test_wait_for_events(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)
    evt1 = cl.enqueue_marker(queue)
    evt2 = cl.enqueue_marker(queue)
    cl.wait_for_events([evt1, evt2])


def test_unload_compiler(platform):
    if (platform._get_cl_version() < (1, 2)
            or cl.get_cl_header_version() < (1, 2)):
        from pytest import skip
        skip("clUnloadPlatformCompiler is only available in OpenCL 1.2")
    _xfail_if_pocl(platform, (0, 13), "pocl does not support unloading compiler")
    if platform.vendor == "Intel(R) Corporation":
        from pytest import skip
        skip("Intel proprietary driver does not support unloading compiler")
    cl.unload_platform_compiler(platform)


def test_platform_get_devices(ctx_factory):
    ctx = ctx_factory()
    platform = ctx.devices[0].platform

    if platform.name == "Apple":
        pytest.xfail("Apple doesn't understand all the values we pass "
                "for dev_type")

    dev_types = [cl.device_type.ACCELERATOR, cl.device_type.ALL,
                 cl.device_type.CPU, cl.device_type.DEFAULT, cl.device_type.GPU]
    if (platform._get_cl_version() >= (1, 2)
            and cl.get_cl_header_version() >= (1, 2)
            and not platform.name.lower().startswith("nvidia")):
        dev_types.append(cl.device_type.CUSTOM)

    for dev_type in dev_types:
        print(dev_type)
        devs = platform.get_devices(dev_type)
        if dev_type in (cl.device_type.DEFAULT,
                        cl.device_type.ALL,
                        getattr(cl.device_type, "CUSTOM", None)):
            continue
        for dev in devs:
            assert dev.type & dev_type == dev_type


def test_user_event(ctx_factory):
    ctx = ctx_factory()
    if (ctx._get_cl_version() < (1, 1)
            and cl.get_cl_header_version() < (1, 1)):
        from pytest import skip
        skip("UserEvent is only available in OpenCL 1.1")

    # https://github.com/pocl/pocl/issues/201
    _xfail_if_pocl(ctx.devices[0].platform, (0, 13),
            "pocl's user events don't work right")

    status = {}

    def event_waiter1(e, key):
        e.wait()
        status[key] = True

    def event_waiter2(e, key):
        cl.wait_for_events([e])
        status[key] = True

    from threading import Thread
    from time import sleep
    evt = cl.UserEvent(ctx)
    Thread(target=event_waiter1, args=(evt, 1)).start()
    sleep(.05)
    if status.get(1, False):
        raise RuntimeError("UserEvent triggered before set_status")
    evt.set_status(cl.command_execution_status.COMPLETE)
    sleep(.05)
    if not status.get(1, False):
        raise RuntimeError("UserEvent.wait timeout")
    assert evt.command_execution_status == cl.command_execution_status.COMPLETE

    evt = cl.UserEvent(ctx)
    Thread(target=event_waiter2, args=(evt, 2)).start()
    sleep(.05)
    if status.get(2, False):
        raise RuntimeError("UserEvent triggered before set_status")
    evt.set_status(cl.command_execution_status.COMPLETE)
    sleep(.05)
    if not status.get(2, False):
        raise RuntimeError("cl.wait_for_events timeout on UserEvent")
    assert evt.command_execution_status == cl.command_execution_status.COMPLETE


def test_buffer_get_host_array(ctx_factory):
    if cl._PYPY:
        # FIXME
        pytest.xfail("Buffer.get_host_array not yet working on pypy")

    ctx = ctx_factory()
    mf = cl.mem_flags

    host_buf = np.random.rand(25).astype(np.float32)
    buf = cl.Buffer(ctx, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=host_buf)
    host_buf2 = buf.get_host_array(25, np.float32)
    assert (host_buf == host_buf2).all()
    assert (host_buf.__array_interface__["data"][0]
            == host_buf.__array_interface__["data"][0])
    assert host_buf2.base is buf

    buf = cl.Buffer(ctx, mf.READ_WRITE | mf.ALLOC_HOST_PTR, size=100)
    try:
        host_buf2 = buf.get_host_array(25, np.float32)
        assert False, ("MemoryObject.get_host_array should not accept buffer "
                       "without USE_HOST_PTR")
    except cl.LogicError:
        pass

    host_buf = np.random.rand(25).astype(np.float32)
    buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host_buf)
    try:
        host_buf2 = buf.get_host_array(25, np.float32)
        assert False, ("MemoryObject.get_host_array should not accept buffer "
                       "without USE_HOST_PTR")
    except cl.LogicError:
        pass


def test_program_valued_get_info(ctx_factory):
    ctx = ctx_factory()

    prg = cl.Program(ctx, """
    __kernel void
    reverse(__global float *out)
    {
        out[get_global_id(0)] *= 2;
    }
    """).build()

    knl = prg.reverse

    assert knl.program == prg
    knl.program.binaries[0]


def test_event_set_callback(ctx_factory):
    import sys
    if sys.platform.startswith("win"):
        pytest.xfail("Event.set_callback not present on Windows")

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    _xfail_if_pocl_gpu(queue.device, "event callbacks")

    if ctx._get_cl_version() < (1, 1):
        pytest.skip("OpenCL 1.1 or newer required for set_callback")

    a_np = np.random.rand(50000).astype(np.float32)
    b_np = np.random.rand(50000).astype(np.float32)

    got_called = []

    def cb(status):
        got_called.append(status)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

    prg = cl.Program(ctx, """
    __kernel void sum(__global const float *a_g, __global const float *b_g,
        __global float *res_g) {
      int gid = get_global_id(0);
      res_g[gid] = a_g[gid] + b_g[gid];
    }
    """).build()

    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)

    uevt = cl.UserEvent(ctx)

    evt = prg.sum(queue, a_np.shape, None, a_g, b_g, res_g, wait_for=[uevt])

    evt.set_callback(cl.command_execution_status.COMPLETE, cb)

    uevt.set_status(cl.command_execution_status.COMPLETE)

    queue.finish()

    counter = 0

    # yuck
    while not got_called:
        from time import sleep
        sleep(0.01)

        # wait up to five seconds (?!)
        counter += 1
        if counter >= 500:
            break

    assert got_called


def test_global_offset(ctx_factory):
    context = ctx_factory()
    queue = cl.CommandQueue(context)

    _xfail_if_pocl_gpu(queue.device, "global offset")

    prg = cl.Program(context, """
        __kernel void mult(__global float *a)
        { a[get_global_id(0)] *= 2; }
        """).build()

    n = 50
    a = np.random.rand(n).astype(np.float32)

    queue = cl.CommandQueue(context)
    mf = cl.mem_flags
    a_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)

    step = 10
    for ofs in range(0, n, step):
        prg.mult(queue, (step,), None, a_buf, global_offset=(ofs,))

    a_2 = np.empty_like(a)
    cl.enqueue_copy(queue, a_2, a_buf)

    assert (a_2 == 2*a).all()


def test_sub_buffers(ctx_factory):
    ctx = ctx_factory()
    if (ctx._get_cl_version() < (1, 1)
            or cl.get_cl_header_version() < (1, 1)):
        from pytest import skip
        skip("sub-buffers are only available in OpenCL 1.1")

    alignment = ctx.devices[0].mem_base_addr_align

    queue = cl.CommandQueue(ctx)

    n = 30000
    a = (np.random.rand(n) * 100).astype(np.uint8)

    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)

    start = (5000 // alignment) * alignment
    stop = start + 20 * alignment

    a_sub_ref = a[start:stop]

    a_sub = np.empty_like(a_sub_ref)
    cl.enqueue_copy(queue, a_sub, a_buf[start:stop])

    assert np.array_equal(a_sub, a_sub_ref)


def test_spirv(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    if (ctx._get_cl_version() < (2, 1)
            or cl.get_cl_header_version() < (2, 1)):
        pytest.skip("SPIR-V program creation only available "
                "in OpenCL 2.1 and higher")

    if queue.device.platform.name == "Portable Computing Language":
        # I'm not sure this is universal, but pocl 1.7 seems to use it.
        if "cl_khr_spirv" not in queue.device.extensions.split():
            pytest.skip("SPIR-V program creation not supported by device")

    n = 50000

    a_dev = cl.clrandom.rand(queue, n, np.float32)
    b_dev = cl.clrandom.rand(queue, n, np.float32)
    dest_dev = cl_array.empty_like(a_dev)

    with open("add-vectors-%d.spv" % queue.device.address_bits, "rb") as spv_file:
        spv = spv_file.read()

    prg = cl.Program(ctx, spv).build()
    if (not prg.all_kernels()
            and queue.device.platform.name.startswith("AMD Accelerated")):
        pytest.skip("SPIR-V program creation on AMD did not result in any kernels")

    prg.sum(queue, a_dev.shape, None, a_dev.data, b_dev.data, dest_dev.data)

    assert la.norm((dest_dev - (a_dev+b_dev)).get()) < 1e-7


def test_coarse_grain_svm(ctx_factory):
    import sys
    is_pypy = "__pypy__" in sys.builtin_module_names

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    _xfail_if_pocl_gpu(queue.device, "SVM")

    dev = ctx.devices[0]

    from pyopencl.characterize import has_coarse_grain_buffer_svm
    from pytest import skip
    if not has_coarse_grain_buffer_svm(queue.device):
        skip("device does not support coarse-grain SVM")

    if ("AMD" in dev.platform.name
            and dev.type & cl.device_type.CPU):
        pytest.xfail("AMD CPU doesn't do coarse-grain SVM")
    if ("AMD" in dev.platform.name
            and dev.type & cl.device_type.GPU):
        pytest.xfail("AMD GPU crashes on SVM unmap")

    n = 3000
    svm_ary = cl.SVM(cl.csvm_empty(ctx, (n,), np.float32, alignment=64))
    if not is_pypy:
        # https://bitbucket.org/pypy/numpy/issues/52
        assert isinstance(svm_ary.mem.base, cl.SVMAllocation)

    cl.enqueue_svm_memfill(queue, svm_ary, np.zeros((), svm_ary.mem.dtype))

    with svm_ary.map_rw(queue) as ary:
        ary.fill(17)
        orig_ary = ary.copy()

    prg = cl.Program(ctx, """
        __kernel void twice(__global float *a_g)
        {
          a_g[get_global_id(0)] *= 2;
        }
        """).build()

    prg.twice(queue, svm_ary.mem.shape, None, svm_ary)

    with svm_ary.map_ro(queue) as ary:
        print(ary)
        assert np.array_equal(orig_ary*2, ary)

    new_ary = np.empty_like(orig_ary)
    new_ary.fill(-1)

    if ctx.devices[0].platform.name != "Portable Computing Language":
        # "Blocking memcpy is unimplemented (clEnqueueSVMMemcpy.c:61)"
        # in pocl up to and including 1.0rc1.

        cl.enqueue_copy(queue, new_ary, svm_ary)
        assert np.array_equal(orig_ary*2, new_ary)

    # {{{ https://github.com/inducer/pyopencl/issues/372

    buf_arr = cl.svm_empty(ctx, cl.svm_mem_flags.READ_ONLY, 10, np.int32)
    out_arr = cl.svm_empty(ctx, cl.svm_mem_flags.READ_WRITE, 10, np.int32)

    svm_buf_arr = cl.SVM(buf_arr)
    svm_out_arr = cl.SVM(out_arr)
    with svm_buf_arr.map_rw(queue) as ary:
        ary.fill(17)

    prg_ro = cl.Program(ctx, r"""
        __kernel void twice_ro(__global int *out_g, __global int *in_g)
        {
          out_g[get_global_id(0)] = 2*in_g[get_global_id(0)];
        }
        """).build()

    prg_ro.twice_ro(queue, buf_arr.shape, None, svm_out_arr, svm_buf_arr)

    with svm_out_arr.map_ro(queue) as ary:
        print(ary)

    # }}}


def test_fine_grain_svm(ctx_factory):
    import sys
    is_pypy = "__pypy__" in sys.builtin_module_names

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    _xfail_if_pocl_gpu(queue.device, "GPU SVM")

    from pyopencl.characterize import has_fine_grain_buffer_svm
    from pytest import skip
    if not has_fine_grain_buffer_svm(queue.device):
        skip("device does not support fine-grain SVM")

    n = 3000
    ary = cl.fsvm_empty(ctx, n, np.float32, alignment=64)

    if not is_pypy:
        # https://bitbucket.org/pypy/numpy/issues/52
        assert isinstance(ary.base, cl.SVMAllocation)

    ary.fill(17)
    orig_ary = ary.copy()

    prg = cl.Program(ctx, """
        __kernel void twice(__global float *a_g)
        {
          a_g[get_global_id(0)] *= 2;
        }
        """).build()

    prg.twice(queue, ary.shape, None, cl.SVM(ary))
    queue.finish()

    print(ary)
    assert np.array_equal(orig_ary*2, ary)


@pytest.mark.parametrize("dtype", [
    np.uint,
    cltypes.uint2,
    ])
def test_map_dtype(ctx_factory, dtype):
    if cl._PYPY:
        # FIXME
        pytest.xfail("enqueue_map_buffer not yet working on pypy")

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    dt = np.dtype(dtype)

    b = pyopencl.Buffer(ctx,
                        pyopencl.mem_flags.READ_ONLY,
                        dt.itemsize)
    array, ev = pyopencl.enqueue_map_buffer(queue, b, pyopencl.map_flags.WRITE, 0,
                                            (1,), dt)
    with array.base:
        print(array.dtype)
        assert array.dtype == dt


def test_compile_link(ctx_factory):
    ctx = ctx_factory()

    if ctx._get_cl_version() < (1, 2) or cl.get_cl_header_version() < (1, 2):
        pytest.skip("Context and ICD loader must understand CL1.2 for compile/link")

    platform = ctx.devices[0].platform
    if platform.name == "Apple":
        pytest.skip("Apple doesn't like our compile/link test")

    queue = cl.CommandQueue(ctx)
    vsink_prg = cl.Program(ctx, """//CL//
        void value_sink(float x)
        {
        }
        """).compile()
    pi_h__prg = cl.Program(ctx, """//CL//
        inline float get_pi()
        {
            return 3.1415f;
        }
        """).compile()
    main_prg = cl.Program(ctx, """//CL//
        #include "pi.h"

        void value_sink(float x);

        __kernel void experiment()
        {
            value_sink(get_pi() + get_global_id(0));
        }
        """).compile(headers=[("pi.h", pi_h__prg)])
    z = cl.link_program(ctx, [vsink_prg, main_prg], devices=ctx.devices)
    z.experiment(queue, (128**2,), (128,))
    queue.finish()


def test_copy_buffer_rect(ctx_factory):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    _xfail_if_pocl_gpu(queue.device, "rectangular copies")

    arr1 = cl_array.zeros(queue, (2, 3), "f")
    arr2 = cl_array.zeros(queue, (4, 5), "f")
    arr1.fill(1)
    cl.enqueue_copy(
            queue, arr2.data, arr1.data,
            src_origin=(0, 0), dst_origin=(1, 1),
            region=arr1.shape[::-1])


def test_threaded_nanny_events(ctx_factory):
    # https://github.com/inducer/pyopencl/issues/296

    import gc
    import threading

    def create_arrays_thread(n1=10, n2=20):
        ctx = ctx_factory()
        queue = cl.CommandQueue(ctx)
        for i1 in range(n2):
            for i in range(n1):
                acl = cl.array.zeros(queue, 10, dtype=np.float32)
                acl.get()
            # Garbage collection triggers the error
            print("collected ", str(gc.collect()))
            print("stats ", gc.get_stats())

    t1 = threading.Thread(target=create_arrays_thread)
    t2 = threading.Thread(target=create_arrays_thread)

    t1.start()
    t2.start()

    t1.join()
    t2.join()


@pytest.mark.parametrize("empty_shape", [(0,), (3, 0, 2)])
def test_empty_ndrange(ctx_factory, empty_shape):
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    if ctx._get_cl_version() < (1, 2) or cl.get_cl_header_version() < (1, 2):
        pytest.skip("OpenCL 1.2 required for empty NDRange suuport")

    a = cl_array.zeros(queue, empty_shape, dtype=np.float32)

    prg = cl.Program(ctx, """
        __kernel void add_two(__global float *a_g)
        {
          a_g[get_global_id(0)] += 2;
        }
        """).build()

    prg.add_two(queue, a.shape, None, a.data, allow_empty_ndrange=True)


if __name__ == "__main__":
    # make sure that import failures get reported, instead of skipping the tests.
    import pyopencl  # noqa

    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
