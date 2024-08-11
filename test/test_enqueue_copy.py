#! /usr/bin/env python

__copyright__ = "Copyright (C) 2016 Shane J. Latham"

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
import pytest

import pyopencl as cl
from pyopencl.characterize import get_pocl_version
from pyopencl.tools import (
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,  # noqa: F401
)


def generate_slice(start, shape):
    return tuple(slice(start[i], start[i]+shape[i]) for i in range(len(start)))


def test_enqueue_copy_rect_2d(ctx_factory, honor_skip=True):
    """
    Test 2D sub-array (slice) copy.
    """
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    if (honor_skip
            and ctx.devices[0].platform.name == "Portable Computing Language"
            and get_pocl_version(ctx.devices[0].platform) <= (0, 13)):
        # https://github.com/pocl/pocl/issues/353
        pytest.skip("PoCL's rectangular copies crash")

    device = queue.device
    if device.platform.vendor == "The pocl project" \
            and device.type & cl.device_type.GPU:
        pytest.xfail("rect copies fail on PoCL + Nvidia,"
                "at least the K40, as of PoCL 1.6, 2021-01-20")

    if honor_skip and queue.device.platform.name == "Apple":
        pytest.xfail("Apple's CL implementation crashes on this.")

    ary_in_shp = 256, 128  # Entire array shape from which sub-array copied to device
    sub_ary_shp = 128, 96  # Sub-array shape to be copied to device
    ary_in_origin = 20, 13  # Sub-array origin
    ary_in_slice = generate_slice(ary_in_origin, sub_ary_shp)

    ary_out_origin = 11, 19  # Origin of sub-array copy from device to host-array
    ary_out_shp = 512, 256  # Entire host-array shape copy sub-array device->host
    ary_out_slice = generate_slice(ary_out_origin, sub_ary_shp)

    buf_in_origin = 7, 3  # Origin of sub-array in device buffer
    buf_in_shp = 300, 200  # shape of device buffer

    buf_out_origin = 31, 17  # Origin of 2nd device buffer
    buf_out_shp = 300, 400  # shape of 2nd device buffer

    # Create host array of random values.
    rng = np.random.default_rng(seed=42)
    h_ary_in = rng.integers(0, 256, ary_in_shp, dtype=np.uint8)

    # Create device buffers
    d_in_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, size=np.prod(buf_in_shp))
    d_out_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, size=np.prod(buf_out_shp))

    # Copy sub-array (rectangular buffer) from host to device
    cl.enqueue_copy(
        queue,
        d_in_buf,
        h_ary_in,
        buffer_origin=buf_in_origin[::-1],
        host_origin=ary_in_origin[::-1],
        region=sub_ary_shp[::-1],
        buffer_pitches=(buf_in_shp[-1],),
        host_pitches=(ary_in_shp[-1],)
    )
    # Copy sub-array (rectangular buffer) from device-buffer to device-buffer
    cl.enqueue_copy(
        queue,
        d_out_buf,
        d_in_buf,
        src_origin=buf_in_origin[::-1],
        dst_origin=buf_out_origin[::-1],
        region=sub_ary_shp[::-1],
        src_pitches=(buf_in_shp[-1],),
        dst_pitches=(buf_out_shp[-1],)
    )

    # Create zero-initialised array to receive sub-array from device
    h_ary_out = np.zeros(ary_out_shp, dtype=h_ary_in.dtype)

    # Copy sub-array (rectangular buffer) from device to host-array.
    cl.enqueue_copy(
        queue,
        h_ary_out,
        d_out_buf,
        buffer_origin=buf_out_origin[::-1],
        host_origin=ary_out_origin[::-1],
        region=sub_ary_shp[::-1],
        buffer_pitches=(buf_out_shp[-1],),
        host_pitches=(ary_out_shp[-1],)
    )
    queue.finish()

    # Check that the sub-array copied to device is
    # the same as the sub-array received from device.
    assert np.all(h_ary_in[ary_in_slice] == h_ary_out[ary_out_slice])


def test_enqueue_copy_rect_3d(ctx_factory, honor_skip=True):
    """
    Test 3D sub-array (slice) copy.
    """
    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    if (honor_skip
            and ctx.devices[0].platform.name == "Portable Computing Language"
            and get_pocl_version(ctx.devices[0].platform) <= (0, 13)):
        # https://github.com/pocl/pocl/issues/353
        pytest.skip("PoCL's rectangular copies crash")

    device = queue.device
    if device.platform.vendor == "The pocl project" \
            and device.type & cl.device_type.GPU:
        pytest.xfail("rect copies fail on PoCL + Nvidia,"
                "at least the K40, as of PoCL 1.6, 2021-01-20")

    if honor_skip and queue.device.platform.name == "Apple":
        pytest.skip("Apple's CL implementation crashes on this.")

    ary_in_shp = 256, 128, 31  # array shape from which sub-array copied to device
    sub_ary_shp = 128, 96, 20  # Sub-array shape to be copied to device
    ary_in_origin = 20, 13, 7  # Sub-array origin
    ary_in_slice = generate_slice(ary_in_origin, sub_ary_shp)

    ary_out_origin = 11, 19, 14  # Origin of sub-array copy from device to host-array
    ary_out_shp = 192, 256, 128  # Entire host-array shape copy sub-array dev->host
    ary_out_slice = generate_slice(ary_out_origin, sub_ary_shp)

    buf_in_origin = 7, 3, 6  # Origin of sub-array in device buffer
    buf_in_shp = 300, 200, 30  # shape of device buffer

    buf_out_origin = 31, 17, 3  # Origin of 2nd device buffer
    buf_out_shp = 300, 400, 40  # shape of 2nd device buffer

    # Create host array of random values.
    rng = np.random.default_rng(seed=42)
    h_ary_in = rng.integers(0, 256, ary_in_shp, dtype=np.uint8)

    # Create device buffers
    d_in_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, size=np.prod(buf_in_shp))
    d_out_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, size=np.prod(buf_out_shp))

    # Copy sub-array (rectangular buffer) from host to device
    cl.enqueue_copy(
        queue,
        d_in_buf,
        h_ary_in,
        buffer_origin=buf_in_origin[::-1],
        host_origin=ary_in_origin[::-1],
        region=sub_ary_shp[::-1],
        buffer_pitches=(buf_in_shp[-1], buf_in_shp[-1]*buf_in_shp[-2]),
        host_pitches=(ary_in_shp[-1], ary_in_shp[-1]*ary_in_shp[-2])
    )
    # Copy sub-array (rectangular buffer) from device-buffer to device-buffer
    cl.enqueue_copy(
        queue,
        d_out_buf,
        d_in_buf,
        src_origin=buf_in_origin[::-1],
        dst_origin=buf_out_origin[::-1],
        region=sub_ary_shp[::-1],
        src_pitches=(buf_in_shp[-1], buf_in_shp[-1]*buf_in_shp[-2]),
        dst_pitches=(buf_out_shp[-1], buf_out_shp[-1]*buf_out_shp[-2])
    )

    # Create zero-initialised array to receive sub-array from device
    h_ary_out = np.zeros(ary_out_shp, dtype=h_ary_in.dtype)

    # Copy sub-array (rectangular buffer) from device to host-array.
    cl.enqueue_copy(
        queue,
        h_ary_out,
        d_out_buf,
        buffer_origin=buf_out_origin[::-1],
        host_origin=ary_out_origin[::-1],
        region=sub_ary_shp[::-1],
        buffer_pitches=(buf_out_shp[-1], buf_out_shp[-1]*buf_out_shp[-2]),
        host_pitches=(ary_out_shp[-1], ary_out_shp[-1]*ary_out_shp[-2])
    )
    queue.finish()

    # Check that the sub-array copied to device is
    # the same as the sub-array received from device.
    assert np.array_equal(h_ary_in[ary_in_slice], h_ary_out[ary_out_slice])


def test_enqueue_copy_buffer_p2p_amd(honor_skip=True):
    platform = cl.get_platforms()[0]
    if honor_skip and platform.vendor != "Advanced Micro Devices, Inc.":
        pytest.skip("AMD-specific test")

    devices = platform.get_devices()
    if len(devices) < 2:
        pytest.skip("Need at least two devices")

    ctx1 = cl.Context([devices[0]])
    ctx2 = cl.Context([devices[1]])

    queue1 = cl.CommandQueue(ctx1)
    queue2 = cl.CommandQueue(ctx2)

    ary_shp = 256, 128, 32  # array shape

    # Create host array of random values.
    rng = np.random.default_rng(seed=42)
    h_ary = rng.integers(0, 256, ary_shp, dtype=np.uint8)

    # Create device buffers
    d_buf1 = cl.Buffer(ctx1, cl.mem_flags.READ_WRITE, size=np.prod(ary_shp))
    d_buf2 = cl.Buffer(ctx2, cl.mem_flags.READ_WRITE, size=np.prod(ary_shp))

    # Copy array from host to device
    cl.enqueue_copy(queue1, d_buf1, h_ary)

    # Copy array from device to device
    cl.enqueue_copy_buffer_p2p_amd(
        platform,
        queue1,
        d_buf1,
        d_buf2,
        np.prod(ary_shp)
    )
    queue1.finish()

    # Create zero-initialised array to receive array from device
    h_ary_out = np.zeros(ary_shp, dtype=h_ary.dtype)

    # Copy array from device to host
    cl.enqueue_copy(queue2, h_ary_out, d_buf2)
    queue2.finish()

    # Check that the arrays are the same
    assert np.array_equal(h_ary, h_ary_out)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
