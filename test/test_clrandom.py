__copyright__ = "Copyright (C) 2018 Matt Wala"

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
import pytest

import pyopencl as cl
import pyopencl.cltypes as cltypes
import pyopencl.clrandom as clrandom
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


def make_ranlux_generator(cl_ctx):
    queue = cl.CommandQueue(cl_ctx)
    return clrandom.RanluxGenerator(queue)


@pytest.mark.parametrize("rng_class", [
    make_ranlux_generator,
    clrandom.PhiloxGenerator,
    clrandom.ThreefryGenerator])
@pytest.mark.parametrize("dtype", [
    np.int32,
    np.int64,
    np.float32,
    np.float64,
    cltypes.float2,
    cltypes.float3,
    cltypes.float4])
def test_clrandom_dtypes(ctx_factory, rng_class, dtype):
    cl_ctx = ctx_factory()
    if dtype == np.float64 and not has_double_support(cl_ctx.devices[0]):
        pytest.skip("double precision not supported on this device")
    rng = rng_class(cl_ctx)

    size = 10

    with cl.CommandQueue(cl_ctx) as queue:
        device = queue.device
        if device.platform.vendor == "The pocl project" \
                and device.type & cl.device_type.GPU \
                and rng_class is make_ranlux_generator:
            pytest.xfail("ranlux test fails on POCL + Nvidia,"
                    "at least the K40, as of pocl 1.6, 2021-01-20")

        rng.uniform(queue, size, dtype)

        if dtype not in (np.int32, np.int64):
            rng.normal(queue, size, dtype)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
