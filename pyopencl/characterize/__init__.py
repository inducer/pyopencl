from __future__ import annotations


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


from pytools import memoize

import pyopencl as cl


class CLCharacterizationWarning(UserWarning):
    pass


@memoize
def has_double_support(dev: cl.Device):
    for ext in dev.extensions.split(" "):
        if ext == "cl_khr_fp64":
            return True
    return False


def has_amd_double_support(dev: cl.Device):
    """"Fix to allow incomplete amd double support in low end boards"""

    for ext in dev.extensions.split(" "):
        if ext == "cl_amd_fp64":
            return True
    return False


def reasonable_work_group_size_multiple(
            dev: cl.Device,
            ctx: cl.Context | None = None
        ):
    try:
        return dev.warp_size_nv
    except Exception:
        pass

    if ctx is None:
        ctx = cl.Context([dev])
    prg = cl.Program(ctx, """
        __kernel void knl(__global float *a)
        {
            a[get_global_id(0)] = 0;
        }
        """)
    prg.build()
    return prg.knl.get_work_group_info(
            cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
            dev)


def nv_compute_capability(dev: cl.Device):
    """If *dev* is an Nvidia GPU :class:`pyopencl.Device`, return a tuple
    *(major, minor)* indicating the device's compute capability.
    """

    try:
        return (dev.compute_capability_major_nv,
                dev.compute_capability_minor_nv)
    except Exception:
        return None


def usable_local_mem_size(dev: cl.Device, nargs: int | None = None):
    """Return an estimate of the usable local memory size.
    :arg nargs: Number of 32-bit arguments passed.
    """

    usable_local_mem_size = dev.local_mem_size

    nv_compute_cap = nv_compute_capability(dev)

    if (nv_compute_cap is not None
            and nv_compute_cap < (2, 0)):
        # pre-Fermi use local mem for parameter passing
        if nargs is None:
            # assume maximum
            usable_local_mem_size -= 256
        else:
            usable_local_mem_size -= 4*nargs

    return usable_local_mem_size


def simultaneous_work_items_on_local_access(dev: cl.Device):
    """Return the number of work items that access local
    memory simultaneously and thereby may conflict with
    each other.
    """
    nv_compute_cap = nv_compute_capability(dev)

    if nv_compute_cap is not None:
        if nv_compute_cap < (2, 0):
            return 16
        else:
            if nv_compute_cap >= (3, 0):
                from warnings import warn
                warn(
                    f"Wildly guessing conflicting local access size on '{dev}'",
                    CLCharacterizationWarning, stacklevel=2)

            return 32

    if dev.type & cl.device_type.GPU:
        from warnings import warn
        warn(
            f"Wildly guessing conflicting local access size on '{dev}'",
            CLCharacterizationWarning, stacklevel=2)
        return 16
    elif dev.type & cl.device_type.CPU:
        return 1
    else:
        from warnings import warn
        warn(
            f"Wildly guessing conflicting local access size on '{dev}'",
            CLCharacterizationWarning, stacklevel=2)
        return 16


def local_memory_access_granularity(dev: cl.Device):
    """Return the number of bytes per bank in local memory."""
    return 4


def local_memory_bank_count(dev: cl.Device):
    """Return the number of banks present in local memory.
    """
    nv_compute_cap = nv_compute_capability(dev)

    if nv_compute_cap is not None:
        if nv_compute_cap < (2, 0):
            return 16
        else:
            if nv_compute_cap >= (3, 0):
                from warnings import warn
                warn(
                    f"Wildly guessing local memory bank count on '{dev}'",
                    CLCharacterizationWarning, stacklevel=2)

            return 32

    if dev.type & cl.device_type.GPU:
        from warnings import warn
        warn(
            f"Wildly guessing local memory bank count on '{dev}'",
            CLCharacterizationWarning, stacklevel=2)
        return 16
    elif dev.type & cl.device_type.CPU:
        if dev.local_mem_type == cl.device_local_mem_type.GLOBAL:
            raise RuntimeError("asking for a bank count is "
                    "meaningless for cache-based lmem")

    from warnings import warn
    warn(
        f"Wildly guessing conflicting local access size on '{dev}'",
        CLCharacterizationWarning, stacklevel=2)
    return 16


def why_not_local_access_conflict_free(dev, itemsize,
        array_shape, array_stored_shape=None):
    """
    :param itemsize: size of accessed data in bytes
    :param array_shape: array dimensions, fastest-moving last
        (C order)

    :returns: a tuple (multiplicity, explanation), where *multiplicity*
        is the number of work items that will conflict on a bank when accessing
        local memory. *explanation* is a string detailing the found conflict.
    """
    # FIXME: Treat 64-bit access on NV CC 2.x + correctly

    if array_stored_shape is None:
        array_stored_shape = array_shape

    rank = len(array_shape)

    array_shape = array_shape[::-1]
    array_stored_shape = array_stored_shape[::-1]

    gran = local_memory_access_granularity(dev)
    if itemsize != gran:
        from warnings import warn
        warn(
            f"Local conflict info might be inaccurate for itemsize != {gran}",
            CLCharacterizationWarning, stacklevel=2)

    sim_wi = simultaneous_work_items_on_local_access(dev)
    bank_count = local_memory_bank_count(dev)

    conflicts = []

    for work_item_axis in range(rank):

        bank_accesses = {}
        for work_item_id in range(sim_wi):
            addr = 0
            addr_mult = itemsize

            idx = []
            left_over_idx = work_item_id
            for axis, (ax_size, ax_stor_size) in enumerate(
                    zip(array_shape, array_stored_shape, strict=True)):

                if axis >= work_item_axis:
                    left_over_idx, ax_idx = divmod(left_over_idx, ax_size)
                    addr += addr_mult*ax_idx
                    idx.append(ax_idx)
                else:
                    idx.append(0)

                addr_mult *= ax_stor_size

            if left_over_idx:
                # out-of-bounds, assume not taking place
                continue

            bank = (addr // gran) % bank_count
            bank_accesses.setdefault(bank, []).append(
                    "w.item {} -> {}".format(work_item_id, idx[::-1]))

        conflict_multiplicity = max(
                len(acc) for acc in bank_accesses.values())

        if conflict_multiplicity > 1:
            for bank, acc in bank_accesses.items():
                if len(acc) == conflict_multiplicity:
                    conflicts.append(
                            (conflict_multiplicity,
                                "%dx conflict on axis %d (from right, 0-based): "
                                "%s access bank %d" % (
                                    conflict_multiplicity,
                                    work_item_axis,
                                    ", ".join(acc), bank)))

    if conflicts:
        return max(conflicts)
    else:
        return 1, None


def get_fast_inaccurate_build_options(dev: cl.Device):
    """Return a list of flags valid on device *dev* that enable fast, but
    potentially inaccurate floating point math.
    """
    result = ["-cl-mad-enable", "-cl-fast-relaxed-math",
        "-cl-no-signed-zeros", ]
    if dev.vendor.startswith("Advanced Micro") or dev.vendor.startswith("NVIDIA"):
        result.append("-cl-strict-aliasing")
    return result


def get_simd_group_size(dev: cl.Device, type_size: int):
    """Return an estimate of how many work items will be executed across SIMD
    lanes. This returns the size of what Nvidia calls a warp and what AMD calls
    a wavefront.

    Only refers to implicit SIMD.

    :arg type_size: number of bytes in vector entry type.
    """
    try:
        return dev.warp_size_nv
    except Exception:
        pass

    lc_plat_vendor = dev.platform.vendor.lower()
    lc_dev_vendor = dev.vendor.lower()
    if "nvidia" in lc_plat_vendor or "nvidia" in lc_dev_vendor:
        return 32

    if ("advanced micro" in lc_plat_vendor or "ati" in lc_plat_vendor
      or "advanced micro" in lc_dev_vendor or "ati" in lc_dev_vendor):
        if dev.type & cl.device_type.GPU:
            # Tomasz Rybak says, in response to reduction misbehaving on the AMD
            # 'Loveland' APU:
            #
            #    Like in CUDA reduction bug (related to Fermi) it again seems
            # to be related to too eager concurrency when reducing results.
            # According to http://oscarbg.blogspot.com/2009/10/news-from-web.html
            # "Actually the wavefront size is only 64 for the highend cards(48XX,
            # 58XX, 57XX), but 32 for the middleend cards and 16 for the lowend
            # cards."
            # IMO we should use PREFERRED_WORK_GROUP_SIZE_MULTIPLE to get
            # non_sync_size. At the same size we lose SIMD CPU optimisation,
            # but I do not know for now how to fix those two at the same time.
            # Attached patch fixes problem on Loveland, not breaking anything on
            # NVIDIA ION.

            # This is therefore our best guess as to the SIMD group size.

            return reasonable_work_group_size_multiple(dev)
        elif dev.type & cl.device_type.CPU:
            return 1
        else:
            raise RuntimeError("unexpected AMD device type")

    if dev.type & cl.device_type.CPU:
        # implicit assumption: Impl. will vectorize
        return 1

    return None


def get_pocl_version(
        platform: cl.Platform,
        fallback_value: tuple[int, int] | None = None
        ) -> tuple[int, int] | None:
    if platform.name != "Portable Computing Language":
        return None

    import re
    version = platform.version
    ver_match = re.match(
            r"^OpenCL [0-9.]+ [Pp]o[Cc][Ll] ([0-9]+)\.([0-9]+)", version)

    if ver_match is None:
        from warnings import warn
        warn(f"PoCL version number did not have expected format: '{version}'",
             stacklevel=2)
        return fallback_value
    else:
        return (int(ver_match.group(1)), int(ver_match.group(2)))


_CHECK_FOR_POCL_ARG_COUNT_BUG_CACHE: dict[cl.Device, bool] = {}


def _check_for_pocl_arg_count_bug(
        dev: cl.Device,
        ctx: cl.Context | None = None) -> bool:
    try:
        return _CHECK_FOR_POCL_ARG_COUNT_BUG_CACHE[dev]
    except KeyError:
        pass

    if ctx is None:
        build_ctx = cl.Context([dev])
    else:
        build_ctx = ctx

    prg = cl.Program(build_ctx, """
            struct two_things
            {
                long a;
                long b;
            };

            __kernel void test_knl(struct two_things x)
            {
            }
            """).build()

    result = prg.test_knl.num_args == 2
    _CHECK_FOR_POCL_ARG_COUNT_BUG_CACHE[dev] = result

    return result


def has_struct_arg_count_bug(dev, ctx=None):
    """Checks whether the device is expected to have the
    `argument counting bug <https://github.com/pocl/pocl/issues/197>`__.
    """

    if dev.platform.name == "Apple" and dev.type & cl.device_type.CPU:
        return "apple"
    if dev.platform.name == "Portable Computing Language":
        pocl_version = get_pocl_version(dev.platform, fallback_value=(0, 14))
        if pocl_version <= (0, 13):
            return "pocl"
        elif pocl_version <= (0, 14) and _check_for_pocl_arg_count_bug(dev, ctx):
            return "pocl"

    return False


# {{{ SVM capabilities

def _may_have_svm(dev):
    has_svm = (dev.platform._get_cl_version() >= (2, 0)
            and cl.get_cl_header_version() >= (2, 0))

    if dev.platform.name == "Portable Computing Language":
        has_svm = (
                get_pocl_version(dev.platform) >= (1, 0)
                and cl.get_cl_header_version() >= (2, 0))

    return has_svm


def has_coarse_grain_buffer_svm(dev):
    return (_may_have_svm(dev)
            and bool(dev.svm_capabilities
                & cl.device_svm_capabilities.COARSE_GRAIN_BUFFER))


def has_fine_grain_buffer_svm(dev):
    return (_may_have_svm(dev)
            and bool(dev.svm_capabilities
                & cl.device_svm_capabilities.FINE_GRAIN_BUFFER))


def has_fine_grain_system_svm(dev):
    return (_may_have_svm(dev)
            and bool(dev.svm_capabilities
                & cl.device_svm_capabilities.FINE_GRAIN_SYSTEM))


def has_fine_grain_buffer_svm_atomics(dev):
    return has_fine_grain_buffer_svm(dev) and bool(dev.svm_capabilities
                & cl.device_svm_capabilities.ATOMICS)


def has_fine_grain_system_svm_atomics(dev):
    return has_fine_grain_system_svm(dev) and bool(dev.svm_capabilities
                & cl.device_svm_capabilities.ATOMICS)

# }}}


def has_src_build_cache(dev: cl.Device) -> bool | None:
    """
    Return *True* if *dev* has internal support for caching builds from source,
    *False* if it doesn't, and *None* if unknown.
    """
    if dev.platform.name == "Portable Computing Language":
        return True

    if nv_compute_capability(dev) is not None:
        return True

    if dev.platform.name == "AMD Accelerated Parallel Processing":
        return False

    return None

# vim: foldmethod=marker
