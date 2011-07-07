from __future__ import division

import pyopencl as cl

def has_double_support(dev):
    for ext in dev.extensions.split(" "):
        if ext == "cl_khr_fp64":
            return True
    return False




def has_amd_double_support(dev):
    """"Fix to allow incomplete amd double support in low end boards"""

    for ext in dev.extensions.split(" "):
        if ext == "cl_amd_fp64":
            return True
    return False




def reasonable_work_group_size_multiple(dev, ctx=None):
    try:
        return dev.warp_size_nv
    except AttributeError:
        pass

    if ctx is None:
        ctx = cl.Context([dev])
    prg = cl.Program(ctx, """
        void knl(float *a)
        {
            a[get_global_id(0)] = 0;
        }
        """)
    return prg.knl.get_work_group_info(
            cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
            dev)




def usable_local_mem_size(dev, nargs=None):
    """Return an estimate of the usable local memory size.
    :arg nargs: Number of 32-bit arguments passed.
    """
    usable_local_mem_size = dev.local_mem_size

    if ("nvidia" in dev.platform.name.lower()
            and (dev.compute_capability_major_nv,
                dev.compute_capability_minor_nv) < (2, 0)):
        # pre-Fermi use local mem for parameter passing
        if nargs is None:
            # assume maximum
            usable_local_mem_size -= 256
        else:
            usable_local_mem_size -= 4*nargs

    return usable_local_mem_size

