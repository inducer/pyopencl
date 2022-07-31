#!/usr/bin/env python

from pyopencl.characterize import (has_coarse_grain_buffer_svm,
                                   has_fine_grain_buffer_svm,
                                   has_fine_grain_system_svm)
import numpy as np
import pyopencl as cl

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

dev = queue.device

print(
    f"Device '{dev.name}' on platform '{dev.platform.name} ({dev.platform.version})'"
    " has the following SVM features:\n"
    f"  Coarse-grained buffer SVM: {has_coarse_grain_buffer_svm(dev)}\n"
    f"  Fine-grained buffer SVM:   {has_fine_grain_buffer_svm(dev)}\n"
    f"  Fine-grained system SVM:   {has_fine_grain_system_svm(dev)}"
    )

prg = cl.Program(ctx, """
__kernel void twice(
    __global float *a_g)
{
  int gid = get_global_id(0);
  a_g[gid] = 2*a_g[gid];
}
""").build()


if has_coarse_grain_buffer_svm(dev):
    print("Testing coarse-grained buffer SVM...", end="")

    svm_ary = cl.SVM(cl.csvm_empty(ctx, 10, np.float32))
    assert isinstance(svm_ary.mem, np.ndarray)

    with svm_ary.map_rw(queue) as ary:
        ary.fill(17)  # use from host
        orig_ary = ary.copy()

    prg.twice(queue, svm_ary.mem.shape, None, svm_ary)
    queue.finish()

    with svm_ary.map_ro(queue) as ary:
        assert np.array_equal(orig_ary*2, ary)

    print(" done.")

if has_fine_grain_buffer_svm(dev):
    print("Testing fine-grained buffer SVM...", end="")

    ary = cl.fsvm_empty(ctx, 10, np.float32)
    assert isinstance(ary.base, cl.SVMAllocation)

    ary.fill(17)
    orig_ary = ary.copy()

    prg.twice(queue, ary.shape, None, cl.SVM(ary))
    queue.finish()

    assert np.array_equal(orig_ary*2, ary)

    print(" done.")

if has_fine_grain_system_svm(dev):
    print("Testing fine-grained system SVM...", end="")

    ary = np.zeros(10, np.float32)
    assert isinstance(ary, np.ndarray)

    ary.fill(17)
    orig_ary = ary.copy()

    prg.twice(queue, ary.shape, None, cl.SVM(ary))
    queue.finish()

    assert np.array_equal(orig_ary*2, ary)

    print(" done.")
