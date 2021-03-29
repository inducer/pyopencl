import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import numpy.linalg as la

n = 5000000
a = np.random.rand(n).astype(np.float32)
b = np.random.rand(n).astype(np.float32)


class SVMAllocator:
    def __init__(self, ctx, flags, alignment):
        self._context = ctx
        self._flags = flags
        self._alignment = alignment

    def __call__(self, nbytes):
        return cl.SVM(cl.svm_empty(
                ctx, self._flags, (nbytes,), np.int8, "C", self._alignment))


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

alloc = SVMAllocator(ctx,
        cl.svm_mem_flags.READ_WRITE | cl.svm_mem_flags.SVM_FINE_GRAIN_BUFFER,
        0)

a_dev = cl_array.to_device(queue, a, allocator=alloc)
print("A_DEV", a_dev.data.mem.nbytes, a_dev.data.mem.__array_interface__)
b_dev = cl_array.to_device(queue, b, allocator=alloc)
dest_dev = cl_array.empty_like(a_dev)
print("DEST", dest_dev.data.mem.__array_interface__)

prg = cl.Program(ctx, """
    __kernel void sum(__global const float *a,
    __global const float *b, __global float *c)
    {
      int gid = get_global_id(0);
      c[gid] = a[gid] + b[gid];
    }
    """).build()

knl = prg.sum  # Use this Kernel object for repeated calls
knl(queue, a.shape, None, a_dev.data, b_dev.data, dest_dev.data)

# PROBLEM: numpy frees the temporary out of (a_dev+b_dev) before
# we're done with it
diff = dest_dev - (a_dev+b_dev)
if 0:
    diff = diff.get()
    np.set_printoptions(linewidth=400)
    print(dest_dev)
    print((a_dev+b_dev).get())
    print(diff)
    print(la.norm(diff))
    print("A_DEV", a_dev.data.mem.__array_interface__)
