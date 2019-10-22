import pyopencl as cl
import weakref


# mapping from buffers to list of
# (queue,jjj
BUFFER_TO_OPS = weakref.WeakKeyDictionary()

# mapping from kernel to dictionary containing {nr: buffer argument}
CURRENT_BUF_ARGS = weakref.WeakKeyDictionary()


prev_enqueue_nd_range_kernel = None
prev_kernel__set_arg_buf = None
prev_kernel_set_arg = None


def my_set_arg(kernel, index, obj):
    print("SET_ARG: %s %d" % (kernel.function_name, index))
    if isinstance(obj, cl.Buffer):
        arg_dict = CURRENT_BUF_ARGS.setdefault(kernel, {})
        arg_dict[index] = weakref.ref(obj)
    return prev_kernel_set_arg(kernel, index, obj)


def my_enqueue_nd_range_kernel(
        queue, kernel, global_size, local_size,
        global_offset=None, wait_for=None, g_times_l=None):
    print("ENQUEUE: %s" % kernel.function_name)
    arg_dict = CURRENT_BUF_ARGS[kernel]
    print(arg_dict)
    return prev_enqueue_nd_range_kernel(
        queue, kernel, global_size, local_size,
        global_offset, wait_for, g_times_l)


def enable():
    global prev_enqueue_nd_range_kernel
    global prev_kernel_set_arg
    global prev_get_cl_header_version

    if prev_enqueue_nd_range_kernel is not None:
        raise RuntimeError("already enabled")

    prev_enqueue_nd_range_kernel = cl.enqueue_nd_range_kernel
    prev_kernel_set_arg = cl.Kernel.set_arg
    prev_get_cl_header_version = cl.get_cl_header_version

    cl.Kernel.set_arg = my_set_arg
    cl.enqueue_nd_range_kernel = my_enqueue_nd_range_kernel

    # I can't be bothered to handle clEnqueueFillBuffer
    cl.get_cl_header_version = lambda: (1, 1)
