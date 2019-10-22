from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2019 Andreas Kloeckner"

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

import pyopencl as cl
import weakref
from collections import namedtuple

OpRecord = namedtuple("OpRecord", [
    "kernel_name",
    "queue",
    ])


# mapping from buffers to list of
# (kernel_name, queue weakref)
BUFFER_TO_OPS = weakref.WeakKeyDictionary()

# mapping from kernel to dictionary containing {nr: buffer argument}
CURRENT_BUF_ARGS = weakref.WeakKeyDictionary()


prev_enqueue_nd_range_kernel = None
prev_kernel__set_arg_buf = None
prev_kernel_set_arg = None


def my_set_arg(kernel, index, obj):
    if isinstance(obj, cl.Buffer):
        arg_dict = CURRENT_BUF_ARGS.setdefault(kernel, {})
        arg_dict[index] = weakref.ref(obj)
    return prev_kernel_set_arg(kernel, index, obj)


def my_enqueue_nd_range_kernel(
        queue, kernel, global_size, local_size,
        global_offset=None, wait_for=None, g_times_l=None):
    evt = prev_enqueue_nd_range_kernel(
        queue, kernel, global_size, local_size,
        global_offset, wait_for, g_times_l)

    arg_dict = CURRENT_BUF_ARGS.get(kernel)
    if arg_dict is not None:
        for buf in arg_dict.values():
            buf = buf()
            if buf is None:
                continue

            prior_ops = BUFFER_TO_OPS.setdefault(buf, [])
            for prior_op in prior_ops:
                prev_queue = prior_op.queue()

                if prev_queue is not None and prev_queue.int_ptr != queue.int_ptr:
                    print("DIFFERENT QUEUES",
                            kernel.function_name, prior_op.kernel_name)

            prior_ops.append(
                    OpRecord(
                        kernel_name=kernel.function_name,
                        queue=weakref.ref(queue),)
                    )

    return evt


class ConcurrencyCheck(object):
    def __enter__(self):
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

    def __exit__(self, exc_type, exc_value, traceback):
        global prev_enqueue_nd_range_kernel
        global prev_kernel_set_arg
        global prev_get_cl_header_version

        cl.enqueue_nd_range_kernel = prev_enqueue_nd_range_kernel
        cl.Kernel.set_arg = prev_kernel_set_arg
        cl.get_cl_header_version = prev_get_cl_header_version

        prev_enqueue_nd_range_kernel = None

        BUFFER_TO_OPS.clear()
        CURRENT_BUF_ARGS.clear()
