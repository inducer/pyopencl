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
import pyopencl.invoker

import traceback
import weakref
from collections import namedtuple


OpRecord = namedtuple("OpRecord", [
    "kernel_name",
    "queue",
    "event"
    ])


# mapping from buffers to list of
# (kernel_name, queue weakref)
BUFFER_TO_OPS = weakref.WeakKeyDictionary()

# mapping from kernel to dictionary containing {nr: buffer argument}
CURRENT_BUF_ARGS = weakref.WeakKeyDictionary()


def add_local_imports_wrapper(gen):
    cl.invoker.add_local_imports(gen)
    # NOTE: need to import pyopencl to be able to wrap it in generated code
    gen("import pyopencl as _cl")
    gen("")


def set_arg_wrapper(cc, kernel, index, obj):
    if cc.verbose:
        # FIXME: should really use logging
        print('set_arg: %s %s' % (kernel.function_name, index))

    if isinstance(obj, cl.Buffer):
        arg_dict = CURRENT_BUF_ARGS.setdefault(kernel, {})
        arg_dict[index] = weakref.ref(obj)
    return cc.prev_kernel_set_arg(kernel, index, obj)


def check_events(wait_for_events, prior_events):
    for evt in wait_for_events:
        if evt in prior_events:
            return True

    return False


def enqueue_nd_range_kernel_wrapper(
        cc, queue, kernel, global_size, local_size,
        global_offset=None, wait_for=None, g_times_l=None):
    if cc.verbose:
        print('enqueue_nd_range_kernel: %s' % (kernel.function_name,))

    evt = cc.prev_enqueue_nd_range_kernel(
        queue, kernel, global_size, local_size,
        global_offset, wait_for, g_times_l)

    arg_dict = CURRENT_BUF_ARGS.get(kernel)
    if arg_dict is None:
        return evt

    for index, buf in arg_dict.items():
        buf = buf()
        if buf is None:
            continue

        prior_ops = BUFFER_TO_OPS.setdefault(buf, [])
        prior_events = []
        for prior_op in prior_ops:
            prev_queue = prior_op.queue()
            if prev_queue is None:
                continue

            if prev_queue.int_ptr != queue.int_ptr:
                if cc.show_traceback:
                    print("Traceback")
                    traceback.print_stack()

                print('DifferentQueuesInKernel: argument %d current kernel `%s` '
                        'previous kernel `%s`' % (
                            index, kernel.function_name, prior_op.kernel_name))

                prior_event = prior_op.event()
                if prior_event is not None:
                    prior_events.append(prior_event)

        if not check_events(wait_for, prior_events):
            print('EventsNotFound')

        prior_ops.append(
                OpRecord(
                    kernel_name=kernel.function_name,
                    queue=weakref.ref(queue),
                    event=weakref.ref(evt),)
                )

    return evt


class ConcurrencyCheck(object):
    prev_enqueue_nd_range_kernel = None
    prev_kernel_set_arg = None
    prev_get_cl_header_version = None

    def __init__(self, show_traceback=True, verbose=True):
        self.show_traceback = show_traceback
        self.verbose = verbose

    def __enter__(self):
        if self.prev_enqueue_nd_range_kernel is not None:
            raise RuntimeError('cannot nest `ConcurrencyCheck`s')

        self.prev_enqueue_nd_range_kernel = cl.enqueue_nd_range_kernel
        self.prev_kernel_set_arg = cl.Kernel.set_arg
        self.prev_get_cl_header_version = cl.get_cl_header_version

        from functools import partial
        cl.Kernel.set_arg = lambda a, b, c: set_arg_wrapper(self, a, b, c)
        cl.enqueue_nd_range_kernel = \
                partial(enqueue_nd_range_kernel_wrapper, self)
        cl.invoker.add_local_imports = \
                add_local_imports_wrapper

        # I can't be bothered to handle clEnqueueFillBuffer
        cl.get_cl_header_version = lambda: (1, 1)

    def __exit__(self, exc_type, exc_value, traceback):
        cl.enqueue_nd_range_kernel = self.prev_enqueue_nd_range_kernel
        cl.Kernel.set_arg = self.prev_kernel_set_arg
        cl.get_cl_header_version = self.prev_get_cl_header_version

        self.prev_enqueue_nd_range_kernel = None

        BUFFER_TO_OPS.clear()
        CURRENT_BUF_ARGS.clear()

# vim: foldmethod=marker
