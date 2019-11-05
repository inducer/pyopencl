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

import logging
logger = logging.getLogger(__name__)

OpRecord = namedtuple("OpRecord", [
    "arg_name",
    "kernel_name",
    "queue",
    "event",
    "stack",
    ])

ArgRecord = namedtuple("ArgRecord", ["name", "buf"])

# mapping from buffers to list of
#   (kernel_name, queue weakref)
BUFFER_TO_OPS = weakref.WeakKeyDictionary()

# mapping from kernel to dictionary containing {nr: buffer argument}
CURRENT_BUF_ARGS = weakref.WeakKeyDictionary()

# list of events for each queue
QUEUE_TO_EVENTS = weakref.WeakKeyDictionary()


# {{{ wrappers

def wrapper_add_local_imports(cc, gen):
    """Wraps :func:`pyopencl.invoker.add_local_imports`"""
    cc.func('add_local_imports')(gen)

    # NOTE: need to import pyopencl to be able to wrap it in generated code
    gen("import pyopencl as _cl")
    gen("")


def wrapper_set_arg(cc, kernel, index, obj):
    """Wraps :meth:`pyopencl.Kernel.set_arg`"""

    logger.debug('set_arg: %s %s', kernel.function_name, index)
    if isinstance(obj, cl.Buffer):
        arg_dict = CURRENT_BUF_ARGS.setdefault(kernel, {})

        try:
            arg_name = kernel.get_arg_info(index, cl.kernel_arg_info.NAME)
        except cl.RuntimeError:
            arg_name = str(index)

        arg_dict[index] = ArgRecord(
                name=arg_name,
                buf=weakref.ref(obj),
                )

    return cc.func('set_arg')(kernel, index, obj)


def wrapper_wait_for_events(cc, wait_for):
    for evt in wait_for:
        queue = evt.get_info(cl.event_info.COMMAND_QUEUE)
        if queue not in QUEUE_TO_EVENTS:
            continue

        if evt in QUEUE_TO_EVENTS[queue]:
            QUEUE_TO_EVENTS[queue].remove(evt)


def wrapper_finish(cc, queue):
    """Wraps :meth:`pyopencl.CommandQueue.finish`"""

    if queue in QUEUE_TO_EVENTS:
        QUEUE_TO_EVENTS[queue].clear()

    return cc.func('finish')(queue)


def wrapper_enqueue_nd_range_kernel(cc,
        queue, kernel, global_size, local_size,
        global_offset=None, wait_for=None, g_times_l=None):
    """Wraps :func:`pyopencl.enqueue_nd_range_kernel`"""

    logger.debug('enqueue_nd_range_kernel: %s', kernel.function_name)
    evt = cc.func('enqueue_nd_range_kernel')(queue, kernel,
            global_size, local_size, global_offset, wait_for, g_times_l)
    QUEUE_TO_EVENTS.setdefault(queue, set()).add(evt)

    arg_dict = CURRENT_BUF_ARGS.get(kernel)
    if arg_dict is not None:
        synced_events = set()
        if wait_for is not None:
            synced_events |= set(wait_for)

        for arg in arg_dict.values():
            logger.debug("%s: arg %s" % (kernel.function_name, arg.name))

            buf = arg.buf()
            if buf is None:
                continue

            prior_ops = BUFFER_TO_OPS.setdefault(buf, [])
            unsynced_events = []
            for op in prior_ops:
                prior_queue = op.queue()
                if prior_queue is None:
                    continue
                if prior_queue.int_ptr == queue.int_ptr:
                    continue

                prior_queue_events = QUEUE_TO_EVENTS.get(prior_queue, set())
                if op.event in prior_queue_events \
                        and op.event not in synced_events:
                    unsynced_events.append((op.arg_name, op.kernel_name))

            logger.debug("unsynced events: %s", list(unsynced_events))

            tb = "".join(traceback.format_stack())
            if unsynced_events:
                cc.concurrency_issues += 1

                from warnings import warn
                warn("\n[%5d] EventsNotSynced: argument `%s` kernel `%s`\n"
                        "%7s previous kernels %s\n"
                        "%7s %d events not found in `wait_for` "
                        "or synced with `queue.finish()` "
                        "or `cl.wait_for_events()`\n" % (
                            cc.concurrency_issues,
                            arg.name, kernel.function_name, " ",
                            ", ".join([str(x) for x in unsynced_events]), " ",
                            len(unsynced_events)),
                        RuntimeWarning, stacklevel=5)

                if cc.show_traceback:
                    print("Traceback\n%s" % tb)
                    import pudb
                    pudb.set_trace()

            prior_ops.append(OpRecord(
                arg_name=arg.name,
                kernel_name=kernel.function_name,
                queue=weakref.ref(queue),
                event=evt,
                stack=tb))

    return evt

# }}}


# {{{

def with_concurrency_check(func):
    def wrapper(func, *args, **kwargs):
        with ConcurrencyCheck(show_traceback=True):
            return func(*args, **kwargs)

    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    global logger
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    from pytools import decorator
    return decorator.decorator(wrapper, func)


class MonkeyPatch(object):
    def __init__(self, cc, obj, attr, wrapper=None):
        def wrapper_partial(*args, **kwargs):
            func = globals()["wrapper_%s" % attr]
            return func(cc, *args, **kwargs)

        if wrapper is None:
            wrapper = wrapper_partial

        func = getattr(obj, attr, None)
        setattr(obj, attr, wrapper)

        self.obj = obj
        self.attr = attr
        self.func = func

    def undo(self):
        if self.func is None:
            delattr(self.obj, self.attr)
        else:
            setattr(self.obj, self.attr, self.func)


class ConcurrencyCheck(object):
    _entered = False

    def __init__(self, show_traceback=False):
        self.show_traceback = show_traceback

        self._overwritten_attrs = {}
        if self._entered:
            raise RuntimeError('cannot nest `ConcurrencyCheck`s')

    def func(self, attr):
        return self._overwritten_attrs[attr].func

    def __enter__(self):
        ConcurrencyCheck._entered = True
        self.concurrency_issues = 0

        def monkeypatch(obj, attr, wrapper=None):
            p = MonkeyPatch(self, obj, attr, wrapper=wrapper)
            self._overwritten_attrs[attr] = p

        # allow monkeypatching in generated code
        monkeypatch(cl.invoker, 'add_local_imports')
        # fix version to avoid handling enqueue_fill_buffer
        # in pyopencl.array.Array._zero_fill::1223
        monkeypatch(cl, 'get_cl_header_version', wrapper=lambda: (1, 1))

        # catch kernel argument buffers
        monkeypatch(cl.Kernel, 'set_arg')
        # catch events
        monkeypatch(cl.Event, '__hash__',
                wrapper=lambda x: x.int_ptr)
        monkeypatch(cl, 'wait_for_events')
        monkeypatch(cl.CommandQueue, 'finish')
        # catch kernel calls to check concurrency
        monkeypatch(cl, 'enqueue_nd_range_kernel')

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for p in self._overwritten_attrs.values():
            p.undo()

        BUFFER_TO_OPS.clear()
        CURRENT_BUF_ARGS.clear()
        QUEUE_TO_EVENTS.clear()

        self._overwritten_attrs.clear()
        ConcurrencyCheck._entered = False

# }}}

# vim: foldmethod=marker
