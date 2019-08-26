"""CL device arrays."""

# pylint:disable=unexpected-keyword-arg  # for @elwise_kernel_runner

from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

import six
from six.moves import range, reduce

import numpy as np
import pyopencl.elementwise as elementwise
import pyopencl as cl
from pytools import memoize_method
from pyopencl.compyte.array import (
        as_strided as _as_strided,
        f_contiguous_strides as _f_contiguous_strides,
        c_contiguous_strides as _c_contiguous_strides,
        equal_strides as _equal_strides,
        ArrayFlags as _ArrayFlags,
        get_common_dtype as _get_common_dtype_base)
from pyopencl.characterize import has_double_support
from pyopencl import cltypes


def _get_common_dtype(obj1, obj2, queue):
    return _get_common_dtype_base(obj1, obj2,
                                  has_double_support(queue.device))


# Work around PyPy not currently supporting the object dtype.
# (Yes, it doesn't even support checking!)
# (as of May 27, 2014 on PyPy 2.3)
try:
    np.dtype(object)

    def _dtype_is_object(t):
        return t == object
except Exception:
    def _dtype_is_object(t):
        return False


class VecLookupWarner(object):
    def __getattr__(self, name):
        from warnings import warn
        warn("pyopencl.array.vec is deprecated. "
             "Please use pyopencl.cltypes for OpenCL vector and scalar types",
             DeprecationWarning, 2)

        if name == "types":
            name = "vec_types"
        elif name == "type_to_scalar_and_count":
            name = "vec_type_to_scalar_and_count"

        return getattr(cltypes, name)


vec = VecLookupWarner()

# {{{ helper functionality


def splay(queue, n, kernel_specific_max_wg_size=None):
    dev = queue.device
    max_work_items = _builtin_min(128, dev.max_work_group_size)

    if kernel_specific_max_wg_size is not None:
        from six.moves.builtins import min
        max_work_items = min(max_work_items, kernel_specific_max_wg_size)

    min_work_items = _builtin_min(32, max_work_items)
    max_groups = dev.max_compute_units * 4 * 8
    # 4 to overfill the device
    # 8 is an Nvidia constant--that's how many
    # groups fit onto one compute device

    if n < min_work_items:
        group_count = 1
        work_items_per_group = min_work_items
    elif n < (max_groups * min_work_items):
        group_count = (n + min_work_items - 1) // min_work_items
        work_items_per_group = min_work_items
    elif n < (max_groups * max_work_items):
        group_count = max_groups
        grp = (n + min_work_items - 1) // min_work_items
        work_items_per_group = (
                (grp + max_groups - 1) // max_groups) * min_work_items
    else:
        group_count = max_groups
        work_items_per_group = max_work_items

    #print "n:%d gc:%d wipg:%d" % (n, group_count, work_items_per_group)
    return (group_count*work_items_per_group,), (work_items_per_group,)


def elwise_kernel_runner(kernel_getter):
    """Take a kernel getter of the same signature as the kernel
    and return a function that invokes that kernel.

    Assumes that the zeroth entry in *args* is an :class:`Array`.
    """

    def kernel_runner(*args, **kwargs):
        repr_ary = args[0]
        queue = kwargs.pop("queue", None) or repr_ary.queue
        wait_for = kwargs.pop("wait_for", None)

        # wait_for must be a copy, because we modify it in-place below
        if wait_for is None:
            wait_for = []
        else:
            wait_for = list(wait_for)

        knl = kernel_getter(*args, **kwargs)

        gs, ls = repr_ary.get_sizes(queue,
                knl.get_work_group_info(
                    cl.kernel_work_group_info.WORK_GROUP_SIZE,
                    queue.device))

        assert isinstance(repr_ary, Array)

        actual_args = []
        for arg in args:
            if isinstance(arg, Array):
                if not arg.flags.forc:
                    raise RuntimeError("only contiguous arrays may "
                            "be used as arguments to this operation")
                actual_args.append(arg.base_data)
                actual_args.append(arg.offset)
                wait_for.extend(arg.events)
            else:
                actual_args.append(arg)
        actual_args.append(repr_ary.size)

        return knl(queue, gs, ls, *actual_args, **dict(wait_for=wait_for))

    try:
        from functools import update_wrapper
    except ImportError:
        return kernel_runner
    else:
        return update_wrapper(kernel_runner, kernel_getter)


class DefaultAllocator(cl.tools.DeferredAllocator):
    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("pyopencl.array.DefaultAllocator is deprecated. "
                "It will be continue to exist throughout the 2013.x "
                "versions of PyOpenCL.",
                DeprecationWarning, 2)
        cl.tools.DeferredAllocator.__init__(self, *args, **kwargs)


def _make_strides(itemsize, shape, order):
    if order in "fF":
        return _f_contiguous_strides(itemsize, shape)
    elif order in "cC":
        return _c_contiguous_strides(itemsize, shape)
    else:
        raise ValueError("invalid order: %s" % order)

# }}}


# {{{ array class

class ArrayHasOffsetError(ValueError):
    """
    .. versionadded:: 2013.1
    """

    def __init__(self, val="The operation you are attempting does not yet "
                "support arrays that start at an offset from the beginning "
                "of their buffer."):
        ValueError.__init__(self, val)


class _copy_queue:  # noqa
    pass


class Array(object):
    """A :class:`numpy.ndarray` work-alike that stores its data and performs
    its computations on the compute device.  *shape* and *dtype* work exactly
    as in :mod:`numpy`.  Arithmetic methods in :class:`Array` support the
    broadcasting of scalars. (e.g. `array+5`)

    *cq* must be a :class:`pyopencl.CommandQueue` or a :class:`pyopencl.Context`.

    If it is a queue, *cq* specifies the queue in which the array carries out
    its computations by default. If a default queue (and thereby overloaded
    operators and many other niceties) are not desired, pass a
    :class:`Context`.

    *allocator* may be `None` or a callable that, upon being called with an
    argument of the number of bytes to be allocated, returns an
    :class:`pyopencl.Buffer` object. (A :class:`pyopencl.tools.MemoryPool`
    instance is one useful example of an object to pass here.)

    .. versionchanged:: 2011.1
        Renamed *context* to *cqa*, made it general-purpose.

        All arguments beyond *order* should be considered keyword-only.

    .. versionchanged:: 2015.2
        Renamed *context* to *cq*, disallowed passing allocators through it.

    .. attribute :: data

        The :class:`pyopencl.MemoryObject` instance created for the memory that
        backs this :class:`Array`.

        .. versionchanged:: 2013.1

            If a non-zero :attr:`offset` has been specified for this array,
            this will fail with :exc:`ArrayHasOffsetError`.

    .. attribute :: base_data

        The :class:`pyopencl.MemoryObject` instance created for the memory that
        backs this :class:`Array`. Unlike :attr:`data`, the base address of
        *base_data* is allowed to be different from the beginning of the array.
        The actual beginning is the base address of *base_data* plus
        :attr:`offset` bytes.

        Unlike :attr:`data`, retrieving :attr:`base_data` always succeeds.

        .. versionadded:: 2013.1

    .. attribute :: offset

        See :attr:`base_data`.

        .. versionadded:: 2013.1

    .. attribute :: shape

        The tuple of lengths of each dimension in the array.

    .. attribute :: ndim

        The number of dimensions in :attr:`shape`.

    .. attribute :: dtype

        The :class:`numpy.dtype` of the items in the GPU array.

    .. attribute :: size

        The number of meaningful entries in the array. Can also be computed by
        multiplying up the numbers in :attr:`shape`.

    .. attribute :: nbytes

        The size of the entire array in bytes. Computed as :attr:`size` times
        ``dtype.itemsize``.

    .. attribute :: strides

        Tuple of bytes to step in each dimension when traversing an array.

    .. attribute :: flags

        Return an object with attributes `c_contiguous`, `f_contiguous` and
        `forc`, which may be used to query contiguity properties in analogy to
        :attr:`numpy.ndarray.flags`.

    .. rubric:: Methods

    .. automethod :: with_queue

    .. automethod :: __len__
    .. automethod :: reshape
    .. automethod :: ravel
    .. automethod :: view
    .. automethod :: squeeze
    .. automethod :: transpose
    .. attribute :: T
    .. automethod :: set
    .. automethod :: get
    .. automethod :: get_async
    .. automethod :: copy

    .. automethod :: __str__
    .. automethod :: __repr__

    .. automethod :: mul_add
    .. automethod :: __add__
    .. automethod :: __sub__
    .. automethod :: __iadd__
    .. automethod :: __isub__
    .. automethod :: __neg__
    .. automethod :: __mul__
    .. automethod :: __div__
    .. automethod :: __rdiv__
    .. automethod :: __pow__

    .. automethod :: __and__
    .. automethod :: __xor__
    .. automethod :: __or__
    .. automethod :: __iand__
    .. automethod :: __ixor__
    .. automethod :: __ior__

    .. automethod :: __abs__
    .. automethod :: __invert__

    .. UNDOC reverse()

    .. automethod :: fill

    .. automethod :: astype

    .. autoattribute :: real
    .. autoattribute :: imag
    .. automethod :: conj

    .. automethod :: __getitem__
    .. automethod :: __setitem__

    .. automethod :: setitem

    .. automethod :: map_to_host

    .. rubric:: Comparisons, conditionals, any, all

    .. versionadded:: 2013.2

    Boolean arrays are stored as :class:`numpy.int8` because ``bool``
    has an unspecified size in the OpenCL spec.

    .. automethod :: __nonzero__

        Only works for device scalars. (i.e. "arrays" with ``shape == ()``.)

    .. automethod :: any
    .. automethod :: all

    .. automethod :: __eq__
    .. automethod :: __ne__
    .. automethod :: __lt__
    .. automethod :: __le__
    .. automethod :: __gt__
    .. automethod :: __ge__

    .. rubric:: Event management

    If an array is used from within an out-of-order queue, it needs to take
    care of its own operation ordering. The facilities in this section make
    this possible.

    .. note::

        Currently, read and write events are not distinguished.
        As a result, e.g., read-only operations will needlessly wait on other
        read-only operations (i.e., executed among asynchronous queues).

    .. versionadded:: 2014.1.1

    .. attribute:: events

        A list of :class:`pyopencl.Event` instances that the current content of
        this array depends on. User code may read, but should never modify this
        list directly. To update this list, instead use the following methods.

    .. automethod:: add_event
    .. automethod:: finish
    """

    __array_priority__ = 100

    def __init__(self, cq, shape, dtype, order="C", allocator=None,
            data=None, offset=0, strides=None, events=None):
        # {{{ backward compatibility

        if isinstance(cq, cl.CommandQueue):
            queue = cq
            context = queue.context

        elif isinstance(cq, cl.Context):
            context = cq
            queue = None

        else:
            raise TypeError("cq may be a queue or a context, not '%s'"
                    % type(cq))

        if allocator is not None:
            # "is" would be wrong because two Python objects are allowed
            # to hold handles to the same context.

            # FIXME It would be nice to check this. But it would require
            # changing the allocator interface. Trust the user for now.

            #assert allocator.context == context
            pass

        # Queue-less arrays do have a purpose in life.
        # They don't do very much, but at least they don't run kernels
        # in random queues.
        #
        # See also :meth:`with_queue`.

        del cq

        # }}}

        # invariant here: allocator, queue set

        # {{{ determine shape, size, and strides
        dtype = np.dtype(dtype)

        try:
            size = 1
            for dim in shape:
                size *= dim
        except TypeError:
            import sys
            if sys.version_info >= (3,):
                admissible_types = (int, np.integer)
            else:
                admissible_types = (np.integer,) + six.integer_types

            if not isinstance(shape, admissible_types):
                raise TypeError("shape must either be iterable or "
                        "castable to an integer")
            size = shape
            shape = (shape,)

        if isinstance(size, np.integer):
            size = size.item()

        if strides is None:
            strides = _make_strides(dtype.itemsize, shape, order)

        else:
            # FIXME: We should possibly perform some plausibility
            # checking on 'strides' here.

            strides = tuple(strides)

        # }}}

        if _dtype_is_object(dtype):
            raise TypeError("object arrays on the compute device are not allowed")

        assert isinstance(shape, tuple)
        assert isinstance(strides, tuple)

        self.queue = queue
        self.shape = shape
        self.dtype = dtype
        self.strides = strides
        if events is None:
            self.events = []
        else:
            self.events = events

        self.size = size
        alloc_nbytes = self.nbytes = self.dtype.itemsize * self.size

        self.allocator = allocator

        if data is None:
            if alloc_nbytes <= 0:
                if alloc_nbytes == 0:
                    # Work around CL not allowing zero-sized buffers.
                    alloc_nbytes = 1

                else:
                    raise ValueError("cannot allocate CL buffer with "
                            "negative size")

            if allocator is None:
                if context is None and queue is not None:
                    context = queue.context

                self.base_data = cl.Buffer(
                        context, cl.mem_flags.READ_WRITE, alloc_nbytes)
            else:
                self.base_data = self.allocator(alloc_nbytes)
        else:
            self.base_data = data

        self.offset = offset
        self.context = context

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def data(self):
        if self.offset:
            raise ArrayHasOffsetError()
        else:
            return self.base_data

    @property
    @memoize_method
    def flags(self):
        return _ArrayFlags(self)

    def _new_with_changes(self, data, offset, shape=None, dtype=None,
            strides=None, queue=_copy_queue, allocator=None):
        """
        :arg data: *None* means allocate a new array.
        """
        if shape is None:
            shape = self.shape
        if dtype is None:
            dtype = self.dtype
        if strides is None:
            strides = self.strides
        if queue is _copy_queue:
            queue = self.queue
        if allocator is None:
            allocator = self.allocator

        # If we're allocating new data, then there's not likely to be
        # a data dependency. Otherwise, the two arrays should probably
        # share the same events list.

        if data is None:
            events = None
        else:
            events = self.events

        if queue is not None:
            return Array(queue, shape, dtype, allocator=allocator,
                    strides=strides, data=data, offset=offset,
                    events=events)
        else:
            return Array(self.context, shape, dtype,
                    strides=strides, data=data, offset=offset,
                    events=events, allocator=allocator)

    def with_queue(self, queue):
        """Return a copy of *self* with the default queue set to *queue*.

        *None* is allowed as a value for *queue*.

        .. versionadded:: 2013.1
        """

        if queue is not None:
            assert queue.context == self.context

        return self._new_with_changes(self.base_data, self.offset,
                queue=queue)

    #@memoize_method FIXME: reenable
    def get_sizes(self, queue, kernel_specific_max_wg_size=None):
        if not self.flags.forc:
            raise NotImplementedError("cannot operate on non-contiguous array")
        return splay(queue, self.size,
                kernel_specific_max_wg_size=kernel_specific_max_wg_size)

    def set(self, ary, queue=None, async_=None, **kwargs):
        """Transfer the contents the :class:`numpy.ndarray` object *ary*
        onto the device.

        *ary* must have the same dtype and size (not necessarily shape) as
        *self*.

        .. versionchanged:: 2017.2.1

            Python 3.7 makes ``async`` a reserved keyword. On older Pythons,
            we will continue to  accept *async* as a parameter, however this
            should be considered deprecated. *async_* is the new, official
            spelling.
        """

        # {{{ handle 'async' deprecation

        async_arg = kwargs.pop("async", None)
        if async_arg is not None:
            if async_ is not None:
                raise TypeError("may not specify both 'async' and 'async_'")
            async_ = async_arg

        if async_ is None:
            async_ = False

        if kwargs:
            raise TypeError("extra keyword arguments specified: %s"
                    % ", ".join(kwargs))

        # }}}

        assert ary.size == self.size
        assert ary.dtype == self.dtype

        if not ary.flags.forc:
            raise RuntimeError("cannot set from non-contiguous array")

        if not _equal_strides(ary.strides, self.strides, self.shape):
            from warnings import warn
            warn("Setting array from one with different "
                    "strides/storage order. This will cease to work "
                    "in 2013.x.",
                    stacklevel=2)

        if self.size:
            event1 = cl.enqueue_copy(queue or self.queue, self.base_data, ary,
                    device_offset=self.offset,
                    is_blocking=not async_)
            self.add_event(event1)

    def _get(self, queue=None, ary=None, async_=None, **kwargs):
        # {{{ handle 'async' deprecation

        async_arg = kwargs.pop("async", None)
        if async_arg is not None:
            if async_ is not None:
                raise TypeError("may not specify both 'async' and 'async_'")
            async_ = async_arg

        if async_ is None:
            async_ = False

        if kwargs:
            raise TypeError("extra keyword arguments specified: %s"
                    % ", ".join(kwargs))

        # }}}

        if ary is None:
            ary = np.empty(self.shape, self.dtype)

            if self.strides != ary.strides:
                ary = _as_strided(ary, strides=self.strides)
        else:
            if ary.size != self.size:
                raise TypeError("'ary' has non-matching size")
            if ary.dtype != self.dtype:
                raise TypeError("'ary' has non-matching type")

            if self.shape != ary.shape:
                from warnings import warn
                warn("get() between arrays of different shape is deprecated "
                        "and will be removed in PyCUDA 2017.x",
                        DeprecationWarning, stacklevel=2)

        assert self.flags.forc, "Array in get() must be contiguous"

        queue = queue or self.queue
        if queue is None:
            raise ValueError("Cannot copy array to host. "
                    "Array has no queue. Use "
                    "'new_array = array.with_queue(queue)' "
                    "to associate one.")

        if self.size:
            event1 = cl.enqueue_copy(queue, ary, self.base_data,
                    device_offset=self.offset,
                    wait_for=self.events, is_blocking=not async_)
            self.add_event(event1)
        else:
            event1 = None

        return ary, event1

    def get(self, queue=None, ary=None, async_=None, **kwargs):
        """Transfer the contents of *self* into *ary* or a newly allocated
        :mod:`numpy.ndarray`. If *ary* is given, it must have the same
        shape and dtype.

        .. versionchanged:: 2019.1

            Calling with `async_=True` was deprecated and replaced by
            :meth:`get_async`.
            The event returned by :meth:`pyopencl.enqueue_copy` is now stored into
            :attr:`events` to ensure data is not modified before the copy is
            complete.

        .. versionchanged:: 2015.2

            *ary* with different shape was deprecated.

        .. versionchanged:: 2017.2.1

            Python 3.7 makes ``async`` a reserved keyword. On older Pythons,
            we will continue to  accept *async* as a parameter, however this
            should be considered deprecated. *async_* is the new, official
            spelling.
        """

        if async_:
            from warnings import warn
            warn("calling pyopencl.Array.get with `async_=True` is deprecated. "
                    "Please use pyopencl.Array.get_async for asynchronous "
                    "device-to-host transfers",
                    DeprecationWarning, 2)

        ary, event1 = self.get(queue=queue, ary=ary, async_=async_, **kwargs)

        return ary

    def get_async(self, queue=None, ary=None, **kwargs):
        """
        Asynchronous version of :meth:`get`, following the same calling convention
        while returning a tuple ``(ary, event)`` containing the host array `ary`
        and the :class:`pyopencl.NannyEvent` `event` returned by
        :meth:`pyopencl.enqueue_copy`.
        """

        return self.get(queue=queue, ary=ary, async_=True, **kwargs)

    def copy(self, queue=_copy_queue):
        """
        :arg queue: The :class:`CommandQueue` for the returned array.

        .. versionchanged:: 2017.1.2
            Updates the queue of the returned array.

        .. versionadded:: 2013.1
        """

        if queue is _copy_queue:
            queue = self.queue

        result = self._new_like_me(queue=queue)

        # result.queue won't be the same as queue if queue is None.
        # We force them to be the same here.
        if result.queue is not queue:
            result = result.with_queue(queue)

        if self.nbytes:
            event1 = cl.enqueue_copy(queue or self.queue,
                    result.base_data, self.base_data,
                    src_offset=self.offset, byte_count=self.nbytes,
                    wait_for=self.events)
            result.add_event(event1)

        return result

    def __str__(self):
        return str(self.get())

    def __repr__(self):
        return repr(self.get())

    def safely_stringify_for_pudb(self):
        return "cl.Array %s %s" % (self.dtype, self.shape)

    def __hash__(self):
        raise TypeError("pyopencl arrays are not hashable.")

    # {{{ kernel invocation wrappers

    @staticmethod
    @elwise_kernel_runner
    def _axpbyz(out, afac, a, bfac, b, queue=None):
        """Compute ``out = selffac * self + otherfac*other``,
        where *other* is an array."""
        assert out.shape == a.shape
        assert out.shape == b.shape

        return elementwise.get_axpbyz_kernel(
                out.context, a.dtype, b.dtype, out.dtype)

    @staticmethod
    @elwise_kernel_runner
    def _axpbz(out, a, x, b, queue=None):
        """Compute ``z = a * x + b``, where *b* is a scalar."""
        a = np.array(a)
        b = np.array(b)
        assert out.shape == x.shape
        return elementwise.get_axpbz_kernel(out.context,
                a.dtype, x.dtype, b.dtype, out.dtype)

    @staticmethod
    @elwise_kernel_runner
    def _elwise_multiply(out, a, b, queue=None):
        assert out.shape == a.shape
        assert out.shape == b.shape
        return elementwise.get_multiply_kernel(
                a.context, a.dtype, b.dtype, out.dtype)

    @staticmethod
    @elwise_kernel_runner
    def _rdiv_scalar(out, ary, other, queue=None):
        other = np.array(other)
        assert out.shape == ary.shape
        return elementwise.get_rdivide_elwise_kernel(
                out.context, ary.dtype, other.dtype, out.dtype)

    @staticmethod
    @elwise_kernel_runner
    def _div(out, self, other, queue=None):
        """Divides an array by another array."""

        assert self.shape == other.shape

        return elementwise.get_divide_kernel(self.context,
                self.dtype, other.dtype, out.dtype)

    @staticmethod
    @elwise_kernel_runner
    def _fill(result, scalar):
        return elementwise.get_fill_kernel(result.context, result.dtype)

    @staticmethod
    @elwise_kernel_runner
    def _abs(result, arg):
        if arg.dtype.kind == "c":
            from pyopencl.elementwise import complex_dtype_to_name
            fname = "%s_abs" % complex_dtype_to_name(arg.dtype)
        elif arg.dtype.kind == "f":
            fname = "fabs"
        elif arg.dtype.kind in ["u", "i"]:
            fname = "abs"
        else:
            raise TypeError("unsupported dtype in _abs()")

        return elementwise.get_unary_func_kernel(
                arg.context, fname, arg.dtype, out_dtype=result.dtype)

    @staticmethod
    @elwise_kernel_runner
    def _real(result, arg):
        from pyopencl.elementwise import complex_dtype_to_name
        fname = "%s_real" % complex_dtype_to_name(arg.dtype)
        return elementwise.get_unary_func_kernel(
                arg.context, fname, arg.dtype, out_dtype=result.dtype)

    @staticmethod
    @elwise_kernel_runner
    def _imag(result, arg):
        from pyopencl.elementwise import complex_dtype_to_name
        fname = "%s_imag" % complex_dtype_to_name(arg.dtype)
        return elementwise.get_unary_func_kernel(
                arg.context, fname, arg.dtype, out_dtype=result.dtype)

    @staticmethod
    @elwise_kernel_runner
    def _conj(result, arg):
        from pyopencl.elementwise import complex_dtype_to_name
        fname = "%s_conj" % complex_dtype_to_name(arg.dtype)
        return elementwise.get_unary_func_kernel(
                arg.context, fname, arg.dtype, out_dtype=result.dtype)

    @staticmethod
    @elwise_kernel_runner
    def _pow_scalar(result, ary, exponent):
        exponent = np.array(exponent)
        return elementwise.get_pow_kernel(result.context,
                ary.dtype, exponent.dtype, result.dtype,
                is_base_array=True, is_exp_array=False)

    @staticmethod
    @elwise_kernel_runner
    def _rpow_scalar(result, base, exponent):
        base = np.array(base)
        return elementwise.get_pow_kernel(result.context,
                base.dtype, exponent.dtype, result.dtype,
                is_base_array=False, is_exp_array=True)

    @staticmethod
    @elwise_kernel_runner
    def _pow_array(result, base, exponent):
        return elementwise.get_pow_kernel(
                result.context, base.dtype, exponent.dtype, result.dtype,
                is_base_array=True, is_exp_array=True)

    @staticmethod
    @elwise_kernel_runner
    def _reverse(result, ary):
        return elementwise.get_reverse_kernel(result.context, ary.dtype)

    @staticmethod
    @elwise_kernel_runner
    def _copy(dest, src):
        return elementwise.get_copy_kernel(
                dest.context, dest.dtype, src.dtype)

    def _new_like_me(self, dtype=None, queue=None):
        strides = None
        if dtype is None:
            dtype = self.dtype

        if dtype == self.dtype:
            strides = self.strides

        queue = queue or self.queue
        if queue is not None:
            return self.__class__(queue, self.shape, dtype,
                    allocator=self.allocator, strides=strides)
        else:
            return self.__class__(self.context, self.shape, dtype,
                    strides=strides, allocator=self.allocator)

    @staticmethod
    @elwise_kernel_runner
    def _scalar_binop(out, a, b, queue=None, op=None):
        return elementwise.get_array_scalar_binop_kernel(
                out.context, op, out.dtype, a.dtype,
                np.array(b).dtype)

    @staticmethod
    @elwise_kernel_runner
    def _array_binop(out, a, b, queue=None, op=None):
        if a.shape != b.shape:
            raise ValueError("shapes of binop arguments do not match")
        return elementwise.get_array_binop_kernel(
                out.context, op, out.dtype, a.dtype, b.dtype)

    @staticmethod
    @elwise_kernel_runner
    def _unop(out, a, queue=None, op=None):
        if out.shape != a.shape:
            raise ValueError("shapes of arguments do not match")
        return elementwise.get_unop_kernel(
                out.context, op, a.dtype, out.dtype)

    # }}}

    # {{{ operators

    def mul_add(self, selffac, other, otherfac, queue=None):
        """Return `selffac * self + otherfac*other`.
        """
        result = self._new_like_me(
                _get_common_dtype(self, other, queue or self.queue))
        result.add_event(
                self._axpbyz(result, selffac, self, otherfac, other))
        return result

    def __add__(self, other):
        """Add an array with an array or an array with a scalar."""

        if isinstance(other, Array):
            # add another vector
            result = self._new_like_me(
                    _get_common_dtype(self, other, self.queue))

            result.add_event(
                    self._axpbyz(result,
                        self.dtype.type(1), self,
                        other.dtype.type(1), other))

            return result
        else:
            # add a scalar
            if other == 0:
                return self.copy()
            else:
                common_dtype = _get_common_dtype(self, other, self.queue)
                result = self._new_like_me(common_dtype)
                result.add_event(
                        self._axpbz(result, self.dtype.type(1),
                            self, common_dtype.type(other)))
                return result

    __radd__ = __add__

    def __sub__(self, other):
        """Substract an array from an array or a scalar from an array."""

        if isinstance(other, Array):
            result = self._new_like_me(
                    _get_common_dtype(self, other, self.queue))
            result.add_event(
                    self._axpbyz(result,
                        self.dtype.type(1), self,
                        other.dtype.type(-1), other))

            return result
        else:
            # subtract a scalar
            if other == 0:
                return self.copy()
            else:
                result = self._new_like_me(
                        _get_common_dtype(self, other, self.queue))
                result.add_event(
                        self._axpbz(result, self.dtype.type(1), self, -other))
                return result

    def __rsub__(self, other):
        """Substracts an array by a scalar or an array::

           x = n - self
        """
        common_dtype = _get_common_dtype(self, other, self.queue)
        # other must be a scalar
        result = self._new_like_me(common_dtype)
        result.add_event(
                self._axpbz(result, self.dtype.type(-1), self,
                    common_dtype.type(other)))
        return result

    def __iadd__(self, other):
        if isinstance(other, Array):
            self.add_event(
                    self._axpbyz(self,
                        self.dtype.type(1), self,
                        other.dtype.type(1), other))
            return self
        else:
            self.add_event(
                    self._axpbz(self, self.dtype.type(1), self, other))
            return self

    def __isub__(self, other):
        if isinstance(other, Array):
            self.add_event(
                    self._axpbyz(self, self.dtype.type(1), self,
                        other.dtype.type(-1), other))
            return self
        else:
            self._axpbz(self, self.dtype.type(1), self, -other)
            return self

    def __neg__(self):
        result = self._new_like_me()
        result.add_event(self._axpbz(result, -1, self, 0))
        return result

    def __mul__(self, other):
        if isinstance(other, Array):
            result = self._new_like_me(
                    _get_common_dtype(self, other, self.queue))
            result.add_event(
                    self._elwise_multiply(result, self, other))
            return result
        else:
            common_dtype = _get_common_dtype(self, other, self.queue)
            result = self._new_like_me(common_dtype)
            result.add_event(
                    self._axpbz(result,
                        common_dtype.type(other), self, self.dtype.type(0)))
            return result

    def __rmul__(self, scalar):
        common_dtype = _get_common_dtype(self, scalar, self.queue)
        result = self._new_like_me(common_dtype)
        result.add_event(
                self._axpbz(result,
                    common_dtype.type(scalar), self, self.dtype.type(0)))
        return result

    def __imul__(self, other):
        if isinstance(other, Array):
            self.add_event(
                    self._elwise_multiply(self, self, other))
        else:
            # scalar
            self.add_event(
                    self._axpbz(self, other, self, self.dtype.type(0)))

        return self

    def __div__(self, other):
        """Divides an array by an array or a scalar, i.e. ``self / other``.
        """
        if isinstance(other, Array):
            result = self._new_like_me(
                    _get_common_dtype(self, other, self.queue))
            result.add_event(self._div(result, self, other))
        else:
            if other == 1:
                return self.copy()
            else:
                # create a new array for the result
                common_dtype = _get_common_dtype(self, other, self.queue)
                result = self._new_like_me(common_dtype)
                result.add_event(
                        self._axpbz(result,
                            common_dtype.type(1/other), self, self.dtype.type(0)))

        return result

    __truediv__ = __div__

    def __rdiv__(self, other):
        """Divides an array by a scalar or an array, i.e. ``other / self``.
        """

        if isinstance(other, Array):
            result = self._new_like_me(
                    _get_common_dtype(self, other, self.queue))
            result.add_event(other._div(result, self))
        else:
            # create a new array for the result
            common_dtype = _get_common_dtype(self, other, self.queue)
            result = self._new_like_me(common_dtype)
            result.add_event(
                    self._rdiv_scalar(result, self, common_dtype.type(other)))

        return result

    __rtruediv__ = __rdiv__

    def __and__(self, other):
        common_dtype = _get_common_dtype(self, other, self.queue)

        if not np.issubdtype(common_dtype, np.integer):
            raise TypeError("Integral types only")

        if isinstance(other, Array):
            result = self._new_like_me(common_dtype)
            result.add_event(self._array_binop(result, self, other, op="&"))
        else:
            # create a new array for the result
            result = self._new_like_me(common_dtype)
            result.add_event(
                    self._scalar_binop(result, self, other, op="&"))

        return result

    __rand__ = __and__  # commutes

    def __or__(self, other):
        common_dtype = _get_common_dtype(self, other, self.queue)

        if not np.issubdtype(common_dtype, np.integer):
            raise TypeError("Integral types only")

        if isinstance(other, Array):
            result = self._new_like_me(common_dtype)
            result.add_event(self._array_binop(result, self, other, op="|"))
        else:
            # create a new array for the result
            result = self._new_like_me(common_dtype)
            result.add_event(
                    self._scalar_binop(result, self, other, op="|"))

        return result

    __ror__ = __or__  # commutes

    def __xor__(self, other):
        common_dtype = _get_common_dtype(self, other, self.queue)

        if not np.issubdtype(common_dtype, np.integer):
            raise TypeError("Integral types only")

        if isinstance(other, Array):
            result = self._new_like_me(common_dtype)
            result.add_event(self._array_binop(result, self, other, op="^"))
        else:
            # create a new array for the result
            result = self._new_like_me(common_dtype)
            result.add_event(
                    self._scalar_binop(result, self, other, op="^"))

        return result

    __rxor__ = __xor__  # commutes

    def __iand__(self, other):
        common_dtype = _get_common_dtype(self, other, self.queue)

        if not np.issubdtype(common_dtype, np.integer):
            raise TypeError("Integral types only")

        if isinstance(other, Array):
            self.add_event(self._array_binop(self, self, other, op="&"))
        else:
            self.add_event(
                    self._scalar_binop(self, self, other, op="&"))

        return self

    def __ior__(self, other):
        common_dtype = _get_common_dtype(self, other, self.queue)

        if not np.issubdtype(common_dtype, np.integer):
            raise TypeError("Integral types only")

        if isinstance(other, Array):
            self.add_event(self._array_binop(self, self, other, op="|"))
        else:
            self.add_event(
                    self._scalar_binop(self, self, other, op="|"))

        return self

    def __ixor__(self, other):
        common_dtype = _get_common_dtype(self, other, self.queue)

        if not np.issubdtype(common_dtype, np.integer):
            raise TypeError("Integral types only")

        if isinstance(other, Array):
            self.add_event(self._array_binop(self, self, other, op="^"))
        else:
            self.add_event(
                    self._scalar_binop(self, self, other, op="^"))

        return self

    def _zero_fill(self, queue=None, wait_for=None):
        queue = queue or self.queue

        if (
                queue._get_cl_version() >= (1, 2)
                and cl.get_cl_header_version() >= (1, 2)):

            self.add_event(
                    cl.enqueue_fill_buffer(queue, self.base_data, np.int8(0),
                        self.offset, self.nbytes, wait_for=wait_for))
        else:
            zero = np.zeros((), self.dtype)
            self.fill(zero, queue=queue)

    def fill(self, value, queue=None, wait_for=None):
        """Fill the array with *scalar*.

        :returns: *self*.
        """

        self.add_event(
                self._fill(self, value, queue=queue, wait_for=wait_for))

        return self

    def __len__(self):
        """Returns the size of the leading dimension of *self*."""
        if len(self.shape):
            return self.shape[0]
        else:
            return TypeError("scalar has no len()")

    def __abs__(self):
        """Return a `Array` of the absolute values of the elements
        of *self*.
        """

        result = self._new_like_me(self.dtype.type(0).real.dtype)
        result.add_event(self._abs(result, self))
        return result

    def __pow__(self, other):
        """Exponentiation by a scalar or elementwise by another
        :class:`Array`.
        """

        if isinstance(other, Array):
            assert self.shape == other.shape

            result = self._new_like_me(
                    _get_common_dtype(self, other, self.queue))
            result.add_event(
                    self._pow_array(result, self, other))
        else:
            result = self._new_like_me(
                    _get_common_dtype(self, other, self.queue))
            result.add_event(self._pow_scalar(result, self, other))

        return result

    def __rpow__(self, other):
        # other must be a scalar
        common_dtype = _get_common_dtype(self, other, self.queue)
        result = self._new_like_me(common_dtype)
        result.add_event(
                self._rpow_scalar(result, common_dtype.type(other), self))
        return result

    def __invert__(self):
        if not np.issubdtype(self.dtype, np.integer):
            raise TypeError("Integral types only")

        result = self._new_like_me()
        result.add_event(self._unop(result, self, op="~"))

        return result

    # }}}

    def reverse(self, queue=None):
        """Return this array in reversed order. The array is treated
        as one-dimensional.
        """

        result = self._new_like_me()
        result.add_event(
                self._reverse(result, self))
        return result

    def astype(self, dtype, queue=None):
        """Return a copy of *self*, cast to *dtype*."""
        if dtype == self.dtype:
            return self.copy()

        result = self._new_like_me(dtype=dtype)
        result.add_event(self._copy(result, self, queue=queue))
        return result

    # {{{ rich comparisons, any, all

    def __nonzero__(self):
        if self.shape == ():
            return bool(self.get())
        else:
            raise ValueError("The truth value of an array with "
                    "more than one element is ambiguous. Use a.any() or a.all()")

    __bool__ = __nonzero__

    def any(self, queue=None, wait_for=None):
        from pyopencl.reduction import get_any_kernel
        krnl = get_any_kernel(self.context, self.dtype)
        if wait_for is None:
            wait_for = []
        result, event1 = krnl(self, queue=queue,
               wait_for=wait_for + self.events, return_event=True)
        result.add_event(event1)
        return result

    def all(self, queue=None, wait_for=None):
        from pyopencl.reduction import get_all_kernel
        krnl = get_all_kernel(self.context, self.dtype)
        if wait_for is None:
            wait_for = []
        result, event1 = krnl(self, queue=queue,
               wait_for=wait_for + self.events, return_event=True)
        result.add_event(event1)
        return result

    @staticmethod
    @elwise_kernel_runner
    def _scalar_comparison(out, a, b, queue=None, op=None):
        return elementwise.get_array_scalar_comparison_kernel(
                out.context, op, a.dtype)

    @staticmethod
    @elwise_kernel_runner
    def _array_comparison(out, a, b, queue=None, op=None):
        if a.shape != b.shape:
            raise ValueError("shapes of comparison arguments do not match")
        return elementwise.get_array_comparison_kernel(
                out.context, op, a.dtype, b.dtype)

    def __eq__(self, other):
        if isinstance(other, Array):
            result = self._new_like_me(np.int8)
            result.add_event(
                    self._array_comparison(result, self, other, op="=="))
            return result
        else:
            result = self._new_like_me(np.int8)
            result.add_event(
                    self._scalar_comparison(result, self, other, op="=="))
            return result

    def __ne__(self, other):
        if isinstance(other, Array):
            result = self._new_like_me(np.int8)
            result.add_event(
                    self._array_comparison(result, self, other, op="!="))
            return result
        else:
            result = self._new_like_me(np.int8)
            result.add_event(
                    self._scalar_comparison(result, self, other, op="!="))
            return result

    def __le__(self, other):
        if isinstance(other, Array):
            result = self._new_like_me(np.int8)
            result.add_event(
                    self._array_comparison(result, self, other, op="<="))
            return result
        else:
            result = self._new_like_me(np.int8)
            self._scalar_comparison(result, self, other, op="<=")
            return result

    def __ge__(self, other):
        if isinstance(other, Array):
            result = self._new_like_me(np.int8)
            result.add_event(
                    self._array_comparison(result, self, other, op=">="))
            return result
        else:
            result = self._new_like_me(np.int8)
            result.add_event(
                    self._scalar_comparison(result, self, other, op=">="))
            return result

    def __lt__(self, other):
        if isinstance(other, Array):
            result = self._new_like_me(np.int8)
            result.add_event(
                    self._array_comparison(result, self, other, op="<"))
            return result
        else:
            result = self._new_like_me(np.int8)
            result.add_event(
                    self._scalar_comparison(result, self, other, op="<"))
            return result

    def __gt__(self, other):
        if isinstance(other, Array):
            result = self._new_like_me(np.int8)
            result.add_event(
                    self._array_comparison(result, self, other, op=">"))
            return result
        else:
            result = self._new_like_me(np.int8)
            result.add_event(
                    self._scalar_comparison(result, self, other, op=">"))
            return result

    # }}}

    # {{{ complex-valued business

    def real(self):
        if self.dtype.kind == "c":
            result = self._new_like_me(self.dtype.type(0).real.dtype)
            result.add_event(
                    self._real(result, self))
            return result
        else:
            return self
    real = property(real, doc=".. versionadded:: 2012.1")

    def imag(self):
        if self.dtype.kind == "c":
            result = self._new_like_me(self.dtype.type(0).real.dtype)
            result.add_event(
                    self._imag(result, self))
            return result
        else:
            return zeros_like(self)
    imag = property(imag, doc=".. versionadded:: 2012.1")

    def conj(self):
        """.. versionadded:: 2012.1"""
        if self.dtype.kind == "c":
            result = self._new_like_me()
            result.add_event(self._conj(result, self))
            return result
        else:
            return self

    # }}}

    # {{{ event management

    def add_event(self, evt):
        """Add *evt* to :attr:`events`. If :attr:`events` is too long, this method
        may implicitly wait for a subset of :attr:`events` and clear them from the
        list.
        """
        n_wait = 4

        self.events.append(evt)

        if len(self.events) > 3*n_wait:
            wait_events = self.events[:n_wait]
            cl.wait_for_events(wait_events)
            del self.events[:n_wait]

    def finish(self):
        """Wait for the entire contents of :attr:`events`, clear it."""

        if self.events:
            cl.wait_for_events(self.events)
            del self.events[:]

    # }}}

    # {{{ views

    def reshape(self, *shape, **kwargs):
        """Returns an array containing the same data with a new shape."""

        order = kwargs.pop("order", "C")
        if kwargs:
            raise TypeError("unexpected keyword arguments: %s"
                    % list(kwargs.keys()))

        if order not in "CF":
            raise ValueError("order must be either 'C' or 'F'")

        # TODO: add more error-checking, perhaps

        # FIXME: The following is overly conservative. As long as we don't change
        # our memory footprint, we're good.

        # if not self.flags.forc:
        #     raise RuntimeError("only contiguous arrays may "
        #             "be used as arguments to this operation")

        if isinstance(shape[0], tuple) or isinstance(shape[0], list):
            shape = tuple(shape[0])

        if -1 in shape:
            shape = list(shape)
            idx = shape.index(-1)
            size = -reduce(lambda x, y: x * y, shape, 1)
            shape[idx] = self.size // size
            if any(s < 0 for s in shape):
                raise ValueError("can only specify one unknown dimension")
            shape = tuple(shape)

        if shape == self.shape:
            return self._new_with_changes(
                    data=self.base_data, offset=self.offset, shape=shape,
                    strides=self.strides)

        import operator
        size = reduce(operator.mul, shape, 1)
        if size != self.size:
            raise ValueError("total size of new array must be unchanged")

        # {{{ determine reshaped strides

        # copied and translated from
        # https://github.com/numpy/numpy/blob/4083883228d61a3b571dec640185b5a5d983bf59/numpy/core/src/multiarray/shape.c  # noqa

        newdims = shape
        newnd = len(newdims)

        # Remove axes with dimension 1 from the old array. They have no effect
        # but would need special cases since their strides do not matter.

        olddims = []
        oldstrides = []
        for oi in range(len(self.shape)):
            s = self.shape[oi]
            if s != 1:
                olddims.append(s)
                oldstrides.append(self.strides[oi])

        oldnd = len(olddims)

        newstrides = [-1]*len(newdims)

        # oi to oj and ni to nj give the axis ranges currently worked with
        oi = 0
        oj = 1
        ni = 0
        nj = 1
        while ni < newnd and oi < oldnd:
            np = newdims[ni]
            op = olddims[oi]

            while np != op:
                if np < op:
                    # Misses trailing 1s, these are handled later
                    np *= newdims[nj]
                    nj += 1
                else:
                    op *= olddims[oj]
                    oj += 1

            # Check whether the original axes can be combined
            for ok in range(oi, oj-1):
                if order == "F":
                    if oldstrides[ok+1] != olddims[ok]*oldstrides[ok]:
                        raise ValueError("cannot reshape without copy")
                else:
                    # C order
                    if (oldstrides[ok] != olddims[ok+1]*oldstrides[ok+1]):
                        raise ValueError("cannot reshape without copy")

            # Calculate new strides for all axes currently worked with
            if order == "F":
                newstrides[ni] = oldstrides[oi]
                for nk in range(ni+1, nj):
                    newstrides[nk] = newstrides[nk - 1]*newdims[nk - 1]
            else:
                # C order
                newstrides[nj - 1] = oldstrides[oj - 1]
                for nk in range(nj-1, ni, -1):
                    newstrides[nk - 1] = newstrides[nk]*newdims[nk]

            ni = nj
            nj += 1

            oi = oj
            oj += 1

        # Set strides corresponding to trailing 1s of the new shape.
        if ni >= 1:
            last_stride = newstrides[ni - 1]
        else:
            last_stride = self.dtype.itemsize

        if order == "F":
            last_stride *= newdims[ni - 1]

        for nk in range(ni, len(shape)):
            newstrides[nk] = last_stride

        # }}}

        return self._new_with_changes(
                data=self.base_data, offset=self.offset, shape=shape,
                strides=tuple(newstrides))

    def ravel(self):
        """Returns flattened array containing the same data."""
        return self.reshape(self.size)

    def view(self, dtype=None):
        """Returns view of array with the same data. If *dtype* is different
        from current dtype, the actual bytes of memory will be reinterpreted.
        """

        if dtype is None:
            dtype = self.dtype

        old_itemsize = self.dtype.itemsize
        itemsize = np.dtype(dtype).itemsize

        from pytools import argmin2
        min_stride_axis = argmin2(
                (axis, abs(stride))
                for axis, stride in enumerate(self.strides))

        if self.shape[min_stride_axis] * old_itemsize % itemsize != 0:
            raise ValueError("new type not compatible with array")

        new_shape = (
                self.shape[:min_stride_axis]
                + (self.shape[min_stride_axis] * old_itemsize // itemsize,)
                + self.shape[min_stride_axis+1:])
        new_strides = (
                self.strides[:min_stride_axis]
                + (self.strides[min_stride_axis] * itemsize // old_itemsize,)
                + self.strides[min_stride_axis+1:])

        return self._new_with_changes(
                self.base_data, self.offset,
                shape=new_shape, dtype=dtype,
                strides=new_strides)

    def squeeze(self):
        """Returns a view of the array with dimensions of
        length 1 removed.

        .. versionadded:: 2015.2
        """
        new_shape = tuple([dim for dim in self.shape if dim > 1])
        new_strides = tuple([self.strides[i]
            for i, dim in enumerate(self.shape) if dim > 1])

        return self._new_with_changes(
                self.base_data, self.offset,
                shape=new_shape, strides=new_strides)

    def transpose(self, axes=None):
        """Permute the dimensions of an array.

        :arg axes: list of ints, optional.
            By default, reverse the dimensions, otherwise permute the axes
            according to the values given.

        :returns: :class:`Array` A view of the array with its axes permuted.

        .. versionadded:: 2015.2
        """

        if axes is None:
            axes = range(self.ndim-1, -1, -1)

        if len(axes) != len(self.shape):
            raise ValueError("axes don't match array")

        new_shape = [self.shape[axes[i]] for i in range(len(axes))]
        new_strides = [self.strides[axes[i]] for i in range(len(axes))]

        return self._new_with_changes(
                self.base_data, self.offset,
                shape=tuple(new_shape),
                strides=tuple(new_strides))

    @property
    def T(self):  # noqa
        """
        .. versionadded:: 2015.2
        """
        return self.transpose()

    # }}}

    def map_to_host(self, queue=None, flags=None, is_blocking=True, wait_for=None):
        """If *is_blocking*, return a :class:`numpy.ndarray` corresponding to the
        same memory as *self*.

        If *is_blocking* is not true, return a tuple ``(ary, evt)``, where
        *ary* is the above-mentioned array.

        The host array is obtained using :func:`pyopencl.enqueue_map_buffer`.
        See there for further details.

        :arg flags: A combination of :class:`pyopencl.map_flags`.
            Defaults to read-write.

        .. versionadded :: 2013.2
        """

        if flags is None:
            flags = cl.map_flags.READ | cl.map_flags.WRITE
        if wait_for is None:
            wait_for = []

        ary, evt = cl.enqueue_map_buffer(
                queue or self.queue, self.base_data, flags, self.offset,
                self.shape, self.dtype, strides=self.strides,
                wait_for=wait_for + self.events, is_blocking=is_blocking)

        if is_blocking:
            return ary
        else:
            return ary, evt

    # {{{ getitem/setitem

    def __getitem__(self, index):
        """
        .. versionadded:: 2013.1
        """

        if isinstance(index, Array):
            if index.dtype.kind != "i":
                raise TypeError(
                        "fancy indexing is only allowed with integers")
            if len(index.shape) != 1:
                raise NotImplementedError(
                        "multidimensional fancy indexing is not supported")
            if len(self.shape) != 1:
                raise NotImplementedError(
                        "fancy indexing into a multi-d array is not supported")

            return take(self, index)

        if not isinstance(index, tuple):
            index = (index,)

        new_shape = []
        new_offset = self.offset
        new_strides = []

        seen_ellipsis = False

        index_axis = 0
        array_axis = 0
        while index_axis < len(index):
            index_entry = index[index_axis]

            if array_axis > len(self.shape):
                raise IndexError("too many axes in index")

            if isinstance(index_entry, slice):
                start, stop, idx_stride = index_entry.indices(
                        self.shape[array_axis])

                array_stride = self.strides[array_axis]

                new_shape.append((abs(stop-start)-1)//abs(idx_stride)+1)
                new_strides.append(idx_stride*array_stride)
                new_offset += array_stride*start

                index_axis += 1
                array_axis += 1

            elif isinstance(index_entry, (int, np.integer)):
                array_shape = self.shape[array_axis]
                if index_entry < 0:
                    index_entry += array_shape

                if not (0 <= index_entry < array_shape):
                    raise IndexError(
                            "subindex in axis %d out of range" % index_axis)

                new_offset += self.strides[array_axis]*index_entry

                index_axis += 1
                array_axis += 1

            elif index_entry is Ellipsis:
                index_axis += 1

                remaining_index_count = len(index) - index_axis
                new_array_axis = len(self.shape) - remaining_index_count
                if new_array_axis < array_axis:
                    raise IndexError("invalid use of ellipsis in index")
                while array_axis < new_array_axis:
                    new_shape.append(self.shape[array_axis])
                    new_strides.append(self.strides[array_axis])
                    array_axis += 1

                if seen_ellipsis:
                    raise IndexError(
                            "more than one ellipsis not allowed in index")
                seen_ellipsis = True

            elif index_entry is np.newaxis:
                new_shape.append(1)
                new_strides.append(0)
                index_axis += 1

            else:
                raise IndexError("invalid subindex in axis %d" % index_axis)

        while array_axis < len(self.shape):
            new_shape.append(self.shape[array_axis])
            new_strides.append(self.strides[array_axis])

            array_axis += 1

        return self._new_with_changes(
                self.base_data, offset=new_offset,
                shape=tuple(new_shape),
                strides=tuple(new_strides))

    def setitem(self, subscript, value, queue=None, wait_for=None):
        """Like :meth:`__setitem__`, but with the ability to specify
        a *queue* and *wait_for*.

        .. versionadded:: 2013.1

        .. versionchanged:: 2013.2

            Added *wait_for*.
        """

        queue = queue or self.queue or value.queue
        if wait_for is None:
            wait_for = []
        wait_for = wait_for + self.events

        if isinstance(subscript, Array):
            if subscript.dtype.kind != "i":
                raise TypeError(
                        "fancy indexing is only allowed with integers")
            if len(subscript.shape) != 1:
                raise NotImplementedError(
                        "multidimensional fancy indexing is not supported")
            if len(self.shape) != 1:
                raise NotImplementedError(
                        "fancy indexing into a multi-d array is not supported")

            multi_put([value], subscript, out=[self], queue=queue,
                    wait_for=wait_for)
            return

        subarray = self[subscript]

        if isinstance(value, np.ndarray):
            if subarray.shape == value.shape and subarray.strides == value.strides:
                self.add_event(
                        cl.enqueue_copy(queue, subarray.base_data,
                            value, device_offset=subarray.offset, wait_for=wait_for))
                return
            else:
                value = to_device(queue, value, self.allocator)

        if isinstance(value, Array):
            if len(subarray.shape) != len(value.shape):
                raise NotImplementedError("broadcasting is not "
                        "supported in __setitem__")
            if subarray.shape != value.shape:
                raise ValueError("cannot assign between arrays of "
                        "differing shapes")
            if subarray.strides != value.strides:
                raise ValueError("cannot assign between arrays of "
                        "differing strides")

            self.add_event(
                    self._copy(subarray, value, queue=queue, wait_for=wait_for))

        else:
            # Let's assume it's a scalar
            subarray.fill(value, queue=queue, wait_for=wait_for)

    def __setitem__(self, subscript, value):
        """Set the slice of *self* identified *subscript* to *value*.

        *value* is allowed to be:

        * A :class:`Array` of the same :attr:`shape` and (for now) :attr:`strides`,
          but with potentially different :attr:`dtype`.
        * A :class:`numpy.ndarray` of the same :attr:`shape` and (for now)
          :attr:`strides`, but with potentially different :attr:`dtype`.
        * A scalar.

        Non-scalar broadcasting is not currently supported.

        .. versionadded:: 2013.1
        """
        self.setitem(subscript, value)

    # }}}

# }}}


# {{{ creation helpers

def as_strided(ary, shape=None, strides=None):
    """Make an :class:`Array` from the given array with the given
    shape and strides.
    """

    # undocumented for the moment

    if shape is None:
        shape = ary.shape
    if strides is None:
        strides = ary.strides

    return Array(ary.queue, shape, ary.dtype, allocator=ary.allocator,
            data=ary.data, strides=strides)


class _same_as_transfer(object):  # noqa
    pass


def to_device(queue, ary, allocator=None, async_=None,
        array_queue=_same_as_transfer, **kwargs):
    """Return a :class:`Array` that is an exact copy of the
    :class:`numpy.ndarray` instance *ary*.

    :arg array_queue: The :class:`CommandQueue` which will
        be stored in the resulting array. Useful
        to make sure there is no implicit queue associated
        with the array by passing *None*.

    See :class:`Array` for the meaning of *allocator*.

    .. versionchanged:: 2015.2
        *array_queue* argument was added.

    .. versionchanged:: 2017.2.1

        Python 3.7 makes ``async`` a reserved keyword. On older Pythons,
        we will continue to  accept *async* as a parameter, however this
        should be considered deprecated. *async_* is the new, official
        spelling.
    """

    # {{{ handle 'async' deprecation

    async_arg = kwargs.pop("async", None)
    if async_arg is not None:
        if async_ is not None:
            raise TypeError("may not specify both 'async' and 'async_'")
        async_ = async_arg

    if async_ is None:
        async_ = False

    if kwargs:
        raise TypeError("extra keyword arguments specified: %s"
                % ", ".join(kwargs))

    # }}}

    if _dtype_is_object(ary.dtype):
        raise RuntimeError("to_device does not work on object arrays.")

    if array_queue is _same_as_transfer:
        first_arg = queue
    else:
        first_arg = queue.context

    result = Array(first_arg, ary.shape, ary.dtype,
                    allocator=allocator, strides=ary.strides)
    result.set(ary, async_=async_, queue=queue)
    return result


empty = Array


def zeros(queue, shape, dtype, order="C", allocator=None):
    """Same as :func:`empty`, but the :class:`Array` is zero-initialized before
    being returned.

    .. versionchanged:: 2011.1
        *context* argument was deprecated.
    """

    result = Array(queue, shape, dtype,
            order=order, allocator=allocator)
    result._zero_fill()
    return result


def empty_like(ary, queue=_copy_queue, allocator=None):
    """Make a new, uninitialized :class:`Array` having the same properties
    as *other_ary*.
    """

    return ary._new_with_changes(data=None, offset=0, queue=queue,
            allocator=allocator)


def zeros_like(ary):
    """Make a new, zero-initialized :class:`Array` having the same properties
    as *other_ary*.
    """

    result = empty_like(ary)
    result._zero_fill()
    return result


@elwise_kernel_runner
def _arange_knl(result, start, step):
    return elementwise.get_arange_kernel(
            result.context, result.dtype)


def arange(queue, *args, **kwargs):
    """arange(queue, [start, ] stop [, step], **kwargs)
    Create a :class:`Array` filled with numbers spaced `step` apart,
    starting from `start` and ending at `stop`. If not given, *start*
    defaults to 0, *step* defaults to 1.

    For floating point arguments, the length of the result is
    `ceil((stop - start)/step)`.  This rule may result in the last
    element of the result being greater than `stop`.

    *dtype* is a required keyword argument.

    .. versionchanged:: 2011.1
        *context* argument was deprecated.

    .. versionchanged:: 2011.2
        *allocator* keyword argument was added.
    """

    # argument processing -----------------------------------------------------

    # Yuck. Thanks, numpy developers. ;)
    from pytools import Record

    class Info(Record):
        pass

    explicit_dtype = False

    inf = Info()
    inf.start = None
    inf.stop = None
    inf.step = None
    inf.dtype = None
    inf.allocator = None
    inf.wait_for = []

    if isinstance(args[-1], np.dtype):
        inf.dtype = args[-1]
        args = args[:-1]
        explicit_dtype = True

    argc = len(args)
    if argc == 0:
        raise ValueError("stop argument required")
    elif argc == 1:
        inf.stop = args[0]
    elif argc == 2:
        inf.start = args[0]
        inf.stop = args[1]
    elif argc == 3:
        inf.start = args[0]
        inf.stop = args[1]
        inf.step = args[2]
    else:
        raise ValueError("too many arguments")

    admissible_names = ["start", "stop", "step", "dtype", "allocator"]
    for k, v in six.iteritems(kwargs):
        if k in admissible_names:
            if getattr(inf, k) is None:
                setattr(inf, k, v)
                if k == "dtype":
                    explicit_dtype = True
            else:
                raise ValueError(
                        "may not specify '%s' by position and keyword" % k)
        else:
            raise ValueError("unexpected keyword argument '%s'" % k)

    if inf.start is None:
        inf.start = 0
    if inf.step is None:
        inf.step = 1
    if inf.dtype is None:
        inf.dtype = np.array([inf.start, inf.stop, inf.step]).dtype

    # actual functionality ----------------------------------------------------
    dtype = np.dtype(inf.dtype)
    start = dtype.type(inf.start)
    step = dtype.type(inf.step)
    stop = dtype.type(inf.stop)
    wait_for = inf.wait_for

    if not explicit_dtype:
        raise TypeError("arange requires a dtype argument")

    from math import ceil
    size = int(ceil((stop-start)/step))

    result = Array(queue, (size,), dtype, allocator=inf.allocator)
    result.add_event(
            _arange_knl(result, start, step, queue=queue, wait_for=wait_for))
    return result

# }}}


# {{{ take/put/concatenate/diff

@elwise_kernel_runner
def _take(result, ary, indices):
    return elementwise.get_take_kernel(
            result.context, result.dtype, indices.dtype)


def take(a, indices, out=None, queue=None, wait_for=None):
    """Return the :class:`Array` ``[a[indices[0]], ..., a[indices[n]]]``.
    For the moment, *a* must be a type that can be bound to a texture.
    """

    queue = queue or a.queue
    if out is None:
        out = Array(queue, indices.shape, a.dtype, allocator=a.allocator)

    assert len(indices.shape) == 1
    out.add_event(
            _take(out, a, indices, queue=queue, wait_for=wait_for))
    return out


def multi_take(arrays, indices, out=None, queue=None):
    if not len(arrays):
        return []

    assert len(indices.shape) == 1

    from pytools import single_valued
    a_dtype = single_valued(a.dtype for a in arrays)
    a_allocator = arrays[0].dtype
    context = indices.context
    queue = queue or indices.queue

    vec_count = len(arrays)

    if out is None:
        out = [Array(context, queue, indices.shape, a_dtype,
            allocator=a_allocator)
                for i in range(vec_count)]
    else:
        if len(out) != len(arrays):
            raise ValueError("out and arrays must have the same length")

    chunk_size = _builtin_min(vec_count, 10)

    def make_func_for_chunk_size(chunk_size):
        knl = elementwise.get_take_kernel(
                indices.context, a_dtype, indices.dtype,
                vec_count=chunk_size)
        knl.set_block_shape(*indices._block)
        return knl

    knl = make_func_for_chunk_size(chunk_size)

    for start_i in range(0, len(arrays), chunk_size):
        chunk_slice = slice(start_i, start_i+chunk_size)

        if start_i + chunk_size > vec_count:
            knl = make_func_for_chunk_size(vec_count-start_i)

        gs, ls = indices.get_sizes(queue,
                knl.get_work_group_info(
                    cl.kernel_work_group_info.WORK_GROUP_SIZE,
                    queue.device))

        wait_for_this = (indices.events
            + _builtin_sum((i.events for i in arrays[chunk_slice]), [])
            + _builtin_sum((o.events for o in out[chunk_slice]), []))
        evt = knl(queue, gs, ls,
                indices.data,
                *([o.data for o in out[chunk_slice]]
                    + [i.data for i in arrays[chunk_slice]]
                    + [indices.size]), wait_for=wait_for_this)
        for o in out[chunk_slice]:
            o.add_event(evt)

    return out


def multi_take_put(arrays, dest_indices, src_indices, dest_shape=None,
        out=None, queue=None, src_offsets=None):
    if not len(arrays):
        return []

    from pytools import single_valued
    a_dtype = single_valued(a.dtype for a in arrays)
    a_allocator = arrays[0].allocator
    context = src_indices.context
    queue = queue or src_indices.queue

    vec_count = len(arrays)

    if out is None:
        out = [Array(queue, dest_shape, a_dtype, allocator=a_allocator)
                for i in range(vec_count)]
    else:
        if a_dtype != single_valued(o.dtype for o in out):
            raise TypeError("arrays and out must have the same dtype")
        if len(out) != vec_count:
            raise ValueError("out and arrays must have the same length")

    if src_indices.dtype != dest_indices.dtype:
        raise TypeError(
                "src_indices and dest_indices must have the same dtype")

    if len(src_indices.shape) != 1:
        raise ValueError("src_indices must be 1D")

    if src_indices.shape != dest_indices.shape:
        raise ValueError(
                "src_indices and dest_indices must have the same shape")

    if src_offsets is None:
        src_offsets_list = []
    else:
        src_offsets_list = src_offsets
        if len(src_offsets) != vec_count:
            raise ValueError(
                    "src_indices and src_offsets must have the same length")

    max_chunk_size = 10

    chunk_size = _builtin_min(vec_count, max_chunk_size)

    def make_func_for_chunk_size(chunk_size):
        return elementwise.get_take_put_kernel(context,
                a_dtype, src_indices.dtype,
                with_offsets=src_offsets is not None,
                vec_count=chunk_size)

    knl = make_func_for_chunk_size(chunk_size)

    for start_i in range(0, len(arrays), chunk_size):
        chunk_slice = slice(start_i, start_i+chunk_size)

        if start_i + chunk_size > vec_count:
            knl = make_func_for_chunk_size(vec_count-start_i)

        gs, ls = src_indices.get_sizes(queue,
                knl.get_work_group_info(
                    cl.kernel_work_group_info.WORK_GROUP_SIZE,
                    queue.device))

        from pytools import flatten
        wait_for_this = (dest_indices.events + src_indices.events
            + _builtin_sum((i.events for i in arrays[chunk_slice]), [])
            + _builtin_sum((o.events for o in out[chunk_slice]), []))
        evt = knl(queue, gs, ls,
                *([o.data for o in out[chunk_slice]]
                    + [dest_indices.base_data,
                        dest_indices.offset,
                        src_indices.base_data,
                        src_indices.offset]
                    + list(flatten(
                        (i.base_data, i.offset)
                        for i in arrays[chunk_slice]))
                    + src_offsets_list[chunk_slice]
                    + [src_indices.size]), wait_for=wait_for_this)
        for o in out[chunk_slice]:
            o.add_event(evt)

    return out


def multi_put(arrays, dest_indices, dest_shape=None, out=None, queue=None,
        wait_for=None):
    if not len(arrays):
        return []

    from pytools import single_valued
    a_dtype = single_valued(a.dtype for a in arrays)
    a_allocator = arrays[0].allocator
    context = dest_indices.context
    queue = queue or dest_indices.queue
    if wait_for is None:
        wait_for = []
    wait_for = wait_for + dest_indices.events

    vec_count = len(arrays)

    if out is None:
        out = [Array(queue, dest_shape, a_dtype, allocator=a_allocator)
                for _ in range(vec_count)]
    else:
        if a_dtype != single_valued(o.dtype for o in out):
            raise TypeError("arrays and out must have the same dtype")
        if len(out) != vec_count:
            raise ValueError("out and arrays must have the same length")

    if len(dest_indices.shape) != 1:
        raise ValueError("dest_indices must be 1D")

    chunk_size = _builtin_min(vec_count, 10)

    # array of bools to specify whether the array of same index in this chunk
    # will be filled with a single value.
    use_fill = np.ndarray((chunk_size,), dtype=np.uint8)
    array_lengths = np.ndarray((chunk_size,), dtype=np.int64)

    def make_func_for_chunk_size(chunk_size):
        knl = elementwise.get_put_kernel(
                context, a_dtype, dest_indices.dtype,
                vec_count=chunk_size)
        return knl

    knl = make_func_for_chunk_size(chunk_size)

    for start_i in range(0, len(arrays), chunk_size):
        chunk_slice = slice(start_i, start_i+chunk_size)
        for fill_idx, ary in enumerate(arrays[chunk_slice]):
            # If there is only one value in the values array for this src array
            # in the chunk then fill every index in `dest_idx` array with it.
            use_fill[fill_idx] = 1 if ary.size == 1 else 0
            array_lengths[fill_idx] = len(ary)
        # Copy the populated `use_fill` array to a buffer on the device.
        use_fill_cla = to_device(queue, use_fill)
        array_lengths_cla = to_device(queue, array_lengths)

        if start_i + chunk_size > vec_count:
            knl = make_func_for_chunk_size(vec_count-start_i)

        gs, ls = dest_indices.get_sizes(queue,
                knl.get_work_group_info(
                    cl.kernel_work_group_info.WORK_GROUP_SIZE,
                    queue.device))

        from pytools import flatten
        wait_for_this = (wait_for
            + _builtin_sum((i.events for i in arrays[chunk_slice]), [])
            + _builtin_sum((o.events for o in out[chunk_slice]), []))
        evt = knl(queue, gs, ls,
                *(
                    list(flatten(
                        (o.base_data, o.offset)
                        for o in out[chunk_slice]))
                    + [dest_indices.base_data, dest_indices.offset]
                    + list(flatten(
                        (i.base_data, i.offset)
                        for i in arrays[chunk_slice]))
                    + [use_fill_cla.base_data, use_fill_cla.offset]
                    + [array_lengths_cla.base_data, array_lengths_cla.offset]
                    + [dest_indices.size]),
                **dict(wait_for=wait_for_this))

        for o in out[chunk_slice]:
            o.add_event(evt)

    return out


def concatenate(arrays, axis=0, queue=None, allocator=None):
    """
    .. versionadded:: 2013.1
    """
    # {{{ find properties of result array

    shape = None

    for i_ary, ary in enumerate(arrays):
        queue = queue or ary.queue
        allocator = allocator or ary.allocator

        if shape is None:
            # first array
            shape = list(ary.shape)
        else:
            if len(ary.shape) != len(shape):
                raise ValueError("%d'th array has different number of axes "
                        "(shold have %d, has %d)"
                        % (i_ary, len(ary.shape), len(shape)))

            ary_shape_list = list(ary.shape)
            if (ary_shape_list[:axis] != shape[:axis]
                    or ary_shape_list[axis+1:] != shape[axis+1:]):
                raise ValueError("%d'th array has residual not matching "
                        "other arrays" % i_ary)

            shape[axis] += ary.shape[axis]

    # }}}

    shape = tuple(shape)
    dtype = np.find_common_type([ary.dtype for ary in arrays], [])
    result = empty(queue, shape, dtype, allocator=allocator)

    full_slice = (slice(None),) * len(shape)

    base_idx = 0
    for ary in arrays:
        my_len = ary.shape[axis]
        result.setitem(
                full_slice[:axis]
                + (slice(base_idx, base_idx+my_len),)
                + full_slice[axis+1:],
                ary)

        base_idx += my_len

    return result


@elwise_kernel_runner
def _diff(result, array):
    return elementwise.get_diff_kernel(array.context, array.dtype)


def diff(array, queue=None, allocator=None):
    """
    .. versionadded:: 2013.2
    """

    if len(array.shape) != 1:
        raise ValueError("multi-D arrays are not supported")

    n, = array.shape

    queue = queue or array.queue
    allocator = allocator or array.allocator

    result = empty(queue, (n-1,), array.dtype, allocator=allocator)
    event1 = _diff(result, array, queue=queue)
    result.add_event(event1)
    return result


def hstack(arrays, queue=None):
    from pyopencl.array import empty

    if len(arrays) == 0:
        return empty(queue, (), dtype=np.float64)

    if queue is None:
        for ary in arrays:
            if ary.queue is not None:
                queue = ary.queue
                break

    from pytools import all_equal, single_valued
    if not all_equal(len(ary.shape) for ary in arrays):
        raise ValueError("arguments must all have the same number of axes")

    lead_shape = single_valued(ary.shape[:-1] for ary in arrays)

    w = _builtin_sum([ary.shape[-1] for ary in arrays])
    result = empty(queue, lead_shape+(w,), arrays[0].dtype)
    index = 0
    for ary in arrays:
        result[..., index:index+ary.shape[-1]] = ary
        index += ary.shape[-1]

    return result

# }}}


# {{{ shape manipulation

def transpose(a, axes=None):
    """Permute the dimensions of an array.

    :arg a: :class:`Array`
    :arg axes: list of ints, optional.
        By default, reverse the dimensions, otherwise permute the axes
        according to the values given.

    :returns: :class:`Array` A view of the array with its axes permuted.
    """
    return a.transpose(axes)


def reshape(a, shape):
    """Gives a new shape to an array without changing its data.

    .. versionadded:: 2015.2
    """

    return a.reshape(shape)

# }}}


# {{{ conditionals

@elwise_kernel_runner
def _if_positive(result, criterion, then_, else_):
    return elementwise.get_if_positive_kernel(
            result.context, criterion.dtype, then_.dtype)


def if_positive(criterion, then_, else_, out=None, queue=None):
    """Return an array like *then_*, which, for the element at index *i*,
    contains *then_[i]* if *criterion[i]>0*, else *else_[i]*.
    """

    if not (criterion.shape == then_.shape == else_.shape):
        raise ValueError("shapes do not match")

    if not (then_.dtype == else_.dtype):
        raise ValueError("dtypes do not match")

    if out is None:
        out = empty_like(then_)
    event1 = _if_positive(out, criterion, then_, else_, queue=queue)
    out.add_event(event1)
    return out


def maximum(a, b, out=None, queue=None):
    """Return the elementwise maximum of *a* and *b*."""

    # silly, but functional
    return if_positive(a.mul_add(1, b, -1, queue=queue), a, b,
            queue=queue, out=out)


def minimum(a, b, out=None, queue=None):
    """Return the elementwise minimum of *a* and *b*."""
    # silly, but functional
    return if_positive(a.mul_add(1, b, -1, queue=queue), b, a,
            queue=queue, out=out)

# }}}


# {{{ reductions
_builtin_sum = sum
_builtin_min = min
_builtin_max = max


def sum(a, dtype=None, queue=None, slice=None):
    """
    .. versionadded:: 2011.1
    """
    from pyopencl.reduction import get_sum_kernel
    krnl = get_sum_kernel(a.context, dtype, a.dtype)
    result, event1 = krnl(a, queue=queue, slice=slice, wait_for=a.events,
            return_event=True)
    result.add_event(event1)
    return result


def dot(a, b, dtype=None, queue=None, slice=None):
    """
    .. versionadded:: 2011.1
    """
    from pyopencl.reduction import get_dot_kernel
    krnl = get_dot_kernel(a.context, dtype, a.dtype, b.dtype)
    result, event1 = krnl(a, b, queue=queue, slice=slice,
            wait_for=a.events + b.events, return_event=True)
    result.add_event(event1)
    return result


def vdot(a, b, dtype=None, queue=None, slice=None):
    """Like :func:`numpy.vdot`.

    .. versionadded:: 2013.1
    """
    from pyopencl.reduction import get_dot_kernel
    krnl = get_dot_kernel(a.context, dtype, a.dtype, b.dtype,
            conjugate_first=True)
    result, event1 = krnl(a, b, queue=queue, slice=slice,
            wait_for=a.events + b.events, return_event=True)
    result.add_event(event1)
    return result


def subset_dot(subset, a, b, dtype=None, queue=None, slice=None):
    """
    .. versionadded:: 2011.1
    """
    from pyopencl.reduction import get_subset_dot_kernel
    krnl = get_subset_dot_kernel(
            a.context, dtype, subset.dtype, a.dtype, b.dtype)
    result, event1 = krnl(subset, a, b, queue=queue, slice=slice,
            wait_for=subset.events + a.events + b.events, return_event=True)
    result.add_event(event1)
    return result


def _make_minmax_kernel(what):
    def f(a, queue=None):
        from pyopencl.reduction import get_minmax_kernel
        krnl = get_minmax_kernel(a.context, what, a.dtype)
        result, event1 = krnl(a, queue=queue, wait_for=a.events,
                return_event=True)
        result.add_event(event1)
        return result

    return f


min = _make_minmax_kernel("min")
min.__doc__ = """
    .. versionadded:: 2011.1
    """

max = _make_minmax_kernel("max")
max.__doc__ = """
    .. versionadded:: 2011.1
    """


def _make_subset_minmax_kernel(what):
    def f(subset, a, queue=None, slice=None):
        from pyopencl.reduction import get_subset_minmax_kernel
        krnl = get_subset_minmax_kernel(a.context, what, a.dtype, subset.dtype)
        result, event1 = krnl(subset, a,  queue=queue, slice=slice,
                wait_for=a.events + subset.events, return_event=True)
        result.add_event(event1)
        return result
    return f


subset_min = _make_subset_minmax_kernel("min")
subset_min.__doc__ = """.. versionadded:: 2011.1"""
subset_max = _make_subset_minmax_kernel("max")
subset_max.__doc__ = """.. versionadded:: 2011.1"""

# }}}


# {{{ scans

def cumsum(a, output_dtype=None, queue=None,
        wait_for=None, return_event=False):
    # undocumented for now

    """
    .. versionadded:: 2013.1
    """

    if output_dtype is None:
        output_dtype = a.dtype
    if wait_for is None:
        wait_for = []

    result = a._new_like_me(output_dtype)

    from pyopencl.scan import get_cumsum_kernel
    krnl = get_cumsum_kernel(a.context, a.dtype, output_dtype)
    evt = krnl(a, result, queue=queue, wait_for=wait_for + a.events)
    result.add_event(evt)

    if return_event:
        return evt, result
    else:
        return result

# }}}

# vim: foldmethod=marker
