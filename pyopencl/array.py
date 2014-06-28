"""CL device arrays."""

from __future__ import division

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


import numpy as np
import pyopencl.elementwise as elementwise
import pyopencl as cl
from pytools import memoize_method
from pyopencl.compyte.array import (
        as_strided as _as_strided,
        f_contiguous_strides as _f_contiguous_strides,
        c_contiguous_strides as _c_contiguous_strides,
        ArrayFlags as _ArrayFlags,
        get_common_dtype as _get_common_dtype_base)
from pyopencl.compyte.dtypes import DTypeDict as _DTypeDict
from pyopencl.characterize import has_double_support


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
except:
    def _dtype_is_object(t):
        return False

# {{{ vector types

class vec:
    pass


def _create_vector_types():
    field_names = ["x", "y", "z", "w"]

    from pyopencl.tools import get_or_register_dtype

    vec.types = {}
    vec.type_to_scalar_and_count = _DTypeDict()

    counts = [2, 3, 4, 8, 16]

    for base_name, base_type in [
            ('char', np.int8),
            ('uchar', np.uint8),
            ('short', np.int16),
            ('ushort', np.uint16),
            ('int', np.int32),
            ('uint', np.uint32),
            ('long', np.int64),
            ('ulong', np.uint64),
            ('float', np.float32),
            ('double', np.float64),
            ]:
        for count in counts:
            name = "%s%d" % (base_name, count)

            titles = field_names[:count]

            padded_count = count
            if count == 3:
                padded_count = 4

            names = ["s%d" % i for i in range(count)]
            while len(names) < padded_count:
                names.append("padding%d" % (len(names)-count))

            if len(titles) < len(names):
                titles.extend((len(names)-len(titles))*[None])

            try:
                dtype = np.dtype(dict(
                    names=names,
                    formats=[base_type]*padded_count,
                    titles=titles))
            except NotImplementedError:
                try:
                    dtype = np.dtype([((n, title), base_type)
                                      for (n, title) in zip(names, titles)])
                except TypeError:
                    dtype = np.dtype([(n, base_type) for (n, title)
                                      in zip(names, titles)])

            get_or_register_dtype(name, dtype)

            setattr(vec, name, dtype)

            def create_array(dtype, count, padded_count, *args, **kwargs):
                if len(args) < count:
                    from warnings import warn
                    warn("default values for make_xxx are deprecated;"
                            " instead specify all parameters or use"
                            " array.vec.zeros_xxx", DeprecationWarning)
                padded_args = tuple(list(args)+[0]*(padded_count-len(args)))
                array = eval("array(padded_args, dtype=dtype)",
                        dict(array=np.array, padded_args=padded_args,
                        dtype=dtype))
                for key, val in kwargs.items():
                    array[key] = val
                return array

            setattr(vec, "make_"+name, staticmethod(eval(
                    "lambda *args, **kwargs: create_array(dtype, %i, %i, "
                    "*args, **kwargs)" % (count, padded_count),
                    dict(create_array=create_array, dtype=dtype))))
            setattr(vec, "filled_"+name, staticmethod(eval(
                    "lambda val: vec.make_%s(*[val]*%i)" % (name, count))))
            setattr(vec, "zeros_"+name,
                    staticmethod(eval("lambda: vec.filled_%s(0)" % (name))))
            setattr(vec, "ones_"+name,
                    staticmethod(eval("lambda: vec.filled_%s(1)" % (name))))

            vec.types[np.dtype(base_type), count] = dtype
            vec.type_to_scalar_and_count[dtype] = np.dtype(base_type), count

_create_vector_types()

# }}}


# {{{ helper functionality

def splay(queue, n, kernel_specific_max_wg_size=None):
    dev = queue.device
    max_work_items = _builtin_min(128, dev.max_work_group_size)

    if kernel_specific_max_wg_size is not None:
        from __builtin__ import min
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


class _copy_queue:
    pass


class Array(object):
    """A :class:`numpy.ndarray` work-alike that stores its data and performs
    its computations on the compute device.  *shape* and *dtype* work exactly
    as in :mod:`numpy`.  Arithmetic methods in :class:`Array` support the
    broadcasting of scalars. (e.g. `array+5`)

    *cqa* must be a :class:`pyopencl.CommandQueue` or a :class:`pyopencl.Context`.

    If it is a queue, *cqa* specifies the queue in which the array carries out
    its computations by default. If a default queue (and thereby overloaded
    operators and many other niceties) are not desired, pass a
    :class:`Context`.

    *cqa* will at some point be renamed *cq*, so it should be considered
    'positional-only'. Arguments starting from 'order' should be considered
    keyword-only.

    *allocator* may be `None` or a callable that, upon being called with an
    argument of the number of bytes to be allocated, returns an
    :class:`pyopencl.Buffer` object. (A :class:`pyopencl.tools.MemoryPool`
    instance is one useful example of an object to pass here.)

    .. versionchanged:: 2011.1
        Renamed *context* to *cqa*, made it general-purpose.

        All arguments beyond *order* should be considered keyword-only.

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
        :attr:`offset` in units of :attr:`dtype`.

        Unlike :attr:`data`, retrieving :attr:`base_data` always succeeds.

        .. versionadded:: 2013.1

    .. attribute :: offset

        See :attr:`base_data`.

        .. versionadded:: 2013.1

    .. attribute :: shape

        The tuple of lengths of each dimension in the array.

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
    .. automethod :: set
    .. automethod :: get
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

    .. automethod :: __abs__

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
    """

    __array_priority__ = 100

    def __init__(self, cqa, shape, dtype, order="C", allocator=None,
            data=None, offset=0, queue=None, strides=None, events=None):
        # {{{ backward compatibility

        from warnings import warn
        if queue is not None:
            warn("Passing the queue to the array through anything but the "
                    "first argument of the Array constructor is deprecated. "
                    "This will be continue to be accepted throughout the "
                    "2013.[0-6] versions of PyOpenCL.",
                    DeprecationWarning, 2)

        if isinstance(cqa, cl.CommandQueue):
            if queue is not None:
                raise TypeError("can't specify queue in 'cqa' and "
                        "'queue' arguments")
            queue = cqa

        elif isinstance(cqa, cl.Context):
            context = cqa

            if queue is not None:
                raise TypeError("may not pass a context and a queue "
                        "(just pass the queue)")
            if allocator is not None:
                # "is" would be wrong because two Python objects are allowed
                # to hold handles to the same context.

                # FIXME It would be nice to check this. But it would require
                # changing the allocator interface. Trust the user for now.

                #assert allocator.context == context
                pass

        else:
            # cqa is assumed to be an allocator
            warn("Passing an allocator for the 'cqa' parameter is deprecated. "
                    "This usage will be continue to be accepted throughout "
                    "the 2013.[0-6] versions of PyOpenCL.",
                    DeprecationWarning, stacklevel=2)
            if allocator is not None:
                raise TypeError("can't specify allocator in 'cqa' and "
                        "'allocator' arguments")

            allocator = cqa

        # Queue-less arrays do have a purpose in life.
        # They don't do very much, but at least they don't run kernels
        # in random queues.
        #
        # See also :meth:`with_queue`.

        # }}}

        # invariant here: allocator, queue set

        # {{{ determine shape and strides
        dtype = np.dtype(dtype)

        try:
            s = 1
            for dim in shape:
                s *= dim
        except TypeError:
            import sys
            if sys.version_info >= (3,):
                admissible_types = (int, np.integer)
            else:
                admissible_types = (int, long, np.integer)

            if not isinstance(shape, admissible_types):
                raise TypeError("shape must either be iterable or "
                        "castable to an integer")
            s = shape
            shape = (shape,)

        if isinstance(s, np.integer):
            # bombs if s is a Python integer
            s = np.asscalar(s)

        if strides is None:
            strides = _make_strides(dtype.itemsize, shape, order)

        else:
            # FIXME: We should possibly perform some plausibility
            # checking on 'strides' here.

            strides = tuple(strides)

        # }}}

        if _dtype_is_object(dtype):
            raise TypeError("object arrays on the compute device are not allowed")

        self.queue = queue
        self.shape = shape
        self.dtype = dtype
        self.strides = strides
        if events is None:
            self.events = []
        else:
            self.events = events

        self.size = s
        alloc_nbytes = self.nbytes = self.dtype.itemsize * self.size

        self.allocator = allocator

        if data is None:
            if not alloc_nbytes:
                # Work around CL not allowing zero-sized buffers.
                alloc_nbytes = 1

            if allocator is None:
                # FIXME remove me when queues become required
                if queue is not None:
                    context = queue.context

                self.base_data = cl.Buffer(
                        context, cl.mem_flags.READ_WRITE, alloc_nbytes)
            else:
                self.base_data = self.allocator(alloc_nbytes)
        else:
            self.base_data = data

        self.offset = offset

    @property
    def context(self):
        return self.base_data.context

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
            strides=None, queue=_copy_queue):
        """
        :arg data: *None* means alocate a new array.
        """
        if shape is None:
            shape = self.shape
        if dtype is None:
            dtype = self.dtype
        if strides is None:
            strides = self.strides
        if queue is _copy_queue:
            queue = self.queue

        if queue is not None:
            return Array(queue, shape, dtype, allocator=self.allocator,
                    strides=strides, data=data, offset=offset,
                    events=self.events)
        else:
            return Array(self.context, shape, dtype, queue=queue,
                    strides=strides, data=data, offset=offset,
                    events=self.events, allocator=self.allocator)

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

    def set(self, ary, queue=None, async=False):
        """Transfer the contents the :class:`numpy.ndarray` object *ary*
        onto the device.

        *ary* must have the same dtype and size (not necessarily shape) as
        *self*.
        """

        assert ary.size == self.size
        assert ary.dtype == self.dtype

        if not ary.flags.forc:
            raise RuntimeError("cannot set from non-contiguous array")

            ary = ary.copy()

        if ary.strides != self.strides:
            from warnings import warn
            warn("Setting array from one with different "
                    "strides/storage order. This will cease to work "
                    "in 2013.x.",
                    stacklevel=2)

        if self.size:
            cl.enqueue_copy(queue or self.queue, self.base_data, ary,
                    device_offset=self.offset,
                    is_blocking=not async)

    def get(self, queue=None, ary=None, async=False):
        """Transfer the contents of *self* into *ary* or a newly allocated
        :mod:`numpy.ndarray`. If *ary* is given, it must have the right
        size (not necessarily shape) and dtype.
        """

        if not self.size:
            return ary or np.empty(self.shape, self.dtype)

        index = 0
        order = None
        prev_stride = None
        new_strides = []
        for stride in self.strides:
            if prev_stride:
                if not order and stride > prev_stride:
                    order = 'F'
                elif not order and stride < prev_stride:
                    order = 'C'
                elif ((order == 'F' and stride < prev_stride) or
                      (order == 'C' and stride > prev_stride)):
                    raise TypeError(
                        "Array must have row- or column-major layout")
            new_strides = map(lambda x: x * self.shape[index], new_strides)
            new_strides.append(1)
            prev_stride = stride
            index += 1
        if not order: order = 'C'
        new_strides = map(lambda x: x * self.dtype.itemsize, new_strides)
        new_strides = list(new_strides)

        element_strides = list(map(lambda x: int(x / self.dtype.itemsize),
                                   self.strides))

        if order == 'F':
            new_strides.reverse()
            element_strides.reverse()

        if ary is None:
            ary = np.empty(self.shape, self.dtype)

            if self.flags.forc:
                ary = _as_strided(ary, strides=self.strides)
                cl.enqueue_copy(queue or self.queue, ary, self.base_data,
                                device_offset=self.offset,
                                is_blocking=not async)
                return ary

            ary = _as_strided(ary, strides = new_strides)

        else:
            if ary.size != self.size:
                raise TypeError("'ary' has non-matching size")
            if ary.dtype != self.dtype:
                raise TypeError("'ary' has non-matching type")

        region = list(self.shape)
        region[0] *= self.dtype.itemsize

        if len(self.shape) == 2:
            row_pitch = element_strides[0] * element_strides[1] * self.dtype.itemsize
            slice_pitch = 1
        elif len(self.shape) == 3:
            row_pitch = element_strides[-1] * element_strides[-2] * self.dtype.itemsize
            slice_pitch = row_pitch * self.shape[0]
        else:
            raise TypeError(
                "Only 2- or 3-D non-contiguous arrays are currently supported")

        cl.enqueue_copy(queue or self.queue, ary, self.base_data,
                        buffer_origin=(self.offset, 0, 0),
                        host_origin=(0, 0, 0),
                        region=tuple(region),
                        buffer_pitches=(row_pitch, slice_pitch),
                        host_pitches=(0,0))

        return ary

    def copy(self, queue=None):
        """.. versionadded:: 2013.1"""

        queue = queue or self.queue
        result = self._new_like_me()
        cl.enqueue_copy(queue, result.base_data, self.base_data,
                src_offset=self.offset, byte_count=self.nbytes)

        return result

    def __str__(self):
        return str(self.get())

    def __repr__(self):
        return repr(self.get())

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
        else:
            if dtype == self.dtype:
                strides = self.strides

        queue = queue or self.queue
        if queue is not None:
            return self.__class__(queue, self.shape, dtype,
                    allocator=self.allocator, strides=strides)
        elif self.allocator is not None:
            return self.__class__(self.allocator, self.shape, dtype,
                    strides=strides)
        else:
            return self.__class__(self.context, self.shape, dtype,
                    strides=strides)

    # }}}

    # {{{ operators

    def mul_add(self, selffac, other, otherfac, queue=None):
        """Return `selffac * self + otherfac*other`.
        """
        result = self._new_like_me(
                _get_common_dtype(self, other, queue or self.queue))
        self._axpbyz(result, selffac, self, otherfac, other)
        return result

    def __add__(self, other):
        """Add an array with an array or an array with a scalar."""

        if isinstance(other, Array):
            # add another vector
            result = self._new_like_me(
                    _get_common_dtype(self, other, self.queue))
            self._axpbyz(result,
                    self.dtype.type(1), self,
                    other.dtype.type(1), other)
            return result
        else:
            # add a scalar
            if other == 0:
                return self.copy()
            else:
                common_dtype = _get_common_dtype(self, other, self.queue)
                result = self._new_like_me(common_dtype)
                self._axpbz(result, self.dtype.type(1),
                        self, common_dtype.type(other))
                return result

    __radd__ = __add__

    def __sub__(self, other):
        """Substract an array from an array or a scalar from an array."""

        if isinstance(other, Array):
            result = self._new_like_me(
                    _get_common_dtype(self, other, self.queue))
            self._axpbyz(result,
                    self.dtype.type(1), self,
                    other.dtype.type(-1), other)
            return result
        else:
            # subtract a scalar
            if other == 0:
                return self.copy()
            else:
                result = self._new_like_me(
                        _get_common_dtype(self, other, self.queue))
                self._axpbz(result, self.dtype.type(1), self, -other)
                return result

    def __rsub__(self, other):
        """Substracts an array by a scalar or an array::

           x = n - self
        """
        common_dtype = _get_common_dtype(self, other, self.queue)
        # other must be a scalar
        result = self._new_like_me(common_dtype)
        self._axpbz(result, self.dtype.type(-1), self,
                common_dtype.type(other))
        return result

    def __iadd__(self, other):
        if isinstance(other, Array):
            self._axpbyz(self,
                    self.dtype.type(1), self,
                    other.dtype.type(1), other)
            return self
        else:
            self._axpbz(self, self.dtype.type(1), self, other)
            return self

    def __isub__(self, other):
        if isinstance(other, Array):
            self._axpbyz(self, self.dtype.type(1), self,
                    other.dtype.type(-1), other)
            return self
        else:
            self._axpbz(self, self.dtype.type(1), self, -other)
            return self

    def __neg__(self):
        result = self._new_like_me()
        self._axpbz(result, -1, self, 0)
        return result

    def __mul__(self, other):
        if isinstance(other, Array):
            result = self._new_like_me(
                    _get_common_dtype(self, other, self.queue))
            self._elwise_multiply(result, self, other)
            return result
        else:
            common_dtype = _get_common_dtype(self, other, self.queue)
            result = self._new_like_me(common_dtype)
            self._axpbz(result,
                    common_dtype.type(other), self, self.dtype.type(0))
            return result

    def __rmul__(self, scalar):
        common_dtype = _get_common_dtype(self, scalar, self.queue)
        result = self._new_like_me(common_dtype)
        self._axpbz(result,
                common_dtype.type(scalar), self, self.dtype.type(0))
        return result

    def __imul__(self, other):
        if isinstance(other, Array):
            self._elwise_multiply(self, self, other)
        else:
            # scalar
            self._axpbz(self, other, self, self.dtype.type(0))

        return self

    def __div__(self, other):
        """Divides an array by an array or a scalar, i.e. ``self / other``.
        """
        if isinstance(other, Array):
            result = self._new_like_me(
                    _get_common_dtype(self, other, self.queue))
            self._div(result, self, other)
        else:
            if other == 1:
                return self.copy()
            else:
                # create a new array for the result
                common_dtype = _get_common_dtype(self, other, self.queue)
                result = self._new_like_me(common_dtype)
                self._axpbz(result,
                        common_dtype.type(1/other), self, self.dtype.type(0))

        return result

    __truediv__ = __div__

    def __rdiv__(self, other):
        """Divides an array by a scalar or an array, i.e. ``other / self``.
        """

        if isinstance(other, Array):
            result = self._new_like_me(
                    _get_common_dtype(self, other, self.queue))
            other._div(result, self)
        else:
            # create a new array for the result
            common_dtype = _get_common_dtype(self, other, self.queue)
            result = self._new_like_me(common_dtype)
            self._rdiv_scalar(result, self, common_dtype.type(other))

        return result

    __rtruediv__ = __rdiv__

    def fill(self, value, queue=None, wait_for=None):
        """Fill the array with *scalar*.

        :returns: *self*.
        """
        self.events.append(
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
        self._abs(result, self)
        return result

    def __pow__(self, other):
        """Exponentiation by a scalar or elementwise by another
        :class:`Array`.
        """

        if isinstance(other, Array):
            assert self.shape == other.shape

            result = self._new_like_me(
                    _get_common_dtype(self, other, self.queue))
            self._pow_array(result, self, other)
        else:
            result = self._new_like_me(
                    _get_common_dtype(self, other, self.queue))
            self._pow_scalar(result, self, other)

        return result

    def __rpow__(self, other):
        # other must be a scalar
        common_dtype = _get_common_dtype(self, other, self.queue)
        result = self._new_like_me(common_dtype)
        self._rpow_scalar(result, common_dtype.type(other), self)
        return result

    # }}}

    def reverse(self, queue=None):
        """Return this array in reversed order. The array is treated
        as one-dimensional.
        """

        result = self._new_like_me()
        self._reverse(result, self)
        return result

    def astype(self, dtype, queue=None):
        """Return a copy of *self*, cast to *dtype*."""
        if dtype == self.dtype:
            return self.copy()

        result = self._new_like_me(dtype=dtype)
        self._copy(result, self, queue=queue)
        return result

    # {{{ rich comparisons, any, all

    def __nonzero__(self):
        if self.shape == ():
            return bool(self.get())
        else:
            raise ValueError("The truth value of an array with "
                    "more than one element is ambiguous. Use a.any() or a.all()")

    def any(self, queue=None, wait_for=None):
        from pyopencl.reduction import get_any_kernel
        krnl = get_any_kernel(self.context, self.dtype)
        return krnl(self, queue=queue, wait_for=wait_for)

    def all(self, queue=None, wait_for=None):
        from pyopencl.reduction import get_all_kernel
        krnl = get_all_kernel(self.context, self.dtype)
        return krnl(self, queue=queue, wait_for=wait_for)

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
            self._array_comparison(result, self, other, op="==")
            return result
        else:
            result = self._new_like_me(np.int8)
            self._scalar_comparison(result, self, other, op="==")
            return result

    def __ne__(self, other):
        if isinstance(other, Array):
            result = self._new_like_me(np.int8)
            self._array_comparison(result, self, other, op="!=")
            return result
        else:
            result = self._new_like_me(np.int8)
            self._scalar_comparison(result, self, other, op="!=")
            return result

    def __le__(self, other):
        if isinstance(other, Array):
            result = self._new_like_me(np.int8)
            self._array_comparison(result, self, other, op="<=")
            return result
        else:
            result = self._new_like_me(np.int8)
            self._scalar_comparison(result, self, other, op="<=")
            return result

    def __ge__(self, other):
        if isinstance(other, Array):
            result = self._new_like_me(np.int8)
            self._array_comparison(result, self, other, op=">=")
            return result
        else:
            result = self._new_like_me(np.int8)
            self._scalar_comparison(result, self, other, op=">=")
            return result

    def __lt__(self, other):
        if isinstance(other, Array):
            result = self._new_like_me(np.int8)
            self._array_comparison(result, self, other, op="<")
            return result
        else:
            result = self._new_like_me(np.int8)
            self._scalar_comparison(result, self, other, op="<")
            return result

    def __gt__(self, other):
        if isinstance(other, Array):
            result = self._new_like_me(np.int8)
            self._array_comparison(result, self, other, op=">")
            return result
        else:
            result = self._new_like_me(np.int8)
            self._scalar_comparison(result, self, other, op=">")
            return result

    # }}}

    # {{{ complex-valued business

    def real(self):
        if self.dtype.kind == "c":
            result = self._new_like_me(self.dtype.type(0).real.dtype)
            self._real(result, self)
            return result
        else:
            return self
    real = property(real, doc=".. versionadded:: 2012.1")

    def imag(self):
        if self.dtype.kind == "c":
            result = self._new_like_me(self.dtype.type(0).real.dtype)
            self._imag(result, self)
            return result
        else:
            return zeros_like(self)
    imag = property(imag, doc=".. versionadded:: 2012.1")

    def conj(self):
        """.. versionadded:: 2012.1"""
        if self.dtype.kind == "c":
            result = self._new_like_me()
            self._conj(result, self)
            return result
        else:
            return self

    # }}}

    def finish(self):
        # undoc
        if self.events:
            cl.wait_for_events(self.events)
            del self.events[:]

    # {{{ views

    def reshape(self, *shape, **kwargs):
        """Returns an array containing the same data with a new shape."""

        order = kwargs.pop("order", "C")
        if kwargs:
            raise TypeError("unexpected keyword arguments: %s"
                    % kwargs.keys())

        # TODO: add more error-checking, perhaps
        if isinstance(shape[0], tuple) or isinstance(shape[0], list):
            shape = tuple(shape[0])

        if shape == self.shape:
            return self

        size = reduce(lambda x, y: x * y, shape, 1)
        if size != self.size:
            raise ValueError("total size of new array must be unchanged")

        return self._new_with_changes(
                data=self.base_data, offset=self.offset, shape=shape,
                strides=_make_strides(self.dtype.itemsize, shape, order))

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

        ary, evt = cl.enqueue_map_buffer(
                queue or self.queue, self.base_data, flags, self.offset,
                self.shape, self.dtype, strides=self.strides, wait_for=wait_for,
                is_blocking=is_blocking)

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

                new_shape.append((stop-start)//idx_stride)
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

        if isinstance(subscript, Array):
            if subscript.dtype.kind != "i":
                raise TypeError(
                        "fancy indexing is only allowed with integers")
            if len(subscript.shape) != 1:
                raise NotImplementedError(
                        "multidimensional fancy indexing is not supported")
            if len(self.shape) != 1:
                raise NotImplementedError(
                        "fancy indexing into a multi-d array is supported")

            multi_put([value], subscript, out=[self], queue=self.queue,
                    wait_for=wait_for)
            return

        queue = queue or self.queue or value.queue

        subarray = self[subscript]

        if isinstance(value, np.ndarray):
            if subarray.shape == value.shape and subarray.strides == value.strides:
                self.events.append(
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

            self._copy(subarray, value, queue=queue, wait_for=wait_for)

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


def as_strided(ary, shape=None, strides=None):
    """Make an :class:`Array` from the given array with the given
    shape and strides.
    """

    # undocumented for the moment

    shape = shape or ary.shape
    strides = strides or ary.strides

    return Array(ary.queue, shape, ary.dtype, allocator=ary.allocator,
            data=ary.data, strides=strides)

# }}}


# {{{ creation helpers

def to_device(queue, ary, allocator=None, async=False):
    """Return a :class:`Array` that is an exact copy of the
    :class:`numpy.ndarray` instance *ary*.

    See :class:`Array` for the meaning of *allocator*.

    .. versionchanged:: 2011.1
        *context* argument was deprecated.
    """

    if _dtype_is_object(ary.dtype):
        raise RuntimeError("to_device does not work on object arrays.")

    result = Array(queue, ary.shape, ary.dtype,
                    allocator=allocator, strides=ary.strides)
    result.set(ary, async=async)
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
    zero = np.zeros((), dtype)
    result.fill(zero)
    return result


def empty_like(ary):
    """Make a new, uninitialized :class:`Array` having the same properties
    as *other_ary*.
    """

    return ary._new_with_changes(data=None, offset=0)


def zeros_like(ary):
    """Make a new, zero-initialized :class:`Array` having the same properties
    as *other_ary*.
    """

    result = empty_like(ary)
    zero = np.zeros((), ary.dtype)
    result.fill(zero)
    return result


@elwise_kernel_runner
def _arange_knl(result, start, step):
    return elementwise.get_arange_kernel(
            result.context, result.dtype)


def arange(queue, *args, **kwargs):
    """Create a :class:`Array` filled with numbers spaced `step` apart,
    starting from `start` and ending at `stop`.

    For floating point arguments, the length of the result is
    `ceil((stop - start)/step)`.  This rule may result in the last
    element of the result being greater than `stop`.

    *dtype*, if not specified, is taken as the largest common type
    of *start*, *stop* and *step*.

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
    for k, v in kwargs.iteritems():
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
    result.events.append(
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
    out.events.append(
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

        knl(queue, gs, ls,
                indices.data,
                *([o.data for o in out[chunk_slice]]
                    + [i.data for i in arrays[chunk_slice]]
                    + [indices.size]))

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
        knl(queue, gs, ls,
                *([o.data for o in out[chunk_slice]]
                    + [dest_indices.base_data,
                        dest_indices.offset,
                        src_indices.base_data,
                        src_indices.offset]
                    + list(flatten(
                        (i.base_data, i.offset)
                        for i in arrays[chunk_slice]))
                    + src_offsets_list[chunk_slice]
                    + [src_indices.size]))

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

    vec_count = len(arrays)

    if out is None:
        out = [Array(queue, dest_shape, a_dtype,
            allocator=a_allocator, queue=queue)
            for i in range(vec_count)]
    else:
        if a_dtype != single_valued(o.dtype for o in out):
            raise TypeError("arrays and out must have the same dtype")
        if len(out) != vec_count:
            raise ValueError("out and arrays must have the same length")

    if len(dest_indices.shape) != 1:
        raise ValueError("dest_indices must be 1D")

    chunk_size = _builtin_min(vec_count, 10)

    def make_func_for_chunk_size(chunk_size):
        knl = elementwise.get_put_kernel(
                context,
                a_dtype, dest_indices.dtype, vec_count=chunk_size)
        return knl

    knl = make_func_for_chunk_size(chunk_size)

    for start_i in range(0, len(arrays), chunk_size):
        chunk_slice = slice(start_i, start_i+chunk_size)

        if start_i + chunk_size > vec_count:
            knl = make_func_for_chunk_size(vec_count-start_i)

        gs, ls = dest_indices.get_sizes(queue,
                knl.get_work_group_info(
                    cl.kernel_work_group_info.WORK_GROUP_SIZE,
                    queue.device))

        from pytools import flatten
        evt = knl(queue, gs, ls,
                *(
                    list(flatten(
                        (o.base_data, o.offset)
                        for o in out[chunk_slice]))
                    + [dest_indices.base_data, dest_indices.offset]
                    + list(flatten(
                        (i.base_data, i.offset)
                        for i in arrays[chunk_slice]))
                    + [dest_indices.size]),
                **dict(wait_for=wait_for))

        # FIXME should wait on incoming events

        for o in out[chunk_slice]:
            o.events.append(evt)

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
    _diff(result, array, queue=queue)
    return result

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
    _if_positive(out, criterion, then_, else_, queue=queue)
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


def sum(a, dtype=None, queue=None):
    """
    .. versionadded:: 2011.1
    """
    from pyopencl.reduction import get_sum_kernel
    krnl = get_sum_kernel(a.context, dtype, a.dtype)
    return krnl(a, queue=queue)


def dot(a, b, dtype=None, queue=None):
    """
    .. versionadded:: 2011.1
    """
    from pyopencl.reduction import get_dot_kernel
    krnl = get_dot_kernel(a.context, dtype, a.dtype, b.dtype)
    return krnl(a, b, queue=queue)


def vdot(a, b, dtype=None, queue=None):
    """Like :func:`numpy.vdot`.

    .. versionadded:: 2013.1
    """
    from pyopencl.reduction import get_dot_kernel
    krnl = get_dot_kernel(a.context, dtype, a.dtype, b.dtype,
            conjugate_first=True)
    return krnl(a, b, queue=queue)


def subset_dot(subset, a, b, dtype=None, queue=None):
    """
    .. versionadded:: 2011.1
    """
    from pyopencl.reduction import get_subset_dot_kernel
    krnl = get_subset_dot_kernel(
            a.context, dtype, subset.dtype, a.dtype, b.dtype)
    return krnl(subset, a, b, queue=queue)


def _make_minmax_kernel(what):
    def f(a, queue=None):
        from pyopencl.reduction import get_minmax_kernel
        krnl = get_minmax_kernel(a.context, what, a.dtype)
        return krnl(a,  queue=queue)

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
    def f(subset, a, queue=None):
        from pyopencl.reduction import get_subset_minmax_kernel
        krnl = get_subset_minmax_kernel(a.context, what, a.dtype, subset.dtype)
        return krnl(subset, a,  queue=queue)

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

    result = a._new_like_me(output_dtype)

    from pyopencl.scan import get_cumsum_kernel
    krnl = get_cumsum_kernel(a.context, a.dtype, output_dtype)
    evt = krnl(a, result, queue=queue, wait_for=wait_for)

    if return_event:
        return evt, result
    else:
        return result

# }}}

# vim: foldmethod=marker
