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




import numpy
import pyopencl.elementwise as elementwise
import pyopencl as cl
#from pytools import memoize_method
from decorator import decorator



def splay(queue, n):
    dev = queue.device
    max_work_items = min(128, dev.max_work_group_size)
    min_work_items = min(32, max_work_items)
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
        work_items_per_group = ((grp + max_groups -1) // max_groups) * min_work_items
    else:
        group_count = max_groups
        work_items_per_group = max_work_items

    #print "n:%d gc:%d wipg:%d" % (n, group_count, work_items_per_group)
    return (group_count*work_items_per_group, 1, 1), \
            (work_items_per_group, 1, 1)




def _get_common_dtype(obj1, obj2):
    return (obj1.dtype.type(0) + obj2.dtype.type(0)).dtype




@decorator
def elwise_kernel_runner(kernel_getter, *args, **kwargs):
    """Take a kernel getter of the same signature as the kernel 
    and return a function that invokes that kernel.

    Assumes that the zeroth entry in *args* is an :class:`Array`.
    """

    knl = kernel_getter(*args)

    repr_ary = args[0]
    assert isinstance(repr_ary, Array)

    queue = kwargs.pop("queue", None) or repr_ary.queue
    gs, ls = repr_ary.get_sizes(queue)

    if kwargs:
        raise TypeError("only the 'queue' keyword argument is supported")

    actual_args = []
    for arg in args:
        if isinstance(arg, Array):
            actual_args.append(arg.data)
        else:
            actual_args.append(arg)
    actual_args.append(repr_ary.mem_size)

    return knl(queue, gs, ls, *actual_args)





class DefaultAllocator:
    def __init__(self, context, flags=cl.mem_flags.READ_WRITE):
        self.context = context
        self.flags = flags

    def __call__(self, size):
        return cl.Buffer(self.context, self.flags, size)




class Array(object):
    """A :mod:`pyopencl` Array is used to do array-based calculation on
    a compute device.

    This is mostly supposed to be a :mod:`numpy`-workalike. Operators
    work on an element-by-element basis, just like :class:`numpy.ndarray`.
    """

    def __init__(self, context, shape, dtype, allocator=None,
            base=None, data=None, queue=None):
        if allocator is None:
            allocator = DefaultAllocator(context)

        try:
            s = 1
            for dim in shape:
                s *= dim
        except TypeError:
            if not isinstance(shape, int):
                raise TypeError("shape must either be iterable or "
                        "castable to an integer")
            s = shape
            shape = (shape,)

        self.context = context
        self.queue = queue

        self.shape = shape
        self.dtype = numpy.dtype(dtype)

        self.mem_size = self.size = s
        self.nbytes = self.dtype.itemsize * self.size

        self.allocator = allocator
        if data is None:
            if self.size:
                self.data = self.allocator(self.size * self.dtype.itemsize)
            else:
                self.data = None

            if base is not None:
                raise ValueError("If data is specified, base must be None.")
        else:
            self.data = data

        self.base = base

    #@memoize_method FIXME: reenable
    def get_sizes(self, queue):
        return splay(self.queue, self.mem_size)

    def set(self, ary, queue=None, async=False):
        assert ary.size == self.size
        assert ary.dtype == self.dtype
        if self.size:
            evt = cl.enqueue_write_buffer(queue or self.queue, self.data, ary)

            if not async:
                evt.wait()

    def get(self, queue=None, ary=None, async=False):
        if ary is None:
            ary = numpy.empty(self.shape, self.dtype)
        else:
            if ary.size != self.size:
                raise TypeError("'ary' has non-matching type")
            if ary.dtype != self.dtype:
                raise TypeError("'ary' has non-matching size")

        if self.size:
            evt = cl.enqueue_read_buffer(queue or self.queue, self.data, ary)

            if not async:
                evt.wait()

        return ary

    def __str__(self):
        return str(self.get())

    def __repr__(self):
        return repr(self.get())

    def __hash__(self):
        raise TypeError("pyopencl arrays are not hashable.")

    # kernel invocation wrappers ----------------------------------------------
    @elwise_kernel_runner
    @staticmethod
    def _axpbyz(out, afac, a, bfac, b, queue=None):
        """Compute ``out = selffac * self + otherfac*other``,
        where `other` is a vector.."""
        assert self.shape == other.shape

        return elementwise.get_axpbyz_kernel(
                self.context, self.dtype, other.dtype, out.dtype)

    @elwise_kernel_runner
    def _axpbz(self, out, selffac, other, queue=None):
        """Compute ``out = selffac * self + other``, where `other` is a scalar."""
        return elementwise.get_axpbz_kernel(self.context, self.dtype)

    @elwise_kernel_runner
    def _elwise_multiply(self, out, other, queue=None):
        return elementwise.get_multiply_kernel(
                self.context, self.dtype, other.dtype, out.dtype)

    @elwise_kernel_runner
    def _rdiv_scalar(self, out, other, queue=None):
        """Divides an array by a scalar::

           y = n / self
        """

        assert self.dtype == numpy.float32

        return elementwise.get_rdivide_elwise_kernel(self.context, self.dtype)

    @elwise_kernel_runner
    def _div(self, out, other, queue=None):
        """Divides an array by another array."""

        assert self.shape == other.shape

        return elementwise.get_divide_kernel(self.context,
                self.dtype, other.dtype, out.dtype)

    @elwise_kernel_runner
    def _fill(result, scalar):
        return elementwise.get_fill_kernel(result.context, result.dtype)

    @elwise_kernel_runner
    def _abs(result, arg):
        return elementwise.get_unary_func_kernel(
                arg.context, fname, arg.dtype)

    @elwise_kernel_runner
    def _pow_scalar(self, ):
        return elementwise.get_pow_kernel(self.context, self.dtype)

    def _new_like_me(self, dtype=None):
        if dtype is None:
            dtype = self.dtype
        return self.__class__(self.context,
                self.shape, dtype, allocator=self.allocator,
                queue=self.queue)

    # operators ---------------------------------------------------------------
    def mul_add(self, selffac, other, otherfac, queue=None):
        """Return `selffac * self + otherfac*other`.
        """
        result = self._new_like_me(_get_common_dtype(self, other))
        self._axpbyz(result, selffac, self, other, otherfac)
        return result

    def __add__(self, other):
        """Add an array with an array or an array with a scalar."""

        if isinstance(other, Array):
            # add another vector
            result = self._new_like_me(_get_common_dtype(self, other))
            self._axpbyz(result, 1, self, 1, other)
            return result
        else:
            # add a scalar
            if other == 0:
                return self
            else:
                result = self._new_like_me()
                self._axpbz(result, 1, other)
                return result

    __radd__ = __add__

    def __sub__(self, other):
        """Substract an array from an array or a scalar from an array."""

        if isinstance(other, Array):
            result = self._new_like_me(_get_common_dtype(self, other))
            self._axpbyz(result, 1, other, -1)
            return result
        else:
            if other == 0:
                return self
            else:
                # create a new array for the result
                result = self._new_like_me()
                return self._axpbz(result, 1, -other)

    def __rsub__(self,other):
        """Substracts an array by a scalar or an array::

           x = n - self
        """
        # other must be a scalar
        result = self._new_like_me()
        return self._axpbz(result, -1, other)

    def __iadd__(self, other):
        self._axpbyz(self, 1, self, 1, other)
        return self

    def __isub__(self, other):
        return self._axpbyz(self, 1, self, -1, other)

    def __neg__(self):
        result = self._new_like_me()
        return self._axpbz(result, -1, self)

    def __mul__(self, other):
        if isinstance(other, Array):
            result = self._new_like_me(_get_common_dtype(self, other))
            self._elwise_multiply(result, other)
            return result
        else:
            result = self._new_like_me()
            self._axpbz(result, other, self, 0)
            return result

    def __rmul__(self, scalar):
        result = self._new_like_me()
        self._axpbz(result, scalar, self, 0)
        return result

    def __imul__(self, scalar):
        self._axpbz(self, scalar, 0)
        return self

    def __div__(self, other):
        """Divides an array by an array or a scalar::

           x = self / n
        """
        if isinstance(other, Array):
            result = self._new_like_me(_get_common_dtype(self, other))
            return self._div(result, other)
        else:
            if other == 1:
                return self
            else:
                # create a new array for the result
                result = self._new_like_me()
                self._axpbz(result, 1/other, 0)
                return result

    __truediv__ = __div__

    def __rdiv__(self,other):
        """Divides an array by a scalar or an array::

           x = n / self
        """

        if isinstance(other, Array):
            result = self._new_like_me(_get_common_dtype(self, other))
            other._div(result, self)
        else:
            if other == 1:
                return self
            else:
                # create a new array for the result
                result = self._new_like_me()
                self._rdiv_scalar(result, other)

        return result

    __rtruediv__ = __div__

    def fill(self, value, queue=None):
        """fills the array with the specified value"""
        self._fill(self, value, queue=queue)
        return self

    def __len__(self):
        """Return the size of the leading dimension of self."""
        if len(self.shape):
            return self.shape[0]
        else:
            return 1

    def __abs__(self):
        """Return a `Array` of the absolute values of the elements
        of `self`.
        """

        if self.dtype.kind == "f":
            fname = "fabs"
        elif self.dtype.kind in ["u", "i"]:
            fname = "abs"
        else:
            raise TypeError("unsupported dtype in __abs__()")

        result = self._new_like_me()
        self._abs(result, self)
        return result

    def __pow__(self, other):
        """Exponentiation by a scalar or elementwise by another
        :class:`Array`.
        """

        if isinstance(other, Array):
            assert self.shape == other.shape

            result = self._new_like_me(_get_common_dtype(self, other))

            knl = elementwise.get_pow_array_kernel(self.context,
                    self.dtype, other.dtype, result.dtype)

            knl(self.queue, self._global_size, self._local_size,
                    self.data, other.data, result.data,
                    self.mem_size)

            return result
        else:
            result = self._new_like_me()
            knl(self.queue, self._global_size, self._local_size,
                    other, self.data, result.data,
                    self.mem_size)

            return result

    def reverse(self, queue=None):
        """Return this array in reversed order. The array is treated
        as one-dimensional.
        """

        result = self._new_like_me()

        knl = elementwise.get_reverse_kernel(self.context, self.dtype)
        knl(queue or self.queue, self._global_size, self._local_size,
                self.data, result.data, self.mem_size)

        return result

    def astype(self, dtype, queue=None):
        if dtype == self.dtype:
            return self

        result = self._new_like_me(dtype=dtype)

        knl = elementwise.get_copy_kernel(self.context, dtype, self.dtype)
        knl(queue or self.queue, self._global_size, self._local_size,
                result.data, self.data, self.mem_size)

        return result


    # slicing -----------------------------------------------------------------
    def __getitem__(self, idx):
        if idx == ():
            return self

        if len(self.shape) > 1:
            raise NotImplementedError("multi-d slicing is not yet implemented")

        if not isinstance(idx, slice):
            raise ValueError("non-slice indexing not supported: %s" % (idx,))

        l, = self.shape
        start, stop, stride = idx.indices(l)

        if stride != 1:
            raise NotImplementedError("strided slicing is not yet implemented")

        return Array(
                shape=((stop-start)//stride,),
                dtype=self.dtype,
                allocator=self.allocator,
                base=self,
                data=int(self.data) + start*self.dtype.itemsize)

    # rich comparisons (or rather, lack thereof) ------------------------------
    def __eq__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        raise NotImplementedError

    def __le__(self, other):
        raise NotImplementedError

    def __ge__(self, other):
        raise NotImplementedError

    def __lt__(self, other):
        raise NotImplementedError

    def __gt__(self, other):
        raise NotImplementedError



def to_device(context, queue, ary, allocator=None, async=False):
    """Converts a numpy array to a :class:`Array`."""
    result = Array(context, ary.shape, ary.dtype, allocator,
            queue=queue)
    result.set(ary, async=async)
    return result


empty = Array

def zeros(context, queue, shape, dtype, allocator=None):
    """Returns an array of the given shape and dtype filled with 0's."""

    result = Array(context, shape, dtype, allocator, queue=queue)
    result.fill(0)
    return result

def empty_like(ary):
    result = Array(ary.context,
            ary.shape, ary.dtype, ary.allocator, queue=ary.queue)
    return result

def zeros_like(ary):
    result = Array(ary.context,
            ary.shape, ary.dtype, ary.allocator, queue=ary.queue)
    result.fill(0)
    return result


def arange(context, queue, *args, **kwargs):
    """Create an array filled with numbers spaced `step` apart,
    starting from `start` and ending at `stop`.

    For floating point arguments, the length of the result is
    `ceil((stop - start)/step)`.  This rule may result in the last
    element of the result being greater than stop.
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

    if isinstance(args[-1], numpy.dtype):
        dtype = args[-1]
        args = args[:-1]
        explicit_dtype = True

    argc = len(args)
    if argc == 0:
        raise ValueError, "stop argument required"
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
        raise ValueError, "too many arguments"

    admissible_names = ["start", "stop", "step", "dtype"]
    for k, v in kwargs.iteritems():
        if k in admissible_names:
            if getattr(inf, k) is None:
                setattr(inf, k, v)
                if k == "dtype":
                    explicit_dtype = True
            else:
                raise ValueError, "may not specify '%s' by position and keyword" % k
        else:
            raise ValueError, "unexpected keyword argument '%s'" % k

    if inf.start is None:
        inf.start = 0
    if inf.step is None:
        inf.step = 1
    if inf.dtype is None:
        inf.dtype = numpy.array([inf.start, inf.stop, inf.step]).dtype

    # actual functionality ----------------------------------------------------
    dtype = numpy.dtype(inf.dtype)
    start = dtype.type(inf.start)
    step = dtype.type(inf.step)
    stop = dtype.type(inf.stop)

    if not explicit_dtype:
        raise TypeError("arange requires dtype argument")

    from math import ceil
    size = int(ceil((stop-start)/step))

    result = Array(context, (size,), dtype, queue=queue)

    knl = elementwise.get_arange_kernel(context, dtype)
    knl(queue, result._global_size, result._local_size,
            result.data, start, step, size)

    return result




def take(a, indices, out=None, queue=None):
    queue = queue or a.queue

    if out is None:
        out = Array(a.context, indices.shape, a.dtype, a.allocator,
                queue=queue)

    assert len(indices.shape) == 1

    knl = elementwise.get_take_kernel(
            a.context, a.dtype, indices.dtype)
    knl(queue, a._global_size, a._local_size,
            indices.data, a.data, out.data, indices.size)

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
        out = [Array(context, queue, indices.shape, a_dtype, a_allocator)
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

        knl(queue, indices._global_size, indices._local_size,
                indices.data,
                *([i.data for i in arrays[chunk_slice]]
                    + [o.data for o in out[chunk_slice]]
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
        out = [Array(context, dest_shape, a_dtype, a_allocator, queue=queue)
                for i in range(vec_count)]
    else:
        if a_dtype != single_valued(o.dtype for o in out):
            raise TypeError("arrays and out must have the same dtype")
        if len(out) != vec_count:
            raise ValueError("out and arrays must have the same length")

    if src_indices.dtype != dest_indices.dtype:
        raise TypeError("src_indices and dest_indices must have the same dtype")

    if len(src_indices.shape) != 1:
        raise ValueError("src_indices must be 1D")

    if src_indices.shape != dest_indices.shape:
        raise ValueError("src_indices and dest_indices must have the same shape")

    if src_offsets is None:
        src_offsets_list = []
    else:
        src_offsets_list = src_offsets
        if len(src_offsets) != vec_count:
            raise ValueError("src_indices and src_offsets must have the same length")

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

        knl(queue, src_indices._global_size, src_indices._local_size,
                dest_indices.data, src_indices.data,
                *([i.data for i in arrays[chunk_slice]]
                    + [o.data for o in out[chunk_slice]]
                    + src_offsets_list[chunk_slice]
                    + [src_indices.size]))

    return out




def multi_put(arrays, dest_indices, dest_shape=None, out=None, queue=None):
    if not len(arrays):
        return []

    from pytools import single_valued
    a_dtype = single_valued(a.dtype for a in arrays)
    a_allocator = arrays[0].allocator
    context = dest_indices.context
    queue = queue or dest_indices.queue

    vec_count = len(arrays)

    if out is None:
        out = [Array(context, dest_shape, a_dtype, a_allocator, queue=queue)
                for i in range(vec_count)]
    else:
        if a_dtype != single_valued(o.dtype for o in out):
            raise TypeError("arrays and out must have the same dtype")
        if len(out) != vec_count:
            raise ValueError("out and arrays must have the same length")

    if len(dest_indices.shape) != 1:
        raise ValueError("src_indices must be 1D")

    chunk_size = _builtin_min(vec_count, 10)

    def make_func_for_chunk_size(chunk_size):
        knl = elementwise.get_put_kernel(
                a_dtype, dest_indices.dtype, vec_count=chunk_size)
        knl.set_block_shape(*dest_indices._block)
        return knl

    knl = make_func_for_chunk_size(chunk_size)

    for start_i in range(0, len(arrays), chunk_size):
        chunk_slice = slice(start_i, start_i+chunk_size)

        if start_i + chunk_size > vec_count:
            knl = make_func_for_chunk_size(vec_count-start_i)

        knl(queue, dest_indices._global_size, dest_indices._local_size,
                dest_indices.data,
                *([o.data for o in out[chunk_slice]]
                    + [i.data for i in arrays[chunk_slice]]
                    + [dest_indices.size]))

    return out




def if_positive(criterion, then_, else_, out=None, queue=None):
    if not (criterion.shape == then_.shape == else_.shape):
        raise ValueError("shapes do not match")

    if not (then_.dtype == else_.dtype):
        raise ValueError("dtypes do not match")

    knl = elementwise.get_if_positive_kernel(
            criterion.context,
            criterion.dtype, then_.dtype)

    if out is None:
        out = empty_like(then_)

    knl(queue or criterion.queue, criterion._global_size, criterion._local_size,
            criterion.data, then_.data, else_.data, out.data,
            criterion.size)

    return out




def maximum(a, b, out=None, queue=None):
    # silly, but functional
    return if_positive(a.mul_add(1, b, -1, queue=queue), a, b,
            queue=queue, out=out)




def minimum(a, b, out=None, queue=None):
    # silly, but functional
    return if_positive(a.mul_add(1, b, -1, queue=queue), b, a,
            queue=queue, out=out)




# reductions ------------------------------------------------------------------
_builtin_min = min
_builtin_max = max

if False:
    # not yet
    def sum(a, dtype=None, queue=None):
        from pyopencl.reduction import get_sum_kernel
        krnl = get_sum_kernel(dtype, a.dtype)
        return krnl(a, queue=queue)

    def dot(a, b, dtype=None, queue=None):
        from pyopencl.reduction import get_dot_kernel
        krnl = get_dot_kernel(dtype, a.dtype, b.dtype)
        return krnl(a, b, queue=queue)

    def subset_dot(subset, a, b, dtype=None, queue=None):
        from pyopencl.reduction import get_subset_dot_kernel
        krnl = get_subset_dot_kernel(dtype, a.dtype, b.dtype)
        return krnl(subset, a, b, queue=queue)

    def _make_minmax_kernel(what):
        def f(a, queue=None):
            from pyopencl.reduction import get_minmax_kernel
            krnl = get_minmax_kernel(what, a.dtype)
            return krnl(a,  queue=queue)

        return f

    min = _make_minmax_kernel("min")
    max = _make_minmax_kernel("max")

    def _make_subset_minmax_kernel(what):
        def f(subset, a, queue=None):
            from pyopencl.reduction import get_subset_minmax_kernel
            import pyopencl.reduction
            krnl = get_subset_minmax_kernel(what, a.dtype)
            return krnl(subset, a,  queue=queue)

        return f

    subset_min = _make_subset_minmax_kernel("min")
    subset_max = _make_subset_minmax_kernel("max")





# vim: foldmethod=marker
