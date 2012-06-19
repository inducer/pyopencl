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
from pyopencl.characterize import has_double_support




def _get_common_dtype(obj1, obj2, queue):
    return _get_common_dtype_base(obj1, obj2,
            has_double_support(queue.device))



# {{{ vector types

class vec:
    pass

def _create_vector_types():
    field_names = ["x", "y", "z", "w"]

    from pyopencl.tools import register_dtype

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
            if len(titles) < count:
                titles.extend((count-len(titles))*[None])

            dtype = np.dtype(dict(
                names=["s%d" % i for i in range(count)],
                formats=[base_type]*count,
                titles=titles))

            register_dtype(dtype, name)

            setattr(vec, name, dtype)

            my_field_names = ",".join(field_names[:count])
            my_field_names_defaulted = ",".join(
                    "%s=0" % fn for fn in field_names[:count])
            setattr(vec, "make_"+name,
                    staticmethod(eval(
                        "lambda %s: array((%s), dtype=my_dtype)"
                        % (my_field_names_defaulted, my_field_names),
                        dict(array=np.array, my_dtype=dtype))))

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
        work_items_per_group = ((grp + max_groups -1) // max_groups) * min_work_items
    else:
        group_count = max_groups
        work_items_per_group = max_work_items

    #print "n:%d gc:%d wipg:%d" % (n, group_count, work_items_per_group)
    return (group_count*work_items_per_group, 1, 1), \
            (work_items_per_group, 1, 1)




def elwise_kernel_runner(kernel_getter):
    """Take a kernel getter of the same signature as the kernel
    and return a function that invokes that kernel.

    Assumes that the zeroth entry in *args* is an :class:`Array`.
    """
    #(Note that the 'return a function' bit is done by @decorator.)

    def kernel_runner(*args, **kwargs):
        repr_ary = args[0]
        queue = kwargs.pop("queue", None) or repr_ary.queue

        knl = kernel_getter(*args)

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
                actual_args.append(arg.data)
            else:
                actual_args.append(arg)
        actual_args.append(repr_ary.mem_size)

        return knl(queue, gs, ls, *actual_args)

    try:
       from functools import update_wrapper
    except ImportError:
        return kernel_runner
    else:
       return update_wrapper(kernel_runner, kernel_getter)






DefaultAllocator = cl.CLAllocator




def _should_be_cqa(what):
    from warnings import warn
    warn("'%s' should be specified as the frst"
            "('cqa') argument, "
            "not in the '%s' keyword argument. "
            "This will be continue to be accepted througout "
            "versions 2011.x of PyOpenCL." % (what, what),
            DeprecationWarning, 3)

# }}}

# {{{ array class

class Array(object):
    """A :mod:`pyopencl` Array is used to do array-based calculation on
    a compute device.

    This is mostly supposed to be a :mod:`numpy`-workalike. Operators
    work on an element-by-element basis, just like :class:`numpy.ndarray`.
    """

    def __init__(self, cqa, shape, dtype, order="C", allocator=None,
            base=None, data=None, queue=None, strides=None):
        # {{{ backward compatibility for pre-cqa days

        if isinstance(cqa, cl.CommandQueue):
            if queue is not None:
                raise TypeError("can't specify queue in 'cqa' and "
                        "'queue' arguments")
            queue = cqa

            if allocator is None:
                context = queue.context
                allocator = DefaultAllocator(context)

        elif isinstance(cqa, cl.Context):
            if queue is not None:
                _should_be_cqa("queue")

            if allocator is not None:
                _should_be_cqa("allocator")
            else:
                allocator = DefaultAllocator(cqa)

        else:
            # cqa is assumed to be an allocator
            if allocator is not None:
                raise TypeError("can't specify allocator in 'cqa' and "
                        "'allocator' arguments")

            allocator = cqa

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

        if strides is None:
            if order == "F":
                strides = _f_contiguous_strides(
                        dtype.itemsize, shape)
            elif order == "C":
                strides = _c_contiguous_strides(
                        dtype.itemsize, shape)
            else:
                raise ValueError("invalid order: %s" % order)
        else:
            # FIXME: We should possibly perform some plausibility
            # checking on 'strides' here.

            strides = tuple(strides)

        # }}}

        self.queue = queue
        self.shape = shape
        self.dtype = dtype
        self.strides = strides

        self.mem_size = self.size = s
        self.nbytes = self.dtype.itemsize * self.size

        self.allocator = allocator
        if data is None:
            if self.size:
                self.data = self.allocator(self.size * self.dtype.itemsize)
            else:
                self.data = None
        else:
            self.data = data

        self.base = base

    @property
    def context(self):
        return self.data.context

    @property
    @memoize_method
    def flags(self):
        return _ArrayFlags(self)

    def _new_with_changes(self, data, shape=None, dtype=None, strides=None, queue=None,
            base=None):
        if shape is None:
            shape = self.shape
        if dtype is None:
            dtype = self.dtype
        if strides is None:
            strides = self.strides
        if queue is None:
            queue = self.queue
        if base is None and data is self.data:
            base = self

        if queue is not None:
            return Array(queue, shape, dtype,
                    allocator=self.allocator, strides=strides, base=base)
        elif self.allocator is not None:
            return Array(self.allocator, shape, dtype, queue=queue,
                    strides=strides, base=base)
        else:
            return Array(self.context, shape, dtype, strides=strides, base=base)

    #@memoize_method FIXME: reenable
    def get_sizes(self, queue, kernel_specific_max_wg_size=None):
        return splay(queue, self.mem_size,
                kernel_specific_max_wg_size=kernel_specific_max_wg_size)

    def set(self, ary, queue=None, async=False):
        assert ary.size == self.size
        assert ary.dtype == self.dtype

        if not ary.flags.forc:
            raise RuntimeError("cannot set from non-contiguous array")

            ary = ary.copy()

        if ary.strides != self.strides:
            from warnings import warn
            warn("Setting array from one with different strides/storage order. "
                    "This will cease to work in 2013.x.",
                    stacklevel=2)

        if self.size:
            cl.enqueue_copy(queue or self.queue, self.data, ary,
                    is_blocking=not async)

    def get(self, queue=None, ary=None, async=False):
        if ary is None:
            ary = np.empty(self.shape, self.dtype)

            ary = _as_strided(ary, strides=self.strides)
        else:
            if ary.size != self.size:
                raise TypeError("'ary' has non-matching type")
            if ary.dtype != self.dtype:
                raise TypeError("'ary' has non-matching size")

        assert self.flags.forc, "Array in get() must be contiguous"

        if self.size:
            cl.enqueue_copy(queue or self.queue, ary, self.data,
                    is_blocking=not async)

        return ary

    def __str__(self):
        return str(self.get())

    def __repr__(self):
        return repr(self.get())

    def __hash__(self):
        raise TypeError("pyopencl arrays are not hashable.")

    # kernel invocation wrappers ----------------------------------------------
    @staticmethod
    @elwise_kernel_runner
    def _axpbyz(out, afac, a, bfac, b, queue=None):
        """Compute ``out = selffac * self + otherfac*other``,
        where `other` is a vector.."""
        assert out.shape == a.shape

        return elementwise.get_axpbyz_kernel(
                out.context, a.dtype, b.dtype, out.dtype)

    @staticmethod
    @elwise_kernel_runner
    def _axpbz(out, a, x, b, queue=None):
        """Compute ``z = a * x + b``, where `b` is a scalar."""
        a = np.array(a)
        b = np.array(b)
        return elementwise.get_axpbz_kernel(out.context,
                a.dtype, x.dtype, b.dtype, out.dtype)

    @staticmethod
    @elwise_kernel_runner
    def _elwise_multiply(out, a, b, queue=None):
        return elementwise.get_multiply_kernel(
                a.context, a.dtype, b.dtype, out.dtype)

    @staticmethod
    @elwise_kernel_runner
    def _rdiv_scalar(out, ary, other, queue=None):
        other = np.array(other)
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

    # operators ---------------------------------------------------------------
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
            result = self._new_like_me(_get_common_dtype(self, other, self.queue))
            self._axpbyz(result,
                    self.dtype.type(1), self,
                    other.dtype.type(1), other)
            return result
        else:
            # add a scalar
            if other == 0:
                return self
            else:
                common_dtype = _get_common_dtype(self, other, self.queue)
                result = self._new_like_me(common_dtype)
                self._axpbz(result, self.dtype.type(1), self, common_dtype.type(other))
                return result

    __radd__ = __add__

    def __sub__(self, other):
        """Substract an array from an array or a scalar from an array."""

        if isinstance(other, Array):
            result = self._new_like_me(_get_common_dtype(self, other, self.queue))
            self._axpbyz(result,
                    self.dtype.type(1), self,
                    other.dtype.type(-1), other)
            return result
        else:
            # subtract a scalar
            if other == 0:
                return self
            else:
                result = self._new_like_me(_get_common_dtype(self, other, self.queue))
                self._axpbz(result, self.dtype.type(1), self, -other)
                return result

    def __rsub__(self,other):
        """Substracts an array by a scalar or an array::

           x = n - self
        """
        common_dtype = _get_common_dtype(self, other, self.queue)
        # other must be a scalar
        result = self._new_like_me(common_dtype)
        self._axpbz(result, self.dtype.type(-1), self, common_dtype.type(other))
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
            self._axpbyz(self, self.dtype.type(1), self, other.dtype.type(-1), other)
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
            result = self._new_like_me(_get_common_dtype(self, other, self.queue))
            self._elwise_multiply(result, self, other)
            return result
        else:
            common_dtype = _get_common_dtype(self, other, self.queue)
            result = self._new_like_me(common_dtype)
            self._axpbz(result, common_dtype.type(other), self, self.dtype.type(0))
            return result

    def __rmul__(self, scalar):
        common_dtype = _get_common_dtype(self, scalar, self.queue)
        result = self._new_like_me(common_dtype)
        self._axpbz(result, common_dtype.type(scalar), self, self.dtype.type(0))
        return result

    def __imul__(self, other):
        if isinstance(other, Array):
            self._elwise_multiply(self, self, other)
        else:
            # scalar
            self._axpbz(self, other, self, self.dtype.type(0))

        return self

    def __div__(self, other):
        """Divides an array by an array or a scalar::

           x = self / n
        """
        if isinstance(other, Array):
            result = self._new_like_me(_get_common_dtype(self, other, self.queue))
            self._div(result, self, other)
        else:
            if other == 1:
                return self
            else:
                # create a new array for the result
                common_dtype = _get_common_dtype(self, other, self.queue)
                result = self._new_like_me(common_dtype)
                self._axpbz(result,
                        common_dtype.type(1/other), self, self.dtype.type(0))

        return result

    __truediv__ = __div__

    def __rdiv__(self,other):
        """Divides an array by a scalar or an array::

           x = n / self
        """

        if isinstance(other, Array):
            result = self._new_like_me(_get_common_dtype(self, other, self.queue))
            other._div(result, self)
        else:
            if other == 1:
                return self
            else:
                # create a new array for the result
                common_dtype = _get_common_dtype(self, other, self.queue)
                result = self._new_like_me(common_dtype)
                self._rdiv_scalar(result, self, common_dtype.type(other))

        return result

    __rtruediv__ = __rdiv__

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

        result = self._new_like_me(self.dtype.type(0).real.dtype)
        self._abs(result, self)
        return result

    def __pow__(self, other):
        """Exponentiation by a scalar or elementwise by another
        :class:`Array`.
        """

        if isinstance(other, Array):
            assert self.shape == other.shape

            result = self._new_like_me(_get_common_dtype(self, other, self.queue))
            self._pow_array(result, self, other)
        else:
            result = self._new_like_me(_get_common_dtype(self, other, self.queue))
            self._pow_scalar(result, self, other)

        return result

    def __rpow__(self, other):
        # other must be a scalar
        common_dtype = _get_common_dtype(self, other, self.queue)
        result = self._new_like_me(common_dtype)
        self._rpow_scalar(result, common_dtype.type(other), self)
        return result

    def reverse(self, queue=None):
        """Return this array in reversed order. The array is treated
        as one-dimensional.
        """

        result = self._new_like_me()
        self._reverse(result, self)
        return result

    def astype(self, dtype, queue=None):
        if dtype == self.dtype:
            return self

        result = self._new_like_me(dtype=dtype)
        self._copy(result, self, queue=queue)
        return result

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

    # {{{ complex-valued business

    @property
    def real(self):
        if self.dtype.kind == "c":
            result = self._new_like_me(self.dtype.type(0).real.dtype)
            self._real(result, self)
            return result
        else:
            return self

    @property
    def imag(self):
        if self.dtype.kind == "c":
            result = self._new_like_me(self.dtype.type(0).real.dtype)
            self._imag(result, self)
            return result
        else:
            return zeros_like(self)

    def conj(self):
        if self.dtype.kind == "c":
            result = self._new_like_me()
            self._conj(result, self)
            return result
        else:
            return self

    # }}}

    # {{{ views

    def reshape(self, *shape):
        # TODO: add more error-checking, perhaps
        if isinstance(shape[0], tuple) or isinstance(shape[0], list):
            shape = tuple(shape[0])
        size = reduce(lambda x, y: x * y, shape, 1)
        if size != self.size:
            raise ValueError("total size of new array must be unchanged")

        return self._new_with_changes(data=self.data, shape=shape)

    def ravel(self):
        return self.reshape(self.size)

    def view(self, dtype=None):
        if dtype is None:
            dtype = self.dtype

        old_itemsize = self.dtype.itemsize
        itemsize = np.dtype(dtype).itemsize

        if self.shape[-1] * old_itemsize % itemsize != 0:
            raise ValueError("new type not compatible with array")

        shape = self.shape[:-1] + (self.shape[-1] * old_itemsize // itemsize,)

        return self._new_with_changes(data=self.data, shape=shape, dtype=dtype)

    # }}

# }}}

# }}}

# {{{ creation helpers

def to_device(*args, **kwargs):
    """Converts a numpy array to a :class:`Array`."""

    def _to_device(queue, ary, allocator=None, async=False):
        if ary.dtype == object:
            raise RuntimeError("to_device does not work on object arrays.")

        result = Array(queue, ary.shape, ary.dtype,
                        allocator=allocator, strides=ary.strides)
        result.set(ary, async=async)
        return result

    if isinstance(args[0], cl.Context):
        from warnings import warn
        warn("Passing a context as first argument is deprecated. "
            "This will be continue to be accepted througout "
            "versions 2011.x of PyOpenCL.",
            DeprecationWarning, 2)
        args = args[1:]

    return _to_device(*args, **kwargs)




empty = Array

def zeros(*args, **kwargs):
    """Returns an array of the given shape and dtype filled with 0's."""

    def _zeros(queue, shape, dtype, order="C", allocator=None):
        result = Array(queue, shape, dtype,
                order=order, allocator=allocator)
        zero = np.zeros((), dtype)
        result.fill(zero)
        return result

    if isinstance(args[0], cl.Context):
        from warnings import warn
        warn("Passing a context as first argument is deprecated. "
            "This will be continue to be accepted througout "
            "versions 2011.x of PyOpenCL.",
            DeprecationWarning, 2)
        args = args[1:]

    return _zeros(*args, **kwargs)

def empty_like(ary):
    return ary._new_with_changes(data=None)

def zeros_like(ary):
    result = empty_like(ary)
    zero = np.zeros((), ary.dtype)
    result.fill(zero)
    return result


@elwise_kernel_runner
def _arange_knl(result, start, step):
    return elementwise.get_arange_kernel(
            result.context, result.dtype)

def _arange(queue, *args, **kwargs):
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

    if isinstance(args[-1], np.dtype):
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

    admissible_names = ["start", "stop", "step", "dtype", "allocator"]
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
        inf.dtype = np.array([inf.start, inf.stop, inf.step]).dtype

    # actual functionality ----------------------------------------------------
    dtype = np.dtype(inf.dtype)
    start = dtype.type(inf.start)
    step = dtype.type(inf.step)
    stop = dtype.type(inf.stop)

    if not explicit_dtype:
        raise TypeError("arange requires dtype argument")

    from math import ceil
    size = int(ceil((stop-start)/step))

    result = Array(queue, (size,), dtype, allocator=inf.allocator)
    _arange_knl(result, start, step, queue=queue)
    return result




def arange(*args, **kwargs):
    """Create an array filled with numbers spaced `step` apart,
    starting from `start` and ending at `stop`.

    For floating point arguments, the length of the result is
    `ceil((stop - start)/step)`.  This rule may result in the last
    element of the result being greater than stop.
    """

    if isinstance(args[0], cl.Context):
        from warnings import warn
        warn("Passing a context as first argument is deprecated. "
            "This will be continue to be accepted througout "
            "versions 2011.x of PyOpenCL.",
            DeprecationWarning, 2)
        args = args[1:]

    return _arange(*args, **kwargs)

# }}}

# {{{ take/put

@elwise_kernel_runner
def _take(result, ary, indices):
    return elementwise.get_take_kernel(
            result.context, result.dtype, indices.dtype)




def take(a, indices, out=None, queue=None):
    queue = queue or a.queue
    if out is None:
        out = Array(queue, indices.shape, a.dtype, allocator=a.allocator)

    assert len(indices.shape) == 1
    _take(out, a, indices, queue=queue)
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

        gs, ls = src_indices.get_sizes(queue,
                knl.get_work_group_info(
                    cl.kernel_work_group_info.WORK_GROUP_SIZE,
                    queue.device))

        knl(queue, gs, ls,
                *([o.data for o in out[chunk_slice]]
                    + [dest_indices.data, src_indices.data]
                    + [i.data for i in arrays[chunk_slice]]
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
        out = [Array(context, dest_shape, a_dtype, allocator=a_allocator, queue=queue)
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

        gs, ls = dest_indices.get_sizes(queue,
                knl.get_work_group_info(
                    cl.kernel_work_group_info.WORK_GROUP_SIZE,
                    queue.device))

        knl(queue, gs, ls,
                *([o.data for o in out[chunk_slice]]
                    + [dest_indices.data]
                    + [i.data for i in arrays[chunk_slice]]
                    + [dest_indices.size]))

    return out

# }}}

# {{{ conditionals

@elwise_kernel_runner
def _if_positive(result, criterion, then_, else_):
    return elementwise.get_if_positive_kernel(
            result.context, criterion.dtype, then_.dtype)




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
    _if_positive(out, criterion, then_, else_)
    return out




def maximum(a, b, out=None, queue=None):
    # silly, but functional
    return if_positive(a.mul_add(1, b, -1, queue=queue), a, b,
            queue=queue, out=out)




def minimum(a, b, out=None, queue=None):
    # silly, but functional
    return if_positive(a.mul_add(1, b, -1, queue=queue), b, a,
            queue=queue, out=out)

# }}}

# {{{ reductions
_builtin_min = min
_builtin_max = max

def sum(a, dtype=None, queue=None):
    from pyopencl.reduction import get_sum_kernel
    krnl = get_sum_kernel(a.context, dtype, a.dtype)
    return krnl(a, queue=queue)

def dot(a, b, dtype=None, queue=None):
    from pyopencl.reduction import get_dot_kernel
    krnl = get_dot_kernel(a.context, dtype, a.dtype, b.dtype)
    return krnl(a, b, queue=queue)

def subset_dot(subset, a, b, dtype=None, queue=None):
    from pyopencl.reduction import get_subset_dot_kernel
    krnl = get_subset_dot_kernel(a.context, dtype, subset.dtype, a.dtype, b.dtype)
    return krnl(subset, a, b, queue=queue)

def _make_minmax_kernel(what):
    def f(a, queue=None):
        from pyopencl.reduction import get_minmax_kernel
        krnl = get_minmax_kernel(a.context, what, a.dtype)
        return krnl(a,  queue=queue)

    return f

min = _make_minmax_kernel("min")
max = _make_minmax_kernel("max")

def _make_subset_minmax_kernel(what):
    def f(subset, a, queue=None):
        from pyopencl.reduction import get_subset_minmax_kernel
        krnl = get_subset_minmax_kernel(a.context, what, a.dtype, subset.dtype)
        return krnl(subset, a,  queue=queue)

    return f

subset_min = _make_subset_minmax_kernel("min")
subset_max = _make_subset_minmax_kernel("max")

# }}}

# vim: foldmethod=marker
