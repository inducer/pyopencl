r"""
.. _memory-pools:

Memory Pools
------------

Memory allocation (e.g. in the form of the :func:`pyopencl.Buffer` constructor)
can be expensive if used frequently. For example, code based on
:class:`pyopencl.array.Array` can easily run into this issue because a fresh
memory area is allocated for each intermediate result.  Memory pools are a
remedy for this problem based on the observation that often many of the block
allocations are of the same sizes as previously used ones.

Then, instead of fully returning the memory to the system and incurring the
associated reallocation overhead, the pool holds on to the memory and uses it
to satisfy future allocations of similarly-sized blocks. The pool reacts
appropriately to out-of-memory conditions as long as all memory allocations
are made through it. Allocations performed from outside of the pool may run
into spurious out-of-memory conditions due to the pool owning much or all of
the available memory.

There are two flavors of allocators and memory pools:

- :ref:`buf-mempool`
- :ref:`svm-mempool`

Using :class:`pyopencl.array.Array`\ s can be used with memory pools in a
straightforward manner::

    mem_pool = pyopencl.tools.MemoryPool(pyopencl.tools.ImmediateAllocator(queue))
    a_dev = cl_array.arange(queue, 2000, dtype=np.float32, allocator=mem_pool)

Likewise, SVM-based allocators are directly usable with
:class:`pyopencl.array.Array`.

.. _buf-mempool:

:class:`~pyopencl.Buffer`-based Allocators and Memory Pools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PooledBuffer

.. autoclass:: AllocatorBase

.. autoclass:: DeferredAllocator

.. autoclass:: ImmediateAllocator

.. autoclass:: MemoryPool

.. _svm-mempool:

:ref:`SVM <svm>`-Based Allocators and Memory Pools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SVM functionality requires OpenCL 2.0.

.. autoclass:: PooledSVM

.. autoclass:: SVMAllocator

.. autoclass:: SVMPool

CL-Object-dependent Caching
---------------------------

.. autofunction:: first_arg_dependent_memoize
.. autofunction:: clear_first_arg_caches

Testing
-------

.. autofunction:: pytest_generate_tests_for_pyopencl

Argument Types
--------------

.. autoclass:: Argument
.. autoclass:: DtypedArgument

.. autoclass:: VectorArg
.. autoclass:: ScalarArg
.. autoclass:: OtherArg

.. autofunction:: parse_arg_list

Device Characterization
-----------------------

.. automodule:: pyopencl.characterize
    :members:

Type aliases
------------

.. currentmodule:: pyopencl._cl

.. class:: AllocatorBase

   See :class:`pyopencl.tools.AllocatorBase`.
"""

from __future__ import annotations


__copyright__ = "Copyright (C) 2010 Andreas Kloeckner"

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

import atexit
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from sys import intern
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Concatenate,
    ParamSpec,
    TypeAlias,
    TypedDict,
    TypeVar,
    cast,
    overload,
)

import numpy as np
from typing_extensions import TypeIs, override

from pytools import Hash, memoize, memoize_method
from pytools.persistent_dict import KeyBuilder as KeyBuilderBase

from pyopencl._cl import bitlog2, get_cl_header_version
from pyopencl.compyte.dtypes import (
    TypeNameNotKnown,
    dtype_to_ctype,
    get_or_register_dtype,
    register_dtype,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence

    import pytest
    from mako.template import Template
    from numpy.typing import DTypeLike, NDArray

# Do not add a pyopencl import here: This will add an import cycle.


def _register_types():
    from pyopencl.compyte.dtypes import TYPE_REGISTRY, fill_registry_with_opencl_c_types

    fill_registry_with_opencl_c_types(TYPE_REGISTRY)

    get_or_register_dtype("cfloat_t", np.complex64)
    get_or_register_dtype("cdouble_t", np.complex128)


_register_types()


# {{{ imported names

from pyopencl._cl import (
    AllocatorBase,
    DeferredAllocator,
    ImmediateAllocator,
    MemoryPool,
    PooledBuffer,
)


if get_cl_header_version() >= (2, 0):
    from pyopencl._cl import PooledSVM, SVMAllocator, SVMPool

# }}}


# {{{ monkeypatch docstrings into imported interfaces

_MEMPOOL_IFACE_DOCS = """
.. note::

    The current implementation of the memory pool will retain allocated
    memory after it is returned by the application and keep it in a bin
    identified by the leading *leading_bits_in_bin_id* bits of the
    allocation size. To ensure that allocations within each bin are
    interchangeable, allocation sizes are rounded up to the largest size
    that shares the leading bits of the requested allocation size.

    The current default value of *leading_bits_in_bin_id* is
    four, but this may change in future versions and is not
    guaranteed.

    *leading_bits_in_bin_id* must be passed by keyword,
    and its role is purely advisory. It is not guaranteed
    that future versions of the pool will use the
    same allocation scheme and/or honor *leading_bits_in_bin_id*.

.. attribute:: held_blocks

    The number of unused blocks being held by this pool.

.. attribute:: active_blocks

    The number of blocks in active use that have been allocated
    through this pool.

.. attribute:: managed_bytes

    "Managed" memory is "active" and "held" memory.

    .. versionadded:: 2021.1.2

.. attribute:: active_bytes

    "Active" bytes are bytes under the control of the application.
    This may be smaller than the actual allocated size reflected
    in :attr:`managed_bytes`.

    .. versionadded:: 2021.1.2


.. method:: free_held

    Free all unused memory that the pool is currently holding.

.. method:: stop_holding

    Instruct the memory to start immediately freeing memory returned
    to it, instead of holding it for future allocations.
    Implicitly calls :meth:`free_held`.
    This is useful as a cleanup action when a memory pool falls out
    of use.
"""


def _monkeypatch_docstrings():
    from pytools.codegen import remove_common_indentation

    PooledBuffer.__doc__ = """
    An object representing a :class:`MemoryPool`-based allocation of
    :class:`~pyopencl.Buffer`-style device memory.  Analogous to
    :class:`~pyopencl.Buffer`, however once this object is deleted, its
    associated device memory is returned to the pool.

    Is a :class:`pyopencl.MemoryObject`.
    """

    AllocatorBase.__doc__ = """
    An interface implemented by various memory allocation functions
    in :mod:`pyopencl`.

    .. automethod:: __call__

        Allocate and return a :class:`pyopencl.Buffer` of the given *size*.
    """

    # {{{ DeferredAllocator

    DeferredAllocator.__doc__ = """
    *mem_flags* takes its values from :class:`pyopencl.mem_flags` and corresponds
    to the *flags* argument of :class:`pyopencl.Buffer`. DeferredAllocator
    has the same semantics as regular OpenCL buffer allocation, i.e. it may
    promise memory to be available that may (in any call to a buffer-using
    CL function) turn out to not exist later on. (Allocations in CL are
    bound to contexts, not devices, and memory availability depends on which
    device the buffer is used with.)

    Implements :class:`AllocatorBase`.

    .. versionchanged :: 2013.1

        ``CLAllocator`` was deprecated and replaced
        by :class:`DeferredAllocator`.

    .. method::  __init__(context, mem_flags=pyopencl.mem_flags.READ_WRITE)

    .. automethod:: __call__

        Allocate a :class:`pyopencl.Buffer` of the given *size*.

        .. versionchanged :: 2020.2

            The allocator will succeed even for allocations of size zero,
            returning *None*.
    """

    # }}}

    # {{{ ImmediateAllocator

    ImmediateAllocator.__doc__ = """
    *mem_flags* takes its values from :class:`pyopencl.mem_flags` and corresponds
    to the *flags* argument of :class:`pyopencl.Buffer`.
    :class:`ImmediateAllocator` will attempt to ensure at allocation time that
    allocated memory is actually available. If no memory is available, an
    out-of-memory error is reported at allocation time.

    Implements :class:`AllocatorBase`.

    .. versionadded:: 2013.1

    .. method:: __init__(queue, mem_flags=pyopencl.mem_flags.READ_WRITE)

    .. automethod:: __call__

        Allocate a :class:`pyopencl.Buffer` of the given *size*.

        .. versionchanged :: 2020.2

            The allocator will succeed even for allocations of size zero,
            returning *None*.
    """

    # }}}

    # {{{ MemoryPool

    MemoryPool.__doc__ = remove_common_indentation("""
    A memory pool for OpenCL device memory in :class:`pyopencl.Buffer` form.
    *allocator* must be an instance of one of the above classes, and should be
    an :class:`ImmediateAllocator`.  The memory pool assumes that allocation
    failures are reported by the allocator immediately, and not in the
    OpenCL-typical deferred manner.

    Implements :class:`AllocatorBase`.

    .. versionchanged:: 2019.1

        Current bin allocation behavior documented, *leading_bits_in_bin_id*
        added.

    .. automethod:: __init__

    .. automethod:: allocate

        Return a :class:`PooledBuffer` of the given *size*.

    .. automethod:: __call__

        Synonym for :meth:`allocate` to match :class:`AllocatorBase`.

        .. versionadded:: 2011.2
    """) + _MEMPOOL_IFACE_DOCS

    # }}}


_monkeypatch_docstrings()


def _monkeypatch_svm_docstrings():
    from pytools.codegen import remove_common_indentation

    # {{{ PooledSVM

    PooledSVM.__doc__ = (  # pyright: ignore[reportPossiblyUnboundVariable]
        """An object representing a :class:`SVMPool`-based allocation of
        :ref:`svm`.  Analogous to :class:`~pyopencl.SVMAllocation`, however once
        this object is deleted, its associated device memory is returned to the
        pool from which it came.

        .. versionadded:: 2022.2

        .. note::

            If the :class:`SVMAllocator` for the :class:`SVMPool` that allocated an
            object of this type is associated with an (in-order)
            :class:`~pyopencl.CommandQueue`, sufficient synchronization is provided
            to ensure operations enqueued before deallocation complete before
            operations from a different use (possibly in a different queue) are
            permitted to start. This applies when :class:`release` is called and
            also when the object is freed automatically by the garbage collector.

        Is a :class:`pyopencl.SVMPointer`.

        Supports structural equality and hashing.

        .. automethod:: release

            Return the held memory to the pool. See the note about synchronization
            behavior during deallocation above.

        .. automethod:: enqueue_release

            Synonymous to :meth:`release`, for consistency with
            :class:`~pyopencl.SVMAllocation`. Note that, unlike
            :meth:`pyopencl.SVMAllocation.enqueue_release`, specifying a queue
            or events to be waited for is not supported.

        .. automethod:: bind_to_queue

            Analogous to :meth:`pyopencl.SVMAllocation.bind_to_queue`.

        .. automethod:: unbind_from_queue

            Analogous to :meth:`pyopencl.SVMAllocation.unbind_from_queue`.
        """)

    # }}}

    # {{{ SVMAllocator

    SVMAllocator.__doc__ = (  # pyright: ignore[reportPossiblyUnboundVariable]
        """
        .. versionadded:: 2022.2

        .. automethod:: __init__

            :arg flags: See :class:`~pyopencl.svm_mem_flags`.
            :arg queue: If not specified, allocations will be freed
                eagerly, irrespective of whether pending/enqueued operations
                are still using the memory.

                If specified, deallocation of memory will be enqueued
                with the given queue, and will only be performed
                after previously-enqueue operations in the queue have
                completed.

                It is an error to specify an out-of-order queue.

                .. warning::

                    Not specifying a queue will typically lead to undesired
                    behavior, including crashes and memory corruption.
                    See the warning in :ref:`svm`.

        .. automethod:: __call__

            Return a :class:`~pyopencl.SVMAllocation` of the given *size*.
        """)

    # }}}

    # {{{ SVMPool

    SVMPool.__doc__ = (  # pyright: ignore[reportPossiblyUnboundVariable]
        remove_common_indentation("""
        A memory pool for OpenCL device memory in :ref:`SVM <svm>` form.
        *allocator* must be an instance of :class:`SVMAllocator`.

        .. versionadded:: 2022.2

        .. automethod:: __init__
        .. automethod:: __call__

            Return a :class:`PooledSVM` of the given *size*.
        """) + _MEMPOOL_IFACE_DOCS)

    # }}}


if get_cl_header_version() >= (2, 0):
    _monkeypatch_svm_docstrings()

# }}}


# {{{ first-arg caches

_first_arg_dependent_caches: list[Mapping[Hashable, object]] = []


HashableT = TypeVar("HashableT", bound="Hashable")
RetT = TypeVar("RetT")
P = ParamSpec("P")


def first_arg_dependent_memoize(
            func: Callable[Concatenate[HashableT, P], RetT]
        ) -> Callable[Concatenate[HashableT, P], RetT]:
    def wrapper(cl_object: HashableT, *args: P.args, **kwargs: P.kwargs) -> RetT:
        """Provides memoization for a function. Typically used to cache
        things that get created inside a :class:`pyopencl.Context`, e.g. programs
        and kernels. Assumes that the first argument of the decorated function is
        an OpenCL object that might go away, such as a :class:`pyopencl.Context` or
        a :class:`pyopencl.CommandQueue`, and based on which we might want to clear
        the cache.

        .. versionadded:: 2011.2
        """
        if kwargs:
            cache_key = (args, frozenset(kwargs.items()))
        else:
            cache_key = (args,)

        ctx_dict: dict[Hashable, dict[Hashable, RetT]]
        try:
            ctx_dict = func._pyopencl_first_arg_dep_memoize_dic  # pyright: ignore[reportFunctionMemberAccess]
        except AttributeError:
            # FIXME: This may keep contexts alive longer than desired.
            # But I guess since the memory in them is freed, who cares.
            ctx_dict = func._pyopencl_first_arg_dep_memoize_dic = {}  # pyright: ignore[reportFunctionMemberAccess]
            _first_arg_dependent_caches.append(ctx_dict)

        try:
            return ctx_dict[cl_object][cache_key]
        except KeyError:
            arg_dict = ctx_dict.setdefault(cl_object, {})
            result = func(cl_object, *args, **kwargs)
            arg_dict[cache_key] = result
            return result

    from functools import update_wrapper
    update_wrapper(wrapper, func)
    return wrapper


context_dependent_memoize = first_arg_dependent_memoize


def first_arg_dependent_memoize_nested(
        nested_func: Callable[Concatenate[Hashable, P], RetT]
        ) -> Callable[Concatenate[Hashable, P], RetT]:
    """Provides memoization for nested functions.

    Typically used to cache things that get created inside a
    :class:`pyopencl.Context`, e.g. programs and kernels. Assumes that the first
    argument of the decorated function is an OpenCL object that might go away,
    such as a :class:`pyopencl.Context` or a :class:`pyopencl.CommandQueue`, and
    will therefore respond to :func:`clear_first_arg_caches`.

    .. versionadded:: 2013.1
    """

    from functools import wraps
    cache_dict_name = intern(
        f"_memoize_inner_dic_{nested_func.__name__}_"
        f"{nested_func.__code__.co_filename}_"
        f"{nested_func.__code__.co_firstlineno}")

    from inspect import currentframe

    # prevent ref cycle
    frame = currentframe()
    cache_context = None
    if frame:
        try:
            caller_frame = frame.f_back
            if caller_frame:
                cache_context = caller_frame.f_globals[caller_frame.f_code.co_name]
        finally:
            # del caller_frame
            pass

    cache_dict: dict[Hashable, dict[Hashable, RetT]]
    try:
        cache_dict = getattr(cache_context, cache_dict_name)
    except AttributeError:
        cache_dict = {}
        _first_arg_dependent_caches.append(cache_dict)
        setattr(cache_context, cache_dict_name, cache_dict)

    @wraps(nested_func)
    def new_nested_func(cl_object: Hashable, *args: P.args, **kwargs: P.kwargs) -> RetT:
        assert not kwargs

        try:
            return cache_dict[cl_object][args]
        except KeyError:
            arg_dict = cache_dict.setdefault(cl_object, {})
            result = nested_func(cl_object, *args, **kwargs)
            arg_dict[args] = result
            return result

    return new_nested_func


def clear_first_arg_caches():
    """Empties all first-argument-dependent memoization caches.

    Also releases all held reference contexts. If it is important to you that the
    program detaches from its context, you might need to call this function to
    free all remaining references to your context.

    .. versionadded:: 2011.2
    """
    for cache in _first_arg_dependent_caches:
        # NOTE: this could be fixed by making the caches a MutableMapping, but
        # that doesn't seem to be correctly covariant in its values, so other
        # parts fail to work nicely..
        cache.clear()  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType]


if TYPE_CHECKING:
    import pyopencl as cl
    from pyopencl.array import Array as CLArray

atexit.register(clear_first_arg_caches)

# }}}


# {{{ pytest fixtures

class _ContextFactory:
    device: cl.Device

    def __init__(self, device: cl.Device):
        self.device = device

    def __call__(self):
        # Get rid of leftovers from past tests.
        # CL implementations are surprisingly limited in how many
        # simultaneous contexts they allow...
        clear_first_arg_caches()

        from gc import collect
        collect()

        import pyopencl as cl
        return cl.Context([self.device])

    @override
    def __str__(self) -> str:
        # Don't show address, so that parallel test collection works
        device = self.device.name.strip()
        platform = self.device.platform.name.strip()
        return f"<context factory for <pyopencl.Device '{device}' on '{platform}'>>"


DeviceOrPlatformT = TypeVar("DeviceOrPlatformT", "cl.Device", "cl.Platform")


def _find_cl_obj(
            objs: Sequence[DeviceOrPlatformT],
            identifier: str
        ) -> DeviceOrPlatformT:
    try:
        num = int(identifier)
    except Exception:
        pass
    else:
        return objs[num]

    for obj in objs:
        if identifier.lower() in (obj.name + " " + obj.vendor).lower():
            return obj
    raise RuntimeError(f"object '{identifier}' not found")


def get_test_platforms_and_devices(
            plat_dev_string: str | None = None
        ):
    """Parse a string of the form 'PYOPENCL_TEST=0:0,1;intel:i5'.

    :return: list of tuples (platform, [device, device, ...])
    """

    import pyopencl as cl

    if plat_dev_string is None:
        import os
        plat_dev_string = os.environ.get("PYOPENCL_TEST", None)

    if plat_dev_string:
        result: list[tuple[cl.Platform, list[cl.Device]]] = []

        for entry in plat_dev_string.split(";"):
            lhsrhs = entry.split(":")

            if len(lhsrhs) == 1:
                platform = _find_cl_obj(cl.get_platforms(), lhsrhs[0])
                result.append((platform, platform.get_devices()))

            elif len(lhsrhs) != 2:
                raise RuntimeError("invalid syntax of PYOPENCL_TEST")
            else:
                plat_str, dev_strs = lhsrhs

                platform = _find_cl_obj(cl.get_platforms(), plat_str)
                devs = platform.get_devices()
                result.append(
                        (platform,
                            [_find_cl_obj(devs, dev_id)
                                for dev_id in dev_strs.split(",")]))

        return result

    else:
        return [
                (platform, platform.get_devices())
                for platform in cl.get_platforms()]


def get_pyopencl_fixture_arg_names(
        metafunc: pytest.Metafunc,
        extra_arg_names: list[str] | None = None) -> list[str]:
    if extra_arg_names is None:
        extra_arg_names = []

    supported_arg_names = [
            "platform", "device",
            "ctx_factory", "ctx_getter",
            *extra_arg_names
            ]

    arg_names: list[str] = []
    for arg in supported_arg_names:
        if arg not in metafunc.fixturenames:
            continue

        if arg == "ctx_getter":
            from warnings import warn
            warn(
                "The 'ctx_getter' arg is deprecated in favor of 'ctx_factory'.",
                DeprecationWarning, stacklevel=2)

        arg_names.append(arg)

    return arg_names


def get_pyopencl_fixture_arg_values() -> tuple[list[dict[str, Any]],
                                               Callable[[Any], str]]:
    import pyopencl as cl

    arg_values: list[dict[str, Any]] = []
    for platform, devices in get_test_platforms_and_devices():
        for device in devices:
            arg_dict = {
                "platform": platform,
                "device": device,
                "ctx_factory": _ContextFactory(device),
                "ctx_getter": _ContextFactory(device)
            }
            arg_values.append(arg_dict)

    def idfn(val: Any) -> str:
        if isinstance(val, cl.Platform):
            # Don't show address, so that parallel test collection works
            return f"<pyopencl.Platform '{val.name}'>"
        else:
            return str(val)

    return arg_values, idfn


def pytest_generate_tests_for_pyopencl(metafunc: pytest.Metafunc) -> None:
    """Using the line::

        from pyopencl.tools import pytest_generate_tests_for_pyopencl
                as pytest_generate_tests

    in your `pytest <https://docs.pytest.org/en/latest/>`__ test scripts allows
    you to use the arguments *ctx_factory*, *device*, or *platform* in your test
    functions, and they will automatically be run for each OpenCL device/platform
    in the system, as appropriate.

    The following two environment variabls is also supported to control
    device/platform choice::

        PYOPENCL_TEST=0:0,1;intel=i5,i7
    """

    arg_names = get_pyopencl_fixture_arg_names(metafunc)
    if not arg_names:
        return

    arg_values, ids = get_pyopencl_fixture_arg_values()
    arg_values = [
            tuple(arg_dict[name] for name in arg_names)
            for arg_dict in arg_values
            ]

    metafunc.parametrize(arg_names, arg_values, ids=ids)

# }}}


# {{{ C argument lists

ArgType: TypeAlias = "np.dtype[Any] | VectorArg"
ArgDType: TypeAlias = "np.dtype[Any] | None"


class Argument(ABC):
    """
    .. automethod:: declarator
    """

    @abstractmethod
    def declarator(self) -> str:
        pass


@dataclass(frozen=True, init=False)
class DtypedArgument(Argument, ABC):
    """
    .. autoattribute:: name
    .. autoattribute:: dtype
    """
    dtype: np.dtype[Any]
    name: str

    def __init__(self, dtype: DTypeLike, name: str) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dtype", np.dtype(dtype))


@dataclass(frozen=True)
class VectorArg(DtypedArgument):
    """Inherits from :class:`DtypedArgument`.

    .. automethod:: __init__
    """
    with_offset: bool

    def __init__(self, dtype: DTypeLike, name: str, with_offset: bool = False):
        super().__init__(dtype, name)
        object.__setattr__(self, "with_offset", with_offset)

    @override
    def declarator(self) -> str:
        if self.with_offset:
            # Two underscores -> less likelihood of a name clash.
            return "__global {} *{}__base, long {}__offset".format(
                    dtype_to_ctype(self.dtype), self.name, self.name)
        else:
            result = "__global {} *{}".format(dtype_to_ctype(self.dtype), self.name)

        return result


@dataclass(frozen=True, init=False)
class ScalarArg(DtypedArgument):
    """Inherits from :class:`DtypedArgument`."""

    @override
    def declarator(self) -> str:
        return "{} {}".format(dtype_to_ctype(self.dtype), self.name)


@dataclass(frozen=True)
class OtherArg(Argument):
    decl: str
    name: str

    @override
    def declarator(self) -> str:
        return self.decl


def parse_c_arg(c_arg: str, with_offset: bool = False) -> DtypedArgument:
    for aspace in ["__local", "__constant"]:
        if aspace in c_arg:
            raise RuntimeError("cannot deal with local or constant "
                    "OpenCL address spaces in C argument lists ")

    c_arg = c_arg.replace("__global", "")

    if with_offset:
        def vec_arg_factory(dtype: DTypeLike, name: str) -> VectorArg:
            return VectorArg(dtype, name, with_offset=True)
    else:
        vec_arg_factory = VectorArg

    from pyopencl.compyte.dtypes import parse_c_arg_backend

    return parse_c_arg_backend(c_arg, ScalarArg, vec_arg_factory)


def parse_arg_list(
        arguments: str | Sequence[str] | Sequence[Argument],
        with_offset: bool = False) -> Sequence[DtypedArgument]:
    """Parse a list of kernel arguments. *arguments* may be a comma-separate
    list of C declarators in a string, a list of strings representing C
    declarators, or :class:`Argument` objects.
    """

    if isinstance(arguments, str):
        arguments = arguments.split(",")

    def parse_single_arg(obj: str | Argument) -> DtypedArgument:
        if isinstance(obj, str):
            from pyopencl.tools import parse_c_arg
            return parse_c_arg(obj, with_offset=with_offset)
        else:
            assert isinstance(obj, DtypedArgument)
            return obj

    return [parse_single_arg(arg) for arg in arguments]


def get_arg_list_arg_types(arg_types: Sequence[Argument]) -> tuple[ArgType, ...]:
    result: list[ArgType] = []

    for arg_type in arg_types:
        if isinstance(arg_type, ScalarArg):
            result.append(arg_type.dtype)
        elif isinstance(arg_type, VectorArg):
            result.append(arg_type)
        else:
            raise RuntimeError(f"arg type not understood: {type(arg_type)}")

    return tuple(result)


def get_arg_list_scalar_arg_dtypes(
        arg_types: Sequence[Argument]
        ) -> Sequence[ArgDType]:
    result: list[ArgDType] = []

    for arg_type in arg_types:
        if isinstance(arg_type, ScalarArg):
            result.append(arg_type.dtype)
        elif isinstance(arg_type, VectorArg):
            result.append(None)
            if arg_type.with_offset:
                result.append(np.dtype(np.int64))
        else:
            raise RuntimeError(f"arg type not understood: {type(arg_type)}")

    return result


def get_arg_offset_adjuster_code(arg_types: Sequence[Argument]) -> str:
    result: list[str] = []

    for arg_type in arg_types:
        if isinstance(arg_type, VectorArg) and arg_type.with_offset:
            name = arg_type.name
            ctype = dtype_to_ctype(arg_type.dtype)
            result.append(
                    f"__global {ctype} *{name} = "
                    f"(__global {ctype} *) "
                    f"((__global char *) {name}__base + {name}__offset);")

    return "\n".join(result)

# }}}


def get_gl_sharing_context_properties() -> list[tuple[cl.context_properties, Any]]:
    import pyopencl as cl

    ctx_props = cl.context_properties

    from OpenGL import platform as gl_platform

    props: list[tuple[cl.context_properties, Any]] = []

    import sys
    if sys.platform in ["linux", "linux2"]:
        from OpenGL import GLX
        props.append(
            (ctx_props.GL_CONTEXT_KHR, GLX.glXGetCurrentContext()))
        props.append(
                (ctx_props.GLX_DISPLAY_KHR,
                    GLX.glXGetCurrentDisplay()))
    elif sys.platform == "win32":
        from OpenGL import WGL
        props.append(
            (ctx_props.GL_CONTEXT_KHR, gl_platform.GetCurrentContext()))
        props.append(
                (ctx_props.WGL_HDC_KHR,
                    WGL.wglGetCurrentDC()))
    elif sys.platform == "darwin":
        props.append(
            (ctx_props.CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
                cl.get_apple_cgl_share_group()))
    else:
        raise NotImplementedError(f"platform '{sys.platform}' not yet supported")

    return props


class _CDeclList:
    def __init__(self, device: cl.Device) -> None:
        self.device: cl.Device = device
        self.declared_dtypes: set[np.dtype[Any]] = set()
        self.declarations: list[str] = []
        self.saw_double: bool = False
        self.saw_complex: bool = False

    def add_dtype(self, dtype: DTypeLike) -> None:
        dtype = np.dtype(dtype)

        if dtype.type in (np.float64, np.complex128):
            self.saw_double = True

        if dtype.kind == "c":
            self.saw_complex = True

        if dtype.kind != "V":
            return

        if dtype in self.declared_dtypes:
            return

        from pyopencl.cltypes import vec_type_to_scalar_and_count

        if dtype in vec_type_to_scalar_and_count:
            return

        if hasattr(dtype, "subdtype") and dtype.subdtype is not None:
            self.add_dtype(dtype.subdtype[0])
            return

        fields = cast("Mapping[str, tuple[np.dtype[Any], int]] | None", dtype.fields)
        if fields is not None:
            for _name, field_data in sorted(fields.items()):
                field_dtype, _offset = field_data[:2]
                self.add_dtype(field_dtype)

        _, cdecl = match_dtype_to_c_struct(
                self.device, dtype_to_ctype(dtype), dtype)

        self.declarations.append(cdecl)
        self.declared_dtypes.add(dtype)

    def visit_arguments(self, arguments: Sequence[Argument]) -> None:
        for arg in arguments:
            if not isinstance(arg, DtypedArgument):
                continue

            dtype = arg.dtype
            if dtype.type in (np.float64, np.complex128):
                self.saw_double = True

            if dtype.kind == "c":
                self.saw_complex = True

    def get_declarations(self) -> str:
        result = "\n\n".join(self.declarations)

        if self.saw_complex:
            result = (
                    "#include <pyopencl-complex.h>\n\n"
                    + result)

        if self.saw_double:
            result = (
                    """
                    #if __OPENCL_C_VERSION__ < 120
                    #pragma OPENCL EXTENSION cl_khr_fp64: enable
                    #endif
                    #define PYOPENCL_DEFINE_CDOUBLE
                    """
                    + result)

        return result


class _DTypeDict(TypedDict):
    names: list[str]
    formats: list[np.dtype[Any]]
    offsets: list[int]
    itemsize: int


@memoize
def match_dtype_to_c_struct(
        device: cl.Device,
        name: str,
        dtype: np.dtype[Any],
        context: cl.Context | None = None) -> tuple[np.dtype[Any], str]:
    """Return a tuple ``(dtype, c_decl)`` such that the C struct declaration
    in ``c_decl`` and the structure :class:`numpy.dtype` instance ``dtype``
    have the same memory layout.

    Note that *dtype* may be modified from the value that was passed in,
    for example to insert padding.

    (As a remark on implementation, this routine runs a small kernel on
    the given *device* to ensure that :mod:`numpy` and C offsets and
    sizes match.)

    .. versionadded:: 2013.1

    This example explains the use of this function::

        >>> import numpy as np
        >>> import pyopencl as cl
        >>> import pyopencl.tools
        >>> ctx = cl.create_some_context()
        >>> dtype = np.dtype([("id", np.uint32), ("value", np.float32)])
        >>> dtype, c_decl = pyopencl.tools.match_dtype_to_c_struct(
        ...     ctx.devices[0], 'id_val', dtype)
        >>> print c_decl
        typedef struct {
          unsigned id;
          float value;
        } id_val;
        >>> print dtype
        [('id', '<u4'), ('value', '<f4')]
        >>> cl.tools.get_or_register_dtype('id_val', dtype)

    As this example shows, it is important to call
    :func:`get_or_register_dtype` on the modified ``dtype`` returned by this
    function, not the original one.
    """
    fields = cast("Mapping[str, tuple[np.dtype[Any], int]] | None", dtype.fields)
    if not fields:
        raise ValueError(f"dtype has no fields: '{dtype}'")

    import pyopencl as cl

    sorted_fields = sorted(
            fields.items(),
            key=lambda name_dtype_offset: name_dtype_offset[1][1])

    c_fields: list[str] = []
    for field_name, dtype_and_offset in sorted_fields:
        field_dtype, _offset = dtype_and_offset[:2]
        if hasattr(field_dtype, "subdtype") and field_dtype.subdtype is not None:
            array_dtype = field_dtype.subdtype[0]
            if hasattr(array_dtype, "subdtype") and array_dtype.subdtype is not None:
                raise NotImplementedError("nested array dtypes are not supported")
            array_dims = field_dtype.subdtype[1]
            dims_str = ""
            try:
                for dim in array_dims:
                    dims_str += f"[{dim}]"
            except TypeError:
                dims_str = f"[{array_dims}]"
            c_fields.append("  {} {}{};".format(
                dtype_to_ctype(array_dtype), field_name, dims_str)
            )
        else:
            c_fields.append(
                    "  {} {};".format(dtype_to_ctype(field_dtype), field_name))

    c_decl = "typedef struct {{\n{}\n}} {};\n\n".format(
            "\n".join(c_fields),
            name)

    cdl = _CDeclList(device)
    for _field_name, dtype_and_offset in sorted_fields:
        field_dtype, _offset = dtype_and_offset[:2]
        cdl.add_dtype(field_dtype)

    pre_decls = cdl.get_declarations()

    offset_code = "\n".join(
            f"result[{i + 1}] = pycl_offsetof({name}, {field_name});"
            for i, (field_name, _) in enumerate(sorted_fields))

    src = rf"""
        #define pycl_offsetof(st, m) \
                 ((uint) ((__local char *) &(dummy.m) \
                 - (__local char *)&dummy ))

        {pre_decls}

        {c_decl}

        __kernel void get_size_and_offsets(__global uint *result)
        {{
            result[0] = sizeof({name});
            __local {name} dummy;
            {offset_code}
        }}
    """

    if context is None:
        context = cl.Context([device])

    queue = cl.CommandQueue(context)

    prg = cl.Program(context, src)
    knl = prg.build(devices=[device]).get_size_and_offsets

    import pyopencl.array as cl_array

    result_buf = cl_array.empty(queue, 1+len(sorted_fields), np.uint32)
    assert result_buf.data is not None

    knl(queue, (1,), (1,), result_buf.data)
    queue.finish()
    size_and_offsets = result_buf.get()

    size = int(size_and_offsets[0])
    offsets = size_and_offsets[1:]

    if any(ofs >= size for ofs in offsets):
        # offsets not plausible

        if dtype.itemsize == size:
            # If sizes match, use numpy's idea of the offsets.
            offsets = [dtype_and_offset[1] for _name, dtype_and_offset in sorted_fields]
        else:
            raise RuntimeError(
                    "OpenCL compiler reported offsetof() past sizeof() for struct "
                    f"layout on '{device}'. This makes no sense, and it usually "
                    "indicates a compiler bug. Refusing to discover struct layout.")

    result_buf.data.release()
    del knl
    del prg
    del queue
    del context

    try:
        dtype_arg_dict = _DTypeDict(
            names=[name for name, _ in sorted_fields],
            formats=[dtype_and_offset[0] for _, dtype_and_offset in sorted_fields],
            offsets=[int(x) for x in offsets],
            itemsize=int(size_and_offsets[0]),
            )
        arg_dtype = np.dtype(dtype_arg_dict)

        if arg_dtype.itemsize != size_and_offsets[0]:
            # "Old" versions of numpy (1.6.x?) silently ignore "itemsize". Boo.
            dtype_arg_dict["names"].append("_pycl_size_fixer")
            dtype_arg_dict["formats"].append(np.dtype(np.uint8))
            dtype_arg_dict["offsets"].append(int(size_and_offsets[0]) - 1)

            arg_dtype = np.dtype(dtype_arg_dict)
    except NotImplementedError:
        def calc_field_type() -> Iterator[tuple[str, str | np.dtype[Any]]]:
            total_size = 0
            padding_count = 0
            for offset, (field_name, dtype_and_offset) in zip(
                    offsets, sorted_fields, strict=True):
                field_dtype, _ = dtype_and_offset[:2]
                if offset > total_size:
                    padding_count += 1
                    yield f"__pycl_padding{padding_count}", f"V{offset - total_size}"

                yield field_name, field_dtype
                total_size = field_dtype.itemsize + offset

        arg_dtype = np.dtype(list(calc_field_type()))

    assert arg_dtype.itemsize == size_and_offsets[0]

    return arg_dtype, c_decl


@memoize
def dtype_to_c_struct(device: cl.Device, dtype: np.dtype[Any]) -> str:
    fields = cast("Mapping[str, tuple[np.dtype[Any], int]] | None", dtype.fields)
    if fields is None:
        return ""

    from pyopencl.cltypes import vec_type_to_scalar_and_count

    if dtype in vec_type_to_scalar_and_count:
        # Vector types are built-in. Don't try to redeclare those.
        return ""

    matched_dtype, c_decl = match_dtype_to_c_struct(
            device, dtype_to_ctype(dtype), dtype)

    matched_fields = cast("Mapping[str, tuple[np.dtype[Any], int]] | None",
                          matched_dtype.fields)
    assert matched_fields is not None

    def dtypes_match() -> bool:
        result = len(fields) == len(matched_fields)

        for name, val in fields.items():
            result = result and matched_fields[name] == val

        return result

    assert dtypes_match()

    return c_decl


# {{{ code generation/templating helper

def _process_code_for_macro(code: str) -> str:
    code = code.replace("//CL//", "\n")

    if "//" in code:
        raise RuntimeError(
            "end-of-line comments ('//') may not be used in code snippets")

    return code.replace("\n", " \\\n")


class _TextTemplate(ABC):
    @abstractmethod
    def render(self, context: dict[str, Any]) -> str:
        pass


@dataclass(frozen=True)
class _SimpleTextTemplate(_TextTemplate):
    txt: str

    @override
    def render(self, context: dict[str, Any]) -> str:
        return self.txt


@dataclass(frozen=True)
class _PrintfTextTemplate(_TextTemplate):
    txt: str

    @override
    def render(self, context: dict[str, Any]) -> str:
        return self.txt % context


@dataclass(frozen=True)
class _MakoTextTemplate(_TextTemplate):
    txt: str
    template: Template = field(init=False)

    def __post_init__(self) -> None:
        from mako.template import Template

        object.__setattr__(self, "template", Template(self.txt, strict_undefined=True))

    @override
    def render(self, context: dict[str, Any]) -> str:
        return self.template.render(**context)


class _ArgumentPlaceholder:
    """A placeholder for subclasses of :class:`DtypedArgument`. This is needed
    because the concrete dtype of the argument is not known at template
    creation time--it may be a type alias that will only be filled in
    at run time. These types take the place of these proto-arguments until
    all types are known.

    See also :class:`_TemplateRenderer.render_arg`.
    """

    target_class: ClassVar[type[DtypedArgument]]

    def __init__(self,
                 typename: DTypeLike,
                 name: str,
                 **extra_kwargs: Any) -> None:
        self.typename: DTypeLike = typename
        self.name: str = name
        self.extra_kwargs: dict[str, Any] = extra_kwargs


class _VectorArgPlaceholder(_ArgumentPlaceholder):
    target_class: ClassVar[type[DtypedArgument]] = VectorArg


class _ScalarArgPlaceholder(_ArgumentPlaceholder):
    target_class: ClassVar[type[DtypedArgument]] = ScalarArg


class _TemplateRenderer:
    def __init__(self,
                 template: KernelTemplateBase,
                 type_aliases: (
                     dict[str, np.dtype[Any]]
                     | Sequence[tuple[str, np.dtype[Any]]]),
                 var_values: dict[str, str] | Sequence[tuple[str, str]],
                 context: cl.Context | None = None,
                 options: Any = None) -> None:
        self.template: KernelTemplateBase = template
        self.type_aliases: dict[str, np.dtype[Any]] = dict(type_aliases)
        self.var_dict: dict[str, str] = dict(var_values)

        for name in self.var_dict:
            if name.startswith("macro_"):
                self.var_dict[name] = _process_code_for_macro(self.var_dict[name])

        self.context: cl.Context | None = context
        self.options: Any = options

    @overload
    def __call__(self, txt: None) -> None: ...

    @overload
    def __call__(self, txt: str) -> str: ...

    def __call__(self, txt: str | None) -> str | None:
        if txt is None:
            return txt

        result = self.template.get_text_template(txt).render(self.var_dict)

        return str(result)

    def get_rendered_kernel(self, txt: str, kernel_name: str) -> cl.Kernel:
        if self.context is None:
            raise ValueError("context not provided -- cannot render kernel")

        import pyopencl as cl
        prg = cl.Program(self.context, self(txt)).build(self.options)

        kernel_name_prefix = self.var_dict.get("kernel_name_prefix")
        if kernel_name_prefix is not None:
            kernel_name = kernel_name_prefix+kernel_name

        return getattr(prg, kernel_name)

    def parse_type(self, typename: Any) -> np.dtype[Any]:
        if isinstance(typename, str):
            try:
                return self.type_aliases[typename]
            except KeyError:
                from pyopencl.compyte.dtypes import NAME_TO_DTYPE
                return NAME_TO_DTYPE[typename]
        else:
            return np.dtype(typename)

    def render_arg(self, arg_placeholder: _ArgumentPlaceholder) -> DtypedArgument:
        return arg_placeholder.target_class(
                self.parse_type(arg_placeholder.typename),
                arg_placeholder.name,
                **arg_placeholder.extra_kwargs)

    _C_COMMENT_FINDER: ClassVar[re.Pattern[str]] = re.compile(r"/\*.*?\*/")

    def render_argument_list(self,
                             *arg_lists: Any,
                             with_offset: bool = False,
                             **kwargs: Any) -> list[Argument]:
        if kwargs:
            raise TypeError("unrecognized kwargs: " + ", ".join(kwargs))

        all_args: list[Any] = []
        for arg_list in arg_lists:
            if isinstance(arg_list, str):
                arg_list = str(
                        self.template
                        .get_text_template(arg_list).render(self.var_dict))
                arg_list = self._C_COMMENT_FINDER.sub("", arg_list)
                arg_list = arg_list.replace("\n", " ")

                all_args.extend(arg_list.split(","))
            else:
                all_args.extend(arg_list)

        if with_offset:
            def vec_arg_factory(
                    typename: DTypeLike,
                    name: str) -> _VectorArgPlaceholder:
                return _VectorArgPlaceholder(typename, name, with_offset=True)
        else:
            vec_arg_factory = _VectorArgPlaceholder

        from pyopencl.compyte.dtypes import parse_c_arg_backend

        parsed_args: list[Argument] = []
        for arg in all_args:
            if isinstance(arg, str):
                arg = arg.strip()
                if not arg:
                    continue

                ph = parse_c_arg_backend(arg,
                        _ScalarArgPlaceholder, vec_arg_factory,
                        name_to_dtype=lambda x: x)  # pyright: ignore[reportArgumentType]
                parsed_arg = self.render_arg(ph)
            elif isinstance(arg, Argument):
                parsed_arg = arg
            elif isinstance(arg, tuple):
                assert isinstance(arg[0], str)
                assert isinstance(arg[1], str)
                parsed_arg = ScalarArg(self.parse_type(arg[0]), arg[1])
            else:
                raise TypeError(f"unexpected argument type: {type(arg)}")

            parsed_args.append(parsed_arg)

        return parsed_args

    def get_type_decl_preamble(self,
                               device: cl.Device,
                               decl_type_names: Sequence[DTypeLike],
                               arguments: Sequence[Argument] | None = None,
                               ) -> str:
        cdl = _CDeclList(device)

        for typename in decl_type_names:
            cdl.add_dtype(self.parse_type(typename))

        if arguments is not None:
            cdl.visit_arguments(arguments)

        for _, tv in sorted(self.type_aliases.items()):
            cdl.add_dtype(tv)

        type_alias_decls = [
                "typedef {} {};".format(dtype_to_ctype(val), name)
                for name, val in sorted(self.type_aliases.items())
                ]

        return cdl.get_declarations() + "\n" + "\n".join(type_alias_decls)


class KernelTemplateBase(ABC):
    def __init__(self, template_processor: str | None = None) -> None:
        self.template_processor: str | None = template_processor

        self.build_cache: dict[Hashable, Any] = {}
        _first_arg_dependent_caches.append(self.build_cache)

    _TEMPLATE_PROCESSOR_PATTERN: ClassVar[re.Pattern[str]] = (
        re.compile(r"^//CL(?::([a-zA-Z0-9_]+))?//")
    )

    @memoize_method
    def get_text_template(self, txt: str) -> _TextTemplate:
        proc_match = self._TEMPLATE_PROCESSOR_PATTERN.match(txt)
        tpl_processor = None

        if proc_match is not None:
            tpl_processor = proc_match.group(1)
            # chop off //CL// mark
            txt = txt[len(proc_match.group(0)):]

        if tpl_processor is None:
            tpl_processor = self.template_processor

        if tpl_processor is None or tpl_processor == "none":
            return _SimpleTextTemplate(txt)
        elif tpl_processor == "printf":
            return _PrintfTextTemplate(txt)
        elif tpl_processor == "mako":
            return _MakoTextTemplate(txt)
        else:
            raise RuntimeError(f"unknown template processor '{tpl_processor}'")

    # TODO: this does not seem to be used anywhere -> deprecate / remove
    def get_preamble(self) -> str:
        return ""

    def get_renderer(self,
                     type_aliases: (
                         dict[str, np.dtype[Any]]
                         | Sequence[tuple[str, np.dtype[Any]]]),
                     var_values: dict[str, str] | Sequence[tuple[str, str]],
                     context: cl.Context | None = None,  # pyright: ignore[reportUnusedParameter]
                     options: Any = None,  # pyright: ignore[reportUnusedParameter]
                     ) -> _TemplateRenderer:
        return _TemplateRenderer(self, type_aliases, var_values)

    @abstractmethod
    def build_inner(self,
                    context: cl.Context,
                    *args: Any,
                    **kwargs: Any) -> Callable[..., cl.Event]:
        pass

    def build(self, context: cl.Context, *args: Any, **kwargs: Any) -> Any:
        """Provide caching for an :meth:`build_inner`."""

        cache_key = (context, args, tuple(sorted(kwargs.items())))
        try:
            return self.build_cache[cache_key]
        except KeyError:
            result = self.build_inner(context, *args, **kwargs)
            self.build_cache[cache_key] = result
            return result

# }}}


# {{{ array_module

# TODO: this is not used anywhere: deprecate + remove

class _CLFakeArrayModule:
    def __init__(self, queue: cl.CommandQueue | None = None) -> None:
        self.queue: cl.CommandQueue | None = queue

    @property
    def ndarray(self) -> type[CLArray]:
        from pyopencl.array import Array
        return Array

    def dot(self, x: CLArray, y: CLArray) -> NDArray[Any]:
        from pyopencl.array import dot
        return dot(x, y, queue=self.queue).get()

    def vdot(self, x: CLArray, y: CLArray) -> NDArray[Any]:
        from pyopencl.array import vdot
        return vdot(x, y, queue=self.queue).get()

    def empty(self,
              shape: int | tuple[int, ...],
              dtype: DTypeLike,
              order: str = "C") -> CLArray:
        from pyopencl.array import empty
        return empty(self.queue, shape, dtype, order=order)

    def hstack(self, arrays: Sequence[CLArray]) -> CLArray:
        from pyopencl.array import hstack
        return hstack(arrays, self.queue)


def array_module(a: Any) -> Any:
    if isinstance(a, np.ndarray):
        return np
    else:
        from pyopencl.array import Array

        if isinstance(a, Array):
            return _CLFakeArrayModule(a.queue)
        else:
            raise TypeError(f"array type not understood: {type(a)}")

# }}}


def is_spirv(s: str | bytes) -> TypeIs[bytes]:
    spirv_magic = b"\x07\x23\x02\x03"
    return (
            isinstance(s, bytes)
            and (
                s[:4] == spirv_magic
                or s[:4] == spirv_magic[::-1]))


# {{{ numpy key types builder

class _NumpyTypesKeyBuilder(KeyBuilderBase):  # pyright: ignore[reportUnusedClass]
    def update_for_VectorArg(self, key_hash: Hash, key: VectorArg) -> None:  # noqa: N802
        self.rec(key_hash, key.dtype)
        self.update_for_str(key_hash, key.name)
        self.rec(key_hash, key.with_offset)

    @override
    def update_for_type(self, key_hash: Hash, key: type) -> None:
        if issubclass(key, np.generic):
            self.update_for_str(key_hash, key.__name__)
            return

        raise TypeError(f"unsupported type for persistent hash keying: {key}")

# }}}


__all__ = [
    "AllocatorBase",
    "AllocatorBase",
    "Argument",
    "DeferredAllocator",
    "DtypedArgument",
    "ImmediateAllocator",
    "MemoryPool",
    "OtherArg",
    "PooledBuffer",
    "PooledSVM",
    "SVMAllocator",
    "SVMPool",
    "ScalarArg",
    "TypeNameNotKnown",
    "VectorArg",
    "bitlog2",
    "clear_first_arg_caches",
    "dtype_to_ctype",
    "first_arg_dependent_memoize",
    "get_or_register_dtype",
    "parse_arg_list",
    "pytest_generate_tests_for_pyopencl",
    "register_dtype",
]

# vim: foldmethod=marker
