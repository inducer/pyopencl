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

from abc import ABC, abstractmethod
from sys import intern
from typing import Any, List, Optional, Union

# Do not add a pyopencl import here: This will add an import cycle.

import numpy as np
from pytools import memoize, memoize_method
from pyopencl._cl import bitlog2, get_cl_header_version  # noqa: F401
from pytools.persistent_dict import KeyBuilder as KeyBuilderBase

import re

from pyopencl.compyte.dtypes import (  # noqa: F401
        get_or_register_dtype, TypeNameNotKnown,
        register_dtype, dtype_to_ctype)


def _register_types():
    from pyopencl.compyte.dtypes import (
            TYPE_REGISTRY, fill_registry_with_opencl_c_types)

    fill_registry_with_opencl_c_types(TYPE_REGISTRY)

    get_or_register_dtype("cfloat_t", np.complex64)
    get_or_register_dtype("cdouble_t", np.complex128)


_register_types()


# {{{ imported names

from pyopencl._cl import (  # noqa: F401
        PooledBuffer, AllocatorBase, DeferredAllocator,
        ImmediateAllocator, MemoryPool,
        )


if get_cl_header_version() >= (2, 0):
    from pyopencl._cl import (  # noqa: F401
            SVMPool,
            PooledSVM,
            SVMAllocator,
            )

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

    PooledSVM.__doc__ = """
    An object representing a :class:`SVMPool`-based allocation of
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
    """

    # }}}

    # {{{ SVMAllocator

    SVMAllocator.__doc__ = """
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
    """

    # }}}

    # {{{ SVMPool

    SVMPool.__doc__ = remove_common_indentation("""
    A memory pool for OpenCL device memory in :ref:`SVM <svm>` form.
    *allocator* must be an instance of :class:`SVMAllocator`.

    .. versionadded:: 2022.2

    .. automethod:: __init__
    .. automethod:: __call__

        Return a :class:`PooledSVM` of the given *size*.
    """) + _MEMPOOL_IFACE_DOCS

    # }}}


if get_cl_header_version() >= (2, 0):
    _monkeypatch_svm_docstrings()

# }}}


# {{{ first-arg caches

_first_arg_dependent_caches = []


def first_arg_dependent_memoize(func):
    def wrapper(cl_object, *args, **kwargs):
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

        try:
            ctx_dict = func._pyopencl_first_arg_dep_memoize_dic
        except AttributeError:
            # FIXME: This may keep contexts alive longer than desired.
            # But I guess since the memory in them is freed, who cares.
            ctx_dict = func._pyopencl_first_arg_dep_memoize_dic = {}
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


def first_arg_dependent_memoize_nested(nested_func):
    """Provides memoization for nested functions. Typically used to cache
    things that get created inside a :class:`pyopencl.Context`, e.g. programs
    and kernels. Assumes that the first argument of the decorated function is
    an OpenCL object that might go away, such as a :class:`pyopencl.Context` or
    a :class:`pyopencl.CommandQueue`, and will therefore respond to
    :func:`clear_first_arg_caches`.

    .. versionadded:: 2013.1

    Requires Python 2.5 or newer.
    """

    from functools import wraps
    cache_dict_name = intern("_memoize_inner_dic_%s_%s_%d"
            % (nested_func.__name__, nested_func.__code__.co_filename,
                nested_func.__code__.co_firstlineno))

    from inspect import currentframe
    # prevent ref cycle
    try:
        caller_frame = currentframe().f_back
        cache_context = caller_frame.f_globals[
                caller_frame.f_code.co_name]
    finally:
        #del caller_frame
        pass

    try:
        cache_dict = getattr(cache_context, cache_dict_name)
    except AttributeError:
        cache_dict = {}
        _first_arg_dependent_caches.append(cache_dict)
        setattr(cache_context, cache_dict_name, cache_dict)

    @wraps(nested_func)
    def new_nested_func(cl_object, *args):
        try:
            return cache_dict[cl_object][args]
        except KeyError:
            arg_dict = cache_dict.setdefault(cl_object, {})
            result = nested_func(cl_object, *args)
            arg_dict[args] = result
            return result

    return new_nested_func


def clear_first_arg_caches():
    """Empties all first-argument-dependent memoization caches. Also releases
    all held reference contexts. If it is important to you that the
    program detaches from its context, you might need to call this
    function to free all remaining references to your context.

    .. versionadded:: 2011.2
    """
    for cache in _first_arg_dependent_caches:
        cache.clear()


import atexit
atexit.register(clear_first_arg_caches)

# }}}


# {{{ pytest fixtures

class _ContextFactory:
    def __init__(self, device):
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

    def __str__(self):
        # Don't show address, so that parallel test collection works
        return ("<context factory for <pyopencl.Device '%s' on '%s'>>" %
                (self.device.name.strip(),
                 self.device.platform.name.strip()))


def get_test_platforms_and_devices(plat_dev_string=None):
    """Parse a string of the form 'PYOPENCL_TEST=0:0,1;intel:i5'.

    :return: list of tuples (platform, [device, device, ...])
    """

    import pyopencl as cl

    if plat_dev_string is None:
        import os
        plat_dev_string = os.environ.get("PYOPENCL_TEST", None)

    def find_cl_obj(objs, identifier):
        try:
            num = int(identifier)
        except Exception:
            pass
        else:
            return objs[num]

        found = False
        for obj in objs:
            if identifier.lower() in (obj.name + " " + obj.vendor).lower():
                return obj
        if not found:
            raise RuntimeError("object '%s' not found" % identifier)

    if plat_dev_string:
        result = []

        for entry in plat_dev_string.split(";"):
            lhsrhs = entry.split(":")

            if len(lhsrhs) == 1:
                platform = find_cl_obj(cl.get_platforms(), lhsrhs[0])
                result.append((platform, platform.get_devices()))

            elif len(lhsrhs) != 2:
                raise RuntimeError("invalid syntax of PYOPENCL_TEST")
            else:
                plat_str, dev_strs = lhsrhs

                platform = find_cl_obj(cl.get_platforms(), plat_str)
                devs = platform.get_devices()
                result.append(
                        (platform,
                            [find_cl_obj(devs, dev_id)
                                for dev_id in dev_strs.split(",")]))

        return result

    else:
        return [
                (platform, platform.get_devices())
                for platform in cl.get_platforms()]


def get_pyopencl_fixture_arg_names(metafunc, extra_arg_names=None):
    if extra_arg_names is None:
        extra_arg_names = []

    supported_arg_names = [
            "platform", "device",
            "ctx_factory", "ctx_getter",
            ] + extra_arg_names

    arg_names = []
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


def get_pyopencl_fixture_arg_values():
    import pyopencl as cl

    arg_values = []
    for platform, devices in get_test_platforms_and_devices():
        for device in devices:
            arg_dict = {
                "platform": platform,
                "device": device,
                "ctx_factory": _ContextFactory(device),
                "ctx_getter": _ContextFactory(device)
            }
            arg_values.append(arg_dict)

    def idfn(val):
        if isinstance(val, cl.Platform):
            # Don't show address, so that parallel test collection works
            return f"<pyopencl.Platform '{val.name}'>"
        else:
            return str(val)

    return arg_values, idfn


def pytest_generate_tests_for_pyopencl(metafunc):
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

class Argument(ABC):
    """
    .. automethod:: declarator
    """

    @abstractmethod
    def declarator(self) -> str:
        pass


class DtypedArgument(Argument):
    """
    .. attribute:: name
    .. attribute:: dtype
    """

    def __init__(self, dtype: Any, name: str) -> None:
        self.dtype = np.dtype(dtype)
        self.name = name

    def __repr__(self) -> str:
        return "{}({!r}, {})".format(
                self.__class__.__name__,
                self.name,
                self.dtype)

    def __eq__(self, other: Any) -> bool:
        return (type(self) is type(other)
                and self.dtype == other.dtype
                and self.name == other.name)

    def __hash__(self) -> int:
        return (
                hash(type(self))
                ^ hash(self.dtype)
                ^ hash(self.name))


class VectorArg(DtypedArgument):
    """Inherits from :class:`DtypedArgument`.

    .. automethod:: __init__
    """

    def __init__(self, dtype: Any, name: str, with_offset: bool = False) -> None:
        super().__init__(dtype, name)
        self.with_offset = with_offset

    def declarator(self) -> str:
        if self.with_offset:
            # Two underscores -> less likelihood of a name clash.
            return "__global {} *{}__base, long {}__offset".format(
                    dtype_to_ctype(self.dtype), self.name, self.name)
        else:
            result = "__global {} *{}".format(dtype_to_ctype(self.dtype), self.name)

        return result

    def __eq__(self, other) -> bool:
        return (super().__eq__(other)
                and self.with_offset == other.with_offset)

    def __hash__(self) -> int:
        return super().__hash__() ^ hash(self.with_offset)


class ScalarArg(DtypedArgument):
    """Inherits from :class:`DtypedArgument`."""

    def declarator(self):
        return "{} {}".format(dtype_to_ctype(self.dtype), self.name)


class OtherArg(Argument):
    def __init__(self, declarator: str, name: str) -> None:
        self.decl = declarator
        self.name = name

    def declarator(self) -> str:
        return self.decl

    def __eq__(self, other) -> bool:
        return (type(self) is type(other)
                and self.decl == other.decl
                and self.name == other.name)

    def __hash__(self) -> int:
        return (
                hash(type(self))
                ^ hash(self.decl)
                ^ hash(self.name))


def parse_c_arg(c_arg: str, with_offset: bool = False) -> DtypedArgument:
    for aspace in ["__local", "__constant"]:
        if aspace in c_arg:
            raise RuntimeError("cannot deal with local or constant "
                    "OpenCL address spaces in C argument lists ")

    c_arg = c_arg.replace("__global", "")

    if with_offset:
        def vec_arg_factory(dtype, name):
            return VectorArg(dtype, name, with_offset=True)
    else:
        vec_arg_factory = VectorArg

    from pyopencl.compyte.dtypes import parse_c_arg_backend
    return parse_c_arg_backend(c_arg, ScalarArg, vec_arg_factory)


def parse_arg_list(
        arguments: Union[str, List[str], List[DtypedArgument]],
        with_offset: bool = False) -> List[DtypedArgument]:
    """Parse a list of kernel arguments. *arguments* may be a comma-separate
    list of C declarators in a string, a list of strings representing C
    declarators, or :class:`Argument` objects.
    """

    if isinstance(arguments, str):
        arguments = arguments.split(",")

    def parse_single_arg(obj: Union[str, DtypedArgument]) -> DtypedArgument:
        if isinstance(obj, str):
            from pyopencl.tools import parse_c_arg
            return parse_c_arg(obj, with_offset=with_offset)
        else:
            assert isinstance(obj, DtypedArgument)
            return obj

    return [parse_single_arg(arg) for arg in arguments]


def get_arg_list_arg_types(arg_types):
    result = []

    for arg_type in arg_types:
        if isinstance(arg_type, ScalarArg):
            result.append(arg_type.dtype)
        elif isinstance(arg_type, VectorArg):
            result.append(arg_type)
        else:
            raise RuntimeError("arg type not understood: %s" % type(arg_type))

    return tuple(result)


def get_arg_list_scalar_arg_dtypes(
        arg_types: List[DtypedArgument]
        ) -> List[Optional[np.dtype]]:
    result: List[Optional[np.dtype]] = []

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


def get_arg_offset_adjuster_code(arg_types):
    result = []

    for arg_type in arg_types:
        if isinstance(arg_type, VectorArg) and arg_type.with_offset:
            result.append("__global %(type)s *%(name)s = "
                    "(__global %(type)s *) "
                    "((__global char *) %(name)s__base + %(name)s__offset);"
                    % {
                        "type": dtype_to_ctype(arg_type.dtype),
                        "name": arg_type.name})

    return "\n".join(result)

# }}}


def get_gl_sharing_context_properties():
    import pyopencl as cl

    ctx_props = cl.context_properties

    from OpenGL import platform as gl_platform

    props = []

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
        raise NotImplementedError("platform '%s' not yet supported"
                % sys.platform)

    return props


class _CDeclList:
    def __init__(self, device):
        self.device = device
        self.declared_dtypes = set()
        self.declarations = []
        self.saw_double = False
        self.saw_complex = False

    def add_dtype(self, dtype):
        dtype = np.dtype(dtype)

        if dtype in (np.float64, np.complex128):
            self.saw_double = True

        if dtype.kind == "c":
            self.saw_complex = True

        if dtype.kind != "V":
            return

        if dtype in self.declared_dtypes:
            return

        import pyopencl.cltypes
        if dtype in pyopencl.cltypes.vec_type_to_scalar_and_count:
            return

        if hasattr(dtype, "subdtype") and dtype.subdtype is not None:
            self.add_dtype(dtype.subdtype[0])
            return

        for _name, field_data in sorted(dtype.fields.items()):
            field_dtype, offset = field_data[:2]
            self.add_dtype(field_dtype)

        _, cdecl = match_dtype_to_c_struct(
                self.device, dtype_to_ctype(dtype), dtype)

        self.declarations.append(cdecl)
        self.declared_dtypes.add(dtype)

    def visit_arguments(self, arguments):
        for arg in arguments:
            dtype = arg.dtype
            if dtype in (np.float64, np.complex128):
                self.saw_double = True

            if dtype.kind == "c":
                self.saw_complex = True

    def get_declarations(self):
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


@memoize
def match_dtype_to_c_struct(device, name, dtype, context=None):
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

    import pyopencl as cl

    fields = sorted(dtype.fields.items(),
            key=lambda name_dtype_offset: name_dtype_offset[1][1])

    c_fields = []
    for field_name, dtype_and_offset in fields:
        field_dtype, offset = dtype_and_offset[:2]
        if hasattr(field_dtype, "subdtype") and field_dtype.subdtype is not None:
            array_dtype = field_dtype.subdtype[0]
            if hasattr(array_dtype, "subdtype") and array_dtype.subdtype is not None:
                raise NotImplementedError("nested array dtypes are not supported")
            array_dims = field_dtype.subdtype[1]
            dims_str = ""
            try:
                for dim in array_dims:
                    dims_str += "[%d]" % dim
            except TypeError:
                dims_str = "[%d]" % array_dims
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
    for _field_name, dtype_and_offset in fields:
        field_dtype, offset = dtype_and_offset[:2]
        cdl.add_dtype(field_dtype)

    pre_decls = cdl.get_declarations()

    offset_code = "\n".join(
            "result[%d] = pycl_offsetof(%s, %s);" % (i+1, name, field_name)
            for i, (field_name, _) in enumerate(fields))

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

    import pyopencl.array  # noqa: F401
    result_buf = cl.array.empty(queue, 1+len(fields), np.uint32)
    knl(queue, (1,), (1,), result_buf.data)
    queue.finish()
    size_and_offsets = result_buf.get()

    size = int(size_and_offsets[0])

    offsets = size_and_offsets[1:]
    if any(ofs >= size for ofs in offsets):
        # offsets not plausible

        if dtype.itemsize == size:
            # If sizes match, use numpy's idea of the offsets.
            offsets = [dtype_and_offset[1]
                    for field_name, dtype_and_offset in fields]
        else:
            raise RuntimeError(
                    "OpenCL compiler reported offsetof() past sizeof() "
                    "for struct layout on '%s'. "
                    "This makes no sense, and it's usually indicates a "
                    "compiler bug. "
                    "Refusing to discover struct layout." % device)

    result_buf.data.release()
    del knl
    del prg
    del queue
    del context

    try:
        dtype_arg_dict = {
            "names": [field_name
                      for field_name, (field_dtype, offset) in fields],
            "formats": [field_dtype
                        for field_name, (field_dtype, offset) in fields],
            "offsets": [int(x) for x in offsets],
            "itemsize": int(size_and_offsets[0]),
            }
        dtype = np.dtype(dtype_arg_dict)
        if dtype.itemsize != size_and_offsets[0]:
            # "Old" versions of numpy (1.6.x?) silently ignore "itemsize". Boo.
            dtype_arg_dict["names"].append("_pycl_size_fixer")
            dtype_arg_dict["formats"].append(np.uint8)
            dtype_arg_dict["offsets"].append(int(size_and_offsets[0])-1)
            dtype = np.dtype(dtype_arg_dict)
    except NotImplementedError:
        def calc_field_type():
            total_size = 0
            padding_count = 0
            for offset, (field_name, (field_dtype, _)) in zip(offsets, fields):
                if offset > total_size:
                    padding_count += 1
                    yield ("__pycl_padding%d" % padding_count,
                           "V%d" % offset - total_size)
                yield field_name, field_dtype
                total_size = field_dtype.itemsize + offset
        dtype = np.dtype(list(calc_field_type()))

    assert dtype.itemsize == size_and_offsets[0]

    return dtype, c_decl


@memoize
def dtype_to_c_struct(device, dtype):
    if dtype.fields is None:
        return ""

    import pyopencl.cltypes
    if dtype in pyopencl.cltypes.vec_type_to_scalar_and_count:
        # Vector types are built-in. Don't try to redeclare those.
        return ""

    matched_dtype, c_decl = match_dtype_to_c_struct(
            device, dtype_to_ctype(dtype), dtype)

    def dtypes_match():
        result = len(dtype.fields) == len(matched_dtype.fields)

        for name, val in dtype.fields.items():
            result = result and matched_dtype.fields[name] == val

        return result

    assert dtypes_match()

    return c_decl


# {{{ code generation/templating helper

def _process_code_for_macro(code):
    code = code.replace("//CL//", "\n")

    if "//" in code:
        raise RuntimeError("end-of-line comments ('//') may not be used in "
                "code snippets")

    return code.replace("\n", " \\\n")


class _SimpleTextTemplate:
    def __init__(self, txt):
        self.txt = txt

    def render(self, context):
        return self.txt


class _PrintfTextTemplate:
    def __init__(self, txt):
        self.txt = txt

    def render(self, context):
        return self.txt % context


class _MakoTextTemplate:
    def __init__(self, txt):
        from mako.template import Template
        self.template = Template(txt, strict_undefined=True)

    def render(self, context):
        return self.template.render(**context)


class _ArgumentPlaceholder:
    """A placeholder for subclasses of :class:`DtypedArgument`. This is needed
    because the concrete dtype of the argument is not known at template
    creation time--it may be a type alias that will only be filled in
    at run time. These types take the place of these proto-arguments until
    all types are known.

    See also :class:`_TemplateRenderer.render_arg`.
    """

    def __init__(self, typename, name, **extra_kwargs):
        self.typename = typename
        self.name = name
        self.extra_kwargs = extra_kwargs


class _VectorArgPlaceholder(_ArgumentPlaceholder):
    target_class = VectorArg


class _ScalarArgPlaceholder(_ArgumentPlaceholder):
    target_class = ScalarArg


class _TemplateRenderer:
    def __init__(self, template, type_aliases, var_values, context=None,
            options=None):
        self.template = template
        self.type_aliases = dict(type_aliases)
        self.var_dict = dict(var_values)

        for name in self.var_dict:
            if name.startswith("macro_"):
                self.var_dict[name] = _process_code_for_macro(
                        self.var_dict[name])

        self.context = context
        self.options = options

    def __call__(self, txt):
        if txt is None:
            return txt

        result = self.template.get_text_template(txt).render(self.var_dict)

        return str(result)

    def get_rendered_kernel(self, txt, kernel_name):
        import pyopencl as cl
        prg = cl.Program(self.context, self(txt)).build(self.options)

        kernel_name_prefix = self.var_dict.get("kernel_name_prefix")
        if kernel_name_prefix is not None:
            kernel_name = kernel_name_prefix+kernel_name

        return getattr(prg, kernel_name)

    def parse_type(self, typename):
        if isinstance(typename, str):
            try:
                return self.type_aliases[typename]
            except KeyError:
                from pyopencl.compyte.dtypes import NAME_TO_DTYPE
                return NAME_TO_DTYPE[typename]
        else:
            return np.dtype(typename)

    def render_arg(self, arg_placeholder):
        return arg_placeholder.target_class(
                self.parse_type(arg_placeholder.typename),
                arg_placeholder.name,
                **arg_placeholder.extra_kwargs)

    _C_COMMENT_FINDER = re.compile(r"/\*.*?\*/")

    def render_argument_list(self, *arg_lists, **kwargs):
        with_offset = kwargs.pop("with_offset", False)
        if kwargs:
            raise TypeError("unrecognized kwargs: " + ", ".join(kwargs))

        all_args = []

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
            def vec_arg_factory(typename, name):
                return _VectorArgPlaceholder(typename, name, with_offset=True)
        else:
            vec_arg_factory = _VectorArgPlaceholder

        from pyopencl.compyte.dtypes import parse_c_arg_backend
        parsed_args = []
        for arg in all_args:
            if isinstance(arg, str):
                arg = arg.strip()
                if not arg:
                    continue

                ph = parse_c_arg_backend(arg,
                        _ScalarArgPlaceholder, vec_arg_factory,
                        name_to_dtype=lambda x: x)
                parsed_arg = self.render_arg(ph)

            elif isinstance(arg, Argument):
                parsed_arg = arg
            elif isinstance(arg, tuple):
                parsed_arg = ScalarArg(self.parse_type(arg[0]), arg[1])

            parsed_args.append(parsed_arg)

        return parsed_args

    def get_type_decl_preamble(self, device, decl_type_names, arguments=None):
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


class KernelTemplateBase:
    def __init__(self, template_processor=None):
        self.template_processor = template_processor

        self.build_cache = {}
        _first_arg_dependent_caches.append(self.build_cache)

    def get_preamble(self):
        pass

    _TEMPLATE_PROCESSOR_PATTERN = re.compile(r"^//CL(?::([a-zA-Z0-9_]+))?//")

    @memoize_method
    def get_text_template(self, txt):
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
            raise RuntimeError(
                    "unknown template processor '%s'" % proc_match.group(1))

    def get_renderer(self, type_aliases, var_values, context=None, options=None):
        return _TemplateRenderer(self, type_aliases, var_values)

    def build_inner(self, context, *args, **kwargs):
        raise NotImplementedError

    def build(self, context, *args, **kwargs):
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

class _CLFakeArrayModule:
    def __init__(self, queue):
        self.queue = queue

    @property
    def ndarray(self):
        from pyopencl.array import Array
        return Array

    def dot(self, x, y):
        from pyopencl.array import dot
        return dot(x, y, queue=self.queue).get()

    def vdot(self, x, y):
        from pyopencl.array import vdot
        return vdot(x, y, queue=self.queue).get()

    def empty(self, shape, dtype, order="C"):
        from pyopencl.array import empty
        return empty(self.queue, shape, dtype, order=order)

    def hstack(self, arrays):
        from pyopencl.array import hstack
        return hstack(arrays, self.queue)


def array_module(a):
    if isinstance(a, np.ndarray):
        return np
    else:
        from pyopencl.array import Array
        if isinstance(a, Array):
            return _CLFakeArrayModule(a.queue)
        else:
            raise TypeError("array type not understood: %s" % type(a))

# }}}


def is_spirv(s):
    spirv_magic = b"\x07\x23\x02\x03"
    return (
            isinstance(s, bytes)
            and (
                s[:4] == spirv_magic
                or s[:4] == spirv_magic[::-1]))


# {{{ numpy key types builder

class _NumpyTypesKeyBuilder(KeyBuilderBase):
    def update_for_VectorArg(self, key_hash, key):  # noqa: N802
        self.rec(key_hash, key.dtype)
        self.update_for_str(key_hash, key.name)
        self.rec(key_hash, key.with_offset)

    def update_for_type(self, key_hash, key):
        if issubclass(key, np.generic):
            self.update_for_str(key_hash, key.__name__)
            return

        raise TypeError("unsupported type for persistent hash keying: %s"
                % type(key))

# }}}

# vim: foldmethod=marker
