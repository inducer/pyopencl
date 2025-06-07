from __future__ import annotations


__copyright__ = "Copyright (C) 2025 University of Illinois Board of Trustees"

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
import inspect as _inspect
from sys import intern
from typing import (
    TYPE_CHECKING,
    Any,
    TextIO,
    TypeVar,
    cast,
)
from warnings import warn

import numpy as np

import pyopencl._cl as _cl


if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from numpy.typing import NDArray

    from pyopencl import SVMMap
    from pyopencl.typing import HasBufferInterface, KernelArg, SVMInnerT, WaitList


CONSTANT_CLASSES = tuple(
        getattr(_cl, name) for name in dir(_cl)
        if _inspect.isclass(getattr(_cl, name))
        and name[0].islower() and name not in ["zip", "map", "range"])


BITFIELD_CONSTANT_CLASSES = (
        _cl.device_type,
        _cl.device_fp_config,
        _cl.device_exec_capabilities,
        _cl.command_queue_properties,
        _cl.mem_flags,
        _cl.map_flags,
        _cl.kernel_arg_type_qualifier,
        _cl.device_affinity_domain,
        _cl.mem_migration_flags,
        _cl.device_svm_capabilities,
        _cl.queue_properties,
        _cl.svm_mem_flags,
        _cl.device_atomic_capabilities,
        _cl.device_device_enqueue_capabilities,
        _cl.version_bits,
        )


def generic_get_cl_version(self: _cl.Platform):
    import re
    version_string = self.version
    match = re.match(r"^OpenCL ([0-9]+)\.([0-9]+) .*$", version_string)
    if match is None:
        raise RuntimeError("%s %s returned non-conformant "
                           "platform version string '%s'" %
                           (type(self).__name__, self, version_string))

    return int(match.group(1)), int(match.group(2))


def platform_repr(self: _cl.Platform):
    return f"<pyopencl.Platform '{self.name}' at 0x{self.int_ptr:x}>"


def device_repr(self: _cl.Device):
    return "<pyopencl.Device '{}' on '{}' at 0x{:x}>".format(
            self.name.strip(), self.platform.name.strip(), self.int_ptr)


def device_hashable_model_and_version_identifier(self: _cl.Device):
    return ("v1", self.vendor, self.vendor_id, self.name, self.version)


def device_persistent_unique_id(self: _cl.Device):
    warn("Device.persistent_unique_id is deprecated. "
            "Use Device.hashable_model_and_version_identifier instead.",
            DeprecationWarning, stacklevel=2)
    return device_hashable_model_and_version_identifier(self)


def context_repr(self: _cl.Context):
    return "<pyopencl.Context at 0x{:x} on {}>".format(self.int_ptr,
            ", ".join(repr(dev) for dev in self.devices))


def context_get_cl_version(self: _cl.Context):
    return self.devices[0].platform._get_cl_version()


def command_queue_enter(self: _cl.CommandQueue):
    return self


def command_queue_exit(self: _cl.CommandQueue, exc_type, exc_val, exc_tb):
    self.finish()
    self._finalize()


def command_queue_get_cl_version(self: _cl.CommandQueue):
    return self.device._get_cl_version()


def program_get_build_logs(self: _cl._Program):
    build_logs = []
    for dev in self.get_info(_cl.program_info.DEVICES):
        try:
            log = self.get_build_info(dev, _cl.program_build_info.LOG)
        except Exception:
            log = "<error retrieving log>"

        build_logs.append((dev, log))

    return build_logs


def program_build(
            self: _cl._Program,
            options_bytes: bytes,
            devices: Sequence[_cl.Device] | None = None
        ) -> _cl._Program:
    err = None
    try:
        self._build(options=options_bytes, devices=devices)
    except _cl.Error as e:
        msg = str(e) + "\n\n" + (75*"="+"\n").join(
                f"Build on {dev}:\n\n{log}"
                for dev, log in self._get_build_logs())
        code = e.code
        routine = e.routine

        err = _cl.RuntimeError(
                _cl._ErrorRecord(
                    msg=msg,
                    code=code,
                    routine=routine))

    if err is not None:
        # Python 3.2 outputs the whole list of currently active exceptions
        # This serves to remove one (redundant) level from that nesting.
        raise err

    message = (75*"="+"\n").join(
            f"Build on {dev} succeeded, but said:\n\n{log}"
            for dev, log in self._get_build_logs()
            if log is not None and log.strip())

    if message:
        if self.kind() == _cl.program_kind.SOURCE:
            build_type = "From-source build"
        elif self.kind() == _cl.program_kind.BINARY:
            build_type = "From-binary build"
        elif self.kind() == _cl.program_kind.IL:
            build_type = "From-IL build"
        else:
            build_type = "Build"

        from pyopencl import compiler_output
        compiler_output("%s succeeded, but resulted in non-empty logs:\n%s"
                % (build_type, message))

    return self


class ProfilingInfoGetter:
    event: _cl.Event

    def __init__(self, event: _cl.Event):
        self.event = event

    def __getattr__(self, name: str):
        info_cls = _cl.profiling_info

        try:
            inf_attr = getattr(info_cls, name.upper())
        except AttributeError as err:
            raise AttributeError("%s has no attribute '%s'"
                    % (type(self), name)) from err
        else:
            return self.event.get_profiling_info(inf_attr)

    QUEUED: int  # pyright: ignore[reportUninitializedInstanceVariable]
    SUBMIT: int  # pyright: ignore[reportUninitializedInstanceVariable]
    START: int  # pyright: ignore[reportUninitializedInstanceVariable]
    END: int  # pyright: ignore[reportUninitializedInstanceVariable]
    COMPLETE: int  # pyright: ignore[reportUninitializedInstanceVariable]


kernel_old_get_info = _cl.Kernel.get_info
kernel_old_get_work_group_info = _cl.Kernel.get_work_group_info


def kernel_set_arg_types(self: _cl.Kernel, arg_types):
    arg_types = tuple(arg_types)

    # {{{ arg counting bug handling

    # For example:
    # https://github.com/pocl/pocl/issues/197
    # (but Apple CPU has a similar bug)

    work_around_arg_count_bug = False
    warn_about_arg_count_bug = False

    from pyopencl.characterize import has_struct_arg_count_bug

    count_bug_per_dev = [
            has_struct_arg_count_bug(dev, self.context)
            for dev in self.context.devices]

    from pytools import single_valued
    if any(count_bug_per_dev):
        if all(count_bug_per_dev):
            work_around_arg_count_bug = single_valued(count_bug_per_dev)
        else:
            warn_about_arg_count_bug = True

    # }}}

    from pyopencl.invoker import generate_enqueue_and_set_args
    self._set_enqueue_and_set_args(
            *generate_enqueue_and_set_args(
                    self.function_name,
                    len(arg_types), self.num_args,
                    arg_types,
                    warn_about_arg_count_bug=warn_about_arg_count_bug,
                    work_around_arg_count_bug=work_around_arg_count_bug,
                    devs=self.context.devices))


def kernel_get_work_group_info(self: _cl.Kernel, param: int, device: _cl.Device):
    try:
        wg_info_cache = self._wg_info_cache
    except AttributeError:
        wg_info_cache = self._wg_info_cache = {}

    cache_key = (param, device.int_ptr)
    try:
        return wg_info_cache[cache_key]
    except KeyError:
        pass

    result = kernel_old_get_work_group_info(self, param, device)
    wg_info_cache[cache_key] = result
    return result


def kernel_capture_call(
            self: _cl.Kernel,
            output_file: str | TextIO,
            queue: _cl.CommandQueue,
            global_size: tuple[int, ...],
            local_size: tuple[int, ...] | None,
            *args: KernelArg,
            wait_for: WaitList = None,
            g_times_l: bool = False,
            allow_empty_ndrange: bool = False,
            global_offset: tuple[int, ...] | None = None,
         ) -> None:
    from pyopencl.capture_call import capture_kernel_call
    capture_kernel_call(self, output_file, queue, global_size, local_size,
            *args,
            wait_for=wait_for,
            g_times_l=g_times_l,
            allow_empty_ndrange=allow_empty_ndrange,
            global_offset=global_offset)


def kernel_get_info(self: _cl.Kernel, param_name: _cl.kernel_info) -> object:
    val = kernel_old_get_info(self, param_name)

    if isinstance(val, _cl._Program):
        from pyopencl import Program
        return Program(val)
    else:
        return val


def image_format_repr(self: _cl.ImageFormat) -> str:
    return "ImageFormat({}, {})".format(
            _cl.channel_order.to_string(self.channel_order,
                "<unknown channel order 0x%x>"),
            _cl.channel_type.to_string(self.channel_data_type,
                "<unknown channel data type 0x%x>"))


def image_format_eq(self: _cl.ImageFormat, other: object):
    return (isinstance(other, _cl.ImageFormat)
            and self.channel_order == other.channel_order
            and self.channel_data_type == other.channel_data_type)


def image_format_ne(self: _cl.ImageFormat, other: object):
    return not image_format_eq(self, other)


def image_format_hash(self: _cl.ImageFormat) -> int:
    return hash((type(self), self.channel_order, self.channel_data_type))


def image_init(self: _cl.Image,
            context: _cl.Context,
            flags: _cl.mem_flags,
            format: _cl.ImageFormat,
            shape: tuple[int, ...] | None = None,
            pitches: tuple[int, ...] | None = None,

            hostbuf: HasBufferInterface | None = None,
            is_array: bool = False,
            buffer: _cl.Buffer | None = None,
            *,
            desc: _cl.ImageDescriptor | None = None,
            _through_create_image: bool = False,
        ) -> None:
    if hostbuf is not None and not \
            (flags & (_cl.mem_flags.USE_HOST_PTR | _cl.mem_flags.COPY_HOST_PTR)):
        warn("'hostbuf' was passed, but no memory flags to make use of it.",
             stacklevel=2)

    if desc is not None:
        if shape is not None:
            raise TypeError("shape may not be passed when using descriptor")
        if pitches is not None:
            raise TypeError("pitches may not be passed when using descriptor")
        if is_array:
            raise TypeError("is_array may not be passed when using descriptor")
        if buffer is not None:
            raise TypeError("is_array may not be passed when using descriptor")

        _cl.Image._custom_init(self, context, flags, format, desc, hostbuf)

        return

    if shape is None and hostbuf is None:
        raise _cl.Error("'shape' must be passed if 'hostbuf' is not given")

    if shape is None and hostbuf is not None:
        shape = hostbuf.shape

    if hostbuf is None and pitches is not None:
        raise _cl.Error("'pitches' may only be given if 'hostbuf' is given")

    if context._get_cl_version() >= (1, 2) and _cl.get_cl_header_version() >= (1, 2):
        if not _through_create_image:
            warn("Non-descriptor Image constructor called. "
                 "This will stop working in 2026. "
                 "Use create_image instead (with the same arguments).",
                 DeprecationWarning, stacklevel=2)

        if buffer is not None and is_array:
            raise ValueError(
                    "'buffer' and 'is_array' are mutually exclusive")

        if len(shape) == 3:
            if buffer is not None:
                raise TypeError(
                        "'buffer' argument is not supported for 3D arrays")
            elif is_array:
                image_type = _cl.mem_object_type.IMAGE2D_ARRAY
            else:
                image_type = _cl.mem_object_type.IMAGE3D

        elif len(shape) == 2:
            if buffer is not None:
                raise TypeError(
                        "'buffer' argument is not supported for 2D arrays")
            elif is_array:
                image_type = _cl.mem_object_type.IMAGE1D_ARRAY
            else:
                image_type = _cl.mem_object_type.IMAGE2D

        elif len(shape) == 1:
            if buffer is not None:
                image_type = _cl.mem_object_type.IMAGE1D_BUFFER
            elif is_array:
                raise TypeError("array of zero-dimensional images not supported")
            else:
                image_type = _cl.mem_object_type.IMAGE1D

        else:
            raise ValueError("images cannot have more than three dimensions")

        desc = _cl.ImageDescriptor() \
            # pylint: disable=possibly-used-before-assignment

        desc.image_type = image_type
        desc.shape = shape  # also sets desc.array_size

        if pitches is None:
            desc.pitches = (0, 0)
        else:
            desc.pitches = pitches

        desc.num_mip_levels = 0  # per CL 1.2 spec
        desc.num_samples = 0  # per CL 1.2 spec
        desc.buffer = buffer

        _cl.Image._custom_init(self, context, flags, format, desc, hostbuf)
    else:
        # legacy init for CL 1.1 and older
        if is_array:
            raise TypeError("'is_array=True' is not supported for CL < 1.2")
        # if num_mip_levels is not None:
            # raise TypeError(
            #       "'num_mip_levels' argument is not supported for CL < 1.2")
        # if num_samples is not None:
            # raise TypeError(
            #        "'num_samples' argument is not supported for CL < 1.2")
        if buffer is not None:
            raise TypeError("'buffer' argument is not supported for CL < 1.2")

        _cl.Image._custom_init(self, context, flags, format, shape,
                pitches, hostbuf)


def image_shape(self: _cl.Image) -> tuple[int, int] | tuple[int, int, int]:
    if self.type == _cl.mem_object_type.IMAGE2D:
        return (self.width, self.height)
    elif self.type == _cl.mem_object_type.IMAGE3D:
        return (self.width, self.height, self.depth)
    else:
        raise _cl.LogicError("only images have shapes")


def error_str(self: _cl.Error) -> str:
    val = self.what
    try:
        val.routine  # noqa: B018
    except AttributeError:
        return str(val)
    else:
        result = ""
        if val.code() != _cl.status_code.SUCCESS:
            result = _cl.status_code.to_string(
                    val.code(), "<unknown error %d>")
        routine = val.routine()
        if routine:
            result = f"{routine} failed: {result}"
        what = val.what()
        if what:
            if result:
                result += " - "
            result += what
        return result


def error_code(self: _cl.Error) -> int:
    return cast("_cl._ErrorRecord", self.args[0]).code()


def error_routine(self: _cl.Error) -> str:
    return cast("_cl._ErrorRecord", self.args[0]).routine()


def error_what(self: _cl.Error) -> _cl._ErrorRecord:
    return cast("_cl._ErrorRecord", self.args[0])


def memory_map_enter(self: _cl.MemoryMap):
    return self


def memory_map_exit(self: _cl.MemoryMap, exc_type, exc_val, exc_tb):
    self.release()


def svmptr_map(
            self: _cl.SVMPointer,
            queue: _cl.CommandQueue,
            *,
            flags: int,
            is_blocking: bool = True,
            wait_for: WaitList = None,
            size: int | None = None
        ) -> SVMMap[NDArray[Any]]:
    """
    :arg is_blocking: If *False*, subsequent code must wait on
        :attr:`SVMMap.event` in the returned object before accessing the
        mapped memory.
    :arg flags: a combination of :class:`pyopencl.map_flags`.
    :arg size: The size of the map in bytes. If not provided, defaults to
        :attr:`size`.

    |std-enqueue-blurb|
    """
    from pyopencl import SVMMap
    return SVMMap(self,
            np.asarray(self.buf),
            queue,
            _cl._enqueue_svm_map(queue, is_blocking, flags, self, wait_for,
                                size=size))


def svmptr_map_ro(
            self: _cl.SVMPointer,
            queue: _cl.CommandQueue,
            *,
            is_blocking: bool = True,
            wait_for: WaitList = None,
            size: int | None = None
        ) -> SVMMap[NDArray[Any]]:
    """Like :meth:`map`, but with *flags* set for a read-only map.
    """

    return self.map(queue, flags=_cl.map_flags.READ,
            is_blocking=is_blocking, wait_for=wait_for, size=size)


def svmptr_map_rw(
            self: _cl.SVMPointer,
            queue: _cl.CommandQueue,
            *,
            is_blocking: bool = True,
            wait_for: WaitList = None,
            size: int | None = None
        ) -> SVMMap[NDArray[Any]]:
    """Like :meth:`map`, but with *flags* set for a read-only map.
    """

    return self.map(queue, flags=_cl.map_flags.READ | _cl.map_flags.WRITE,
            is_blocking=is_blocking, wait_for=wait_for, size=size)


def svmptr__enqueue_unmap(
            self: _cl.SVMPointer,
            queue: _cl.CommandQueue,
            wait_for: WaitList = None
        ) -> _cl.Event:
    return _cl._enqueue_svm_unmap(queue, self, wait_for)


def svmptr_as_buffer(
            self: _cl.SVMPointer,
            ctx: _cl.Context,
            *,
            flags: int | None = None,
            size: int | None = None
        ) -> _cl.Buffer:
    """
    :arg ctx: a :class:`Context`
    :arg flags: a combination of :class:`pyopencl.map_flags`, defaults to
        read-write.
    :arg size: The size of the map in bytes. If not provided, defaults to
        :attr:`size`.
    :returns: a :class:`Buffer` corresponding to *self*.

    The memory referred to by this object must not be freed before
    the returned :class:`Buffer` is released.
    """

    if flags is None:
        flags = _cl.mem_flags.READ_WRITE | _cl.mem_flags.USE_HOST_PTR

    if size is None:
        size = self.size

    assert self.buf is not None
    return _cl.Buffer(ctx, flags, size=size, hostbuf=self.buf)


def svm_map(
            self: _cl.SVM[SVMInnerT],
            queue: _cl.CommandQueue,
            flags: int,
            is_blocking: bool = True,
            wait_for: WaitList = None
        ) -> SVMMap[SVMInnerT]:

    """
    :arg is_blocking: If *False*, subsequent code must wait on
        :attr:`SVMMap.event` in the returned object before accessing the
        mapped memory.
    :arg flags: a combination of :class:`pyopencl.map_flags`.
    :returns: an :class:`SVMMap` instance

    This differs from the inherited :class:`SVMPointer.map` in that no size
    can be specified, and that :attr:`mem` is the exact array produced
    when the :class:`SVMMap` is used as a context manager.

    |std-enqueue-blurb|
    """
    from pyopencl import SVMMap
    return SVMMap(
            self,
            self.mem,
            queue,
            _cl._enqueue_svm_map(queue, is_blocking, flags, self, wait_for))


def svm_map_ro(
            self: _cl.SVM[SVMInnerT],
            queue: _cl.CommandQueue,
            is_blocking: bool = True,
            wait_for: WaitList = None,
        ) -> SVMMap[SVMInnerT]:
    """Like :meth:`map`, but with *flags* set for a read-only map."""

    return self.map(queue, _cl.map_flags.READ,
            is_blocking=is_blocking, wait_for=wait_for)


def svm_map_rw(
            self: _cl.SVM[SVMInnerT],
            queue: _cl.CommandQueue,
            is_blocking: bool = True,
            wait_for: WaitList = None,
        ) -> SVMMap[SVMInnerT]:
    """Like :meth:`map`, but with *flags* set for a read-only map."""

    return self.map(queue, _cl.map_flags.READ | _cl.map_flags.WRITE,
            is_blocking=is_blocking, wait_for=wait_for)


def svm__enqueue_unmap(
            self: _cl.SVM[SVMInnerT],
            queue: _cl.CommandQueue
            ,
            wait_for: WaitList = None
        ) -> _cl.Event:
    return _cl._enqueue_svm_unmap(queue, self, wait_for)


def to_string(
            cls: type,
            value: int,
            default_format: str | None = None
        ) -> str:
    if cls._is_bitfield:
        names: list[str] = []
        for name in dir(cls):
            attr = cast("int", getattr(cls, name))
            if not isinstance(attr, int):
                continue
            if attr == value or attr & value:
                names.append(name)
        if names:
            return " | ".join(names)
    else:
        for name in dir(cls):
            if (not name.startswith("_")
                    and getattr(cls, name) == value):
                return name

    if default_format is None:
        raise ValueError("a name for value %d was not found in %s"
                % (value, cls.__name__))
    else:
        return default_format % value


def _add_functionality():
    # {{{ Platform

    _cl.Platform.__repr__ = platform_repr
    _cl.Platform._get_cl_version = generic_get_cl_version

    # }}}

    # {{{ Device

    _cl.Device.__repr__ = device_repr

    # undocumented for now:
    _cl.Device._get_cl_version = generic_get_cl_version
    _cl.Device.hashable_model_and_version_identifier = property(
            device_hashable_model_and_version_identifier)
    _cl.Device.persistent_unique_id = property(device_persistent_unique_id)

    # }}}

    # {{{ Context

    _cl.Context.__repr__ = context_repr
    from pytools import memoize_method
    _cl.Context._get_cl_version = memoize_method(context_get_cl_version)

    # }}}

    # {{{ CommandQueue

    _cl.CommandQueue.__enter__ = command_queue_enter
    _cl.CommandQueue.__exit__ = command_queue_exit
    _cl.CommandQueue._get_cl_version = memoize_method(command_queue_get_cl_version)

    # }}}

    # {{{ _Program (the internal, non-caching version)

    _cl._Program._get_build_logs = program_get_build_logs
    _cl._Program.build = program_build

    # }}}

    # {{{ Event

    _cl.Event.profile = property(ProfilingInfoGetter)

    # }}}

    # {{{ Kernel

    _cl.Kernel.get_work_group_info = kernel_get_work_group_info

    # FIXME: Possibly deprecate this version
    _cl.Kernel.set_scalar_arg_dtypes = kernel_set_arg_types
    _cl.Kernel.set_arg_types = kernel_set_arg_types

    _cl.Kernel.capture_call = kernel_capture_call
    _cl.Kernel.get_info = kernel_get_info

    # }}}

    # {{{ ImageFormat

    _cl.ImageFormat.__repr__ = image_format_repr
    _cl.ImageFormat.__eq__ = image_format_eq
    _cl.ImageFormat.__ne__ = image_format_ne
    _cl.ImageFormat.__hash__ = image_format_hash

    # }}}

    # {{{ Image

    _cl.Image.__init__ = image_init
    _cl.Image.shape = property(image_shape)

    # }}}

    # {{{ Error

    _cl.Error.__str__ = error_str
    _cl.Error.code = property(error_code)
    _cl.Error.routine = property(error_routine)
    _cl.Error.what = property(error_what)

    # }}}

    # {{{ MemoryMap

    _cl.MemoryMap.__doc__ = """
        This class may also be used as a context manager in a ``with`` statement.
        The memory corresponding to this object will be unmapped when
        this object is deleted or :meth:`release` is called.

        .. automethod:: release
        """
    _cl.MemoryMap.__enter__ = memory_map_enter
    _cl.MemoryMap.__exit__ = memory_map_exit

    # }}}

    # {{{ SVMPointer

    if _cl.get_cl_header_version() >= (2, 0):
        _cl.SVMPointer.__doc__ = """A base class for things that can be passed to
            functions that allow an SVM pointer, e.g. kernel enqueues and memory
            copies.

            Objects of this type cannot currently be directly created or
            implemented in Python.  To obtain objects implementing this type,
            consider its subtypes :class:`SVMAllocation` and :class:`SVM`.


            .. property:: svm_ptr

                Gives the SVM pointer as an :class:`int`.

            .. property:: size

                An :class:`int` denoting the size in bytes, or *None*, if the size
                of the SVM pointed to is not known.

                *Most* objects of this type (e.g. instances of
                :class:`SVMAllocation` and :class:`SVM` know their size, so that,
                for example :class:`enqueue_copy` will automatically copy an entire
                :class:`SVMAllocation` when a size is not explicitly specified.

            .. automethod:: map
            .. automethod:: map_ro
            .. automethod:: map_rw
            .. automethod:: as_buffer
            .. property:: buf

                An opaque object implementing the :c:func:`Python buffer protocol
                <PyObject_GetBuffer>`. It exposes the pointed-to memory as
                a one-dimensional buffer of bytes, with the size matching
                :attr:`size`.

                No guarantee is provided that two references to this attribute
                result in the same object.
            """

    if _cl.get_cl_header_version() >= (2, 0):
        _cl.SVMPointer.map = svmptr_map
        _cl.SVMPointer.map_ro = svmptr_map_ro
        _cl.SVMPointer.map_rw = svmptr_map_rw
        _cl.SVMPointer._enqueue_unmap = svmptr__enqueue_unmap
        _cl.SVMPointer.as_buffer = svmptr_as_buffer

    # }}}

    # {{{ SVMAllocation

    if _cl.get_cl_header_version() >= (2, 0):
        _cl.SVMAllocation.__doc__ = """
            Is a :class:`SVMPointer`.

            .. versionadded:: 2016.2

            .. automethod:: __init__

                :arg flags: See :class:`svm_mem_flags`.
                :arg queue: If not specified, the allocation will be freed
                    eagerly, irrespective of whether pending/enqueued operations
                    are still using this memory.

                    If specified, deallocation of the memory will be enqueued
                    with the given queue, and will only be performed
                    after previously-enqueue operations in the queue have
                    completed.

                    It is an error to specify an out-of-order queue.

                    .. warning::

                        Not specifying a queue will typically lead to undesired
                        behavior, including crashes and memory corruption.
                        See the warning in :ref:`svm`.

            .. automethod:: enqueue_release

                Enqueue the release of this allocation into *queue*.
                If *queue* is not specified, enqueue the deallocation
                into the queue provided at allocation time or via
                :class:`bind_to_queue`.

            .. automethod:: bind_to_queue

                Change the queue used for implicit enqueue of deallocation
                to *queue*. Sufficient synchronization is ensured by
                enqueuing a marker into the old queue and waiting on this
                marker in the new queue.

            .. automethod:: unbind_from_queue

                Configure the allocation to no longer implicitly enqueue
                memory allocation. If such a queue was previously provided,
                :meth:`~CommandQueue.finish` is automatically called on it.
            """

    # }}}

    # {{{ SVM

    if _cl.get_cl_header_version() >= (2, 0):
        _cl.SVM.__doc__ = """Tags an object exhibiting the Python buffer interface
            (such as a :class:`numpy.ndarray`) as referring to shared virtual
            memory.

            Is a :class:`SVMPointer`, hence objects of this type may be passed
            to kernel calls and :func:`enqueue_copy`, and all methods declared
            there are also available there. Note that :meth:`map` differs
            slightly from :meth:`SVMPointer.map`.

            Depending on the features of the OpenCL implementation, the following
            types of objects may be passed to/wrapped in this type:

            *   fine-grain shared memory as returned by (e.g.) :func:`fsvm_empty`,
                if the implementation supports fine-grained shared virtual memory.
                This memory may directly be passed to a kernel::

                    ary = cl.fsvm_empty(ctx, 1000, np.float32)
                    assert isinstance(ary, np.ndarray)

                    prg.twice(queue, ary.shape, None, cl.SVM(ary))
                    queue.finish() # synchronize
                    print(ary) # access from host

                Observe how mapping (as needed in coarse-grain SVM) is no longer
                necessary.

            *   any :class:`numpy.ndarray` (or other Python object with a buffer
                interface) if the implementation supports fine-grained *system*
                shared virtual memory.

                This is how plain :mod:`numpy` arrays may directly be passed to a
                kernel::

                    ary = np.zeros(1000, np.float32)
                    prg.twice(queue, ary.shape, None, cl.SVM(ary))
                    queue.finish() # synchronize
                    print(ary) # access from host

            *   coarse-grain shared memory as returned by (e.g.) :func:`csvm_empty`
                for any implementation of OpenCL 2.0.

                .. note::

                    Applications making use of coarse-grain SVM may be better
                    served by opaque-style SVM. See :ref:`opaque-svm`.

                This is how coarse-grain SVM may be used from both host and device::

                    svm_ary = cl.SVM(
                        cl.csvm_empty(ctx, 1000, np.float32, alignment=64))
                    assert isinstance(svm_ary.mem, np.ndarray)

                    with svm_ary.map_rw(queue) as ary:
                        ary.fill(17)  # use from host

                    prg.twice(queue, svm_ary.mem.shape, None, svm_ary)

            Coarse-grain shared-memory *must* be mapped into host address space
            using :meth:`~SVMPointer.map` before being accessed through the
            :mod:`numpy` interface.

            .. note::

                This object merely serves as a 'tag' that changes the behavior
                of functions to which it is passed. It has no special management
                relationship to the memory it tags. For example, it is permissible
                to grab a :class:`numpy.ndarray` out of :attr:`SVM.mem` of one
                :class:`SVM` instance and use the array to construct another.
                Neither of the tags need to be kept alive.

            .. versionadded:: 2016.2

            .. attribute:: mem

                The wrapped object.

            .. automethod:: __init__
            .. automethod:: map
            .. automethod:: map_ro
            .. automethod:: map_rw
            """

    # }}}

    if _cl.get_cl_header_version() >= (2, 0):
        _cl.SVM.map = svm_map
        _cl.SVM.map_ro = svm_map_ro
        _cl.SVM.map_rw = svm_map_rw
        _cl.SVM._enqueue_unmap = svm__enqueue_unmap

    # }}}

    for cls in CONSTANT_CLASSES:
        cls._is_bitfield = cls in BITFIELD_CONSTANT_CLASSES
        cls.to_string = classmethod(to_string)


_add_functionality()


# ORDER DEPENDENCY: Some of the above may override get_info, the effect needs
# to be visible through the attributes. So get_info attr creation needs to happen
# after the overriding is complete.

T = TypeVar("T")


InfoT = TypeVar("InfoT")


def make_getinfo(
            info_method: Callable[[T, InfoT], object],
            info_constant: InfoT
        ) -> property:
    def result(self: T) -> object:
        return info_method(self, info_constant)

    return property(result)


def make_cacheable_getinfo(
            info_method: Callable[[T, InfoT], object],
            cache_attr: str,
            info_constant: InfoT
        ) -> property:
    def result(self: T):
        try:
            return getattr(self, cache_attr)
        except AttributeError:
            pass

        result = info_method(self, info_constant)
        setattr(self, cache_attr, result)
        return result

    return property(result)


def add_get_info(
            cls: type[T],
            info_method: Callable[[T, InfoT], object],
            info_class: type[InfoT],
            cacheable_attrs: Collection[str] = (),
        ) -> None:
    for info_name, _info_value in info_class.__dict__.items():
        if info_name == "to_string" or info_name.startswith("_"):
            continue

        info_lower = info_name.lower()
        info_constant = cast("InfoT", getattr(info_class, info_name))
        if info_name in cacheable_attrs:
            cache_attr = intern("_info_cache_"+info_lower)
            setattr(cls, info_lower, make_cacheable_getinfo(
                info_method, cache_attr, info_constant))
        else:
            setattr(cls, info_lower, make_getinfo(info_method, info_constant))

    # }}}

    if _cl.have_gl():
        def gl_object_get_gl_object(self):
            return self.get_gl_object_info()[1]

        _cl.GLBuffer.gl_object = property(gl_object_get_gl_object)
        _cl.GLTexture.gl_object = property(gl_object_get_gl_object)


def _add_all_get_info():
    add_get_info(_cl.Platform, _cl.Platform.get_info, _cl.platform_info)
    add_get_info(_cl.Device, _cl.Device.get_info, _cl.device_info,
                 ["PLATFORM", "MAX_WORK_GROUP_SIZE", "MAX_COMPUTE_UNITS"])
    add_get_info(_cl.Context, _cl.Context.get_info, _cl.context_info)
    add_get_info(_cl.CommandQueue, _cl.CommandQueue.get_info, _cl.command_queue_info,
                 ["CONTEXT", "DEVICE"])
    add_get_info(_cl.Event, _cl.Event.get_info, _cl.event_info)
    add_get_info(_cl.MemoryObjectHolder, _cl.MemoryObjectHolder.get_info, _cl.mem_info)
    add_get_info(_cl.Image, _cl.Image.get_image_info, _cl.image_info)
    add_get_info(_cl.Pipe, _cl.Pipe.get_pipe_info, _cl.pipe_info)
    add_get_info(_cl.Kernel, _cl.Kernel.get_info, _cl.kernel_info)
    add_get_info(_cl.Sampler, _cl.Sampler.get_info, _cl.sampler_info)


_add_all_get_info()
