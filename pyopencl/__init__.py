from __future__ import annotations


__copyright__ = "Copyright (C) 2009-15 Andreas Kloeckner"

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

from typing_extensions import override
from dataclasses import dataclass
import logging
from typing import (
    TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar, cast,
    overload)
from collections.abc import Callable
from collections.abc import Sequence
from warnings import warn

# must import, otherwise dtype registry will not be fully populated
import pyopencl.cltypes
from pyopencl.version import VERSION, VERSION_STATUS, VERSION_TEXT


__version__ = VERSION_TEXT

logger = logging.getLogger(__name__)

# This tells ocl-icd where to find shipped OpenCL ICDs, cf.
# https://github.com/isuruf/ocl-icd/commit/3862386b51930f95d9ad1089f7157a98165d5a6b
# via
# https://github.com/inducer/pyopencl/blob/0b3d0ef92497e6838eea300b974f385f94cb5100/scripts/build-wheels.sh#L43-L44
import os
os.environ["PYOPENCL_HOME"] = os.path.dirname(os.path.abspath(__file__))

try:
    import pyopencl._cl as _cl
except ImportError:
    from os.path import dirname, join, realpath
    if realpath(join(os.getcwd(), "pyopencl")) == realpath(dirname(__file__)):
        warn(
            "It looks like you are importing PyOpenCL from "
            "its source directory. This likely won't work.",
            stacklevel=2)
    raise

import numpy as np

import sys

_PYPY = "__pypy__" in sys.builtin_module_names

from pyopencl.typing import (
        DTypeT,
        HasBufferInterface,
        SVMInnerT,
        WaitList,
        )
from pyopencl._cl import (  # noqa: F401
        get_cl_header_version,
        program_kind,
        status_code,
        platform_info,
        device_type,
        device_info,
        device_topology_type_amd,
        device_fp_config,
        device_mem_cache_type,
        device_local_mem_type,
        device_exec_capabilities,
        device_svm_capabilities,

        command_queue_properties,
        context_info,
        gl_context_info,
        context_properties,
        command_queue_info,
        queue_properties,

        mem_flags,
        svm_mem_flags,

        channel_order,
        channel_type,
        mem_object_type,
        mem_info,
        image_info,
        pipe_info,
        pipe_properties,
        addressing_mode,
        filter_mode,
        sampler_info,
        sampler_properties,
        map_flags,
        program_info,
        program_build_info,
        program_binary_type,

        kernel_info,
        kernel_arg_info,
        kernel_arg_address_qualifier,
        kernel_arg_access_qualifier,
        kernel_arg_type_qualifier,
        kernel_work_group_info,
        kernel_sub_group_info,

        event_info,
        command_type,
        command_execution_status,
        profiling_info,
        mem_migration_flags,
        device_partition_property,
        device_affinity_domain,
        device_atomic_capabilities,
        device_device_enqueue_capabilities,

        version_bits,
        khronos_vendor_id,

        Error, MemoryError, LogicError, RuntimeError,

        Platform,
        get_platforms,

        Device,
        Context,
        CommandQueue,
        LocalMemory,
        MemoryObjectHolder,
        MemoryObject,
        MemoryMap,
        Buffer,

        _Program,
        Kernel,

        Event,
        wait_for_events,
        NannyEvent,

        enqueue_nd_range_kernel,

        _enqueue_marker,

        _enqueue_read_buffer,
        _enqueue_write_buffer,
        _enqueue_copy_buffer,
        _enqueue_read_buffer_rect,
        _enqueue_write_buffer_rect,
        _enqueue_copy_buffer_rect,

        _enqueue_read_image,
        _enqueue_copy_image,
        _enqueue_write_image,
        _enqueue_copy_image_to_buffer,
        _enqueue_copy_buffer_to_image,

        have_gl,

        ImageFormat,
        get_supported_image_formats,

        Image,
        Sampler,

        # This class is available unconditionally, even though CL only
        # has it on CL2.0 and newer.
        Pipe,
        )


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pyopencl._cl import (
        DeviceTopologyAmd,
        enqueue_copy_buffer_p2p_amd,
        enqueue_map_buffer,
        enqueue_map_image,
        UserEvent, ImageDescriptor,
        SVM, SVMAllocation, SVMPointer,
        # _enqueue_barrier_with_wait_list, _enqueue_fill_buffer,
        # _enqueue_marker_with_wait_list,
        enqueue_fill_image,
        enqueue_migrate_mem_objects, unload_platform_compiler,
        GLBuffer, GLRenderBuffer, GLTexture, gl_object_type, gl_texture_info,
        get_apple_cgl_share_group,
        enqueue_acquire_gl_objects,
        enqueue_release_gl_objects,
        )
else:
    try:
        from pyopencl._cl import DeviceTopologyAmd
        from pyopencl._cl import enqueue_copy_buffer_p2p_amd
    except ImportError:
        pass

    if not _PYPY:
        # FIXME: Add back to default set when pypy support catches up
        from pyopencl._cl import enqueue_map_buffer
        from pyopencl._cl import enqueue_map_image

    if get_cl_header_version() >= (1, 1):
        from pyopencl._cl import UserEvent
    if get_cl_header_version() >= (1, 2):
        from pyopencl._cl import ImageDescriptor
        from pyopencl._cl import (  # noqa: F401
            _enqueue_barrier_with_wait_list, _enqueue_fill_buffer,
            _enqueue_marker_with_wait_list, enqueue_fill_image,
            enqueue_migrate_mem_objects, unload_platform_compiler)

    if get_cl_header_version() >= (2, 0):
        from pyopencl._cl import SVM, SVMAllocation, SVMPointer

    if _cl.have_gl():
        from pyopencl._cl import (
            GLBuffer, GLRenderBuffer, GLTexture, gl_object_type, gl_texture_info)

        try:
            from pyopencl._cl import get_apple_cgl_share_group
        except ImportError:
            pass

        try:
            from pyopencl._cl import enqueue_acquire_gl_objects
            from pyopencl._cl import enqueue_release_gl_objects
        except ImportError:
            pass

import pyopencl._monkeypatch


# {{{ diagnostics

class CompilerWarning(UserWarning):
    pass


class CommandQueueUsedAfterExit(UserWarning):
    pass


def compiler_output(text: str) -> None:
    from pytools import strtobool
    if strtobool(os.environ.get("PYOPENCL_COMPILER_OUTPUT", "False")):
        warn(text, CompilerWarning, stacklevel=3)
    else:
        warn("Non-empty compiler output encountered. Set the "
                "environment variable PYOPENCL_COMPILER_OUTPUT=1 "
                "to see more.", CompilerWarning, stacklevel=3)

# }}}


# {{{ find pyopencl shipped source code

def _find_pyopencl_include_path() -> str:
    from os.path import abspath, dirname, exists, join

    # Try to find the include path in the same directory as this file
    include_path = join(abspath(dirname(__file__)), "cl")
    if not exists(include_path):
        from importlib.resources import files

        include_path = str(files("pyopencl") / "cl")
        if not exists(include_path):
            raise OSError("Unable to find PyOpenCL include path")

    # Quote the path if it contains a space and is not quoted already.
    # See https://github.com/inducer/pyopencl/issues/250 for discussion.
    if " " in include_path and not include_path.startswith('"'):
        return '"' + include_path + '"'
    else:
        return include_path

# }}}


# {{{ build option munging

def _split_options_if_necessary(
            options: str | Sequence[str]
        ) -> Sequence[str]:
    if isinstance(options, str):
        import shlex

        options = shlex.split(options)

    return options


def _find_include_path(options: Sequence[str]) -> list[str]:
    def unquote(path: str):
        if path.startswith('"') and path.endswith('"'):
            return path[1:-1]
        else:
            return path

    include_path = ["."]

    option_idx = 0
    while option_idx < len(options):
        option = options[option_idx].strip()
        if option.startswith("-I") or option.startswith("/I"):
            if len(option) == 2:
                if option_idx+1 < len(options):
                    include_path.append(unquote(options[option_idx+1]))
                option_idx += 2
            else:
                include_path.append(unquote(option[2:].lstrip()))
                option_idx += 1
        else:
            option_idx += 1

    # }}}

    return include_path


def _options_to_bytestring(options: Sequence[str | bytes]):
    def encode_if_necessary(s: str | bytes) -> bytes:
        if isinstance(s, str):
            return s.encode("utf-8")
        else:
            return s

    return b" ".join(encode_if_necessary(s) for s in options)


# }}}


# {{{ Program (wrapper around _Program, adds caching support)

from pytools import strtobool


_PYOPENCL_NO_CACHE = strtobool(os.environ.get("PYOPENCL_NO_CACHE", "false"))

_DEFAULT_BUILD_OPTIONS: list[str] = []
_DEFAULT_INCLUDE_OPTIONS: list[str] = ["-I", _find_pyopencl_include_path()]

# map of platform.name to build options list
_PLAT_BUILD_OPTIONS: dict[str, list[str]] = {
        "Oclgrind": ["-D", "PYOPENCL_USING_OCLGRIND"],
        }


def enable_debugging(platform_or_context: Platform | Context) -> None:
    """Enables debugging for all code subsequently compiled by
    PyOpenCL on the passed *platform*. Alternatively, a context
    may be passed.
    """

    if isinstance(platform_or_context, Context):
        platform = platform_or_context.devices[0].platform
    else:
        platform = platform_or_context

    if "AMD Accelerated" in platform.name:
        _PLAT_BUILD_OPTIONS.setdefault(platform.name, []).extend(
                ["-g", "-O0"])
        os.environ["CPU_MAX_COMPUTE_UNITS"] = "1"
    else:
        warn(f"Do not know how to enable debugging on '{platform.name}'",
             stacklevel=2)


class RepeatedKernelRetrieval(UserWarning):
    pass


RetT = TypeVar("RetT")


class Program:
    _prg: _Program | None
    _context: Context
    _source: str | bytes
    _build_duration_info: tuple[str, bool, float] | None

    @overload
    def __init__(self, arg1: _Program) -> None: ...

    @overload
    def __init__(self, arg1: Context, arg2: str | bytes) -> None: ...

    @overload
    def __init__(
            self,
            arg1: Context,
            arg2: Sequence[Device],
            arg3: Sequence[bytes]
        ) -> None: ...

    def __init__(self, arg1, arg2=None, arg3=None):
        self._knl_retrieval_count: dict[str, int] = {}

        if arg2 is None:
            # 1-argument form: program
            self._prg = cast("_Program", arg1)
            self._context = cast("Context", self._prg.get_info(program_info.CONTEXT))

        elif arg3 is None:
            # 2-argument form: context, source
            context, source = cast("tuple[Context, str | bytes]", (arg1, arg2))

            from pyopencl.tools import is_spirv
            if is_spirv(source):
                # FIXME no caching in SPIR-V case
                self._context = context
                self._prg = _cl._create_program_with_il(context, source)
                return

            self._context = context
            self._source = source
            self._prg = None

        else:
            context, devices, binaries = arg1, arg2, arg3
            self._context = context
            self._prg = _cl._Program(context, devices, binaries)

        self._build_duration_info = None

    def _get_prg(self) -> _Program:
        if self._prg is not None:
            return self._prg
        else:
            # "no program" can only happen in from-source case.
            warn("Pre-build attribute access defeats compiler caching.",
                    stacklevel=3)

            self._prg = _cl._Program(self._context, self._source)
            return self._prg

    def get_info(self, arg: program_info) -> object:
        return self._get_prg().get_info(arg)

    def get_build_info(self, *args, **kwargs):
        return self._get_prg().get_build_info(*args, **kwargs)

    def all_kernels(self) -> Sequence[Kernel]:
        return self._get_prg().all_kernels()

    @property
    def int_ptr(self):
        return self._get_prg().int_ptr
    int_ptr.__doc__ = _cl._Program.int_ptr.__doc__

    @staticmethod
    def from_int_ptr(int_ptr_value: int, retain: bool = True):
        return Program(_cl._Program.from_int_ptr(int_ptr_value, retain))
    from_int_ptr.__doc__ = _cl._Program.from_int_ptr.__doc__

    def __getattr__(self, attr: str) -> Kernel:
        try:
            knl = Kernel(self, attr)
            # Nvidia does not raise errors even for invalid names,
            # but this will give an error if the kernel is invalid.
            knl.num_args  # noqa: B018

            count = self._knl_retrieval_count[attr] = (
                self._knl_retrieval_count.get(attr, 0) + 1)

            if count == 2:
                # https://github.com/inducer/pyopencl/issues/831
                # https://github.com/inducer/pyopencl/issues/830#issuecomment-2913538384
                warn(f"Kernel '{attr}' has been retrieved more than once. "
                     "Each retrieval creates a new, independent kernel, "
                     "at possibly considerable expense. "
                     "To avoid the expense, reuse the retrieved kernel instance. "
                     "To avoid this warning, use cl.Kernel(prg, name).",
                     RepeatedKernelRetrieval, stacklevel=2)

            if self._build_duration_info is not None:
                build_descr, _was_cached, duration = self._build_duration_info
                if duration > 0.2:
                    logger.info(
                        "build program: kernel '%s' was part of a "
                        "lengthy %s (%.2f s)", attr, build_descr, duration)

                # don't whine about build times more than once.
                self._build_duration_info = None

            return knl
        except LogicError as err:
            raise AttributeError("'%s' was not found as a program "
                    "info attribute or as a kernel name" % attr) from err

    # {{{ build

    @classmethod
    def _process_build_options(cls,
                context: Context,
                options: str | Sequence[str] | None,
                _add_include_path: bool = False
            ) -> tuple[bytes, Sequence[str]]:
        if options is None:
            options = []

        options = _split_options_if_necessary(options)

        options = (
            *options,
            *_DEFAULT_BUILD_OPTIONS,
            *_DEFAULT_INCLUDE_OPTIONS,
            *_PLAT_BUILD_OPTIONS.get(context.devices[0].platform.name, []))

        forced_options = os.environ.get("PYOPENCL_BUILD_OPTIONS")
        if forced_options:
            options = (
                *options,
                *forced_options.split())

        return (
                _options_to_bytestring(options),
                _find_include_path(options))

    def build(self,
              options: str | Sequence[str] | None = None,
              devices: Sequence[Device] | None = None,
              cache_dir: str | None = None,
          ):
        options_bytes, include_path = self._process_build_options(
                self._context, options)

        if cache_dir is None:
            cache_dir = getattr(self._context, "cache_dir", None)

        build_descr = None
        from pyopencl.characterize import has_src_build_cache

        if (
                (_PYOPENCL_NO_CACHE or has_src_build_cache(self._context.devices[0]))
                and self._prg is None):
            if _PYOPENCL_NO_CACHE:
                build_descr = "uncached source build (cache disabled by user)"
            else:
                build_descr = "uncached source build (assuming cached by ICD)"

            self._prg = _cl._Program(self._context, self._source)

        from time import time
        start_time = time()
        was_cached = False

        if self._prg is not None:
            # uncached

            if build_descr is None:
                build_descr = "uncached source build"

            self._build_and_catch_errors(
                    lambda: self._prg.build(options_bytes, devices),
                    options_bytes=options_bytes)

        else:
            # cached

            from pyopencl.cache import create_built_program_from_source_cached
            self._prg, was_cached = self._build_and_catch_errors(
                    lambda: create_built_program_from_source_cached(
                        self._context, self._source, options_bytes, devices,
                        cache_dir=cache_dir, include_path=include_path),
                    options_bytes=options_bytes, source=self._source)

            if was_cached:
                build_descr = "cache retrieval"
            else:
                build_descr = "source build resulting from a binary cache miss"

            del self._context

        end_time = time()

        self._build_duration_info = (build_descr, was_cached, end_time-start_time)

        return self

    def _build_and_catch_errors(self,
                build_func: Callable[[], RetT],
                options_bytes: bytes,
                source: str | None = None,
            ):
        try:
            return build_func()
        except RuntimeError as e:
            msg = str(e)
            if options_bytes:
                msg = msg + "\n(options: %s)" % options_bytes.decode("utf-8")

            if source is not None:
                from tempfile import NamedTemporaryFile
                srcfile = NamedTemporaryFile(mode="wt", delete=False, suffix=".cl")
                try:
                    srcfile.write(source)
                finally:
                    srcfile.close()

                msg = msg + "\n(source saved as %s)" % srcfile.name

            code = e.code
            routine = e.routine

            err = RuntimeError(
                    _cl._ErrorRecord(
                        msg=msg,
                        code=code,
                        routine=routine))

        # Python 3.2 outputs the whole list of currently active exceptions
        # This serves to remove one (redundant) level from that nesting.
        raise err

    # }}}

    def compile(self,
                options: str | Sequence[str] | None = None,
                devices: Sequence[Device] | None = None,
                headers: Sequence[tuple[str, Program]] | None = None
            ):
        if headers is None:
            headers = []

        options_bytes, _ = self._process_build_options(self._context, options)

        self._get_prg().compile(options_bytes, devices,
                [(name, prg._get_prg()) for name, prg in headers])
        return self

    @override
    def __eq__(self, other: object):
        return (
            isinstance(other, Program)
            and self._get_prg() == other._get_prg())

    @override
    def __hash__(self):
        return hash(self._get_prg())

    reference_count: int  # pyright: ignore[reportUninitializedInstanceVariable]
    context: Context    # pyright: ignore[reportUninitializedInstanceVariable]
    num_devices: int    # pyright: ignore[reportUninitializedInstanceVariable]
    devices: Sequence[Device]    # pyright: ignore[reportUninitializedInstanceVariable]
    source: str    # pyright: ignore[reportUninitializedInstanceVariable]
    binary_sizes: int    # pyright: ignore[reportUninitializedInstanceVariable]
    binaries: Sequence[bytes]    # pyright: ignore[reportUninitializedInstanceVariable]
    num_kernels: int    # pyright: ignore[reportUninitializedInstanceVariable]
    kernel_names: str    # pyright: ignore[reportUninitializedInstanceVariable]
    il: bytes    # pyright: ignore[reportUninitializedInstanceVariable]
    scope_global_ctors_present: bool    # pyright: ignore[reportUninitializedInstanceVariable]
    scope_global_dtors_present: bool    # pyright: ignore[reportUninitializedInstanceVariable]


pyopencl._monkeypatch.add_get_info(Program, Program.get_info, _cl.program_info)


def create_program_with_built_in_kernels(context, devices, kernel_names):
    if not isinstance(kernel_names, str):
        kernel_names = ":".join(kernel_names)

    return Program(_Program.create_with_built_in_kernels(
        context, devices, kernel_names))


def link_program(context, programs, options=None, devices=None):
    if options is None:
        options = []

    options_bytes = _options_to_bytestring(_split_options_if_necessary(options))
    programs = [prg._get_prg() for prg in programs]
    raw_prg = _Program.link(context, programs, options_bytes, devices)
    return Program(raw_prg)

# }}}


# {{{ _OverriddenArrayInterfaceSVMAllocation

if get_cl_header_version() >= (2, 0):
    class _OverriddenArrayInterfaceSVMAllocation(SVMAllocation):
        def __init__(self, ctx, size, alignment, flags, *, _interface,
                queue=None):
            """
            :arg ctx: a :class:`Context`
            :arg flags: some of :class:`svm_mem_flags`.
            """
            super().__init__(ctx, size, alignment, flags, queue)

            # mem_flags.READ_ONLY applies to kernels, not the host
            read_write = True
            _interface["data"] = (int(self.svm_ptr), not read_write)

            self.__array_interface__ = _interface

# }}}


# {{{ create_image

def create_image(context, flags, format, shape=None, pitches=None,
        hostbuf=None, is_array=False, buffer=None) -> Image:
    """
    See :class:`mem_flags` for values of *flags*.
    *shape* is a 2- or 3-tuple. *format* is an instance of :class:`ImageFormat`.
    *pitches* is a 1-tuple for 2D images and a 2-tuple for 3D images, indicating
    the distance in bytes from one scan line to the next, and from one 2D image
    slice to the next.

    If *hostbuf* is given and *shape* is *None*, then *hostbuf.shape* is
    used as the *shape* parameter.

    :class:`Image` inherits from :class:`MemoryObject`.

    .. note::

        If you want to load images from :class:`numpy.ndarray` instances or read images
        back into them, be aware that OpenCL images expect the *x* dimension to vary
        fastest, whereas in the default (C) order of :mod:`numpy` arrays, the last index
        varies fastest. If your array is arranged in the wrong order in memory,
        there are two possible fixes for this:

        * Convert the array to Fortran (column-major) order using :func:`numpy.asarray`.

        * Pass *ary.T.copy()* to the image creation function.

    .. versionadded:: 2024.3
    """

    return Image(context, flags, format, shape=shape, pitches=pitches,
        hostbuf=hostbuf, is_array=is_array, buffer=buffer,
        _through_create_image=True)

# }}}


# {{{ create_some_context

def choose_devices(interactive: bool | None = None,
                   answers: list[str] | None = None) -> list[Device]:
    """
    Choose :class:`Device` instances 'somehow'.

    :arg interactive: If multiple choices for platform and/or device exist,
        *interactive* is ``True`` (or ``None`` and ``sys.stdin.isatty()``
        returns ``True``), then the user is queried about which device should be
        chosen. Otherwise, a device is chosen in an implementation-defined
        manner.
    :arg answers: A sequence of strings that will be used to answer the
        platform/device selection questions.

    :returns: a list of :class:`Device` instances.
    """

    if answers is None:
        if "PYOPENCL_CTX" in os.environ:
            ctx_spec = os.environ["PYOPENCL_CTX"]
            answers = ctx_spec.split(":")

        if "PYOPENCL_TEST" in os.environ:
            from pyopencl.tools import get_test_platforms_and_devices
            for _plat, devs in get_test_platforms_and_devices():
                for dev in devs:
                    return [dev]

    if answers is not None:
        pre_provided_answers = answers
        answers = answers[:]
    else:
        pre_provided_answers = None

    user_inputs = []

    if interactive is None:
        interactive = True
        try:
            if not sys.stdin.isatty():
                interactive = False
        except Exception:
            interactive = False

    def cc_print(s):
        if interactive:
            print(s)

    def get_input(prompt):
        if answers:
            return str(answers.pop(0))
        elif not interactive:
            return ""
        else:
            user_input = input(prompt)
            user_inputs.append(user_input)
            return user_input

    # {{{ pick a platform

    try:
        platforms = get_platforms()
    except LogicError as e:
        if "PLATFORM_NOT_FOUND_KHR" in str(e):
            # With the cl_khr_icd extension, clGetPlatformIDs fails if no platform
            # is available:
            # https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clGetPlatformIDs.html
            raise RuntimeError("no CL platforms available to ICD loader. "
                               "Install a CL driver "
                               "('ICD', such as pocl, rocm, Intel CL) to fix this. "
                               "See pyopencl docs for help: "
                               "https://documen.tician.de/pyopencl/"
                               "misc.html#installation") from e
        else:
            raise

    if not platforms:
        raise Error("no platforms found")
    else:
        if not answers:
            cc_print("Choose platform:")
            for i, pf in enumerate(platforms):
                cc_print("[%d] %s" % (i, pf))

        answer = get_input("Choice [0]:")
        if not answer:
            platform = platforms[0]
        else:
            platform = None
            try:
                int_choice = int(answer)
            except ValueError:
                pass
            else:
                if 0 <= int_choice < len(platforms):
                    platform = platforms[int_choice]

            if platform is None:
                answer = answer.lower()
                for pf in platforms:
                    if answer in pf.name.lower():
                        platform = pf
                if platform is None:
                    raise RuntimeError("input did not match any platform")

    # }}}

    # {{{ pick a device

    devices = platform.get_devices()

    def parse_device(choice):
        try:
            int_choice = int(choice)
        except ValueError:
            pass
        else:
            if 0 <= int_choice < len(devices):
                return devices[int_choice]

        choice = choice.lower()
        for dev in devices:
            if choice in dev.name.lower():
                return dev
        raise RuntimeError("input did not match any device")

    if not devices:
        raise Error("no devices found")
    elif len(devices) == 1 and not answers:
        cc_print(f"Choosing only available device: {devices[0]}")
        pass
    else:
        if not answers:
            cc_print("Choose device(s):")
            for i, dev in enumerate(devices):
                cc_print("[%d] %s" % (i, dev))

        answer = get_input("Choice, comma-separated [0]:")
        if not answer:
            devices = [devices[0]]
        else:
            devices = [parse_device(i) for i in answer.split(",")]

    # }}}

    if user_inputs:
        if pre_provided_answers is not None:
            user_inputs = pre_provided_answers + user_inputs
        cc_print("Set the environment variable PYOPENCL_CTX='%s' to "
                "avoid being asked again." % ":".join(user_inputs))

    if answers:
        raise RuntimeError("not all provided choices were used by "
                "choose_devices. (left over: '%s')" % ":".join(answers))

    return devices


def create_some_context(interactive: bool | None = None,
                        answers: list[str] | None = None) -> Context:
    """
    Create a :class:`Context` 'somehow'.

    :arg interactive: If multiple choices for platform and/or device exist,
        *interactive* is ``True`` (or ``None`` and ``sys.stdin.isatty()``
        returns ``True``), then the user is queried about which device should be
        chosen. Otherwise, a device is chosen in an implementation-defined
        manner.
    :arg answers: A sequence of strings that will be used to answer the
        platform/device selection questions.

    :returns: an instance of :class:`Context`.
    """
    devices = choose_devices(interactive, answers)

    return Context(devices)


_csc = create_some_context

# }}}


# {{{ SVMMap

@dataclass
class SVMMap(Generic[SVMInnerT]):
    """
    Returned by :func:`SVMPointer.map` and :func:`SVM.map`.
    This class may also be used as a context manager in a ``with`` statement.
    :meth:`release` will be called upon exit from the ``with`` region.
    The value returned to the ``as`` part of the context manager is the
    mapped Python object (e.g. a :mod:`numpy` array).

    .. versionadded:: 2016.2

    .. property:: event

        The :class:`Event` returned when mapping the memory.

    .. automethod:: release

    """
    svm: SVM[SVMInnerT] | None
    array: SVMInnerT
    queue: CommandQueue
    event: Event

    def __del__(self):
        if self.svm is not None:
            self.release()

    def __enter__(self):
        return self.array

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def release(self,
                queue: CommandQueue | None = None,
                wait_for: WaitList = None
            ) -> Event:
        """
        :arg queue: a :class:`pyopencl.CommandQueue`. Defaults to the one
            with which the map was created, if not specified.
        :returns: a :class:`pyopencl.Event`

        |std-enqueue-blurb|
        """

        assert self.svm is not None
        evt = self.svm._enqueue_unmap(queue or self.queue, wait_for)
        self.svm = None

        return evt

# }}}


# {{{ enqueue_copy

_IMAGE_MEM_OBJ_TYPES = [mem_object_type.IMAGE2D, mem_object_type.IMAGE3D]
if get_cl_header_version() >= (1, 2):
    _IMAGE_MEM_OBJ_TYPES.append(mem_object_type.IMAGE2D_ARRAY)


@overload
def enqueue_copy(
        queue: CommandQueue,
        dest: Buffer,
        src: HasBufferInterface,
        *,
        dst_offset: int = 0,
        is_blocking: bool = True,
        wait_for: WaitList = None
    ) -> Event: ...

@overload
def enqueue_copy(
        queue: CommandQueue,
        dest: HasBufferInterface,
        src: Buffer,
        *,
        src_offset: int = 0,
        is_blocking: bool = True,
        wait_for: WaitList = None
    ) -> Event: ...

@overload
def enqueue_copy(
        queue: CommandQueue,
        dest: Buffer,
        src: Buffer,
        *,
        src_offset: int = 0,
        dst_offset: int = 0,
        byte_count: int | None = None,
        wait_for: WaitList = None
    ) -> Event: ...

@overload
def enqueue_copy(
        queue: CommandQueue,
        dest: Buffer,
        src: HasBufferInterface,
        *,
        origin: tuple[int, ...],
        host_origin: tuple[int, ...],
        region: tuple[int, ...],
        buffer_pitches: tuple[int, ...] | None = None,
        host_pitches: tuple[int, ...] | None = None,
        is_blocking: bool = True,
        wait_for: WaitList = None
    ) -> Event: ...

@overload
def enqueue_copy(
        queue: CommandQueue,
        dest: HasBufferInterface,
        src: Buffer,
        *,
        origin: tuple[int, ...],
        host_origin: tuple[int, ...],
        region: tuple[int, ...],
        buffer_pitches: tuple[int, ...] | None = None,
        host_pitches: tuple[int, ...] | None = None,
        is_blocking: bool = True,
        wait_for: WaitList = None
    ) -> Event: ...

@overload
def enqueue_copy(
        queue: CommandQueue,
        dest: Buffer,
        src: Buffer,
        *,
        src_origin: tuple[int, ...],
        dst_origin: tuple[int, ...],
        region: tuple[int, ...],
        src_pitches: tuple[int, ...] | None = None,
        dst_pitches: tuple[int, ...] | None = None,
        wait_for: WaitList = None
    ) -> Event: ...

@overload
def enqueue_copy(
        queue: CommandQueue,
        dest: HasBufferInterface,
        src: Image,
        *,
        origin: tuple[int, ...],
        region: tuple[int, ...],
        pitches: tuple[int, ...] | None = None,
        is_blocking: bool = True,
        wait_for: WaitList = None
    ) -> Event: ...

@overload
def enqueue_copy(
        queue: CommandQueue,
        dest: Image,
        src: HasBufferInterface,
        *,
        origin: tuple[int, ...],
        region: tuple[int, ...],
        pitches: tuple[int, ...] | None = None,
        is_blocking: bool = True,
        wait_for: WaitList = None
    ) -> Event: ...

@overload
def enqueue_copy(
        queue: CommandQueue,
        dest: Image,
        src: Buffer,
        *,
        origin: tuple[int, ...],
        region: tuple[int, ...],
        pitches: tuple[int, ...] | None = None,
        wait_for: WaitList = None
    ) -> Event: ...

@overload
def enqueue_copy(
        queue: CommandQueue,
        dest: Buffer,
        src: Image,
        *,
        origin: tuple[int, ...],
        region: tuple[int, ...],
        pitches: tuple[int, ...] | None = None,
        wait_for: WaitList = None
    ) -> Event: ...

@overload
def enqueue_copy(
        queue: CommandQueue,
        dest: Image,
        src: Image,
        *,
        src_origin: tuple[int, ...],
        dest_origin: tuple[int, ...],
        region: tuple[int, ...],
        wait_for: WaitList = None
    ) -> Event: ...

@overload
def enqueue_copy(
        queue: CommandQueue,
        dest: SVMPointer | HasBufferInterface,
        src: SVMPointer | HasBufferInterface,
        *,
        byte_count: int | None = None,

        # do not use, must be zero
        src_offset: int = 0,
        dst_offset: int = 0,

        is_blocking: bool = True,
        wait_for: WaitList = None
    ) -> Event: ...


def enqueue_copy(queue: CommandQueue, dest: Any, src: Any, **kwargs: Any) -> Event:
    """Copy from :class:`Image`, :class:`Buffer` or the host to
    :class:`Image`, :class:`Buffer` or the host. (Note: host-to-host
    copies are unsupported.)

    The following keyword arguments are available:

    :arg wait_for: (optional, default empty)
    :arg is_blocking: Wait for completion. Defaults to *True*.
      (Available on any copy involving host memory)
    :return: A :class:`NannyEvent` if the transfer involved a
        host-side buffer, otherwise an :class:`Event`.

    .. note::

        Be aware that the deletion of the :class:`NannyEvent` that is
        returned by the function if the transfer involved a host-side buffer
        will block until the transfer is complete, so be sure to keep a
        reference to this :class:`Event` until the
        transfer has completed.

    .. note::

        Two types of 'buffer' occur in the arguments to this function,
        :class:`Buffer` and 'host-side buffers'. The latter are
        defined by Python and commonly called `buffer objects
        <https://docs.python.org/3/c-api/buffer.html>`__. :mod:`numpy`
        arrays are a very common example.
        Make sure to always be clear on whether a :class:`Buffer` or a
        Python buffer object is needed.

    .. ------------------------------------------------------------------------
    .. rubric :: Transfer :class:`Buffer` ↔ host
    .. ------------------------------------------------------------------------

    :arg src_offset: offset in bytes (optional)

        May only be nonzero if applied on the device side.

    :arg dst_offset: offset in bytes (optional)

        May only be nonzero if applied on the device side.

    .. note::

        The size of the transfer is controlled by the size of the
        of the host-side buffer. If the host-side buffer
        is a :class:`numpy.ndarray`, you can control the transfer size by
        transferring into a smaller 'view' of the target array, like this::

            cl.enqueue_copy(queue, large_dest_numpy_array[:15], src_buffer)

    .. ------------------------------------------------------------------------
    .. rubric :: Transfer :class:`Buffer` ↔ :class:`Buffer`
    .. ------------------------------------------------------------------------

    :arg byte_count: (optional) If not specified, defaults to the
        size of the source in versions 2012.x and earlier,
        and to the minimum of the size of the source and target
        from 2013.1 on.
    :arg src_offset: (optional)
    :arg dst_offset: (optional)

    .. ------------------------------------------------------------------------
    .. rubric :: Rectangular :class:`Buffer` ↔  host transfers (CL 1.1 and newer)
    .. ------------------------------------------------------------------------

    :arg buffer_origin: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg host_origin: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg region: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg buffer_pitches: :class:`tuple` of :class:`int` of length
        two or shorter. (optional, "tightly-packed" if unspecified)
    :arg host_pitches: :class:`tuple` of :class:`int` of length
        two or shorter. (optional, "tightly-packed" if unspecified)

    .. ------------------------------------------------------------------------
    .. rubric :: Rectangular :class:`Buffer` ↔  :class:`Buffer`
        transfers (CL 1.1 and newer)
    .. ------------------------------------------------------------------------

    :arg src_origin: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg dst_origin: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg region: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg src_pitches: :class:`tuple` of :class:`int` of length
        two or shorter. (optional, "tightly-packed" if unspecified)
    :arg dst_pitches: :class:`tuple` of :class:`int` of length
        two or shorter. (optional, "tightly-packed" if unspecified)

    .. ------------------------------------------------------------------------
    .. rubric :: Transfer :class:`Image` ↔ host
    .. ------------------------------------------------------------------------

    :arg origin: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg region: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg pitches: :class:`tuple` of :class:`int` of length
        two or shorter. (optional)

    .. ------------------------------------------------------------------------
    .. rubric :: Transfer :class:`Buffer` ↔ :class:`Image`
    .. ------------------------------------------------------------------------

    :arg offset: offset in buffer (mandatory)
    :arg origin: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg region: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)

    .. ------------------------------------------------------------------------
    .. rubric :: Transfer :class:`Image` ↔ :class:`Image`
    .. ------------------------------------------------------------------------

    :arg src_origin: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg dest_origin: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg region: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)

    .. ------------------------------------------------------------------------
    .. rubric :: Transfer :class:`SVMPointer`/host ↔ :class:`SVMPointer`/host
    .. ------------------------------------------------------------------------

    :arg byte_count: (optional) If not specified, defaults to the
        size of the source in versions 2012.x and earlier,
        and to the minimum of the size of the source and target
        from 2013.1 on.

    |std-enqueue-blurb|

    .. versionadded:: 2011.1
    """

    if isinstance(dest, MemoryObjectHolder):
        if dest.type == mem_object_type.BUFFER:
            if isinstance(src, MemoryObjectHolder):
                if src.type == mem_object_type.BUFFER:
                    # {{{ buffer -> buffer

                    if "src_origin" in kwargs:
                        # rectangular
                        return _cl._enqueue_copy_buffer_rect(
                                queue, src, dest, **kwargs)
                    else:
                        # linear
                        dest_offset = kwargs.pop("dest_offset", None)
                        if dest_offset is not None:
                            if "dst_offset" in kwargs:
                                raise TypeError("may not specify both 'dst_offset' "
                                                "and 'dest_offset'")

                            warn("The 'dest_offset' argument of enqueue_copy "
                                 "is deprecated. Use 'dst_offset' instead. "
                                 "'dest_offset' will stop working in 2023.x.",
                                 DeprecationWarning, stacklevel=2)

                            kwargs["dst_offset"] = dest_offset

                        return _cl._enqueue_copy_buffer(queue, src, dest, **kwargs)

                    # }}}
                elif src.type in _IMAGE_MEM_OBJ_TYPES:
                    return _cl._enqueue_copy_image_to_buffer(
                            queue, src, dest, **kwargs)
                else:
                    raise ValueError("invalid src mem object type")
            else:
                # {{{ host -> buffer

                if "buffer_origin" in kwargs:
                    return _cl._enqueue_write_buffer_rect(queue, dest, src, **kwargs)
                else:
                    device_offset = kwargs.pop("device_offset", None)
                    if device_offset is not None:
                        if "dst_offset" in kwargs:
                            raise TypeError("may not specify both 'device_offset' "
                                            "and 'dst_offset'")

                        warn("The 'device_offset' argument of enqueue_copy "
                                "is deprecated. Use 'dst_offset' instead. "
                                "'dst_offset' will stop working in 2023.x.",
                                DeprecationWarning, stacklevel=2)

                        kwargs["dst_offset"] = device_offset

                    return _cl._enqueue_write_buffer(queue, dest, src, **kwargs)

                # }}}

        elif dest.type in _IMAGE_MEM_OBJ_TYPES:
            # {{{ ... -> image

            if isinstance(src, MemoryObjectHolder):
                if src.type == mem_object_type.BUFFER:
                    return _cl._enqueue_copy_buffer_to_image(
                            queue, src, dest, **kwargs)
                elif src.type in _IMAGE_MEM_OBJ_TYPES:
                    return _cl._enqueue_copy_image(queue, src, dest, **kwargs)
                else:
                    raise ValueError("invalid src mem object type")
            else:
                # assume from-host
                origin = kwargs.pop("origin")
                region = kwargs.pop("region")

                pitches = kwargs.pop("pitches", (0, 0))
                if len(pitches) == 1:
                    kwargs["row_pitch"], = pitches
                else:
                    kwargs["row_pitch"], kwargs["slice_pitch"] = pitches

                return _cl._enqueue_write_image(
                        queue, dest, origin, region, src, **kwargs)

            # }}}
        else:
            raise ValueError("invalid dest mem object type")

    elif get_cl_header_version() >= (2, 0) and isinstance(dest, SVMPointer):
        # {{{ ... ->  SVM

        if not isinstance(src, SVMPointer):
            src = SVM(src)

        is_blocking = kwargs.pop("is_blocking", True)

        # These are NOT documented. They only support consistency with the
        # Buffer-based API for the sake of the Array.
        if kwargs.pop("src_offset", 0) != 0:
            raise ValueError("src_offset must be 0")
        if kwargs.pop("dst_offset", 0) != 0:
            raise ValueError("dst_offset must be 0")

        return _cl._enqueue_svm_memcpy(queue, is_blocking, dest, src, **kwargs)

        # }}}

    else:
        # assume to-host

        if isinstance(src, MemoryObjectHolder):
            if src.type == mem_object_type.BUFFER:
                if "buffer_origin" in kwargs:
                    return _cl._enqueue_read_buffer_rect(queue, src, dest, **kwargs)
                else:
                    device_offset = kwargs.pop("device_offset", None)
                    if device_offset is not None:
                        if "src_offset" in kwargs:
                            raise TypeError("may not specify both 'device_offset' "
                                            "and 'src_offset'")

                        warn("The 'device_offset' argument of enqueue_copy "
                                "is deprecated. Use 'src_offset' instead. "
                                "'dst_offset' will stop working in 2023.x.",
                                DeprecationWarning, stacklevel=2)

                        kwargs["src_offset"] = device_offset

                    return _cl._enqueue_read_buffer(queue, src, dest, **kwargs)

            elif src.type in _IMAGE_MEM_OBJ_TYPES:
                origin = kwargs.pop("origin")
                region = kwargs.pop("region")

                pitches = kwargs.pop("pitches", (0, 0))
                if len(pitches) == 1:
                    kwargs["row_pitch"], = pitches
                else:
                    kwargs["row_pitch"], kwargs["slice_pitch"] = pitches

                return _cl._enqueue_read_image(
                        queue, src, origin, region, dest, **kwargs)
            else:
                raise ValueError("invalid src mem object type")
        elif isinstance(src, SVMPointer):
            # {{{ svm -> host

            # dest is not a SVM instance, otherwise we'd be in the branch above

            # This is NOT documented. They only support consistency with the
            # Buffer-based API for the sake of the Array.
            if kwargs.pop("src_offset", 0) != 0:
                raise ValueError("src_offset must be 0")

            is_blocking = kwargs.pop("is_blocking", True)
            return _cl._enqueue_svm_memcpy(
                    queue, is_blocking, SVM(dest), src, **kwargs)

            # }}}
        else:
            # assume from-host
            raise TypeError("enqueue_copy cannot perform host-to-host transfers")

# }}}


# {{{ enqueue_fill

def enqueue_fill(queue: CommandQueue,
        dest: MemoryObjectHolder | SVMPointer,
        pattern: HasBufferInterface,
        size: int,
        *, offset: int = 0,
        wait_for: WaitList = None) -> Event:
    """
    .. versionadded:: 2022.2
    """
    if isinstance(dest, MemoryObjectHolder):
        return enqueue_fill_buffer(queue, dest, pattern, offset, size, wait_for)
    elif isinstance(dest, SVMPointer):
        if offset:
            raise NotImplementedError("enqueue_fill with SVM does not yet support "
                    "offsets")
        return enqueue_svm_memfill(queue, dest, pattern, size, wait_for)
    else:
        raise TypeError(f"enqueue_fill does not know how to fill '{type(dest)}'")

# }}}


# {{{ image creation

DTYPE_TO_CHANNEL_TYPE = {
    np.dtype(np.float32): channel_type.FLOAT,
    np.dtype(np.int16): channel_type.SIGNED_INT16,
    np.dtype(np.int32): channel_type.SIGNED_INT32,
    np.dtype(np.int8): channel_type.SIGNED_INT8,
    np.dtype(np.uint16): channel_type.UNSIGNED_INT16,
    np.dtype(np.uint32): channel_type.UNSIGNED_INT32,
    np.dtype(np.uint8): channel_type.UNSIGNED_INT8,
    }
try:
    np.float16  # noqa: B018
except Exception:
    pass
else:
    DTYPE_TO_CHANNEL_TYPE[np.dtype(np.float16)] = channel_type.HALF_FLOAT

DTYPE_TO_CHANNEL_TYPE_NORM = {
    np.dtype(np.int16): channel_type.SNORM_INT16,
    np.dtype(np.int8): channel_type.SNORM_INT8,
    np.dtype(np.uint16): channel_type.UNORM_INT16,
    np.dtype(np.uint8): channel_type.UNORM_INT8,
    }


def image_from_array(
            ctx: Context,
            ary: NDArray[Any],
            num_channels: int | None = None,
            mode: Literal["r"] | Literal["w"] = "r",
            norm_int: bool = False
        ) -> Image:
    if not ary.flags.c_contiguous:
        raise ValueError("array must be C-contiguous")

    dtype = ary.dtype
    if num_channels is None:

        try:
            dtype, num_channels = \
                    pyopencl.cltypes.vec_type_to_scalar_and_count[dtype]
        except KeyError:
            # It must be a scalar type then.
            num_channels = 1

        shape = ary.shape
        strides = ary.strides

    elif num_channels == 1:
        shape = ary.shape
        strides = ary.strides
    else:
        if ary.shape[-1] != num_channels:
            raise RuntimeError("last dimension must be equal to number of channels")

        shape = ary.shape[:-1]
        strides = ary.strides[:-1]

    if mode == "r":
        mode_flags = mem_flags.READ_ONLY
    elif mode == "w":
        mode_flags = mem_flags.WRITE_ONLY
    else:
        raise ValueError("invalid value '%s' for 'mode'" % mode)

    img_format = {
            1: channel_order.R,
            2: channel_order.RG,
            3: channel_order.RGB,
            4: channel_order.RGBA,
            }[num_channels]

    assert ary.strides[-1] == ary.dtype.itemsize

    if norm_int:
        channel_type = DTYPE_TO_CHANNEL_TYPE_NORM[dtype]
    else:
        channel_type = DTYPE_TO_CHANNEL_TYPE[dtype]

    return create_image(ctx, mode_flags | mem_flags.COPY_HOST_PTR,
            ImageFormat(img_format, channel_type),
            shape=shape[::-1], pitches=strides[::-1][1:],
            hostbuf=ary)

# }}}


# {{{ enqueue_* compatibility shims

def enqueue_marker(queue: CommandQueue, wait_for: WaitList = None) -> Event:
    if queue._get_cl_version() >= (1, 2) and get_cl_header_version() >= (1, 2):
        return _cl._enqueue_marker_with_wait_list(queue, wait_for)
    else:
        if wait_for:
            _cl._enqueue_wait_for_events(queue, wait_for)
        return _cl._enqueue_marker(queue)


def enqueue_barrier(queue: CommandQueue, wait_for: WaitList = None) -> Event:
    if queue._get_cl_version() >= (1, 2) and get_cl_header_version() >= (1, 2):
        return _cl._enqueue_barrier_with_wait_list(queue, wait_for)
    else:
        _cl._enqueue_barrier(queue)
        if wait_for:
            _cl._enqueue_wait_for_events(queue, wait_for)
        return _cl._enqueue_marker(queue)


def enqueue_fill_buffer(
            queue: CommandQueue,
            mem: MemoryObjectHolder,
            pattern: HasBufferInterface,
            offset: int,
            size: int,
            wait_for: WaitList = None
        ) -> Event:
    if not (queue._get_cl_version() >= (1, 2) and get_cl_header_version() >= (1, 2)):
        warn(
            "The context for this queue does not declare OpenCL 1.2 support, so "
            "the next thing you might see is a crash",
            stacklevel=2)

    if _PYPY and isinstance(pattern, np.generic):
        pattern = np.asarray(pattern)

    return _cl._enqueue_fill_buffer(queue, mem, pattern, offset, size,
                                    wait_for)

# }}}


# {{{ numpy-like svm allocation

def enqueue_svm_memfill(
            queue: CommandQueue,
            dest: SVMPointer,
            pattern: HasBufferInterface,
            byte_count: int | None = None,
            wait_for: WaitList = None
        ) -> Event:
    """Fill shared virtual memory with a pattern.

    :arg dest: a Python buffer object, or any implementation of :class:`SVMPointer`.
    :arg pattern: a Python buffer object (e.g. a :class:`numpy.ndarray` with the
        fill pattern to be used.
    :arg byte_count: The size of the memory to be fill. Defaults to the
        entirety of *dest*.

    |std-enqueue-blurb|

    .. versionadded:: 2016.2
    """

    if not isinstance(dest, SVMPointer):
        dest = SVM(dest)

    return _cl._enqueue_svm_memfill(
            queue, dest, pattern, byte_count=byte_count, wait_for=wait_for)


def enqueue_svm_migrate_mem(
            queue: CommandQueue,
            svms: Sequence[SVMPointer],
            flags: svm_mem_flags,
            wait_for: WaitList = None,
        ):
    """
    :arg svms: a collection of Python buffer objects (e.g. :mod:`numpy`
        arrays), or any implementation of :class:`SVMPointer`.
    :arg flags: a combination of :class:`mem_migration_flags`

    |std-enqueue-blurb|

    .. versionadded:: 2016.2

    This function requires OpenCL 2.1.
    """

    return _cl._enqueue_svm_migrate_mem(queue, svms, flags, wait_for)


def svm_empty(
            ctx: Context,
            flags: svm_mem_flags,
            shape: int | tuple[int, ...],
            dtype: DTypeT,
            order: Literal["F"] | Literal["C"] = "C",
            alignment: int | None = None,
            queue: CommandQueue | None = None,
        ) -> np.ndarray[tuple[int, ...], DTypeT]:
    """Allocate an empty :class:`numpy.ndarray` of the given *shape*, *dtype*
    and *order*. (See :func:`numpy.empty` for the meaning of these arguments.)
    The array will be allocated in shared virtual memory belonging
    to *ctx*.

    :arg ctx: a :class:`Context`
    :arg flags: a combination of flags from :class:`svm_mem_flags`.
    :arg alignment: the number of bytes to which the beginning of the memory
        is aligned. Defaults to the :attr:`numpy.dtype.itemsize` of *dtype*.

    :returns: a :class:`numpy.ndarray` whose :attr:`numpy.ndarray.base` attribute
        is a :class:`SVMAllocation`.

    To pass the resulting array to an OpenCL kernel or :func:`enqueue_copy`, you
    will likely want to wrap the returned array in an :class:`SVM` tag.

    .. versionadded:: 2016.2

    .. versionchanged:: 2022.2

        *queue* argument added.
    """

    dtype = np.dtype(dtype)

    try:
        shape = cast("tuple[int, ...]", shape)
        s = 1
        for dim in shape:
            s *= dim
    except TypeError as err:
        admissible_types = (int, np.integer)

        if not isinstance(shape, admissible_types):
            raise TypeError("shape must either be iterable or "
                    "castable to an integer") from err
        s = shape
        shape = (shape,)

    itemsize = dtype.itemsize
    nbytes = s * itemsize

    from pyopencl.compyte.array import c_contiguous_strides, f_contiguous_strides

    if order in "fF":
        strides = f_contiguous_strides(itemsize, shape)
    elif order in "cC":
        strides = c_contiguous_strides(itemsize, shape)
    else:
        raise ValueError("order not recognized: %s" % order)

    descr = dtype.descr

    interface = {
        "version": 3,
        "shape": shape,
        "strides": strides,
        }

    if len(descr) == 1:
        interface["typestr"] = descr[0][1]
    else:
        interface["typestr"] = "V%d" % itemsize
        interface["descr"] = descr

    if alignment is None:
        alignment = itemsize

    svm_alloc = _OverriddenArrayInterfaceSVMAllocation(
            ctx, nbytes, alignment, flags, _interface=interface,
            queue=queue)
    return np.asarray(svm_alloc)


def svm_empty_like(
            ctx: Context,
            flags: svm_mem_flags,
            ary: np.ndarray[tuple[int, ...], DTypeT],
            alignment: int | None = None,
        ) -> np.ndarray[tuple[int, ...], DTypeT]:
    """Allocate an empty :class:`numpy.ndarray` like the existing
    :class:`numpy.ndarray` *ary*.  The array will be allocated in shared
    virtual memory belonging to *ctx*.

    :arg alignment: the number of bytes to which the beginning of the memory
        is aligned. Defaults to the :attr:`numpy.dtype.itemsize` of *dtype*.

    :returns: a :class:`numpy.ndarray` whose :attr:`numpy.ndarray.base` attribute
        is a :class:`SVMAllocation`.

    To pass the resulting array to an OpenCL kernel or :func:`enqueue_copy`, you
    will likely want to wrap the returned array in an :class:`SVM` tag.

    .. versionadded:: 2016.2
    """
    if ary.flags.c_contiguous:
        order = "C"
    elif ary.flags.f_contiguous:
        order = "F"
    else:
        raise ValueError("array is neither C- nor Fortran-contiguous")

    return svm_empty(ctx, flags, ary.shape, ary.dtype, order,
            alignment=alignment)


def csvm_empty(
            ctx: Context,
            shape: int | tuple[int, ...],
            dtype: DTypeT,
            order: Literal["F"] | Literal["C"] = "C",
            alignment: int | None = None,
            queue: CommandQueue | None = None,
        ) -> np.ndarray[tuple[int, ...], DTypeT]:
    """
    Like :func:`svm_empty`, but with *flags* set for a coarse-grain read-write
    buffer.

    .. versionadded:: 2016.2
    """
    return svm_empty(ctx, svm_mem_flags.READ_WRITE, shape, dtype, order,
                     alignment, queue=queue)


def csvm_empty_like(
            ctx: Context,
            ary: np.ndarray[tuple[int, ...], DTypeT],
            alignment: int | None = None,
        ) -> np.ndarray[tuple[int, ...], DTypeT]:
    """
    Like :func:`svm_empty_like`, but with *flags* set for a coarse-grain
    read-write buffer.

    .. versionadded:: 2016.2
    """
    return svm_empty_like(ctx, svm_mem_flags.READ_WRITE, ary, alignment)


def fsvm_empty(
            ctx: Context,
            shape: int | tuple[int, ...],
            dtype: DTypeT,
            order: Literal["F"] | Literal["C"] = "C",
            alignment: int | None = None,
            queue: CommandQueue | None = None,
        ) -> np.ndarray[tuple[int, ...], DTypeT]:
    """
    Like :func:`svm_empty`, but with *flags* set for a fine-grain read-write
    buffer.

    .. versionadded:: 2016.2
    """
    return svm_empty(ctx,
            svm_mem_flags.READ_WRITE | svm_mem_flags.SVM_FINE_GRAIN_BUFFER,
            shape, dtype, order, alignment, queue)


def fsvm_empty_like(
            ctx: Context,
            ary: np.ndarray[tuple[int, ...], DTypeT],
            alignment: int | None = None,
        ) -> np.ndarray[tuple[int, ...], DTypeT]:
    """
    Like :func:`svm_empty_like`, but with *flags* set for a fine-grain
    read-write buffer.

    .. versionadded:: 2016.2
    """
    return svm_empty_like(
            ctx,
            svm_mem_flags.READ_WRITE | svm_mem_flags.SVM_FINE_GRAIN_BUFFER,
            ary,
            alignment)

# }}}


_KERNEL_ARG_CLASSES: tuple[type, ...] = (
        MemoryObjectHolder,
        Sampler,
        CommandQueue,
        LocalMemory,
        *([SVM] if get_cl_header_version() >= (2, 0) else [])
        )


CtxFactory: TypeAlias = Callable[[], Context]


__all__ = [
    "SVM",
    "VERSION",
    "VERSION_STATUS",
    "VERSION_TEXT",
    "Buffer",
    "CommandQueue",
    "Context",
    "CtxFactory",
    "Device",
    "DeviceTopologyAmd",
    "Error",
    "Event",
    "GLBuffer",
    "GLRenderBuffer",
    "GLTexture",
    "Image",
    "ImageDescriptor",
    "ImageFormat",
    "Kernel",
    "LocalMemory",
    "LogicError",
    "MemoryError",
    "MemoryMap",
    "MemoryObject",
    "MemoryObjectHolder",
    "NannyEvent",
    "Pipe",
    "Platform",
    "Program",
    "RuntimeError",
    "SVMAllocation",
    "SVMAllocation",
    "SVMMap",
    "SVMPointer",
    "Sampler",
    "UserEvent",
    "WaitList",
    "_csc",
    "addressing_mode",
    "channel_order",
    "channel_type",
    "choose_devices",
    "command_execution_status",
    "command_queue_info",
    "command_queue_properties",
    "command_type",
    "context_info",
    "context_properties",
    "create_image",
    "create_program_with_built_in_kernels",
    "create_some_context",
    "csvm_empty",
    "csvm_empty_like",
    "device_affinity_domain",
    "device_atomic_capabilities",
    "device_device_enqueue_capabilities",
    "device_exec_capabilities",
    "device_fp_config",
    "device_info",
    "device_local_mem_type",
    "device_mem_cache_type",
    "device_partition_property",
    "device_svm_capabilities",
    "device_topology_type_amd",
    "device_type",
    "enable_debugging",
    "enqueue_acquire_gl_objects",
    "enqueue_barrier",
    "enqueue_copy",
    "enqueue_copy_buffer_p2p_amd",
    "enqueue_fill",
    "enqueue_fill_buffer",
    "enqueue_fill_image",
    "enqueue_map_buffer",
    "enqueue_map_image",
    "enqueue_marker",
    "enqueue_migrate_mem_objects",
    "enqueue_nd_range_kernel",
    "enqueue_release_gl_objects",
    "enqueue_svm_memfill",
    "enqueue_svm_migrate_mem",
    "event_info",
    "filter_mode",
    "fsvm_empty",
    "fsvm_empty_like",
    "get_apple_cgl_share_group",
    "get_cl_header_version",
    "get_platforms",
    "get_supported_image_formats",
    "gl_context_info",
    "gl_object_type",
    "gl_texture_info",
    "have_gl",
    "image_from_array",
    "image_info",
    "kernel_arg_access_qualifier",
    "kernel_arg_address_qualifier",
    "kernel_arg_info",
    "kernel_arg_type_qualifier",
    "kernel_info",
    "kernel_sub_group_info",
    "kernel_work_group_info",
    "khronos_vendor_id",
    "link_program",
    "map_flags",
    "mem_flags",
    "mem_info",
    "mem_migration_flags",
    "mem_object_type",
    "pipe_info",
    "pipe_properties",
    "platform_info",
    "profiling_info",
    "program_binary_type",
    "program_build_info",
    "program_info",
    "program_kind",
    "queue_properties",
    "sampler_info",
    "sampler_properties",
    "status_code",
    "svm_empty",
    "svm_empty_like",
    "svm_mem_flags",
    "unload_platform_compiler",
    "version_bits",
    "wait_for_events",
]


# vim: foldmethod=marker
