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

import logging
from sys import intern
from typing import Any, Sequence
from warnings import warn

# must import, otherwise dtype registry will not be fully populated
import pyopencl.cltypes
from pyopencl.version import VERSION, VERSION_STATUS, VERSION_TEXT  # noqa: F401


__version__ = VERSION_TEXT

logger = logging.getLogger(__name__)

# This supports ocl-icd find shipped OpenCL ICDs, cf.
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


try:
    from pyopencl._cl import DeviceTopologyAmd  # noqa: F401
    from pyopencl._cl import enqueue_copy_buffer_p2p_amd  # noqa: F401
except ImportError:
    pass

if not _PYPY:
    # FIXME: Add back to default set when pypy support catches up
    from pyopencl._cl import enqueue_map_buffer  # noqa: F401
    from pyopencl._cl import enqueue_map_image  # noqa: F401

if get_cl_header_version() >= (1, 1):
    from pyopencl._cl import UserEvent  # noqa: F401
if get_cl_header_version() >= (1, 2):
    from pyopencl._cl import ImageDescriptor
    from pyopencl._cl import (  # noqa: F401
        _enqueue_barrier_with_wait_list, _enqueue_fill_buffer,
        _enqueue_marker_with_wait_list, enqueue_fill_image,
        enqueue_migrate_mem_objects, unload_platform_compiler)

if get_cl_header_version() >= (2, 0):
    from pyopencl._cl import SVM, SVMAllocation, SVMPointer

if _cl.have_gl():
    from pyopencl._cl import (  # noqa: F401
        GLBuffer, GLRenderBuffer, GLTexture, gl_object_type, gl_texture_info)

    try:
        from pyopencl._cl import get_apple_cgl_share_group  # noqa: F401
    except ImportError:
        pass

    try:
        from pyopencl._cl import enqueue_acquire_gl_objects  # noqa: F401
        from pyopencl._cl import enqueue_release_gl_objects  # noqa: F401
    except ImportError:
        pass

import inspect as _inspect


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
        try:
            # NOTE: only available in Python >=3.9
            from importlib.resources import files
        except ImportError:
            from importlib_resources import files  # type: ignore[no-redef]

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

def _split_options_if_necessary(options):
    if isinstance(options, str):
        import shlex

        options = shlex.split(options)

    return options


def _find_include_path(options):
    def unquote(path):
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


def _options_to_bytestring(options):
    def encode_if_necessary(s):
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


def enable_debugging(platform_or_context):
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


class Program:
    def __init__(self, arg1, arg2=None, arg3=None):
        if arg2 is None:
            # 1-argument form: program
            self._prg = arg1
            self._context = self._prg.get_info(program_info.CONTEXT)

        elif arg3 is None:
            # 2-argument form: context, source
            context, source = arg1, arg2

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
            context, device, binaries = arg1, arg2, arg3
            self._context = context
            self._prg = _cl._Program(context, device, binaries)

        self._build_duration_info = None

    def _get_prg(self):
        if self._prg is not None:
            return self._prg
        else:
            # "no program" can only happen in from-source case.
            warn("Pre-build attribute access defeats compiler caching.",
                    stacklevel=3)

            self._prg = _cl._Program(self._context, self._source)
            return self._prg

    def get_info(self, arg):
        return self._get_prg().get_info(arg)

    def get_build_info(self, *args, **kwargs):
        return self._get_prg().get_build_info(*args, **kwargs)

    def all_kernels(self):
        return self._get_prg().all_kernels()

    @property
    def int_ptr(self):
        return self._get_prg().int_ptr
    int_ptr.__doc__ = _cl._Program.int_ptr.__doc__

    @staticmethod
    def from_int_ptr(int_ptr_value, retain=True):
        return Program(_cl._Program.from_int_ptr(int_ptr_value, retain))
    from_int_ptr.__doc__ = _cl._Program.from_int_ptr.__doc__

    def __getattr__(self, attr):
        try:
            knl = Kernel(self, attr)
            # Nvidia does not raise errors even for invalid names,
            # but this will give an error if the kernel is invalid.
            knl.num_args  # noqa: B018

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
    def _process_build_options(cls, context, options, _add_include_path=False):
        if options is None:
            options = []
        if isinstance(options, tuple):
            options = list(options)

        options = _split_options_if_necessary(options)

        options = (options
                + _DEFAULT_BUILD_OPTIONS
                + _DEFAULT_INCLUDE_OPTIONS
                + _PLAT_BUILD_OPTIONS.get(
                    context.devices[0].platform.name, []))

        forced_options = os.environ.get("PYOPENCL_BUILD_OPTIONS")
        if forced_options:
            options = options + forced_options.split()

        return (
                _options_to_bytestring(options),
                _find_include_path(options))

    def build(self, options=None, devices=None, cache_dir=None):
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

    def _build_and_catch_errors(self, build_func, options_bytes, source=None):
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

    def compile(self, options=None, devices=None, headers=None):
        if headers is None:
            headers = []

        options_bytes, _ = self._process_build_options(self._context, options)

        self._get_prg().compile(options_bytes, devices,
                [(name, prg._get_prg()) for name, prg in headers])
        return self

    def __eq__(self, other):
        return self._get_prg() == other._get_prg()

    def __ne__(self, other):
        return self._get_prg() == other._get_prg()

    def __hash__(self):
        return hash(self._get_prg())


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


# {{{ monkeypatch C++ wrappers to add functionality

def _add_functionality():
    def generic_get_cl_version(self):
        import re
        version_string = self.version
        match = re.match(r"^OpenCL ([0-9]+)\.([0-9]+) .*$", version_string)
        if match is None:
            raise RuntimeError("%s %s returned non-conformant "
                               "platform version string '%s'" %
                               (type(self).__name__, self, version_string))

        return int(match.group(1)), int(match.group(2))

    # {{{ Platform

    def platform_repr(self):
        return f"<pyopencl.Platform '{self.name}' at 0x{self.int_ptr:x}>"

    Platform.__repr__ = platform_repr
    Platform._get_cl_version = generic_get_cl_version

    # }}}

    # {{{ Device

    def device_repr(self):
        return "<pyopencl.Device '{}' on '{}' at 0x{:x}>".format(
                self.name.strip(), self.platform.name.strip(), self.int_ptr)

    def device_hashable_model_and_version_identifier(self):
        return ("v1", self.vendor, self.vendor_id, self.name, self.version)

    def device_persistent_unique_id(self):
        warn("Device.persistent_unique_id is deprecated. "
                "Use Device.hashable_model_and_version_identifier instead.",
                DeprecationWarning, stacklevel=2)
        return device_hashable_model_and_version_identifier(self)

    Device.__repr__ = device_repr

    # undocumented for now:
    Device._get_cl_version = generic_get_cl_version
    Device.hashable_model_and_version_identifier = property(
            device_hashable_model_and_version_identifier)
    Device.persistent_unique_id = property(device_persistent_unique_id)

    # }}}

    # {{{ Context

    def context_repr(self):
        return "<pyopencl.Context at 0x{:x} on {}>".format(self.int_ptr,
                ", ".join(repr(dev) for dev in self.devices))

    def context_get_cl_version(self):
        return self.devices[0].platform._get_cl_version()

    Context.__repr__ = context_repr
    from pytools import memoize_method
    Context._get_cl_version = memoize_method(context_get_cl_version)

    # }}}

    # {{{ CommandQueue

    def command_queue_enter(self):
        return self

    def command_queue_exit(self, exc_type, exc_val, exc_tb):
        self.finish()
        self._finalize()

    def command_queue_get_cl_version(self):
        return self.device._get_cl_version()

    CommandQueue.__enter__ = command_queue_enter
    CommandQueue.__exit__ = command_queue_exit
    CommandQueue._get_cl_version = memoize_method(command_queue_get_cl_version)

    # }}}

    # {{{ _Program (the internal, non-caching version)

    def program_get_build_logs(self):
        build_logs = []
        for dev in self.get_info(_cl.program_info.DEVICES):
            try:
                log = self.get_build_info(dev, program_build_info.LOG)
            except Exception:
                log = "<error retrieving log>"

            build_logs.append((dev, log))

        return build_logs

    def program_build(self, options_bytes, devices=None):
        err = None
        try:
            self._build(options=options_bytes, devices=devices)
        except Error as e:
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
            if self.kind() == program_kind.SOURCE:
                build_type = "From-source build"
            elif self.kind() == program_kind.BINARY:
                build_type = "From-binary build"
            elif self.kind() == program_kind.IL:
                build_type = "From-IL build"
            else:
                build_type = "Build"

            compiler_output("%s succeeded, but resulted in non-empty logs:\n%s"
                    % (build_type, message))

        return self

    _cl._Program._get_build_logs = program_get_build_logs
    _cl._Program.build = program_build

    # }}}

    # {{{ Event
    class ProfilingInfoGetter:
        def __init__(self, event):
            self.event = event

        def __getattr__(self, name):
            info_cls = _cl.profiling_info

            try:
                inf_attr = getattr(info_cls, name.upper())
            except AttributeError as err:
                raise AttributeError("%s has no attribute '%s'"
                        % (type(self), name)) from err
            else:
                return self.event.get_profiling_info(inf_attr)

    _cl.Event.profile = property(ProfilingInfoGetter)

    # }}}

    # {{{ Kernel

    kernel_old_get_info = Kernel.get_info
    kernel_old_get_work_group_info = Kernel.get_work_group_info

    def kernel_set_arg_types(self, arg_types):
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

    def kernel_get_work_group_info(self, param, device):
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

    def kernel_capture_call(self, output_file, queue, global_size, local_size,
            *args, **kwargs):
        from pyopencl.capture_call import capture_kernel_call
        capture_kernel_call(self, output_file, queue, global_size, local_size,
                *args, **kwargs)

    def kernel_get_info(self, param_name):
        val = kernel_old_get_info(self, param_name)

        if isinstance(val, _Program):
            return Program(val)
        else:
            return val

    Kernel.get_work_group_info = kernel_get_work_group_info

    # FIXME: Possibly deprecate this version
    Kernel.set_scalar_arg_dtypes = kernel_set_arg_types
    Kernel.set_arg_types = kernel_set_arg_types

    Kernel.capture_call = kernel_capture_call
    Kernel.get_info = kernel_get_info

    # }}}

    # {{{ ImageFormat

    def image_format_repr(self):
        return "ImageFormat({}, {})".format(
                channel_order.to_string(self.channel_order,
                    "<unknown channel order 0x%x>"),
                channel_type.to_string(self.channel_data_type,
                    "<unknown channel data type 0x%x>"))

    def image_format_eq(self, other):
        return (self.channel_order == other.channel_order
                and self.channel_data_type == other.channel_data_type)

    def image_format_ne(self, other):
        return not image_format_eq(self, other)

    def image_format_hash(self):
        return hash((type(self), self.channel_order, self.channel_data_type))

    ImageFormat.__repr__ = image_format_repr
    ImageFormat.__eq__ = image_format_eq
    ImageFormat.__ne__ = image_format_ne
    ImageFormat.__hash__ = image_format_hash

    # }}}

    # {{{ Image

    def image_init(
                self, context, flags, format, shape=None, pitches=None,
                hostbuf=None, is_array=False, buffer=None, *,
                desc: ImageDescriptor | None = None,
                _through_create_image: bool = False,
            ) -> None:
        if hostbuf is not None and not \
                (flags & (mem_flags.USE_HOST_PTR | mem_flags.COPY_HOST_PTR)):
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

            Image._custom_init(self, context, flags, format, desc, hostbuf)

            return

        if shape is None and hostbuf is None:
            raise Error("'shape' must be passed if 'hostbuf' is not given")

        if shape is None and hostbuf is not None:
            shape = hostbuf.shape

        if hostbuf is None and pitches is not None:
            raise Error("'pitches' may only be given if 'hostbuf' is given")

        if context._get_cl_version() >= (1, 2) and get_cl_header_version() >= (1, 2):
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
                    image_type = mem_object_type.IMAGE2D_ARRAY
                else:
                    image_type = mem_object_type.IMAGE3D

            elif len(shape) == 2:
                if buffer is not None:
                    raise TypeError(
                            "'buffer' argument is not supported for 2D arrays")
                elif is_array:
                    image_type = mem_object_type.IMAGE1D_ARRAY
                else:
                    image_type = mem_object_type.IMAGE2D

            elif len(shape) == 1:
                if buffer is not None:
                    image_type = mem_object_type.IMAGE1D_BUFFER
                elif is_array:
                    raise TypeError("array of zero-dimensional images not supported")
                else:
                    image_type = mem_object_type.IMAGE1D

            else:
                raise ValueError("images cannot have more than three dimensions")

            desc = ImageDescriptor() \
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

            Image._custom_init(self, context, flags, format, desc, hostbuf)
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

            Image._custom_init(self, context, flags, format, shape,
                    pitches, hostbuf)

    class _ImageInfoGetter:
        def __init__(self, event):
            warn(
                "Image.image.attr is deprecated and will go away in 2021. "
                "Use Image.attr directly, instead.", stacklevel=2)

            self.event = event

        def __getattr__(self, name):
            try:
                inf_attr = getattr(_cl.image_info, name.upper())
            except AttributeError as err:
                raise AttributeError("%s has no attribute '%s'"
                        % (type(self), name)) from err
            else:
                return self.event.get_image_info(inf_attr)

    def image_shape(self):
        if self.type == mem_object_type.IMAGE2D:
            return (self.width, self.height)
        elif self.type == mem_object_type.IMAGE3D:
            return (self.width, self.height, self.depth)
        else:
            raise LogicError("only images have shapes")

    Image.__init__ = image_init
    Image.image = property(_ImageInfoGetter)
    Image.shape = property(image_shape)

    # }}}

    # {{{ Error

    def error_str(self):
        val = self.what
        try:
            val.routine  # noqa: B018
        except AttributeError:
            return str(val)
        else:
            result = ""
            if val.code() != status_code.SUCCESS:
                result = status_code.to_string(
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

    def error_code(self):
        return self.args[0].code()

    def error_routine(self):
        return self.args[0].routine()

    def error_what(self):
        return self.args[0]

    Error.__str__ = error_str
    Error.code = property(error_code)
    Error.routine = property(error_routine)
    Error.what = property(error_what)

    # }}}

    # {{{ MemoryMap

    def memory_map_enter(self):
        return self

    def memory_map_exit(self, exc_type, exc_val, exc_tb):
        self.release()

    MemoryMap.__doc__ = """
        This class may also be used as a context manager in a ``with`` statement.
        The memory corresponding to this object will be unmapped when
        this object is deleted or :meth:`release` is called.

        .. automethod:: release
        """
    MemoryMap.__enter__ = memory_map_enter
    MemoryMap.__exit__ = memory_map_exit

    # }}}

    # {{{ SVMPointer

    if get_cl_header_version() >= (2, 0):
        SVMPointer.__doc__ = """A base class for things that can be passed to
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

    def svmptr_map(self, queue: CommandQueue, *, flags: int, is_blocking: bool =
                   True, wait_for: Sequence[Event] | None = None,
                   size: Event | None = None) -> SVMMap:
        """
        :arg is_blocking: If *False*, subsequent code must wait on
            :attr:`SVMMap.event` in the returned object before accessing the
            mapped memory.
        :arg flags: a combination of :class:`pyopencl.map_flags`.
        :arg size: The size of the map in bytes. If not provided, defaults to
            :attr:`size`.

        |std-enqueue-blurb|
        """
        return SVMMap(self,
                np.asarray(self.buf),
                queue,
                _cl._enqueue_svm_map(queue, is_blocking, flags, self, wait_for,
                                    size=size))

    def svmptr_map_ro(self, queue: CommandQueue, *, is_blocking: bool = True,
                      wait_for: Sequence[Event] | None = None,
                      size: int | None = None) -> SVMMap:
        """Like :meth:`map`, but with *flags* set for a read-only map.
        """

        return self.map(queue, flags=map_flags.READ,
                is_blocking=is_blocking, wait_for=wait_for, size=size)

    def svmptr_map_rw(self, queue: CommandQueue, *, is_blocking: bool = True,
                      wait_for: Sequence[Event] | None = None,
                      size: int | None = None) -> SVMMap:
        """Like :meth:`map`, but with *flags* set for a read-only map.
        """

        return self.map(queue, flags=map_flags.READ | map_flags.WRITE,
                is_blocking=is_blocking, wait_for=wait_for, size=size)

    def svmptr__enqueue_unmap(self, queue, wait_for=None):
        return _cl._enqueue_svm_unmap(queue, self, wait_for)

    def svmptr_as_buffer(self, ctx: Context, *, flags: int | None = None,
                         size: int | None = None) -> Buffer:
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
            flags = mem_flags.READ_WRITE | mem_flags.USE_HOST_PTR

        if size is None:
            size = self.size

        return Buffer(ctx, flags, size=size, hostbuf=self.buf)

    if get_cl_header_version() >= (2, 0):
        SVMPointer.map = svmptr_map
        SVMPointer.map_ro = svmptr_map_ro
        SVMPointer.map_rw = svmptr_map_rw
        SVMPointer._enqueue_unmap = svmptr__enqueue_unmap
        SVMPointer.as_buffer = svmptr_as_buffer

    # }}}

    # {{{ SVMAllocation

    if get_cl_header_version() >= (2, 0):
        SVMAllocation.__doc__ = """
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

    if get_cl_header_version() >= (2, 0):
        SVM.__doc__ = """Tags an object exhibiting the Python buffer interface
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

    def svm_map(self, queue, flags, is_blocking=True, wait_for=None):
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
        return SVMMap(
                self,
                self.mem,
                queue,
                _cl._enqueue_svm_map(queue, is_blocking, flags, self, wait_for))

    def svm_map_ro(self, queue, is_blocking=True, wait_for=None):
        """Like :meth:`map`, but with *flags* set for a read-only map."""

        return self.map(queue, map_flags.READ,
                is_blocking=is_blocking, wait_for=wait_for)

    def svm_map_rw(self, queue, is_blocking=True, wait_for=None):
        """Like :meth:`map`, but with *flags* set for a read-only map."""

        return self.map(queue, map_flags.READ | map_flags.WRITE,
                is_blocking=is_blocking, wait_for=wait_for)

    def svm__enqueue_unmap(self, queue, wait_for=None):
        return _cl._enqueue_svm_unmap(queue, self, wait_for)

    if get_cl_header_version() >= (2, 0):
        SVM.map = svm_map
        SVM.map_ro = svm_map_ro
        SVM.map_rw = svm_map_rw
        SVM._enqueue_unmap = svm__enqueue_unmap

    # }}}

    # ORDER DEPENDENCY: Some of the above may override get_info, the effect needs
    # to be visible through the attributes. So get_info attr creation needs to happen
    # after the overriding is complete.
    cls_to_info_cls = {
            _cl.Platform: (_cl.Platform.get_info, _cl.platform_info, []),
            _cl.Device: (_cl.Device.get_info, _cl.device_info,
                ["PLATFORM", "MAX_WORK_GROUP_SIZE", "MAX_COMPUTE_UNITS"]),
            _cl.Context: (_cl.Context.get_info, _cl.context_info, []),
            _cl.CommandQueue: (_cl.CommandQueue.get_info, _cl.command_queue_info,
                ["CONTEXT", "DEVICE"]),
            _cl.Event: (_cl.Event.get_info, _cl.event_info, []),
            _cl.MemoryObjectHolder:
            (MemoryObjectHolder.get_info, _cl.mem_info, []),
            Image: (_cl.Image.get_image_info, _cl.image_info, []),
            Pipe: (_cl.Pipe.get_pipe_info, _cl.pipe_info, []),
            Program: (Program.get_info, _cl.program_info, []),
            Kernel: (Kernel.get_info, _cl.kernel_info, []),
            _cl.Sampler: (Sampler.get_info, _cl.sampler_info, []),
            }

    def to_string(cls, value, default_format=None):
        if cls._is_bitfield:
            names = []
            for name in dir(cls):
                attr = getattr(cls, name)
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

    for cls in CONSTANT_CLASSES:
        cls._is_bitfield = cls in BITFIELD_CONSTANT_CLASSES
        cls.to_string = classmethod(to_string)

    # {{{ get_info attributes -------------------------------------------------

    def make_getinfo(info_method, info_name, info_attr):
        def result(self):
            return info_method(self, info_attr)

        return property(result)

    def make_cacheable_getinfo(info_method, info_name, cache_attr, info_attr):
        def result(self):
            try:
                return getattr(self, cache_attr)
            except AttributeError:
                pass

            result = info_method(self, info_attr)
            setattr(self, cache_attr, result)
            return result

        return property(result)

    for cls, (info_method, info_class, cacheable_attrs) \
            in cls_to_info_cls.items():
        for info_name, _info_value in info_class.__dict__.items():
            if info_name == "to_string" or info_name.startswith("_"):
                continue

            info_lower = info_name.lower()
            info_constant = getattr(info_class, info_name)
            if info_name in cacheable_attrs:
                cache_attr = intern("_info_cache_"+info_lower)
                setattr(cls, info_lower, make_cacheable_getinfo(
                    info_method, info_lower, cache_attr, info_constant))
            else:
                setattr(cls, info_lower, make_getinfo(
                        info_method, info_name, info_constant))

    # }}}

    if _cl.have_gl():
        def gl_object_get_gl_object(self):
            return self.get_gl_object_info()[1]

        GLBuffer.gl_object = property(gl_object_get_gl_object)
        GLTexture.gl_object = property(gl_object_get_gl_object)


_add_functionality()

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

class SVMMap:
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
    def __init__(self, svm, array, queue, event):
        self.svm = svm
        self.array = array
        self.queue = queue
        self.event = event

    def __del__(self):
        if self.svm is not None:
            self.release()

    def __enter__(self):
        return self.array

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def release(self, queue=None, wait_for=None):
        """
        :arg queue: a :class:`pyopencl.CommandQueue`. Defaults to the one
            with which the map was created, if not specified.
        :returns: a :class:`pyopencl.Event`

        |std-enqueue-blurb|
        """

        evt = self.svm._enqueue_unmap(self.queue)
        self.svm = None

        return evt

# }}}


# {{{ enqueue_copy

_IMAGE_MEM_OBJ_TYPES = [mem_object_type.IMAGE2D, mem_object_type.IMAGE3D]
if get_cl_header_version() >= (1, 2):
    _IMAGE_MEM_OBJ_TYPES.append(mem_object_type.IMAGE2D_ARRAY)


def enqueue_copy(queue, dest, src, **kwargs):
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
        dest: MemoryObject | SVMPointer,
        pattern: Any, size: int, *, offset: int = 0,
        wait_for: Sequence[Event] | None = None) -> Event:
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


def image_from_array(ctx, ary, num_channels=None, mode="r", norm_int=False):
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

def enqueue_marker(queue, wait_for=None):
    if queue._get_cl_version() >= (1, 2) and get_cl_header_version() >= (1, 2):
        return _cl._enqueue_marker_with_wait_list(queue, wait_for)
    else:
        if wait_for:
            _cl._enqueue_wait_for_events(queue, wait_for)
        return _cl._enqueue_marker(queue)


def enqueue_barrier(queue, wait_for=None):
    if queue._get_cl_version() >= (1, 2) and get_cl_header_version() >= (1, 2):
        return _cl._enqueue_barrier_with_wait_list(queue, wait_for)
    else:
        _cl._enqueue_barrier(queue)
        if wait_for:
            _cl._enqueue_wait_for_events(queue, wait_for)
        return _cl._enqueue_marker(queue)


def enqueue_fill_buffer(queue, mem, pattern, offset, size, wait_for=None):
    if not (queue._get_cl_version() >= (1, 2) and get_cl_header_version() >= (1, 2)):
        warn(
            "The context for this queue does not declare OpenCL 1.2 support, so "
            "the next thing you might see is a crash",
            stacklevel=2)

    if _PYPY and isinstance(pattern, np.generic):
        pattern = np.asarray(pattern)

    return _cl._enqueue_fill_buffer(queue, mem, pattern, offset, size, wait_for)

# }}}


# {{{ numpy-like svm allocation

def enqueue_svm_memfill(queue, dest, pattern, byte_count=None, wait_for=None):
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


def enqueue_svm_migratemem(queue, svms, flags, wait_for=None):
    """
    :arg svms: a collection of Python buffer objects (e.g. :mod:`numpy`
        arrays), or any implementation of :class:`SVMPointer`.
    :arg flags: a combination of :class:`mem_migration_flags`

    |std-enqueue-blurb|

    .. versionadded:: 2016.2

    This function requires OpenCL 2.1.
    """

    return _cl._enqueue_svm_migratemem(queue, svms, flags, wait_for)


def svm_empty(ctx, flags, shape, dtype, order="C", alignment=None, queue=None):
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


def svm_empty_like(ctx, flags, ary, alignment=None):
    """Allocate an empty :class:`numpy.ndarray` like the existing
    :class:`numpy.ndarray` *ary*.  The array will be allocated in shared
    virtual memory belonging to *ctx*.

    :arg ctx: a :class:`Context`
    :arg flags: a combination of flags from :class:`svm_mem_flags`.
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


def csvm_empty(ctx, shape, dtype, order="C", alignment=None):
    """
    Like :func:`svm_empty`, but with *flags* set for a coarse-grain read-write
    buffer.

    .. versionadded:: 2016.2
    """
    return svm_empty(ctx, svm_mem_flags.READ_WRITE, shape, dtype, order, alignment)


def csvm_empty_like(ctx, ary, alignment=None):
    """
    Like :func:`svm_empty_like`, but with *flags* set for a coarse-grain
    read-write buffer.

    .. versionadded:: 2016.2
    """
    return svm_empty_like(ctx, svm_mem_flags.READ_WRITE, ary)


def fsvm_empty(ctx, shape, dtype, order="C", alignment=None):
    """
    Like :func:`svm_empty`, but with *flags* set for a fine-grain read-write
    buffer.

    .. versionadded:: 2016.2
    """
    return svm_empty(ctx,
            svm_mem_flags.READ_WRITE | svm_mem_flags.SVM_FINE_GRAIN_BUFFER,
            shape, dtype, order, alignment)


def fsvm_empty_like(ctx, ary, alignment=None):
    """
    Like :func:`svm_empty_like`, but with *flags* set for a fine-grain
    read-write buffer.

    .. versionadded:: 2016.2
    """
    return svm_empty_like(
            ctx,
            svm_mem_flags.READ_WRITE | svm_mem_flags.SVM_FINE_GRAIN_BUFFER,
            ary)

# }}}


_KERNEL_ARG_CLASSES: tuple[type, ...] = (
        MemoryObjectHolder,
        Sampler,
        CommandQueue,
        LocalMemory,
        )
if get_cl_header_version() >= (2, 0):
    _KERNEL_ARG_CLASSES = (*_KERNEL_ARG_CLASSES, SVM)


# vim: foldmethod=marker
