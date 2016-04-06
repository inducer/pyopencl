# -*- coding: utf-8 -*-

from __future__ import division, absolute_import, print_function

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

import re
import six
import sys
from six.moves import input, range, intern

from pyopencl.version import VERSION, VERSION_STATUS, VERSION_TEXT  # noqa
from pytools import Record as _Record

try:
    import pyopencl.cffi_cl as _cl
except ImportError:
    import os
    from os.path import dirname, join, realpath
    if realpath(join(os.getcwd(), "pyopencl")) == realpath(dirname(__file__)):
        from warnings import warn
        warn("It looks like you are importing PyOpenCL from "
                "its source directory. This likely won't work.")
    raise

_CPY2 = _cl._CPY2
_CPY26 = _cl._CPY2 and sys.version_info < (2, 7)

import numpy as np

from pyopencl.cffi_cl import (  # noqa
        get_cl_header_version,
        program_kind,
        status_code,
        platform_info,
        device_type,
        device_info,
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
        addressing_mode,
        filter_mode,
        sampler_info,
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

        event_info,
        command_type,
        command_execution_status,
        profiling_info,
        mem_migration_flags,
        mem_migration_flags_ext,
        device_partition_property,
        device_affinity_domain,
        gl_object_type,
        gl_texture_info,
        migrate_mem_object_flags_ext,

        Error, MemoryError, LogicError, RuntimeError,

        Platform,
        get_platforms,
        unload_platform_compiler,

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
        UserEvent,

        enqueue_nd_range_kernel,
        enqueue_task,

        _enqueue_marker_with_wait_list,
        _enqueue_marker,
        _enqueue_barrier_with_wait_list,

        enqueue_migrate_mem_objects,
        enqueue_migrate_mem_object_ext,

        _enqueue_barrier_with_wait_list,
        _enqueue_read_buffer,
        _enqueue_write_buffer,
        _enqueue_copy_buffer,
        _enqueue_read_buffer_rect,
        _enqueue_write_buffer_rect,
        _enqueue_copy_buffer_rect,

        enqueue_map_buffer,
        _enqueue_fill_buffer,
        _enqueue_read_image,
        _enqueue_copy_image,
        _enqueue_write_image,
        enqueue_map_image,
        enqueue_fill_image,
        _enqueue_copy_image_to_buffer,
        _enqueue_copy_buffer_to_image,

        have_gl,
        _GLObject,
        GLBuffer,
        GLRenderBuffer,

        ImageFormat,
        get_supported_image_formats,

        ImageDescriptor,
        Image,
        Sampler,
        GLTexture,
        )

if _cl.have_gl():
    try:
        from pyopencl.cffi_cl import get_apple_cgl_share_group  # noqa
    except ImportError:
        pass

    try:
        from pyopencl.cffi_cl import (  # noqa
            enqueue_acquire_gl_objects,
            enqueue_release_gl_objects,
        )
    except ImportError:
        pass


import inspect as _inspect

CONSTANT_CLASSES = [
        getattr(_cl, name) for name in dir(_cl)
        if _inspect.isclass(getattr(_cl, name))
        and name[0].islower() and name not in ["zip", "map", "range"]]


# {{{ diagnostics

class CompilerWarning(UserWarning):
    pass


def compiler_output(text):
    import os
    from warnings import warn
    if int(os.environ.get("PYOPENCL_COMPILER_OUTPUT", "0")):
        warn(text, CompilerWarning)
    else:
        warn("Non-empty compiler output encountered. Set the "
                "environment variable PYOPENCL_COMPILER_OUTPUT=1 "
                "to see more.", CompilerWarning)


class _ErrorRecord(_Record):
    pass

# }}}


# {{{ arg packing helpers

_size_t_char = ({
    8: 'Q',
    4: 'L',
    2: 'H',
    1: 'B',
})[_cl._ffi.sizeof('size_t')]
_type_char_map = {
    'n': _size_t_char.lower(),
    'N': _size_t_char
}
del _size_t_char

# }}}


# {{{ find pyopencl shipped source code

def _find_pyopencl_include_path():
    from pkg_resources import Requirement, resource_filename, DistributionNotFound
    try:
        # Try to find the resource with pkg_resources (the recommended
        # setuptools approach)
        return resource_filename(Requirement.parse("pyopencl2"), "pyopencl/cl")
    except DistributionNotFound:
        # If pkg_resources can't find it (e.g. if the module is part of a
        # frozen application), try to find the include path in the same
        # directory as this file
        from os.path import join, abspath, dirname, exists

        include_path = join(abspath(dirname(__file__)), "cl")
        # If that doesn't exist, just re-raise the exception caught from
        # resource_filename.
        if not exists(include_path):
            raise

        return include_path

# }}}


# {{{ Program (including caching support)

_DEFAULT_BUILD_OPTIONS = []
_DEFAULT_INCLUDE_OPTIONS = ["-I", _find_pyopencl_include_path()]

# map of platform.name to build options list
_PLAT_BUILD_OPTIONS = {}


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
        import os
        os.environ["CPU_MAX_COMPUTE_UNITS"] = "1"
    else:
        from warnings import warn
        warn("do not know how to enable debugging on '%s'"
                % platform.name)


class Program(object):
    def __init__(self, arg1, arg2=None, arg3=None):
        if arg2 is None:
            # 1-argument form: program
            self._prg = arg1

        elif arg3 is None:
            # 2-argument form: context, source
            context, source = arg1, arg2

            import sys
            if isinstance(source, six.text_type) and sys.version_info < (3,):
                from warnings import warn
                warn("Received OpenCL source code in Unicode, "
                     "should be ASCII string. Attempting conversion.",
                     stacklevel=2)
                source = source.encode()

            self._context = context
            self._source = source
            self._prg = None

        else:
            context, device, binaries = arg1, arg2, arg3
            self._context = context
            self._prg = _cl._Program(context, device, binaries)

    def _get_prg(self):
        if self._prg is not None:
            return self._prg
        else:
            # "no program" can only happen in from-source case.
            from warnings import warn
            warn("Pre-build attribute access defeats compiler caching.",
                    stacklevel=3)

            self._prg = _cl._Program(self._context, self._source)
            del self._context
            return self._prg

    def get_info(self, arg):
        return self._get_prg().get_info(arg)

    def get_build_info(self, *args, **kwargs):
        return self._get_prg().get_build_info(*args, **kwargs)

    def all_kernels(self):
        return self._get_prg().all_kernels()

    def int_ptr(self):
        return self._get_prg().int_ptr
    int_ptr = property(int_ptr, doc=_cl._Program.int_ptr.__doc__)

    def from_int_ptr(int_ptr_value):
        return Program(_cl._Program.from_int_ptr(int_ptr_value))
    from_int_ptr.__doc__ = _cl._Program.from_int_ptr.__doc__
    from_int_ptr = staticmethod(from_int_ptr)

    def __getattr__(self, attr):
        try:
            knl = Kernel(self, attr)
            # Nvidia does not raise errors even for invalid names,
            # but this will give an error if the kernel is invalid.
            knl.num_args
            knl._source = getattr(self, "_source", None)
            return knl
        except LogicError:
            raise AttributeError("'%s' was not found as a program "
                    "info attribute or as a kernel name" % attr)

    # {{{ build

    if six.PY3:
        _find_unsafe_re_opts = re.ASCII
    else:
        _find_unsafe_re_opts = 0

    _find_unsafe = re.compile(br'[^\w@%+=:,./-]', _find_unsafe_re_opts).search

    @classmethod
    def _shlex_quote(cls, s):
        """Return a shell-escaped version of the string *s*."""

        # Stolen from https://hg.python.org/cpython/file/default/Lib/shlex.py#l276

        if not s:
            return "''"

        if cls._find_unsafe(s) is None:
            return s

        # use single quotes, and put single quotes into double quotes
        # the string $'b is then quoted as '$'"'"'b'
        import sys
        if sys.platform.startswith("win"):
            # not sure how to escape that
            assert b'"' not in s
            return b'"' + s + b'"'
        else:
            return b"'" + s.replace(b"'", b"'\"'\"'") + b"'"

    @classmethod
    def _process_build_options(cls, context, options):
        if isinstance(options, six.string_types):
            import shlex
            if six.PY2:
                # shlex.split takes bytes (py2 str) on py2
                if isinstance(options, six.text_type):
                    options = options.encode("utf-8")
            else:
                # shlex.split takes unicode (py3 str) on py3
                if isinstance(options, six.binary_type):
                    options = options.decode("utf-8")

            options = shlex.split(options)

        def encode_if_necessary(s):
            if isinstance(s, six.text_type):
                return s.encode("utf-8")
            else:
                return s

        options = (options
                + _DEFAULT_BUILD_OPTIONS
                + _DEFAULT_INCLUDE_OPTIONS
                + _PLAT_BUILD_OPTIONS.get(
                    context.devices[0].platform.name, []))

        import os
        forced_options = os.environ.get("PYOPENCL_BUILD_OPTIONS")
        if forced_options:
            options = options + forced_options.split()

        # {{{ find include path

        include_path = ["."]

        option_idx = 0
        while option_idx < len(options):
            option = options[option_idx].strip()
            if option.startswith("-I") or option.startswith("/I"):
                if len(option) == 2:
                    if option_idx+1 < len(options):
                        include_path.append(options[option_idx+1])
                    option_idx += 2
                else:
                    include_path.append(option[2:].lstrip())
                    option_idx += 1
            else:
                option_idx += 1

        # }}}

        options = [encode_if_necessary(s) for s in options]

        options = [cls._shlex_quote(s) for s in options]

        return b" ".join(options), include_path

    def build(self, options=[], devices=None, cache_dir=None):
        options_bytes, include_path = self._process_build_options(
                self._context, options)

        if cache_dir is None:
            cache_dir = getattr(self._context, 'cache_dir', None)

        import os
        if os.environ.get("PYOPENCL_NO_CACHE") and self._prg is None:
            self._prg = _cl._Program(self._context, self._source)

        if self._prg is not None:
            # uncached

            self._build_and_catch_errors(
                    lambda: self._prg.build(options_bytes, devices),
                    options_bytes=options_bytes)

        else:
            # cached

            from pyopencl.cache import create_built_program_from_source_cached
            self._prg = self._build_and_catch_errors(
                    lambda: create_built_program_from_source_cached(
                        self._context, self._source, options_bytes, devices,
                        cache_dir=cache_dir, include_path=include_path),
                    options_bytes=options_bytes, source=self._source)

            del self._context

        return self

    def _build_and_catch_errors(self, build_func, options_bytes, source=None):
        try:
            return build_func()
        except _cl.RuntimeError as e:
            what = e.what
            if options_bytes:
                what = what + "\n(options: %s)" % options_bytes.decode("utf-8")

            if source is not None:
                from tempfile import NamedTemporaryFile
                srcfile = NamedTemporaryFile(mode="wt", delete=False, suffix=".cl")
                try:
                    srcfile.write(source)
                finally:
                    srcfile.close()

                what = what + "\n(source saved as %s)" % srcfile.name

            code = e.code
            routine = e.routine

            err = _cl.RuntimeError(
                    _ErrorRecord(
                        what=lambda: what,
                        code=lambda: code,
                        routine=lambda: routine))

        # Python 3.2 outputs the whole list of currently active exceptions
        # This serves to remove one (redundant) level from that nesting.
        raise err

    # }}}

    def compile(self, options=[], devices=None, headers=[]):
        options_bytes, _ = self._process_build_options(self._context, options)

        return self._prg.compile(options_bytes, devices, headers)

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


def link_program(context, programs, options=[], devices=None):
    options_bytes, _ = Program._process_build_options(context, options)
    return Program(_Program.link(context, programs, options_bytes, devices))

# }}}


def _add_functionality():
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
            Program: (Program.get_info, _cl.program_info, []),
            Kernel: (Kernel.get_info, _cl.kernel_info, []),
            _cl.Sampler: (Sampler.get_info, _cl.sampler_info, []),
            }

    def to_string(cls, value, default_format=None):
        for name in dir(cls):
            if (not name.startswith("_") and getattr(cls, name) == value):
                return name

        if default_format is None:
            raise ValueError("a name for value %d was not found in %s"
                    % (value, cls.__name__))
        else:
            return default_format % value

    for cls in CONSTANT_CLASSES:
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
            in six.iteritems(cls_to_info_cls):
        for info_name, info_value in six.iteritems(info_class.__dict__):
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

    # {{{ Platform

    def platform_repr(self):
        return "<pyopencl.Platform '%s' at 0x%x>" % (self.name, self.int_ptr)

    def platform_get_cl_version(self):
        import re
        version_string = self.version
        match = re.match(r"^OpenCL ([0-9]+)\.([0-9]+) .*$", version_string)
        if match is None:
            raise RuntimeError("platform %s returned non-conformant "
                               "platform version string '%s'" %
                               (self, version_string))

        return int(match.group(1)), int(match.group(2))

    Platform.__repr__ = platform_repr
    Platform._get_cl_version = platform_get_cl_version

    # }}}

    # {{{ Device

    def device_repr(self):
        return "<pyopencl.Device '%s' on '%s' at 0x%x>" % (
                self.name.strip(), self.platform.name.strip(), self.int_ptr)

    def device_persistent_unique_id(self):
        return (self.vendor, self.vendor_id, self.name, self.version)

    Device.__repr__ = device_repr

    # undocumented for now:
    Device.persistent_unique_id = property(device_persistent_unique_id)

    # }}}

    # {{{ Context

    def context_repr(self):
        return "<pyopencl.Context at 0x%x on %s>" % (self.int_ptr,
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

    def command_queue_get_cl_version(self):
        return self.context._get_cl_version()

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
            except:
                log = "<error retrieving log>"

            build_logs.append((dev, log))

        return build_logs

    def program_build(self, options_bytes, devices=None):
        err = None
        try:
            self._build(options=options_bytes, devices=devices)
        except Error as e:
            what = e.what + "\n\n" + (75*"="+"\n").join(
                    "Build on %s:\n\n%s" % (dev, log)
                    for dev, log in self._get_build_logs())
            code = e.code
            routine = e.routine

            err = _cl.RuntimeError(
                    _ErrorRecord(
                        what=lambda: what,
                        code=lambda: code,
                        routine=lambda: routine))

        if err is not None:
            # Python 3.2 outputs the whole list of currently active exceptions
            # This serves to remove one (redundant) level from that nesting.
            raise err

        message = (75*"="+"\n").join(
                "Build on %s succeeded, but said:\n\n%s" % (dev, log)
                for dev, log in self._get_build_logs()
                if log is not None and log.strip())

        if message:
            if self.kind() == program_kind.SOURCE:
                build_type = "From-source build"
            elif self.kind() == program_kind.BINARY:
                build_type = "From-binary build"
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
            except AttributeError:
                raise AttributeError("%s has no attribute '%s'"
                        % (type(self), name))
            else:
                return self.event.get_profiling_info(inf_attr)

    _cl.Event.profile = property(ProfilingInfoGetter)

    # }}}

    # {{{ Kernel

    kernel_old_init = Kernel.__init__
    kernel_old_get_work_group_info = Kernel.get_work_group_info

    def kernel_init(self, prg, name):
        if not isinstance(prg, _cl._Program):
            prg = prg._get_prg()

        kernel_old_init(self, prg, name)

        self._setup(prg)

    def kernel__setup(self, prg):
        self._source = getattr(prg, "_source", None)

        self._generate_naive_call()
        self._wg_info_cache = {}
        return self

    def kernel_get_work_group_info(self, param, device):
        try:
            return self._wg_info_cache[param, device]
        except KeyError:
            pass

        result = kernel_old_get_work_group_info(self, param, device)
        self._wg_info_cache[param, device] = result
        return result

    # {{{ code generation for __call__, set_args

    def kernel__set_set_args_body(self, body, num_passed_args):
        from pytools.py_codegen import (
                PythonFunctionGenerator,
                PythonCodeGenerator,
                Indentation)

        arg_names = ["arg%d" % i for i in range(num_passed_args)]

        # {{{ wrap in error handler

        err_gen = PythonCodeGenerator()

        def gen_error_handler():
            err_gen("""
                if current_arg is not None:
                    args = [{args}]
                    advice = ""
                    from pyopencl.array import Array
                    if isinstance(args[current_arg], Array):
                        advice = " (perhaps you meant to pass 'array.data' " \
                            "instead of the array itself?)"

                    raise _cl.LogicError(
                            "when processing argument #%d (1-based): %s%s"
                            % (current_arg+1, str(e), advice))
                else:
                    raise
                """
                .format(args=", ".join(arg_names)))
            err_gen("")

        err_gen("try:")
        with Indentation(err_gen):
            err_gen.extend(body)
        err_gen("except TypeError as e:")
        with Indentation(err_gen):
            gen_error_handler()
        err_gen("except _cl.LogicError as e:")
        with Indentation(err_gen):
            gen_error_handler()

        # }}}

        def add_preamble(gen):
            gen.add_to_preamble(
                "import numpy as np")
            gen.add_to_preamble(
                "import pyopencl.cffi_cl as _cl")
            gen.add_to_preamble(
                "from pyopencl.cffi_cl import _lib, "
                "_ffi, _handle_error, _CLKernelArg")
            gen.add_to_preamble("from pyopencl import status_code")
            gen.add_to_preamble("from struct import pack")
            gen.add_to_preamble("")

        # {{{ generate _enqueue

        gen = PythonFunctionGenerator("enqueue_knl_%s" % self.function_name,
                ["self", "queue", "global_size", "local_size"]
                + arg_names
                + ["global_offset=None", "g_times_l=None", "wait_for=None"])

        add_preamble(gen)
        gen.extend(err_gen)

        gen("""
            return _cl.enqueue_nd_range_kernel(queue, self, global_size, local_size,
                    global_offset, wait_for, g_times_l=g_times_l)
            """)

        self._enqueue = gen.get_function()

        # }}}

        # {{{ generate set_args

        gen = PythonFunctionGenerator("_set_args", ["self"] + arg_names)

        add_preamble(gen)
        gen.extend(err_gen)

        self._set_args = gen.get_function()

        # }}}

    def kernel__generate_buffer_arg_setter(self, gen, arg_idx, buf_var):
        from pytools.py_codegen import Indentation

        if _CPY2:
            # https://github.com/numpy/numpy/issues/5381
            gen("if isinstance({buf_var}, np.generic):".format(buf_var=buf_var))
            with Indentation(gen):
                gen("{buf_var} = np.getbuffer({buf_var})".format(buf_var=buf_var))

        gen("""
            c_buf, sz, _ = _cl._c_buffer_from_obj({buf_var})
            status = _lib.kernel__set_arg_buf(self.ptr, {arg_idx}, c_buf, sz)
            if status != _ffi.NULL:
                _handle_error(status)
            """
            .format(arg_idx=arg_idx, buf_var=buf_var))

    def kernel__generate_bytes_arg_setter(self, gen, arg_idx, buf_var):
        gen("""
            status = _lib.kernel__set_arg_buf(self.ptr, {arg_idx},
                {buf_var}, len({buf_var}))
            if status != _ffi.NULL:
                _handle_error(status)
            """
            .format(arg_idx=arg_idx, buf_var=buf_var))

    def kernel__generate_generic_arg_handler(self, gen, arg_idx, arg_var):
        from pytools.py_codegen import Indentation

        gen("""
            if {arg_var} is None:
                status = _lib.kernel__set_arg_null(self.ptr, {arg_idx})
                if status != _ffi.NULL:
                    _handle_error(status)
            elif isinstance({arg_var}, _CLKernelArg):
                self.set_arg({arg_idx}, {arg_var})
            """
            .format(arg_idx=arg_idx, arg_var=arg_var))

        gen("else:")
        with Indentation(gen):
            self._generate_buffer_arg_setter(gen, arg_idx, arg_var)

    def kernel__generate_naive_call(self):
        num_args = self.num_args

        from pytools.py_codegen import PythonCodeGenerator
        gen = PythonCodeGenerator()

        if num_args == 0:
            gen("pass")

        for i in range(num_args):
            gen("# process argument {arg_idx}".format(arg_idx=i))
            gen("")
            gen("current_arg = {arg_idx}".format(arg_idx=i))
            self._generate_generic_arg_handler(gen, i, "arg%d" % i)
            gen("")

        self._set_set_args_body(gen, num_args)

    def kernel_set_scalar_arg_dtypes(self, scalar_arg_dtypes):
        self._scalar_arg_dtypes = scalar_arg_dtypes

        # {{{ arg counting bug handling

        # For example:
        # https://github.com/pocl/pocl/issues/197
        # (but Apple CPU has a similar bug)

        work_around_arg_count_bug = False
        warn_about_arg_count_bug = False

        from pyopencl.characterize import has_struct_arg_count_bug

        count_bug_per_dev = [
                has_struct_arg_count_bug(dev)
                for dev in self.context.devices]

        from pytools import single_valued
        if any(count_bug_per_dev):
            if all(count_bug_per_dev):
                work_around_arg_count_bug = single_valued(count_bug_per_dev)
            else:
                warn_about_arg_count_bug = True

        fp_arg_count = 0

        # }}}

        cl_arg_idx = 0

        from pytools.py_codegen import PythonCodeGenerator
        gen = PythonCodeGenerator()

        if not scalar_arg_dtypes:
            gen("pass")

        for arg_idx, arg_dtype in enumerate(scalar_arg_dtypes):
            gen("# process argument {arg_idx}".format(arg_idx=arg_idx))
            gen("")
            gen("current_arg = {arg_idx}".format(arg_idx=arg_idx))
            arg_var = "arg%d" % arg_idx

            if arg_dtype is None:
                self._generate_generic_arg_handler(gen, cl_arg_idx, arg_var)
                cl_arg_idx += 1
                gen("")
                continue

            arg_dtype = np.dtype(arg_dtype)

            if arg_dtype.char == "V":
                self._generate_generic_arg_handler(gen, cl_arg_idx, arg_var)
                cl_arg_idx += 1

            elif arg_dtype.kind == "c":
                if warn_about_arg_count_bug:
                    warn("{knl_name}: arguments include complex numbers, and "
                            "some (but not all) of the target devices mishandle "
                            "struct kernel arguments (hence the workaround is "
                            "disabled".format(
                                knl_name=self.function_name, stacklevel=2))

                if arg_dtype == np.complex64:
                    arg_char = "f"
                elif arg_dtype == np.complex128:
                    arg_char = "d"
                else:
                    raise TypeError("unexpected complex type: %s" % arg_dtype)

                if (work_around_arg_count_bug == "pocl"
                        and arg_dtype == np.complex128
                        and fp_arg_count + 2 <= 8):
                    gen(
                            "buf = pack('{arg_char}', {arg_var}.real)"
                            .format(arg_char=arg_char, arg_var=arg_var))
                    self._generate_bytes_arg_setter(gen, cl_arg_idx, "buf")
                    cl_arg_idx += 1
                    gen("current_arg = current_arg + 1000")
                    gen(
                            "buf = pack('{arg_char}', {arg_var}.imag)"
                            .format(arg_char=arg_char, arg_var=arg_var))
                    self._generate_bytes_arg_setter(gen, cl_arg_idx, "buf")
                    cl_arg_idx += 1

                elif (work_around_arg_count_bug == "apple"
                        and arg_dtype == np.complex128
                        and fp_arg_count + 2 <= 8):
                    raise NotImplementedError("No work-around to "
                            "Apple's broken structs-as-kernel arg "
                            "handling has been found. "
                            "Cannot pass complex numbers to kernels.")

                else:
                    gen(
                            "buf = pack('{arg_char}{arg_char}', "
                            "{arg_var}.real, {arg_var}.imag)"
                            .format(arg_char=arg_char, arg_var=arg_var))
                    self._generate_bytes_arg_setter(gen, cl_arg_idx, "buf")
                    cl_arg_idx += 1

                fp_arg_count += 2

            elif arg_dtype.char in "IL" and _CPY26:
                # Prevent SystemError: ../Objects/longobject.c:336: bad
                # argument to internal function

                gen(
                        "buf = pack('{arg_char}', long({arg_var}))"
                        .format(arg_char=arg_dtype.char, arg_var=arg_var))
                self._generate_bytes_arg_setter(gen, cl_arg_idx, "buf")
                cl_arg_idx += 1

            else:
                if arg_dtype.kind == "f":
                    fp_arg_count += 1

                arg_char = arg_dtype.char
                arg_char = _type_char_map.get(arg_char, arg_char)
                gen(
                        "buf = pack('{arg_char}', {arg_var})"
                        .format(
                            arg_char=arg_char,
                            arg_var=arg_var))
                self._generate_bytes_arg_setter(gen, cl_arg_idx, "buf")
                cl_arg_idx += 1

            gen("")

        if cl_arg_idx != self.num_args:
            raise TypeError(
                "length of argument list (%d) and "
                "CL-generated number of arguments (%d) do not agree"
                % (cl_arg_idx, self.num_args))

        self._set_set_args_body(gen, len(scalar_arg_dtypes))

    # }}}

    def kernel_set_args(self, *args, **kwargs):
        # Need to dupicate the 'self' argument for dynamically generated  method
        return self._set_args(self, *args, **kwargs)

    def kernel_call(self, queue, global_size, local_size, *args, **kwargs):
        # __call__ can't be overridden directly, so we need this
        # trampoline hack.
        return self._enqueue(self, queue, global_size, local_size, *args, **kwargs)

    def kernel_capture_call(self, filename, queue, global_size, local_size,
            *args, **kwargs):
        from pyopencl.capture_call import capture_kernel_call
        capture_kernel_call(self, filename, queue, global_size, local_size,
                *args, **kwargs)

    Kernel.__init__ = kernel_init
    Kernel._setup = kernel__setup
    Kernel.get_work_group_info = kernel_get_work_group_info
    Kernel._set_set_args_body = kernel__set_set_args_body
    Kernel._generate_buffer_arg_setter = kernel__generate_buffer_arg_setter
    Kernel._generate_bytes_arg_setter = kernel__generate_bytes_arg_setter
    Kernel._generate_generic_arg_handler = kernel__generate_generic_arg_handler
    Kernel._generate_naive_call = kernel__generate_naive_call
    Kernel.set_scalar_arg_dtypes = kernel_set_scalar_arg_dtypes
    Kernel.set_args = kernel_set_args
    Kernel.__call__ = kernel_call
    Kernel.capture_call = kernel_capture_call

    # }}}

    # {{{ ImageFormat

    def image_format_repr(self):
        return "ImageFormat(%s, %s)" % (
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

    image_old_init = Image.__init__

    def image_init(self, context, flags, format, shape=None, pitches=None,
            hostbuf=None, is_array=False, buffer=None):

        if shape is None and hostbuf is None:
            raise Error("'shape' must be passed if 'hostbuf' is not given")

        if shape is None and hostbuf is not None:
            shape = hostbuf.shape

        if hostbuf is not None and not \
                (flags & (mem_flags.USE_HOST_PTR | mem_flags.COPY_HOST_PTR)):
            from warnings import warn
            warn("'hostbuf' was passed, but no memory flags to make use of it.")

        if hostbuf is None and pitches is not None:
            raise Error("'pitches' may only be given if 'hostbuf' is given")

        if context._get_cl_version() >= (1, 2) and get_cl_header_version() >= (1, 2):
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

            desc = ImageDescriptor()

            desc.image_type = image_type
            desc.shape = shape  # also sets desc.array_size

            if pitches is None:
                desc.pitches = (0, 0)
            else:
                desc.pitches = pitches

            desc.num_mip_levels = 0  # per CL 1.2 spec
            desc.num_samples = 0  # per CL 1.2 spec
            desc.buffer = buffer

            image_old_init(self, context, flags, format, desc, hostbuf)
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

            image_old_init(self, context, flags, format, shape,
                    pitches, hostbuf)

    class _ImageInfoGetter:
        def __init__(self, event):
            from warnings import warn
            warn("Image.image.attr is deprecated. "
                    "Use Image.attr directly, instead.")

            self.event = event

        def __getattr__(self, name):
            try:
                inf_attr = getattr(_cl.image_info, name.upper())
            except AttributeError:
                raise AttributeError("%s has no attribute '%s'"
                        % (type(self), name))
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
        val = self.args[0]
        try:
            val.routine
        except AttributeError:
            return str(val)
        else:
            result = ""
            if val.code() != status_code.SUCCESS:
                result = status_code.to_string(
                        val.code(), "<unknown error %d>")
            routine = val.routine()
            if routine:
                result = "%s failed: %s" % (
                    routine.lower().replace("_", " "),
                    result)
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
        return self.args[0].what()

    Error.__str__ = error_str
    Error.code = property(error_code)
    Error.routine = property(error_routine)
    Error.what = property(error_what)

    # }}}

    if _cl.have_gl():
        def gl_object_get_gl_object(self):
            return self.get_gl_object_info()[1]

        GLBuffer.gl_object = property(gl_object_get_gl_object)
        GLTexture.gl_object = property(gl_object_get_gl_object)

_add_functionality()


# {{{ convenience

def create_some_context(interactive=None, answers=None, cache_dir=None):
    import os
    if answers is None:
        if "PYOPENCL_CTX" in os.environ:
            ctx_spec = os.environ["PYOPENCL_CTX"]
            answers = ctx_spec.split(":")

        if "PYOPENCL_TEST" in os.environ:
            from pyopencl.tools import get_test_platforms_and_devices
            for plat, devs in get_test_platforms_and_devices():
                for dev in devs:
                    return Context([dev], cache_dir=cache_dir)

    if answers is not None:
        pre_provided_answers = answers
        answers = answers[:]
    else:
        pre_provided_answers = None

    user_inputs = []

    if interactive is None:
        interactive = True
        try:
            import sys
            if not sys.stdin.isatty():
                interactive = False
        except:
            interactive = False

    def cc_print(s):
        if interactive:
            print(s)

    def get_input(prompt):
        if answers:
            return str(answers.pop(0))
        elif not interactive:
            return ''
        else:
            user_input = input(prompt)
            user_inputs.append(user_input)
            return user_input

    # {{{ pick a platform

    platforms = get_platforms()

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
                for i, pf in enumerate(platforms):
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
        for i, dev in enumerate(devices):
            if choice in dev.name.lower():
                return dev
        raise RuntimeError("input did not match any device")

    if not devices:
        raise Error("no devices found")
    elif len(devices) == 1:
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
                "create_some_context. (left over: '%s')" % ":".join(answers))

    return Context(devices, cache_dir=cache_dir)

_csc = create_some_context


def _mark_copy_deprecated(func):
    def new_func(*args, **kwargs):
        from warnings import warn
        warn("'%s' has been deprecated in version 2011.1. Please use "
                "enqueue_copy() instead." % func.__name__[1:], DeprecationWarning,
                stacklevel=2)
        return func(*args, **kwargs)

    try:
        from functools import update_wrapper
    except ImportError:
        pass
    else:
        try:
            update_wrapper(new_func, func)
        except AttributeError:
            pass

    return new_func


enqueue_read_image = _mark_copy_deprecated(_cl._enqueue_read_image)
enqueue_write_image = _mark_copy_deprecated(_cl._enqueue_write_image)
enqueue_copy_image = _mark_copy_deprecated(_cl._enqueue_copy_image)
enqueue_copy_image_to_buffer = _mark_copy_deprecated(
        _cl._enqueue_copy_image_to_buffer)
enqueue_copy_buffer_to_image = _mark_copy_deprecated(
        _cl._enqueue_copy_buffer_to_image)
enqueue_read_buffer = _mark_copy_deprecated(_cl._enqueue_read_buffer)
enqueue_write_buffer = _mark_copy_deprecated(_cl._enqueue_write_buffer)
enqueue_copy_buffer = _mark_copy_deprecated(_cl._enqueue_copy_buffer)


if _cl.get_cl_header_version() >= (1, 1):
    enqueue_read_buffer_rect = _mark_copy_deprecated(_cl._enqueue_read_buffer_rect)
    enqueue_write_buffer_rect = _mark_copy_deprecated(_cl._enqueue_write_buffer_rect)
    enqueue_copy_buffer_rect = _mark_copy_deprecated(_cl._enqueue_copy_buffer_rect)


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

        Two types of 'buffer' occur in the arguments to this function,
        :class:`Buffer` and 'host-side buffers'. The latter are
        defined by Python and commonly called `buffer objects
        <https://docs.python.org/3.4/c-api/buffer.html>`_.
        Make sure to always be clear on whether a :class:`Buffer` or a
        Python buffer object is needed.

    .. ------------------------------------------------------------------------
    .. rubric :: Transfer :class:`Buffer` ↔ host
    .. ------------------------------------------------------------------------

    :arg device_offset: offset in bytes (optional)

    .. note::

        The size of the transfer is controlled by the size of the
        of the host-side buffer. If the host-side buffer
        is a :class:`numpy.ndarray`, you can control the transfer size by
        transfering into a smaller 'view' of the target array, like this::

            cl.enqueue_copy(queue, large_dest_numpy_array[:15], src_buffer)

    .. ------------------------------------------------------------------------
    .. rubric :: Transfer :class:`Buffer` ↔ :class:`Buffer`
    .. ------------------------------------------------------------------------

    :arg byte_count: (optional) If not specified, defaults to the
        size of the source in versions 2012.x and earlier,
        and to the minimum of the size of the source and target
        from 2013.1 on.
    :arg src_offset: (optional)
    :arg dest_offset: (optional)

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

    |std-enqueue-blurb|

    .. versionadded:: 2011.1
    """

    if isinstance(dest, MemoryObjectHolder):
        if dest.type == mem_object_type.BUFFER:
            if isinstance(src, MemoryObjectHolder):
                if src.type == mem_object_type.BUFFER:
                    if "src_origin" in kwargs:
                        return _cl._enqueue_copy_buffer_rect(
                                queue, src, dest, **kwargs)
                    else:
                        kwargs["dst_offset"] = kwargs.pop("dest_offset", 0)
                        return _cl._enqueue_copy_buffer(queue, src, dest, **kwargs)
                elif src.type in [mem_object_type.IMAGE2D, mem_object_type.IMAGE3D]:
                    return _cl._enqueue_copy_image_to_buffer(
                            queue, src, dest, **kwargs)
                else:
                    raise ValueError("invalid src mem object type")
            else:
                # assume from-host
                if "buffer_origin" in kwargs:
                    return _cl._enqueue_write_buffer_rect(queue, dest, src, **kwargs)
                else:
                    return _cl._enqueue_write_buffer(queue, dest, src, **kwargs)

        elif dest.type in [mem_object_type.IMAGE2D, mem_object_type.IMAGE3D]:
            if isinstance(src, MemoryObjectHolder):
                if src.type == mem_object_type.BUFFER:
                    return _cl._enqueue_copy_buffer_to_image(
                            queue, src, dest, **kwargs)
                elif src.type in [mem_object_type.IMAGE2D, mem_object_type.IMAGE3D]:
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
        else:
            raise ValueError("invalid dest mem object type")

    else:
        # assume to-host

        if isinstance(src, MemoryObjectHolder):
            if src.type == mem_object_type.BUFFER:
                if "buffer_origin" in kwargs:
                    return _cl._enqueue_read_buffer_rect(queue, src, dest, **kwargs)
                else:
                    return _cl._enqueue_read_buffer(queue, src, dest, **kwargs)
            elif src.type in [mem_object_type.IMAGE2D, mem_object_type.IMAGE3D]:
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
        else:
            # assume from-host
            raise TypeError("enqueue_copy cannot perform host-to-host transfers")

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
    np.float16
except:
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

        from pyopencl.array import vec
        try:
            dtype, num_channels = vec.type_to_scalar_and_count[dtype]
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

    return Image(ctx, mode_flags | mem_flags.COPY_HOST_PTR,
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
        from warnings import warn
        warn("The context for this queue does not declare OpenCL 1.2 support, so "
                "the next thing you might see is a crash")
    return _cl._enqueue_fill_buffer(queue, mem, pattern, offset, size, wait_for)

# }}}


# vim: foldmethod=marker
