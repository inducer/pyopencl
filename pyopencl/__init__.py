# -*- coding: utf-8 -*-

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

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

from pyopencl.version import VERSION, VERSION_STATUS, VERSION_TEXT  # noqa

try:
    import pyopencl._cl as _cl
except ImportError:
    import os
    from os.path import dirname, join, realpath
    if realpath(join(os.getcwd(), "pyopencl")) == realpath(dirname(__file__)):
        from warnings import warn
        warn("It looks like you are importing PyOpenCL from "
                "its source directory. This likely won't work.")
    raise


import numpy as np
from pyopencl._cl import *  # noqa
import inspect as _inspect

CONSTANT_CLASSES = [
        getattr(_cl, name) for name in dir(_cl)
        if _inspect.isclass(getattr(_cl, name))
        and name[0].islower()]


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


# {{{ find pyopencl shipped source code

def _find_pyopencl_include_path():
    from pkg_resources import Requirement, resource_filename
    return resource_filename(Requirement.parse("pyopencl"), "pyopencl/cl")

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
            if isinstance(source, unicode) and sys.version_info < (3,):
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

    def build(self, options=[], devices=None, cache_dir=None):
        if isinstance(options, str):
            options = [options]

        options = (options
                + _DEFAULT_BUILD_OPTIONS
                + _DEFAULT_INCLUDE_OPTIONS
                + _PLAT_BUILD_OPTIONS.get(
                    self._context.devices[0].platform.name, []))

        import os
        forced_options = os.environ.get("PYOPENCL_BUILD_OPTIONS")
        if forced_options:
            options = options + forced_options.split()

        if os.environ.get("PYOPENCL_NO_CACHE") and self._prg is None:
            self._prg = _cl._Program(self._context, self._source)

        if self._prg is not None:
            # uncached

            self._build_and_catch_errors(
                    lambda: self._prg.build(" ".join(options), devices),
                    options=options)

        else:
            # cached

            from pyopencl.cache import create_built_program_from_source_cached
            self._prg = self._build_and_catch_errors(
                    lambda: create_built_program_from_source_cached(
                        self._context, self._source, options, devices,
                        cache_dir=cache_dir),
                    options=options, source=self._source)

            del self._context

        return self

    def _build_and_catch_errors(self, build_func, options, source=None):
        try:
            return build_func()
        except _cl.RuntimeError, e:
            from pytools import Record

            class ErrorRecord(Record):
                pass

            what = e.what
            if options:
                what = what + "\n(options: %s)" % " ".join(options)

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
                    ErrorRecord(
                        what=lambda: what,
                        code=lambda: code,
                        routine=lambda: routine))

        # Python 3.2 outputs the whole list of currently active exceptions
        # This serves to remove one (redundant) level from that nesting.
        raise err

    # }}}

    def compile(self, options=[], devices=None, headers=[]):
        options = " ".join(options)
        return self._prg.compile(options, devices, headers)

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
    options = " ".join(options)
    return Program(_Program.link(context, programs, options, devices))

# }}}


def _add_functionality():
    cls_to_info_cls = {
            _cl.Platform:
                (_cl.Platform.get_info, _cl.platform_info),
            _cl.Device:
                (_cl.Device.get_info, _cl.device_info),
            _cl.Context:
                (_cl.Context.get_info, _cl.context_info),
            _cl.CommandQueue:
                (_cl.CommandQueue.get_info, _cl.command_queue_info),
            _cl.Event:
                (_cl.Event.get_info, _cl.event_info),
            _cl.MemoryObjectHolder:
                (MemoryObjectHolder.get_info, _cl.mem_info),
            Image:
                (_cl.Image.get_image_info, _cl.image_info),
            Program:
                (Program.get_info, _cl.program_info),
            Kernel:
                (Kernel.get_info, _cl.kernel_info),
            _cl.Sampler:
                (Sampler.get_info, _cl.sampler_info),
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

    def make_getinfo(info_method, info_attr):
        def result(self):
            return info_method(self, info_attr)

        return property(result)

    for cls, (info_method, info_class) in cls_to_info_cls.iteritems():
        for info_name, info_value in info_class.__dict__.iteritems():
            if info_name == "to_string" or info_name.startswith("_"):
                continue

            setattr(cls, info_name.lower(), make_getinfo(
                    info_method, getattr(info_class, info_name)))

    # }}}

    # {{{ Platform

    def platform_repr(self):
        return "<pyopencl.Platform '%s' at 0x%x>" % (self.name, self.int_ptr)

    Platform.__repr__ = platform_repr

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
        import re
        platform = self.devices[0].platform
        plat_version_string = platform.version
        match = re.match(r"^OpenCL ([0-9]+)\.([0-9]+) .*$",
                plat_version_string)
        if match is None:
            raise RuntimeError("platform %s returned non-conformant "
                    "platform version string '%s'" % (platform, plat_version_string))

        return int(match.group(1)), int(match.group(2))

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

    def program_build(self, options=[], devices=None):
        if isinstance(options, list):
            options = " ".join(options)

        err = None
        try:
            self._build(options=options, devices=devices)
        except Exception, e:
            from pytools import Record

            class ErrorRecord(Record):
                pass

            what = e.what + "\n\n" + (75*"="+"\n").join(
                    "Build on %s:\n\n%s" % (dev, log)
                    for dev, log in self._get_build_logs())
            code = e.code
            routine = e.routine

            err = _cl.RuntimeError(
                    ErrorRecord(
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

    def kernel_init(self, prg, name):
        if not isinstance(prg, _cl._Program):
            prg = prg._get_prg()

        kernel_old_init(self, prg, name)
        self._source = getattr(prg, "_source", None)

    def kernel_call(self, queue, global_size, local_size, *args, **kwargs):
        global_offset = kwargs.pop("global_offset", None)
        g_times_l = kwargs.pop("g_times_l", False)
        wait_for = kwargs.pop("wait_for", None)

        if kwargs:
            raise TypeError(
                    "Kernel.__call__ recived unexpected keyword arguments: %s"
                    % ", ".join(kwargs.keys()))

        self.set_args(*args)

        return enqueue_nd_range_kernel(queue, self, global_size, local_size,
                global_offset, wait_for, g_times_l=g_times_l)

    def kernel_set_scalar_arg_dtypes(self, arg_dtypes):
        assert len(arg_dtypes) == self.num_args, (
                "length of argument type array (%d) and "
                "CL-generated number of arguments (%d) do not agree"
                % (len(arg_dtypes), self.num_args))

        arg_type_chars = []

        for arg_dtype in arg_dtypes:
            if arg_dtype is None:
                arg_type_chars.append(None)
            else:
                arg_type_chars.append(np.dtype(arg_dtype).char)

        self._arg_type_chars = arg_type_chars

    def kernel_set_args(self, *args):
        assert len(args) == self.num_args, (
                "length of argument list (%d) and "
                "CL-generated number of arguments (%d) do not agree"
                % (len(args), self.num_args))

        i = None
        try:
            try:
                arg_type_chars = self.__dict__["_arg_type_chars"]
            except KeyError:
                for i, arg in enumerate(args):
                    self.set_arg(i, arg)
            else:
                from pyopencl._pvt_struct import pack

                for i, (arg, arg_type_char) in enumerate(
                        zip(args, arg_type_chars)):
                    if arg_type_char and arg_type_char != "V":
                        self.set_arg(i, pack(arg_type_char, arg))
                    else:
                        self.set_arg(i, arg)
        except LogicError, e:
            if i is not None:
                advice = ""
                from pyopencl.array import Array
                if isinstance(args[i], Array):
                    advice = " (perhaps you meant to pass 'array.data' " \
                        "instead of the array itself?)"

                raise LogicError(
                        "when processing argument #%d (1-based): %s%s"
                        % (i+1, str(e), advice))
            else:
                raise

    def kernel_capture_call(self, filename, queue, global_size, local_size,
            *args, **kwargs):
        from pyopencl.capture_call import capture_kernel_call
        capture_kernel_call(self, filename, queue, global_size, local_size,
                *args, **kwargs)

    Kernel.__init__ = kernel_init
    Kernel.__call__ = kernel_call
    Kernel.set_scalar_arg_dtypes = kernel_set_scalar_arg_dtypes
    Kernel.set_args = kernel_set_args
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
            #if num_mip_levels is not None:
                #raise TypeError(
                #      "'num_mip_levels' argument is not supported for CL < 1.2")
            #if num_samples is not None:
                #raise TypeError(
                #       "'num_samples' argument is not supported for CL < 1.2")
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
            result = "%s failed: %s" % (val.routine(),
                    status_code.to_string(val.code(), "<unknown error %d>")
                    .lower().replace("_", " "))
            if val.what():
                result += " - " + val.what()
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

def create_some_context(interactive=None, answers=None):
    import os
    if answers is None and "PYOPENCL_CTX" in os.environ:
        ctx_spec = os.environ["PYOPENCL_CTX"]
        answers = ctx_spec.split(":")

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
            print s

    def get_input(prompt):
        if answers:
            return str(answers.pop(0))
        elif not interactive:
            return ''
        else:
            user_input = raw_input(prompt)
            user_inputs.append(user_input)
            return user_input

    # {{{ pick a platform

    platforms = get_platforms()

    if not platforms:
        raise Error("no platforms found")
    elif len(platforms) == 1:
        platform, = platforms
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

    return Context(devices)

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
    DTYPE_TO_CHANNEL_TYPE[np.dtype(np.float16)] = channel_type.HALF_FLOAT,

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
    return _cl._enqueue_fill_buffer(queue, mem, pattern, offset,
            size, wait_for=None)



# }}}


# vim: foldmethod=marker
