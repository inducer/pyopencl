from pyopencl.version import VERSION, VERSION_STATUS, VERSION_TEXT

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
from pyopencl._cl import *
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



# {{{ Kernel

class Kernel(_cl._Kernel):
    def __init__(self, prg, name):
        _cl._Kernel.__init__(self, prg._get_prg(), name)

# }}}

# {{{ Program (including caching support)

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
                source = str(source)

            self._context = context
            self._source = source
            self._prg = None

        else:
            # 3-argument form: context, devices, binaries
            self._prg = _cl._Program(arg1, arg2, arg3)

    def _get_prg(self):
        if self._prg is not None:
            return self._prg
        else:
            # "no program" can only happen in from-source case.
            from warnings import warn
            warn("Pre-build attribute access defeats compiler caching.", stacklevel=3)

            self._prg = _cl._Program(self._context, self._source)
            del self._context
            del self._source
            return self._prg

    def get_info(self, arg):
        return self._get_prg().get_info(arg)

    def get_build_info(self, *args, **kwargs):
        return self._get_prg().get_build_info(*args, **kwargs)

    def all_kernels(self):
        return self._get_prg().all_kernels()

    def __getattr__(self, attr):
        try:
            knl = Kernel(self, attr)
            # Nvidia does not raise errors even for invalid names,
            # but this will give an error if the kernel is invalid.
            knl.num_args
            return knl
        except LogicError:
            raise AttributeError("'%s' was not found as a program "
                    "info attribute or as a kernel name" % attr)

    def build(self, options=[], devices=None, cache_dir=None):
        if isinstance(options, str):
            options = [options]

        options = options + ["-I", _find_pyopencl_include_path()]

        if self._prg is not None:
            if isinstance(options, list):
                options = " ".join(options)

            self._prg._build(options, devices)
        else:
            from pyopencl.cache import create_built_program_from_source_cached
            self._prg = create_built_program_from_source_cached(
                    self._context, self._source, options, devices,
                    cache_dir=cache_dir)

        return self

    def compile(self, options=[], devices=None, headers=[]):
        options = " ".join(options)
        return self._prg().compile(options, devices, headers)

def create_program_with_built_in_kernels(context, devices, kernel_names):
    if not isinstance(kernel_names, str):
        kernel_names = ":".join(kernel_names)

    return Program(_Program.create_with_built_in_kernels(
        context, devices, kernel_names))

def link_program(context, programs, options=[], devices=None):
    options = " ".join(options)
    return Program(_Program.link(context, programs, options, devices))

# }}}

# {{{ Image

class Image(_cl._ImageBase):
    def __init__(self, context, flags, format, shape=None, pitches=None,
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

        if get_cl_header_version() >= (1,2):
            if buffer is not None and is_array:
                    raise ValueError("'buffer' and 'is_array' are mutually exclusive")

            if len(shape) == 3:
                if buffer is not None:
                    raise TypeError("'buffer' argument is not supported for 3D arrays")
                elif is_array:
                    image_type = mem_object_type.IMAGE2D_ARRAY
                else:
                    image_type = mem_object_type.IMAGE3D

            elif len(shape) == 2:
                if buffer is not None:
                    raise TypeError("'buffer' argument is not supported for 2D arrays")
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
            desc.shape = shape # also sets desc.array_size

            if pitches is None:
                desc.pitches = (0, 0)
            else:
                desc.pitches = pitches

            desc.num_mip_levels = 0 # per CL 1.2 spec
            desc.num_samples = 0 # per CL 1.2 spec
            desc.buffer = buffer

            _cl._ImageBase.__init__(self, context, flags, format, desc, hostbuf)
        else:
            # legacy init for CL 1.1 and older
            if is_array:
                raise TypeError("'is_array=True' is not supported for CL < 1.2")
            #if num_mip_levels is not None:
                #raise TypeError("'num_mip_levels' argument is not supported for CL < 1.2")
            #if num_samples is not None:
                #raise TypeError("'num_samples' argument is not supported for CL < 1.2")
            if buffer is not None:
                raise TypeError("'buffer' argument is not supported for CL < 1.2")

            _cl._ImageBase.__init__(self, context, flags, format, shape, pitches, hostbuf)

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

    image = property(_ImageInfoGetter)

    @property
    def shape(self):
        if self.type == mem_object_type.IMAGE2D:
            return (self.width, self.height)
        elif self.type == mem_object_type.IMAGE3D:
            return (self.width, self.height, self.depth)
        else:
            raise LogicError("only images have shapes")

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
                (MemoryObjectHolder.get_info,_cl.mem_info),
            Image:
                (_cl._ImageBase.get_image_info, _cl.image_info),
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
        return "<pyopencl.Platform '%s' at 0x%x>" % (self.name, self.obj_ptr)

    Platform.__repr__ = platform_repr

    # }}}

    # {{{ Device
    def device_repr(self):
        return "<pyopencl.Device '%s' on '%s' at 0x%x>" % (
                self.name.strip(), self.platform.name.strip(), self.obj_ptr)

    Device.__repr__ = device_repr

    # }}}

    # {{{ Context
    def context_repr(self):
        return "<pyopencl.Context at 0x%x on %s>" % (self.obj_ptr,
                ", ".join(repr(dev) for dev in self.devices))

    Context.__repr__ = context_repr

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
                        what=lambda : what,
                        code=lambda : code,
                        routine=lambda : routine))

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
    def kernel_call(self, queue, global_size, *args, **kwargs):
        global_offset = kwargs.pop("global_offset", None)
        had_local_size = "local_size" in kwargs
        local_size = kwargs.pop("local_size", None)
        g_times_l = kwargs.pop("g_times_l", False)
        wait_for = kwargs.pop("wait_for", None)

        if kwargs:
            raise TypeError(
                    "Kernel.__call__ recived unexpected keyword arguments: %s"
                    % ", ".join(kwargs.keys()))

        if had_local_size:
            from warnings import warn
            warn("The local_size keyword argument is deprecated and will be "
                    "removed in pyopencl 2012.x. Pass the local "
                    "size as the third positional argument instead.",
                    DeprecationWarning, stacklevel=2)

        if isinstance(args[0], (type(None), tuple)) and not had_local_size:
            local_size = args[0]
            args = args[1:]
        elif not had_local_size:
            from warnings import warn
            warn("PyOpenCL Warning: There was an API change "
                    "in Kernel.__call__() in pyopencl 0.92. "
                    "local_size was moved from keyword argument to third "
                    "positional argument in pyopencl 0.92. "
                    "You didn't pass local_size, but you still need to insert "
                    "'None' as a third argument. "
                    "Your present usage is deprecated and will stop "
                    "working in pyopencl 2012.x.",
                    DeprecationWarning, stacklevel=2)

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
                    advice = " (perhaps you meant to pass 'array.data' instead of the array itself?)"

                raise LogicError(
                        "when processing argument #%d (1-based): %s%s"
                        % (i+1, str(e), advice))
            else:
                raise

    Kernel.__call__ = kernel_call
    Kernel.set_scalar_arg_dtypes = kernel_set_scalar_arg_dtypes
    Kernel.set_args = kernel_set_args

    # }}}

    # {{{ ImageFormat

    def image_format_repr(self):
        return "ImageFormat(%s, %s)" % (
                channel_order.to_string(self.channel_order, 
                    "<unknown channel order 0x%x>"),
                channel_type.to_string(self.channel_data_type,
                    "<unknown channel data type 0x%x>"))

    ImageFormat.__repr__ = image_format_repr

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



# {{{ find pyopencl shipped source code

def _find_pyopencl_include_path():
    from imp import find_module
    import sys
    file, pathname, descr = find_module("pyopencl")

    # Who knew Python installation is so uniform and predictable?
    from os.path import join, exists
    possible_include_paths = [
            join(pathname, "..", "include", "pyopencl"),
            join(pathname, "..", "src", "cl"),
            join(pathname, "..", "..", "..", "src", "cl"),
            join(pathname, "..", "..", "..", "..", "include", "pyopencl"),
            join(pathname, "..", "..", "..", "include", "pyopencl"),
            ]

    if sys.platform in ("linux2", "darwin"):
        possible_include_paths.extend([
            join(sys.prefix, "include" , "pyopencl"),
            "/usr/include/pyopencl",
            "/usr/local/include/pyopencl"
            ])

    for inc_path in possible_include_paths:
        if exists(inc_path):
            return inc_path

    raise RuntimeError("could not find path to PyOpenCL's CL"
            " header files, searched in : %s"
            % '\n'.join(possible_include_paths))

# }}}

# {{{ convenience -------------------------------------------------------------
def create_some_context(interactive=True, answers=None):
    import os
    if answers is None and "PYOPENCL_CTX" in os.environ:
        ctx_spec = os.environ["PYOPENCL_CTX"]
        answers = ctx_spec.split(":")

    user_inputs = []

    def get_input(prompt):
        if answers:
            return str(answers.pop(0))
        else:
            user_input = raw_input(prompt)
            user_inputs.append(user_input)
            return user_input

    try:
        import sys
        if not sys.stdin.isatty():
            interactive = False
    except:
        interactive = False

    platforms = get_platforms()

    if not platforms:
        raise Error("no platforms found")
    elif len(platforms) == 1 or not interactive:
        platform = platforms[0]
    else:
        if not answers:
            print "Choose platform:"
            for i, pf in enumerate(platforms):
                print "[%d] %s" % (i, pf)

        answer = get_input("Choice [0]:")
        if not answer:
            platform = platforms[0]
        else:
            try:
                choice = int(answer)
            except ValueError:
                answer = answer.lower()
                platform = None
                for i, pf in enumerate(platforms):
                    if answer in pf.name.lower():
                        platform = pf
                if platform is None:
                    raise RuntimeError("input did not match any platform")

            else:
                platform = platforms[choice]

    devices = platform.get_devices()

    def parse_device(choice):
        try:
            choice = int(choice)
        except ValueError:
            choice = choice.lower()
            for i, dev in enumerate(devices):
                if choice in dev.name.lower():
                    return dev
            raise RuntimeError("input did not match any platform")
        else:
            return devices[choice]

    if not devices:
        raise Error("no devices found")
    elif len(devices) == 1 or not interactive:
        pass
    else:
        if not answers:
            print "Choose device(s):"
            for i, dev in enumerate(devices):
                print "[%d] %s" % (i, dev)

        answer = get_input("Choice, comma-separated [0]:")
        if not answer:
            devices = [devices[0]]
        else:
            devices = [parse_device(i) for i in answer.split(",")]

    if user_inputs:
        print("Set the environment variable PYOPENCL_CTX='%s' to "
                "avoid being asked again." % ":".join(user_inputs))
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
enqueue_copy_image_to_buffer = _mark_copy_deprecated(_cl._enqueue_copy_image_to_buffer)
enqueue_copy_buffer_to_image = _mark_copy_deprecated(_cl._enqueue_copy_buffer_to_image)
enqueue_read_buffer = _mark_copy_deprecated(_cl._enqueue_read_buffer)
enqueue_write_buffer = _mark_copy_deprecated(_cl._enqueue_write_buffer)
enqueue_copy_buffer = _mark_copy_deprecated(_cl._enqueue_copy_buffer)

if _cl.get_cl_header_version() >= (1,1):
    enqueue_read_buffer_rect = _mark_copy_deprecated(_cl._enqueue_read_buffer_rect)
    enqueue_write_buffer_rect = _mark_copy_deprecated(_cl._enqueue_write_buffer_rect)
    enqueue_copy_buffer_rect = _mark_copy_deprecated(_cl._enqueue_copy_buffer_rect)

def enqueue_copy(queue, dest, src, **kwargs):
    if isinstance(dest, MemoryObjectHolder):
        if dest.type == mem_object_type.BUFFER:
            if isinstance(src, MemoryObjectHolder):
                if src.type == mem_object_type.BUFFER:
                    if "src_origin" in kwargs:
                        return _cl._enqueue_copy_buffer_rect(queue, src, dest, **kwargs)
                    else:
                        kwargs["dst_offset"] = kwargs.pop("dest_offset", 0)
                        return _cl._enqueue_copy_buffer(queue, src, dest, **kwargs)
                elif src.type in [mem_object_type.IMAGE2D, mem_object_type.IMAGE3D]:
                    return _cl._enqueue_copy_image_to_buffer(queue, src, dest, **kwargs)
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
                    return _cl._enqueue_copy_buffer_to_image(queue, src, dest, **kwargs)
                elif src.type in [mem_object_type.IMAGE2D, mem_object_type.IMAGE3D]:
                    return _cl._enqueue_copy_image(queue, src, dest, **kwargs)
                else:
                    raise ValueError("invalid src mem object type")
            else:
                # assume from-host
                origin = kwargs.pop("origin")
                region = kwargs.pop("region")

                pitches = kwargs.pop("pitches", (0,0))
                if len(pitches) == 1:
                    kwargs["row_pitch"], = pitches
                else:
                    kwargs["row_pitch"], kwargs["slice_pitch"] = pitches

                return _cl._enqueue_write_image(queue, dest, origin, region, src, **kwargs)
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

                pitches = kwargs.pop("pitches", (0,0))
                if len(pitches) == 1:
                    kwargs["row_pitch"], = pitches
                else:
                    kwargs["row_pitch"], kwargs["slice_pitch"] = pitches

                return _cl._enqueue_read_image(queue, src, origin, region, dest, **kwargs)
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

def image_from_array(ctx, ary, num_channels, mode="r", norm_int=False):
    # FIXME what about vector types?

    if not ary.flags.c_contiguous:
        raise ValueError("array must be C-contiguous")

    if num_channels == 1:
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
        channel_type = DTYPE_TO_CHANNEL_TYPE_NORM[ary.dtype]
    else:
        channel_type = DTYPE_TO_CHANNEL_TYPE[ary.dtype]

    return Image(ctx, mode_flags | mem_flags.COPY_HOST_PTR,
            ImageFormat(img_format, channel_type),
            shape=shape[::-1], pitches=strides[::-1][1:],
            hostbuf=ary)

# }}}




# vim: foldmethod=marker
