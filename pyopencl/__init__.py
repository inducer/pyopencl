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
from decorator import decorator as _decorator

CONSTANT_CLASSES = [
        getattr(_cl, name) for name in dir(_cl)
        if _inspect.isclass(getattr(_cl, name))
        and name[0].islower()]

def _add_functionality():
    cls_to_info_cls = {
            _cl.Platform: [
                (_cl.Platform.get_info, _cl.platform_info),
                ],
            _cl.Device: [
                (_cl.Device.get_info, _cl.device_info)
                ],
            _cl.Context: [
                (_cl.Context.get_info, _cl.context_info),
                ],
            _cl.CommandQueue: [
                (_cl.CommandQueue.get_info, _cl.command_queue_info)
                ],
            _cl.Event: [
                (_cl.Event.get_info, _cl.event_info),
                ],
            _cl.MemoryObject: [
                (MemoryObject.get_info,_cl.mem_info),
                ],
            _cl.Image: [
                (Image.get_image_info, _cl.image_info),
                (MemoryObject.get_info,_cl.mem_info),
                ],
            _cl.Kernel: [
                (Kernel.get_info, _cl.kernel_info),
                ],
            _cl.Sampler: [
                (Sampler.get_info, _cl.sampler_info),
                ],
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
    def make_getattr(info_classes):
        name_to_info = dict(
                (intern(info_name.lower()), (info_method, info_value))
                for info_method, info_class in info_classes[::-1]
                for info_name, info_value in
                  info_class.__dict__.iteritems()
                if info_name != "to_string" and not info_name.startswith("_")
                )

        def result(self, name):
            try:
                inf_method, inf_attr = name_to_info[name]
            except KeyError:
                raise AttributeError("%s has no attribute '%s'"
                        % (type(self), name))
            else:
                return inf_method(self, inf_attr)

        return result

    for cls, info_classes in cls_to_info_cls.iteritems():
        cls.__getattr__ = make_getattr(info_classes)

    # }}}

    # {{{ Platform
    def platform_repr(self):
        return "<pyopencl.Platform '%s' at 0x%x>" % (self.name, self.obj_ptr)

    Platform.__repr__ = platform_repr

    # }}}

    # {{{ Device
    def device_repr(self):
        return "<pyopencl.Device '%s' at 0x%x>" % (self.name, self.obj_ptr)

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
            # Python 3.2 outputs the whole tree of currently active exceptions
            # This serves to remove one (redundant) level from that nesting.
            raise err

        message = (75*"="+"\n").join(
                "Build on %s succeeded, but said:\n\n%s" % (dev, log) 
                for dev, log in self._get_build_logs()
                if log is not None and log.strip())

        if message:
            from warnings import warn
            warn("Build succeeded, but resulted in non-empty logs:\n"+message)

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
        arg_type_chars = []

        for arg_dtype in arg_dtypes:
            if arg_dtype is None:
                arg_type_chars.append(None)
            else:
                arg_type_chars.append(np.dtype(arg_dtype).char)

        self._arg_type_chars = arg_type_chars

    def kernel_set_args(self, *args):
        i = None
        try:
            try:
                arg_type_chars = self.__dict__["_arg_type_chars"]
            except KeyError:
                for i, arg in enumerate(args):
                    self.set_arg(i, arg)
            else:
                from struct import pack

                if len(args) != len(arg_type_chars):
                    raise ValueError("length of argument type array and "
                            "length of argument list do not agree")
                for i, (arg, arg_type_char) in enumerate(
                        zip(args, arg_type_chars)):
                    if arg_type_char and arg_type_char != "V":
                        self.set_arg(i, pack(arg_type_char, arg))
                    else:
                        self.set_arg(i, arg)
        except LogicError, e:
            if i is not None:
                raise LogicError(
                        "when processing argument #%d (1-based): %s"
                        % (i+1, str(e)))
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
                    "<unknown channel order %d>"),
                channel_type.to_string(self.channel_data_type,
                    "<unknown channel data type %d>"))

    ImageFormat.__repr__ = image_format_repr

    # }}}

    # {{{ Image

    class ImageInfoGetter:
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

    _cl.Image.image = property(ImageInfoGetter)
    _cl.Image.shape = property(image_shape)

    # }}}

    # {{{ Event

    def event_wait(self):
        wait_for_events([self])
        return self

    Event.wait = event_wait

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




# {{{ Program (including caching support)

class Program(object):
    def __init__(self, context, arg1, arg2=None):
        if arg2 is None:
            source = arg1

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
            self._prg = _cl._Program(context, arg1, arg2)

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

    def __getattr__(self, attr):
        try:
            pi_attr = getattr(_cl.program_info, attr.upper())
        except AttributeError:
            try:
                knl = Kernel(self._get_prg(), attr)
                # Nvidia does not raise errors even for invalid names,
                # but this will give an error if the kernel is invalid.
                knl.num_args
                return knl
            except LogicError:
                raise AttributeError("'%s' was not found as a program "
                        "info attribute or as a kernel name" % attr)
        else:
            return self.get_info(pi_attr)

    def build(self, options=[], devices=None, cache_dir=None):
        if self._prg is not None:
            self._prg._build(options, devices)
        else:
            from pyopencl.cache import create_built_program_from_source_cached
            self._prg = create_built_program_from_source_cached(
                    self._context, self._source, options, devices,
                    cache_dir=cache_dir)

        return self

    # }}}

# {{{ convenience -------------------------------------------------------------
def create_some_context(interactive=True, answers=None):
    def get_input(prompt):
        if answers:
            return str(answers.pop(0))
        else:
            return raw_input(prompt)

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
            choice = 0
        else:
            choice = int(answer)

        platform = platforms[choice]

    devices = platform.get_devices()

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
            devices = [devices[int(i)] for i in answer.split(",")]

    return Context(devices)




def _make_context_creator(answers):
    def func():
        return create_some_context(answers=answers)

    return func



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
    if isinstance(dest, Buffer):
        if isinstance(src, Buffer):
            if "src_origin" in kwargs:
                return _cl._enqueue_copy_buffer_rect(queue, src, dest, **kwargs)
            else:
                kwargs["dst_offset"] = kwargs.pop("dest_offset", 0)
                return _cl._enqueue_copy_buffer(queue, src, dest, **kwargs)
        elif isinstance(src, Image):
            return _cl._enqueue_copy_image_to_buffer(queue, src, dest, **kwargs)
        else:
            # assume from-host
            if "buffer_origin" in kwargs:
                return _cl._enqueue_write_buffer_rect(queue, dest, src, **kwargs)
            else:
                return _cl._enqueue_write_buffer(queue, dest, src, **kwargs)

    elif isinstance(dest, Image):
        if isinstance(src, Buffer):
            return _cl._enqueue_copy_buffer_to_image(queue, src, dest, **kwargs)
        elif isinstance(src, Image):
            return _cl._enqueue_copy_image(queue, src, dest, **kwargs)
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
        # assume to-host

        if isinstance(src, Buffer):
            if "buffer_origin" in kwargs:
                return _cl._enqueue_read_buffer_rect(queue, src, dest, **kwargs)
            else:
                return _cl._enqueue_read_buffer(queue, src, dest, **kwargs)
        elif isinstance(src, Image):
            origin = kwargs.pop("origin")
            region = kwargs.pop("region")

            pitches = kwargs.pop("pitches", (0,0))
            if len(pitches) == 1:
                kwargs["row_pitch"], = pitches
            else:
                kwargs["row_pitch"], kwargs["slice_pitch"] = pitches

            return _cl._enqueue_read_image(queue, src, origin, region, dest, **kwargs)
        else:
            # assume from-host
            raise TypeError("enqueue_copy cannot perform host-to-host transfers")

# }}}




# vim: foldmethod=marker
