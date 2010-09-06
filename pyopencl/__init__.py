from pyopencl.version import VERSION, VERSION_STATUS, VERSION_TEXT

import pyopencl._cl as _cl
from pyopencl._cl import *
import inspect as _inspect

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

    def to_string(cls, value):
        for name in dir(cls):
            if (not name.startswith("_") and getattr(cls, name) == value):
                return name

        raise ValueError("a name for value %d was not found in %s"
                % (value, cls.__name__))

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

    # {{{ Program
    def program_getattr(self, attr):
        try:
            pi_attr = getattr(_cl.program_info, attr.upper())
        except AttributeError:
            try:
                knl = Kernel(self, attr)
                # Nvidia does not raise errors even for invalid names,
                # but this will give an error if the kernel is invalid.
                knl.num_args
                return knl
            except LogicError:
                raise AttributeError("'%s' was not found as a program info attribute or as a kernel name"
                        % attr)
        else:
            return self.get_info(pi_attr)

    def program_build(self, options="", devices=None):
        try:
            self._build(options=options, devices=devices)
        except Exception, e:
            build_logs = []
            for dev in self.devices:
                try:
                    log = self.get_build_info(dev, program_build_info.LOG)
                except:
                    log = "<error retrieving log>"

                build_logs.append((dev, log))

            raise _cl.RuntimeError(
                    str(e) + "\n\n" + (75*"="+"\n").join(
                        "Build on %s:\n\n%s" % (dev, log) for dev, log in build_logs))

        return self

    Program.__getattr__ = program_getattr
    Program.build = program_build

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
        wait_for = kwargs.pop("wait_for", None)

        if kwargs:
            raise TypeError(
                    "Kernel.__call__ recived unexpected keyword arguments: %s"
                    % ", ".join(kwargs.keys()))

        if had_local_size:
            from warnings import warn
            warn("The local_size keyword argument is deprecated and will be "
                    "removed in pyopencl 0.94. Pass the local "
                    "size as the third positional argument instead.",
                    DeprecationWarning, stacklevel=2)

        from types import NoneType
        if isinstance(args[0], (NoneType, tuple)) and not had_local_size:
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
                    "working in pyopencl 0.94.",
                    DeprecationWarning, stacklevel=2)

        self.set_args(*args)

        return enqueue_nd_range_kernel(queue, self, global_size, local_size,
                global_offset, wait_for)

    def kernel_set_scalar_arg_dtypes(self, arg_dtypes):
        arg_type_chars = []

        for arg_dtype in arg_dtypes:
            if arg_dtype is None:
                arg_type_chars.append(None)
            else:
                import numpy
                arg_type_chars.append(numpy.dtype(arg_dtype).char)

        self._arg_type_chars = arg_type_chars

    def kernel_set_args(self, *args):
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
                if arg_type_char:
                    self.set_arg(i, pack(arg_type_char, arg))
                else:
                    self.set_arg(i, arg)

    Kernel.__call__ = kernel_call
    Kernel.set_scalar_arg_dtypes = kernel_set_scalar_arg_dtypes
    Kernel.set_args = kernel_set_args

    # }}}

    # {{{ ImageFormat
    def image_format_repr(self):
        return "ImageFormat(%s, %s)" % (
                channel_order.to_string(self.channel_order),
                channel_type.to_string(self.channel_data_type))

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

    if _cl.have_gl():
        def gl_object_get_gl_object(self):
            return self.get_gl_object_info()[1]

        GLBuffer.gl_object = property(gl_object_get_gl_object)
        GLTexture.gl_object = property(gl_object_get_gl_object)

_add_functionality()




# {{{ convenience -------------------------------------------------------------
def create_some_context(interactive=True):
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
        print "Choose platform:"
        for i, pf in enumerate(platforms):
            print "[%d] %s" % (i, pf)

        answer = raw_input("Choice [0]:")
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
        print "Choose device(s):"
        for i, dev in enumerate(devices):
            print "[%d] %s" % (i, dev)

        answer = raw_input("Choice, comma-separated [0]:")
        if not answer:
            devices = [devices[0]]
        else:
            devices = [devices[int(i)] for i in answer.split(",")]

    return Context(devices)

# }}}


# vim: foldmethod=marker
