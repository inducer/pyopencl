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
            _cl.Platform: _cl.platform_info,
            _cl.Device: _cl.device_info,
            _cl.Context: _cl.context_info,
            _cl.CommandQueue: _cl.command_queue_info,
            _cl.Event: _cl.event_info,
            _cl.MemoryObject: _cl.mem_info,
            _cl.Kernel: _cl.kernel_info,
            }

    def to_string(cls, value):
        for name in dir(cls):
            if (not name.startswith("_") and getattr(cls, name) == value):
                return name

        raise ValueError("a name for value %d was not found in %s"
                % (value, cls.__name__))

    addressing_mode.to_string = classmethod(to_string)
    channel_order.to_string = classmethod(to_string)
    channel_type.to_string = classmethod(to_string)
    command_execution_status.to_string = classmethod(to_string)
    command_queue_info.to_string = classmethod(to_string)
    command_queue_properties.to_string = classmethod(to_string)
    context_info.to_string = classmethod(to_string)
    context_properties.to_string = classmethod(to_string)
    device_exec_capabilities.to_string = classmethod(to_string)
    device_fp_config.to_string = classmethod(to_string)
    device_info.to_string = classmethod(to_string)
    device_local_mem_type.to_string = classmethod(to_string)
    device_mem_cache_type.to_string = classmethod(to_string)
    device_type.to_string = classmethod(to_string)
    event_info.to_string = classmethod(to_string)
    filter_mode.to_string = classmethod(to_string)
    image_info.to_string = classmethod(to_string)
    kernel_info.to_string = classmethod(to_string)
    kernel_work_group_info.to_string = classmethod(to_string)
    map_flags.to_string = classmethod(to_string)
    mem_info.to_string = classmethod(to_string)
    mem_object_type.to_string = classmethod(to_string)
    platform_info.to_string = classmethod(to_string)
    profiling_info.to_string = classmethod(to_string)
    program_build_info.to_string = classmethod(to_string)
    program_info.to_string = classmethod(to_string)
    sampler_info.to_string = classmethod(to_string)

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

    class ImageInfoGetter:
        def __init__(self, mem):
            self.mem = mem

        def __getattr__(self, name):
            info_cls = _cl.image_info

            try:
                inf_attr = getattr(info_cls, name.upper())
            except AttributeError:
                raise AttributeError("%s has no attribute '%s'"
                        % (type(self), name))
            else:
                return self.mem.get_image_info(inf_attr)

    _cl.MemoryObject.image = property(ImageInfoGetter)

    def make_getattr(info_cls):
        def result(self, name):
            try:
                inf_attr = getattr(info_cls, name.upper())
            except AttributeError:
                raise AttributeError("%s has no attribute '%s'"
                        % (type(self), name))
            else:
                return self.get_info(inf_attr)

        return result

    for cls, info_cls in cls_to_info_cls.iteritems():
        cls.__getattr__ = make_getattr(info_cls)

    def program_getattr(self, attr):
        try:
            pi_attr = getattr(program_info, attr.upper())
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

    Program.__getattr__ = program_getattr

    def kernel_call(self, queue, global_size, *args, **kwargs):
        for i, arg in enumerate(args):
            self.set_arg(i, arg)

        global_offset = kwargs.pop("global_offset", None)
        local_size = kwargs.pop("local_size", None)
        wait_for = kwargs.pop("wait_for", None)

        if kwargs:
            raise TypeError(
                    "Kernel.__call__ recived unexpected keyword arguments: %s"
                    % ", ".join(kwargs.keys()))

        return enqueue_nd_range_kernel(queue, self, global_size, local_size,
                global_offset, wait_for)

    Kernel.__call__ = kernel_call

    def image_format_repr(self):
        return "ImageFormat(%s, %s)" % (
                channel_order.to_string(self.channel_order),
                channel_type.to_string(self.channel_data_type))

    ImageFormat.__repr__ = image_format_repr

    def event_wait(self):
        wait_for_events([self])
        return self

    Event.wait = event_wait




_add_functionality()
