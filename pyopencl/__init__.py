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
                return Kernel(self, attr)
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

    def event_wait(self):
        wait_for_events([self])
        return self

    Event.wait = event_wait




_add_functionality()
