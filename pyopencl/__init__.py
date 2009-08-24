VERSION = (0, 90)
VERSION_STATUS = "alpha"
VERSION_TEXT = ".".join(str(x) for x in VERSION) + VERSION_STATUS

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
            _cl.Program: _cl.program_info,
            _cl.Kernel: _cl.kernel_info,
            }

    def make_getattr(info_cls):
        def result(self, name):
            return self.get_info(getattr(info_cls, name.upper()))

        return result

    for cls, info_cls in cls_to_info_cls.iteritems():
        cls.__getattr__ = make_getattr(info_cls)

_add_functionality()
