from cffi import FFI

import os.path

current_directory = os.path.dirname(__file__)

_ffi = FFI()
def _get_verifier(**kwargs):
    from cffi.verifier import Verifier
    ver = Verifier(
        _ffi,
        """
        #include <wrap_cl.h>
        """,
        tmpdir="pyopencl/__cffi__",
        modulename='_cffi_wrapcl',
        **kwargs)
    ver.compile_module()
    return ver
    
