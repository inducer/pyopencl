from __future__ import division

from IPython.core.magic import (magics_class, Magics, cell_magic)

import pyopencl as cl


def _try_to_utf8(text):
    if isinstance(text, unicode):
        return text.encode("utf8")
    return text


@magics_class
class PyOpenCLMagics(Magics):
    @cell_magic
    def cl_kernel(self, line, cell):
        try:
            ctx = self.shell.user_ns["cl_ctx"]
        except KeyError:
            ctx = None

        if not isinstance(ctx, cl.Context):
            ctx = None

        if ctx is None:
            try:
                ctx = self.shell.user_ns["ctx"]
            except KeyError:
                ctx = None

        if ctx is None or not isinstance(ctx, cl.Context):
            raise RuntimeError("unable to locate cl context, which must be "
                    "present in namespace as 'cl_ctx' or 'ctx'")

        prg = cl.Program(ctx, _try_to_utf8(cell)).build(options=_try_to_utf8(line).strip())

        for knl in prg.all_kernels():
            self.shell.user_ns[knl.function_name] = knl


def load_ipython_extension(ip):
    ip.register_magics(PyOpenCLMagics)
