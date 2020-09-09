from IPython.core.magic import (magics_class, Magics, cell_magic, line_magic)

import pyopencl as cl
import sys


def _try_to_utf8(text):
    if isinstance(text, str):
        return text.encode("utf8")
    return text


@magics_class
class PyOpenCLMagics(Magics):
    def _run_kernel(self, kernel, options):
        if sys.version_info < (3,):
            kernel = _try_to_utf8(kernel)
            options = _try_to_utf8(options).strip()

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

        prg = cl.Program(ctx, kernel).build(options=options.split())

        for knl in prg.all_kernels():
            self.shell.user_ns[knl.function_name] = knl

    @cell_magic
    def cl_kernel(self, line, cell):
        kernel = cell

        opts, args = self.parse_options(line, "o:")
        build_options = opts.get("o", "")

        self._run_kernel(kernel, build_options)

    def _load_kernel_and_options(self, line):
        opts, args = self.parse_options(line, "o:f:")

        build_options = opts.get("o")
        kernel = self.shell.find_user_code(opts.get("f") or args)

        return kernel, build_options

    @line_magic
    def cl_kernel_from_file(self, line):
        kernel, build_options = self._load_kernel_and_options(line)
        self._run_kernel(kernel, build_options)

    @line_magic
    def cl_load_edit_kernel(self, line):
        kernel, build_options = self._load_kernel_and_options(line)
        header = "%%cl_kernel"

        if build_options:
            header = f'{header} -o "{build_options}"'

        content = f"{header}\n\n{kernel}"

        self.shell.set_next_input(content)


def load_ipython_extension(ip):
    ip.register_magics(PyOpenCLMagics)
