__copyright__ = "Copyright (C) 2022 Isuru Fernando"

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

"""
cldis.py

A script to compile and print the native code for a OpenCL kernel.

Usage: python cldis.py prog.cl <ptx/sass/asm> <build options>
"""

import glob
import os
import re
import subprocess
import sys
import tempfile


def main(ctx, tmp_dir, cl_str, output=None, build_options=()):
    device = ctx.devices[0]
    platform = device.platform
    if platform.name == "NVIDIA CUDA":
        supported_outputs = ["ptx", "sass"]
    elif platform.name == "Portable Computing Language":
        if device.name.startswith("NVIDIA"):
            supported_outputs = ["ptx", "sass"]
        elif device.name.startswith("pthread") or device.name.startswith("cpu"):
            supported_outputs = ["asm"]
        else:
            raise NotImplementedError(f"Unknown pocl device '{device.name}'")
    else:
        raise NotImplementedError(f"Unknown opencl device '{device}'")
    if output is None:
        output = supported_outputs[0]
    else:
        assert output in supported_outputs

    prg = cl.Program(ctx, cl_str).build(options=build_options,
            cache_dir=os.path.join(tmp_dir, "cache"))

    for binary in prg.binaries:
        if output in ["ptx", "sass"]:
            res = binary[binary.index(b"// Generated"):].decode("utf-8")
            if output == "sass":
                with open(os.path.join(tmp_dir, "cl.ptx"), "w") as f:
                    f.write(res)
                tgt = re.findall(r".target sm_[0-9]*", res, re.MULTILINE)[0]
                gpu_name = tgt[8:]
                subprocess.check_call(["ptxas", "cl.ptx", "--verbose",
                    f"--gpu-name={gpu_name}", "--warn-on-spills"], cwd=tmp_dir)
                res = subprocess.check_output(["cuobjdump", "-sass", "elf.o"],
                        cwd=tmp_dir).decode("utf-8")

        elif output == "asm" and platform.name == "Portable Computing Language":
            so = glob.glob(f"{tmp_dir}/**/*.so", recursive=True)[0]
            res = subprocess.check_output(["objdump", "-d", so]).decode("utf-8")

        print(res)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.environ["POCL_CACHE_DIR"] = os.path.join(tmp_dir, "pocl_cache")
        import pyopencl as cl
        ctx = cl.create_some_context()
        cl_file = sys.argv[1]
        with open(cl_file) as f:
            cl_str = f.read()
        output = sys.argv[2] if len(sys.argv) >= 3 else None
        build_options = sys.argv[3:] if len(sys.argv) >= 4 else []
        main(ctx, tmp_dir, cl_str, output, build_options)
