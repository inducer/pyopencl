#!/usr/bin/env python
# -*- coding: latin-1 -*-

__copyright__ = """
Copyright (C) 2009-14 Andreas Kloeckner
Copyright (C) 2013 Marko Bencun
"""

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


import sys


def get_config_schema():
    from aksetup_helper import ConfigSchema, Option, \
            IncludeDir, LibraryDir, Libraries, \
            Switch, StringListOption

    if 'darwin' in sys.platform:
        import platform
        osx_ver, _, _ = platform.mac_ver()
        osx_ver = '.'.join(osx_ver.split('.')[:2])

        sysroot_paths = [
                "/Applications/Xcode.app/Contents/Developer/Platforms/"
                "MacOSX.platform/Developer/SDKs/MacOSX%s.sdk" % osx_ver,
                "/Developer/SDKs/MacOSX%s.sdk" % osx_ver
                ]

        default_libs = []
        default_cxxflags = ['-arch', 'i386', '-arch', 'x86_64']

        from os.path import isdir
        for srp in sysroot_paths:
            if isdir(srp):
                default_cxxflags.extend(['-isysroot', srp])
                break

        default_ldflags = default_cxxflags[:] + ["-Wl,-framework,OpenCL"]

    else:
        default_libs = ["OpenCL"]
        default_cxxflags = []
        default_ldflags = []

    return ConfigSchema([
        Switch("CL_TRACE", False, "Enable OpenCL API tracing"),
        Switch("CL_ENABLE_GL", False, "Enable OpenCL<->OpenGL interoperability"),
        Switch("CL_ENABLE_DEVICE_FISSION", True,
            "Enable device fission extension, if present"),
        Option("CL_PRETEND_VERSION", None,
            "Dotted CL version (e.g. 1.2) which you'd like to use."),

        IncludeDir("CL", []),
        LibraryDir("CL", []),
        Libraries("CL", default_libs),

        StringListOption("CXXFLAGS", default_cxxflags,
            help="Any extra C++ compiler options to include"),
        StringListOption("LDFLAGS", default_ldflags,
            help="Any extra linker options to include"),
        ])


def main():
    from aksetup_helper import (hack_distutils, get_config, setup,
            check_git_submodules)
    from setuptools import Extension
    check_git_submodules()

    hack_distutils()
    conf = get_config(get_config_schema(),
            warn_about_no_config=False)
    EXTRA_DEFINES = {}

    EXTRA_DEFINES["PYGPU_PACKAGE"] = "pyopencl"
    EXTRA_DEFINES["PYGPU_PYOPENCL"] = "1"

    if conf["CL_TRACE"]:
        EXTRA_DEFINES["PYOPENCL_TRACE"] = 1

    if conf["CL_ENABLE_GL"]:
        EXTRA_DEFINES["HAVE_GL"] = 1

    if conf["CL_ENABLE_DEVICE_FISSION"]:
        EXTRA_DEFINES["PYOPENCL_USE_DEVICE_FISSION"] = 1
    if conf["CL_PRETEND_VERSION"]:
        try:
            major, minor = [int(x) for x in conf["CL_PRETEND_VERSION"].split(".")]
            EXTRA_DEFINES["PYOPENCL_PRETEND_CL_VERSION"] = \
                    0x1000*major + 0x10 * minor
        except:
            print("CL_PRETEND_VERSION must be of the form M.N, "
                    "with two integers M and N")
            raise

    ver_dic = {}
    version_file = open("pyopencl/version.py")
    try:
        version_file_contents = version_file.read()
    finally:
        version_file.close()

    exec(compile(version_file_contents, "pyopencl/version.py", 'exec'), ver_dic)

    SEPARATOR = "-"*75
    try:
        from distutils.command.build_py import build_py_2to3 as build_py
    except ImportError:
        # 2.x
        from distutils.command.build_py import build_py

    try:
        import mako  # noqa
    except ImportError:
        print(SEPARATOR)
        print("Mako is not installed.")
        print(SEPARATOR)
        print("That is not a problem, as most of PyOpenCL will be just fine ")
        print("without it.Some higher-level parts of pyopencl (such as ")
        print("pyopencl.reduction) will not function without the templating engine ")
        print("Mako [1] being installed. If you would like this functionality to ")
        print("work, you might want to install Mako after you finish ")
        print("installing PyOpenCL.")
        print("")
        print("[1] http://www.makotemplates.org/")
        print(SEPARATOR)
        print("Hit Ctrl-C now if you'd like to think about the situation.")
        print(SEPARATOR)

        from aksetup_helper import count_down_delay
        count_down_delay(delay=5)

    might_be_cuda = False
    for inc_dir in conf["CL_INC_DIR"]:
        inc_dir = inc_dir.lower()
        if "nv" in inc_dir or "cuda" in inc_dir:
            might_be_cuda = True

    if might_be_cuda and conf["CL_ENABLE_DEVICE_FISSION"]:
        print(SEPARATOR)
        print("You might be compiling against Nvidia CUDA with device "
                "fission enabled.")
        print(SEPARATOR)
        print("That is not a problem on CUDA 4.0 and newer. If you are "
                "using CUDA 3.2,")
        print("your build will break, because Nvidia shipped a broken CL header in")
        print("in your version. The fix is to set CL_ENABLE_DEVICE_FISSION to False")
        print("in your PyOpenCL configuration.")
        print(SEPARATOR)
        print("Hit Ctrl-C now if you'd like to think about the situation.")
        print(SEPARATOR)

        from aksetup_helper import count_down_delay
        count_down_delay(delay=5)

    # from pyopencl._cffi import _get_verifier

    # for development: clean cache such that the extension is rebuilt
    if 0:
        import os.path
        current_directory = os.path.dirname(__file__)

        shutil.rmtree(os.path.join(current_directory,
            'pyopencl', '__pycache__/'), ignore_errors=True)

    setup(name="pyopencl",
            # metadata
            version=ver_dic["VERSION_TEXT"],
            description="Python wrapper for OpenCL",
            long_description=open("README.rst", "rt").read(),
            author="Andreas Kloeckner",
            author_email="inform@tiker.net",
            license="MIT",
            url="http://mathema.tician.de/software/pyopencl",
            classifiers=[
                'Environment :: Console',
                'Development Status :: 5 - Production/Stable',
                'Intended Audience :: Developers',
                'Intended Audience :: Other Audience',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: MIT License',
                'Natural Language :: English',
                'Programming Language :: C++',
                'Programming Language :: Python',
                'Programming Language :: Python :: 2',
                'Programming Language :: Python :: 2.6',
                'Programming Language :: Python :: 2.7',
                'Programming Language :: Python :: 3',
                'Programming Language :: Python :: 3.2',
                'Programming Language :: Python :: 3.3',
                'Topic :: Scientific/Engineering',
                'Topic :: Scientific/Engineering :: Mathematics',
                'Topic :: Scientific/Engineering :: Physics',
                ],

            # build info
            packages=["pyopencl", "pyopencl.characterize", "pyopencl.compyte"],

            setup_requires=[
                "numpy",
                ],

            install_requires=[
                "pytools>=2014.2",
                "pytest>=2",
                "decorator>=3.2.0",
                "cffi>=0.7.2",
                # "Mako>=0.3.6",
                ],

            ext_package="pyopencl",
            ext_modules=[
                Extension("_wrapcl",
                               ["src/c_wrapper/wrap_cl.cpp",
                                "src/c_wrapper/wrap_constants.cpp",
                                "src/c_wrapper/bitlog.cpp",
                                "src/c_wrapper/pyhelper.cpp",
                                "src/c_wrapper/async.cpp",
                                "src/c_wrapper/platform.cpp",
                                "src/c_wrapper/device.cpp",
                                "src/c_wrapper/context.cpp",
                                "src/c_wrapper/command_queue.cpp",
                                "src/c_wrapper/event.cpp",
                                "src/c_wrapper/memory_object.cpp",
                                "src/c_wrapper/image.cpp",
                                "src/c_wrapper/gl_obj.cpp",
                                "src/c_wrapper/memory_map.cpp",
                                "src/c_wrapper/buffer.cpp",
                                "src/c_wrapper/sampler.cpp",
                                "src/c_wrapper/program.cpp",
                                "src/c_wrapper/kernel.cpp",
                                "src/c_wrapper/debug.cpp",
                                ],
                               include_dirs=(
                                   conf["CL_INC_DIR"]
                                   + ["src/c_wrapper/",
                                      "pyopencl/c_wrapper/"]),
                               library_dirs=conf["CL_LIB_DIR"],
                               libraries=conf["CL_LIBNAME"],
                               define_macros=list(EXTRA_DEFINES.items()),
                               extra_compile_args=(['-std=c++0x']
                                                   + conf["CXXFLAGS"]),
                               extra_link_args=conf["LDFLAGS"])
            ],

            include_package_data=True,
            package_data={
                    "pyopencl": [
                        "cl/*.cl",
                        "cl/*.h",
                        "c_wrapper/wrap_cl_core.h",
                        ] + (["c_wrapper/wrap_cl_gl_core.h"]
                             if conf["CL_ENABLE_GL"] else [])
                    },

            # 2to3 invocation
            cmdclass={'build_py': build_py},
            zip_safe=False)


if __name__ == '__main__':
    main()
