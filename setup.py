#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function

__copyright__ = """
Copyright (C) 2009-15 Andreas Kloeckner
Copyright (C) 2013 Marko Bencun
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

    default_cxxflags = ['-std=c++0x']

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
        default_cxxflags = default_cxxflags + [
                '-stdlib=libc++', '-mmacosx-version-min=10.7',
                '-arch', 'i386', '-arch', 'x86_64'
                ]

        from os.path import isdir
        for srp in sysroot_paths:
            if isdir(srp):
                default_cxxflags.extend(['-isysroot', srp])
                break

        default_ldflags = default_cxxflags[:] + ["-Wl,-framework,OpenCL"]

    else:
        default_libs = ["OpenCL"]
        default_ldflags = []

    return ConfigSchema([
        Switch("CL_TRACE", False, "Enable OpenCL API tracing"),
        Switch("CL_ENABLE_GL", False, "Enable OpenCL<->OpenGL interoperability"),
        Switch(
            "CL_USE_SHIPPED_EXT", True,
            "Use the pyopencl version of CL/cl_ext.h which includes" +
            " a broader range of vendor-specific OpenCL extension attributes" +
            " than the standard Khronos (or vendor specific) CL/cl_ext.h."),
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
    from setuptools import find_packages
    from aksetup_helper import (hack_distutils, get_config, setup,
            check_git_submodules)
    check_git_submodules()

    hack_distutils()
    conf = get_config(get_config_schema(),
            warn_about_no_config=False)

    extra_defines = {}

    extra_defines["PYGPU_PACKAGE"] = "pyopencl"
    extra_defines["PYGPU_PYOPENCL"] = "1"

    if conf["CL_TRACE"]:
        extra_defines["PYOPENCL_TRACE"] = 1

    if conf["CL_ENABLE_GL"]:
        extra_defines["HAVE_GL"] = 1

    if conf["CL_USE_SHIPPED_EXT"]:
        extra_defines["PYOPENCL_USE_SHIPPED_EXT"] = 1

    if conf["CL_PRETEND_VERSION"]:
        try:
            major, minor = [int(x) for x in conf["CL_PRETEND_VERSION"].split(".")]
            extra_defines["PYOPENCL_PRETEND_CL_VERSION"] = \
                    0x1000*major + 0x10 * minor
        except:
            print("CL_PRETEND_VERSION must be of the form M.N, "
                    "with two integers M and N")
            raise

    conf["EXTRA_DEFINES"] = extra_defines

    ver_dic = {}
    version_file = open("pyopencl/version.py")
    try:
        version_file_contents = version_file.read()
    finally:
        version_file.close()

    exec(compile(version_file_contents, "pyopencl/version.py", 'exec'), ver_dic)

    separator = "-"*75

    try:
        import mako  # noqa
    except ImportError:
        print(separator)
        print("Mako is not installed.")
        print(separator)
        print("That is not a problem, as most of PyOpenCL will be just fine ")
        print("without it. Some higher-level parts of pyopencl (such as ")
        print("pyopencl.reduction) will not function without the templating engine ")
        print("Mako [1] being installed. If you would like this functionality to ")
        print("work, you might want to install Mako after you finish ")
        print("installing PyOpenCL.")
        print("")
        print("Simply type")
        print("python -m pip install mako")
        print("either now or after the installation completes to fix this.")
        print("")
        print("[1] http://www.makotemplates.org/")
        print(separator)
        print("Hit Ctrl-C now if you'd like to think about the situation.")
        print(separator)

        from aksetup_helper import count_down_delay
        count_down_delay(delay=5)

    # {{{ write cffi build script

    with open("cffi_build.py.in", "rt") as f:
        build_script_template = f.read()

    format_args = {}
    for k, v in conf.items():
        format_args[k] = repr(v)

    build_script = build_script_template.format(**format_args)

    with open("cffi_build.py", "wt") as f:
        f.write(build_script)

    # }}}

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
            packages=find_packages(),

            setup_requires=[
                "numpy",
                "cffi>=1.1.0",
                ],

            install_requires=[
                "numpy",
                "pytools>=2015.1.2",
                "pytest>=2",
                "decorator>=3.2.0",
                "cffi>=1.1.0",
                "appdirs>=1.4.0",
                "six>=1.9.0",
                # "Mako>=0.3.6",
                ],

            cffi_modules=["cffi_build.py:ffi"],

            include_package_data=True,
            package_data={
                    "pyopencl": [
                        "cl/*.cl",
                        "cl/*.h",
                        "cl/pyopencl-random123/*.cl",
                        "cl/pyopencl-random123/*.h",
                        ]
                    },

            zip_safe=False)


if __name__ == '__main__':
    main()
