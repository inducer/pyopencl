#!/usr/bin/env python


__copyright__ = """
Copyright (C) 2009-15 Andreas Kloeckner
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


import os
import sys
from os.path import exists


def get_config_schema():
    from aksetup_helper import (ConfigSchema, Option,
            IncludeDir, LibraryDir, Libraries,
            Switch, StringListOption)

    default_cxxflags = [
            # Required for pybind11:
            # https://pybind11.readthedocs.io/en/stable/faq.html#someclass-declared-with-greater-visibility-than-the-type-of-its-field-someclass-member-wattributes
            "-fvisibility=hidden"
            ]

    if "darwin" in sys.platform:
        import platform
        osx_ver, _, _ = platform.mac_ver()
        osx_ver = ".".join(osx_ver.split(".")[:2])

        sysroot_paths = [
                "/Applications/Xcode.app/Contents/Developer/Platforms/"
                "MacOSX.platform/Developer/SDKs/MacOSX%s.sdk" % osx_ver,
                "/Developer/SDKs/MacOSX%s.sdk" % osx_ver
                ]

        default_libs = []
        default_cxxflags = default_cxxflags + [
                "-stdlib=libc++", "-mmacosx-version-min=10.7"]

        from os.path import isdir
        for srp in sysroot_paths:
            if isdir(srp):
                default_cxxflags.extend(["-isysroot", srp])
                break

        default_ldflags = default_cxxflags[:] + ["-Wl,-framework,OpenCL"]

    else:
        default_libs = ["OpenCL"]
        if "linux" in sys.platform:
            # Requested in
            # https://github.com/inducer/pyopencl/issues/132#issuecomment-314713573
            # to make life with Altera FPGAs less painful by default.
            default_ldflags = ["-Wl,--no-as-needed"]
        else:
            default_ldflags = []

    return ConfigSchema([
        Switch("CL_TRACE", False, "Enable OpenCL API tracing"),
        Switch("CL_ENABLE_GL", False, "Enable OpenCL<->OpenGL interoperability"),
        Switch(
            "CL_USE_SHIPPED_EXT", False,
            "Use the pyopencl version of CL/cl_ext.h which includes"
            + " a broader range of vendor-specific OpenCL extension attributes"
            + " than the standard Khronos (or vendor specific) CL/cl_ext.h."),

        # This is here because the ocl-icd wheel declares but doesn't define
        # clSetProgramSpecializationConstant as of 2021-05-22, and providing
        # configuration options for this to pip install isn't easy. Being
        # able to specify this means being able to build a PyOpenCL via pip install
        # that'll work with an outdated ICD loader.
        #
        # (https://stackoverflow.com/q/677577)
        # x-ref: https://github.com/inducer/meshmode/pull/194/checks?check_run_id=2647133257#step:5:27  # noqa: E501
        # x-ref: https://github.com/isuruf/ocl-wheels/
        Option("CL_PRETEND_VERSION",
            os.environ.get("PYOPENCL_CL_PRETEND_VERSION", None),
            "Dotted CL version (e.g. 1.2) which you'd like to use."),

        IncludeDir("CL", []),
        LibraryDir("CL", []),
        Libraries("CL", default_libs),

        StringListOption("CXXFLAGS", default_cxxflags,
            help="Any extra C++ compiler options to include"),
        StringListOption("LDFLAGS", default_ldflags,
            help="Any extra linker options to include"),
        ])


SEPARATOR = "-"*75


def main():
    from setuptools import find_packages
    from aksetup_helper import (hack_distutils, get_config, setup,
            check_pybind11, check_git_submodules, ExtensionUsingNumpy,
            get_pybind_include, PybindBuildExtCommand)
    check_pybind11()
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
        except Exception:
            print("CL_PRETEND_VERSION must be of the form M.N, "
                    "with two integers M and N")
            raise

    conf["EXTRA_DEFINES"] = extra_defines

    INCLUDE_DIRS = conf["CL_INC_DIR"] + ["pybind11/include"]  # noqa: N806

    ver_dic = {}
    version_file = open("pyopencl/version.py")
    try:
        version_file_contents = version_file.read()
    finally:
        version_file.close()

    exec(compile(version_file_contents, "pyopencl/version.py", "exec"), ver_dic)

    try:
        import mako  # noqa
    except ImportError:
        print(SEPARATOR)
        print("Mako is not installed.")
        print(SEPARATOR)
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
        print(SEPARATOR)
        print("Hit Ctrl-C now if you'd like to think about the situation.")
        print(SEPARATOR)

        from aksetup_helper import count_down_delay
        count_down_delay(delay=5)

    if not exists("pyopencl/compyte/dtypes.py"):
        print(75 * "-")
        print("You are missing important files from the pyopencl distribution.")
        print(75 * "-")
        print("You may have downloaded a zip or tar file from Github.")
        print("Those do not work, and I am unable to prevent Github from showing")
        print("them. Delete that file, and get an actual release file from the")
        print("Python package index:")
        print()
        print("https://pypi.python.org/pypi/pyopencl")
        sys.exit(1)

    setup(name="pyopencl",
            # metadata
            version=ver_dic["VERSION_TEXT"],
            description="Python wrapper for OpenCL",
            long_description=open("README.rst").read(),
            author="Andreas Kloeckner",
            author_email="inform@tiker.net",
            license="MIT",
            url="http://mathema.tician.de/software/pyopencl",
            classifiers=[
                "Environment :: Console",
                "Development Status :: 5 - Production/Stable",
                "Intended Audience :: Developers",
                "Intended Audience :: Other Audience",
                "Intended Audience :: Science/Research",
                "License :: OSI Approved :: MIT License",
                "Natural Language :: English",
                "Programming Language :: C++",
                "Programming Language :: Python",
                "Programming Language :: Python :: 3",
                "Topic :: Scientific/Engineering",
                "Topic :: Scientific/Engineering :: Mathematics",
                "Topic :: Scientific/Engineering :: Physics",
                ],

            # build info
            packages=find_packages(),

            ext_modules=[
                ExtensionUsingNumpy("pyopencl._cl",
                    [
                        "src/wrap_constants.cpp",
                        "src/wrap_cl.cpp",
                        "src/wrap_cl_part_1.cpp",
                        "src/wrap_cl_part_2.cpp",
                        "src/wrap_mempool.cpp",
                        "src/bitlog.cpp",
                        ],
                    include_dirs=INCLUDE_DIRS + [
                        get_pybind_include(),
                        ],
                    library_dirs=conf["CL_LIB_DIR"],
                    libraries=conf["CL_LIBNAME"],
                    define_macros=list(conf["EXTRA_DEFINES"].items()),
                    extra_compile_args=conf["CXXFLAGS"],
                    extra_link_args=conf["LDFLAGS"],
                    language="c++",
                    ),
                ],

            setup_requires=[
                "pybind11>=2.5.0",
                "numpy",
                ],

            python_requires="~=3.6",
            install_requires=[
                "numpy",
                "pytools>=2021.2.7",
                "platformdirs>=2.2.0",
                # "Mako>=0.3.6",
                ],
            extras_require={
                "pocl":  ["pocl_binary_distribution>=1.2"],
                "oclgrind":  ["oclgrind_binary_distribution>=18.3"],
            },
            include_package_data=True,
            package_data={
                    "pyopencl": [
                        "cl/*.cl",
                        "cl/*.h",
                        "cl/pyopencl-random123/*.cl",
                        "cl/pyopencl-random123/*.h",
                        ]
                    },

            cmdclass={"build_ext": PybindBuildExtCommand},
            zip_safe=False)


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
