#!/usr/bin/env python


__copyright__ = """
Copyright (C) 2009-15 Andreas Kloeckner
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
from os.path import exists, join


sys.path.append(os.path.dirname(__file__))


def get_config_schema():
    from aksetup_helper import (
        ConfigSchema, IncludeDir, Libraries, LibraryDir, Option, Switch)

    return ConfigSchema([
        Switch("CL_TRACE", False, "Enable OpenCL API tracing"),
        Switch("CL_ENABLE_GL", False, "Enable OpenCL<->OpenGL interoperability"),
        Switch(
            "CL_USE_SHIPPED_EXT", False,
            "Use the pyopencl version of CL/cl_ext.h which includes"
            + " a broader range of vendor-specific OpenCL extension attributes"
            + " than the standard Khronos (or vendor specific) CL/cl_ext.h."),

        IncludeDir("CL", []),
        LibraryDir("CL", []),
        Libraries("CL", []),

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

        ])


SEPARATOR = "-"*75


def main():
    try:
        import nanobind  # noqa: F401
        from setuptools import find_packages
        from skbuild import setup
    except ImportError:
        print("The preferred way to invoke 'setup.py' is via pip, as in 'pip "
              "install .'. If you wish to run the setup script directly, you must "
              "first install the build dependencies listed in pyproject.toml!",
              file=sys.stderr)
        raise

    # {{{ import aksetup_helper bits

    prev_path = sys.path[:]
    # FIXME skbuild seems to remove this. Why?
    sys.path.append(".")

    from aksetup_helper import check_git_submodules, get_config

    sys.path = prev_path

    # }}}

    check_git_submodules()

    conf = get_config(get_config_schema(), warn_about_no_config=False)

    cmake_args = []

    if conf["CL_TRACE"]:
        cmake_args.append("-DPYOPENCL_TRACE:BOOL=ON")

    if conf["CL_ENABLE_GL"]:
        cmake_args.append("-DPYOPENCL_ENABLE_GL:BOOL=ON")

    if conf["CL_USE_SHIPPED_EXT"]:
        cmake_args.append("-DPYOPENCL_USE_SHIPPED_EXT:BOOL=ON")

    if conf["CL_LIB_DIR"] or conf["CL_LIBNAME"]:
        if conf["CL_LIBNAME"] and not conf["CL_LIB_DIR"]:
            raise ValueError("must specify CL_LIBNAME if CL_LIB_DIR is provided")
        if conf["CL_LIB_DIR"] and not conf["CL_LIBNAME"]:
            from warnings import warn
            warn("Only CL_LIB_DIR specified, assuming 'OpenCL' as library name. "
                    "Specify 'CL_LIBNAME' if this is undesired.")
            conf["CL_LIBNAME"] = ["OpenCL"]

        if len(conf["CL_LIBNAME"]) != 1:
            raise ValueError("Specifying more than one value for 'CL_LIBNAME' "
                    "is no longer supported.")
        if len(conf["CL_LIB_DIR"]) != 1:
            raise ValueError("Specifying more than one value for 'CL_LIB_DIR' "
                    "is no longer supported.")

        lib_dir, = conf["CL_LIB_DIR"]
        libname, = conf["CL_LIBNAME"]

        cl_library = None

        for prefix in ["lib", ""]:
            for suffix in [".so", ".dylib", ".lib"]:
                try_cl_library = join(lib_dir, prefix+libname+suffix)
                if exists(try_cl_library):
                    cl_library = try_cl_library

        if cl_library is None:
            cl_library = join(lib_dir, "lib"+libname)

        cmake_args.append(f"-DOpenCL_LIBRARY={cl_library}")

    if conf["CL_INC_DIR"]:
        cmake_args.append(f"-DOpenCL_INCLUDE_DIR:LIST="
                f"{';'.join(conf['CL_INC_DIR'])}")

    if conf["CL_PRETEND_VERSION"]:
        try:
            major, minor = [int(x) for x in conf["CL_PRETEND_VERSION"].split(".")]
            cmake_args.append(
                    "-DPYOPENCL_PRETEND_CL_VERSION:INT="
                    f"{0x1000*major + 0x10 * minor}")
        except Exception:
            print("CL_PRETEND_VERSION must be of the form M.N, "
                    "with two integers M and N")
            raise

    ver_dic = {}
    version_file = open("pyopencl/version.py")
    try:
        version_file_contents = version_file.read()
    finally:
        version_file.close()

    exec(compile(version_file_contents, "pyopencl/version.py", "exec"), ver_dic)

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

            packages=find_packages(),

            cmake_args=cmake_args,

            python_requires="~=3.8",
            install_requires=[
                "numpy",
                "pytools>=2022.1.13",
                "platformdirs>=2.2.0",
                "importlib-resources; python_version<'3.9'",
                # "Mako>=0.3.6",
                ],
            extras_require={
                "pocl":  ["pocl_binary_distribution>=1.2"],
                "oclgrind":  ["oclgrind_binary_distribution>=18.3"],
                "test": ["pytest>=7.0.0", "Mako"],
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

            cmake_install_dir="pyopencl",
            zip_safe=False)


if __name__ == "__main__":
    main()

# vim: foldmethod=marker
