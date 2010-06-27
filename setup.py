#!/usr/bin/env python
# -*- coding: latin-1 -*-



def get_config_schema():
    from aksetup_helper import ConfigSchema, Option, \
            IncludeDir, LibraryDir, Libraries, BoostLibraries, \
            Switch, StringListOption, make_boost_base_options

    import sys
    if 'darwin' in sys.platform:
        default_libs = []
        default_cxxflags = ['-arch', 'i386', '-arch', 'x86_64',
                '-isysroot', '/Developer/SDKs/MacOSX10.6.sdk']
        default_ldflags = default_cxxflags[:] + ["-Wl,-framework,OpenCL"]
    else:
        default_libs = ["OpenCL"]
        default_cxxflags = []
        default_ldflags = []

    return ConfigSchema(make_boost_base_options() + [
        BoostLibraries("python"),

        Switch("CL_TRACE", False, "Enable OpenCL API tracing"),
        Switch("CL_ENABLE_GL", False, "Enable OpenCL<->OpenGL interoperability"),

        IncludeDir("CL", []),
        LibraryDir("CL", []),
        Libraries("CL", default_libs),

        StringListOption("CXXFLAGS", default_cxxflags, 
            help="Any extra C++ compiler options to include"),
        StringListOption("LDFLAGS", default_ldflags, 
            help="Any extra linker options to include"),
        ])




def main():
    import glob
    from aksetup_helper import hack_distutils, get_config, setup, \
            NumpyExtension

    hack_distutils()
    conf = get_config(get_config_schema())

    LIBRARY_DIRS = conf["BOOST_LIB_DIR"]
    LIBRARIES = conf["BOOST_PYTHON_LIBNAME"]

    from os.path import dirname, join, normpath

    EXTRA_DEFINES = { }
    EXTRA_INCLUDE_DIRS = []
    EXTRA_LIBRARY_DIRS = []
    EXTRA_LIBRARIES = []

    if conf["CL_TRACE"]:
        EXTRA_DEFINES["PYOPENCL_TRACE"] = 1

    INCLUDE_DIRS = ['src/cpp'] + conf["BOOST_INC_DIR"] + conf["CL_INC_DIR"]

    ext_kwargs = dict()

    if conf["CL_ENABLE_GL"]:
        EXTRA_DEFINES["HAVE_GL"] = 1

    ver_dic = {}
    execfile("pyopencl/version.py", ver_dic)

    setup(name="pyopencl",
            # metadata
            version=ver_dic["VERSION_TEXT"],
            description="Python wrapper for OpenCL",
            long_description="""
            PyOpenCL lets you access GPUs and other massively parallel compute
            devices from Python. It tries to offer computing goodness in the
            spirit of its sister project `PyCUDA <http://mathema.tician.de/software/pycuda>`_:

            * Object cleanup tied to lifetime of objects. This idiom, often
              called
              `RAII <http://en.wikipedia.org/wiki/Resource_Acquisition_Is_Initialization>`_
              in C++, makes it much easier to write correct, leak- and
              crash-free code.

            * Completeness. PyOpenCL puts the full power of OpenCL's API at
              your disposal, if you wish.  Every obscure `get_info()` query and 
              all CL calls are accessible.

            * Automatic Error Checking. All CL errors are automatically
              translated into Python exceptions.

            * Speed. PyOpenCL's base layer is written in C++, so all the niceties
              above are virtually free.

            * Helpful and complete `Documentation <http://documen.tician.de/pyopencl>`_
              as well as a `Wiki <http://wiki.tiker.net/PyOpenCL>`_.

            * Liberal license. PyOpenCL is open-source under the 
              `MIT license <http://en.wikipedia.org/wiki/MIT_License>`_
              and free for commercial, academic, and private use.

            * Broad support. PyOpenCL was tested and works with Apple's, AMD's, and Nvidia's 
              CL implementations.
            """,
            author=u"Andreas Kloeckner",
            author_email="inform@tiker.net",
            license = "MIT",
            url="http://mathema.tician.de/software/pyopencl",
            classifiers=[
              'Environment :: Console',
              'Development Status :: 4 - Beta',
              'Intended Audience :: Developers',
              'Intended Audience :: Other Audience',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: Apache Software License',
              'Natural Language :: English',
              'Programming Language :: C++',
              'Programming Language :: Python',
              'Topic :: Scientific/Engineering',
              'Topic :: Scientific/Engineering :: Mathematics',
              'Topic :: Scientific/Engineering :: Physics',
              ],

            # build info
            packages=["pyopencl"],

            install_requires=[
                "pytools>=7",
                "py>=1.0.2",
                "decorator>=3.2.0",
                ],

            ext_package="pyopencl",
            ext_modules=[
                NumpyExtension("_cl", 
                    [
                        "src/wrapper/wrap_cl.cpp", 
                        ], 
                    include_dirs=INCLUDE_DIRS + EXTRA_INCLUDE_DIRS,
                    library_dirs=LIBRARY_DIRS + conf["CL_LIB_DIR"],
                    libraries=LIBRARIES + conf["CL_LIBNAME"],
                    define_macros=list(EXTRA_DEFINES.iteritems()),
                    extra_compile_args=conf["CXXFLAGS"],
                    extra_link_args=conf["LDFLAGS"],
                    ),
                ])




if __name__ == '__main__':
    main()
