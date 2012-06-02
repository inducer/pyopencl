#!/usr/bin/env python
# -*- coding: latin-1 -*-



def get_config_schema():
    from aksetup_helper import ConfigSchema, Option, \
            IncludeDir, LibraryDir, Libraries, BoostLibraries, \
            Switch, StringListOption, make_boost_base_options

    import sys
    if 'darwin' in sys.platform:
        import platform
        osx_ver, _, _ = platform.mac_ver()
        osx_ver = '.'.join(osx_ver.split('.')[:2])

        sysroot_paths = [
                "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX%s.sdk" % osx_ver,
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

    return ConfigSchema(make_boost_base_options() + [
        BoostLibraries("python"),

        Switch("USE_SHIPPED_BOOST", True, "Use included Boost library"),

        Switch("CL_TRACE", False, "Enable OpenCL API tracing"),
        Switch("CL_ENABLE_GL", False, "Enable OpenCL<->OpenGL interoperability"),
        Switch("CL_ENABLE_DEVICE_FISSION", True, "Enable device fission extension, if present"),
        Option("CL_PRETEND_VERSION", None, "Dotted CL version (e.g. 1.2) which you'd like to use."),

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
    from aksetup_helper import (hack_distutils, get_config, setup,
            NumpyExtension, set_up_shipped_boost_if_requested,
            check_git_submodules)

    check_git_submodules()

    hack_distutils()
    conf = get_config(get_config_schema(),
            warn_about_no_config=False)
    EXTRA_OBJECTS, EXTRA_DEFINES = set_up_shipped_boost_if_requested("pyopencl", conf)

    LIBRARY_DIRS = conf["BOOST_LIB_DIR"]
    LIBRARIES = conf["BOOST_PYTHON_LIBNAME"]

    from os.path import dirname, join, normpath

    EXTRA_INCLUDE_DIRS = []
    EXTRA_LIBRARY_DIRS = []
    EXTRA_LIBRARIES = []

    EXTRA_DEFINES["PYGPU_PACKAGE"] = "pyopencl"
    EXTRA_DEFINES["PYGPU_PYOPENCL"] = "1"

    if conf["CL_TRACE"]:
        EXTRA_DEFINES["PYOPENCL_TRACE"] = 1

    INCLUDE_DIRS = conf["BOOST_INC_DIR"] + conf["CL_INC_DIR"]

    ext_kwargs = dict()

    if conf["CL_ENABLE_GL"]:
        EXTRA_DEFINES["HAVE_GL"] = 1

    if conf["CL_ENABLE_DEVICE_FISSION"]:
        EXTRA_DEFINES["PYOPENCL_USE_DEVICE_FISSION"] = 1
    if conf["CL_PRETEND_VERSION"]:
        try:
            major, minor = [int(x) for x in conf["CL_PRETEND_VERSION"].split(".")]
            EXTRA_DEFINES["PYOPENCL_PRETEND_CL_VERSION"] = 0x1000*major + 0x10 * minor
        except:
            print("CL_PRETEND_VERSION must be of the form M.N, with two integers M and N")
            raise

    ver_dic = {}
    version_file = open("pyopencl/version.py")
    try:
        version_file_contents = version_file.read()
    finally:
        version_file.close()

    exec(compile(version_file_contents, "pyopencl/version.py", 'exec'), ver_dic)

    try:
        from distutils.command.build_py import build_py_2to3 as build_py
    except ImportError:
        # 2.x
        from distutils.command.build_py import build_py

    try:
        import mako
    except ImportError:
        print("-------------------------------------------------------------------------")
        print("Mako is not installed.")
        print("-------------------------------------------------------------------------")
        print("That is not a problem, as most of PyOpenCL will be just fine without it.")
        print("Some higher-level parts of pyopencl (such as pyopencl.reduction)")
        print("will not function without the templating engine Mako [1] being installed.")
        print("If you would like this functionality to work, you might want to install")
        print("Mako after you finish installing PyOpenCL.")
        print("")
        print("[1] http://www.makotemplates.org/")
        print("-------------------------------------------------------------------------")
        print("Hit Ctrl-C now if you'd like to think about the situation.")
        print("-------------------------------------------------------------------------")

        from aksetup_helper import count_down_delay
        count_down_delay(delay=5)

    might_be_cuda = False
    for inc_dir in conf["CL_INC_DIR"]:
        inc_dir = inc_dir.lower()
        if "nv" in inc_dir or "cuda" in inc_dir:
            might_be_cuda = True

    if might_be_cuda and conf["CL_ENABLE_DEVICE_FISSION"]:
        print("-------------------------------------------------------------------------")
        print("You might be compiling against Nvidia CUDA with device fission enabled.")
        print("-------------------------------------------------------------------------")
        print("That is not a problem on CUDA 4.0 and newer. If you are using CUDA 3.2,")
        print("your build will break, because Nvidia shipped a broken CL header in")
        print("in your version. The fix is to set CL_ENABLE_DEVICE_FISSION to False")
        print("in your PyOpenCL configuration.")
        print("-------------------------------------------------------------------------")
        print("Hit Ctrl-C now if you'd like to think about the situation.")
        print("-------------------------------------------------------------------------")

        from aksetup_helper import count_down_delay
        count_down_delay(delay=5)

    import sys
    if sys.version_info >= (3,):
        pvt_struct_source = "src/wrapper/_pvt_struct_v3.cpp"
    else:
        pvt_struct_source = "src/wrapper/_pvt_struct_v2.cpp"

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

            Like PyOpenCL? (And perhaps use it for `bitcoin
            <http://bitcoin.org>`_ mining?) Leave a (bitcoin) tip:
            1HGPQitv27CdENBcH1bstu5B3zeqXRDwtY
            """,
            author="Andreas Kloeckner",
            author_email="inform@tiker.net",
            license = "MIT",
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
            'Programming Language :: Python :: 3',
              'Topic :: Scientific/Engineering',
              'Topic :: Scientific/Engineering :: Mathematics',
              'Topic :: Scientific/Engineering :: Physics',
              ],

            # build info
            packages=["pyopencl", "pyopencl.characterize", "pyopencl.compyte"],

            install_requires=[
                "pytools>=2011.2",
                "pytest>=2",
                "decorator>=3.2.0",
                # "Mako>=0.3.6",
                ],

            ext_package="pyopencl",
            ext_modules=[
                NumpyExtension("_cl", 
                    [
                        "src/wrapper/wrap_cl.cpp", 
                        "src/wrapper/wrap_cl_part_1.cpp", 
                        "src/wrapper/wrap_cl_part_2.cpp", 
                        "src/wrapper/wrap_constants.cpp", 
                        "src/wrapper/wrap_mempool.cpp", 
                        "src/wrapper/bitlog.cpp", 
                        ]+EXTRA_OBJECTS, 
                    include_dirs=INCLUDE_DIRS + EXTRA_INCLUDE_DIRS,
                    library_dirs=LIBRARY_DIRS + conf["CL_LIB_DIR"],
                    libraries=LIBRARIES + conf["CL_LIBNAME"],
                    define_macros=list(EXTRA_DEFINES.items()),
                    extra_compile_args=conf["CXXFLAGS"],
                    extra_link_args=conf["LDFLAGS"],
                    ),
                NumpyExtension("_pvt_struct",
                    [pvt_struct_source],
                    extra_compile_args=conf["CXXFLAGS"],
                    extra_link_args=conf["LDFLAGS"],
                    ),
                ],

            data_files=[
                ("include/pyopencl",
                    glob.glob("src/cl/*.cl") + glob.glob("src/cl/*.h"))
                ],

            # 2to3 invocation
            cmdclass={'build_py': build_py})




if __name__ == '__main__':
    main()
