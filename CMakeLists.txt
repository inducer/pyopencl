cmake_minimum_required(VERSION 3.17...3.26)

project(pyopencl LANGUAGES CXX VERSION ${SKBUILD_PROJECT_VERSION})

if(NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build'. Running
  it directly will almost certainly not produce the desired result. If
  you are a user trying to install this package, please use the command
  below, which will install all necessary build dependencies, compile
  the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to re-run the above
  after editing C++ files.")
endif()

# {{{ Options

option(PYOPENCL_TRACE "Enable OpenCL tracing" $ENV{PYOPENCL_TRACE})
option(PYOPENCL_ENABLE_GL "Enable OpenGL interoperability" $ENV{PYOPENCL_ENABLE_GL})
option(PYOPENCL_USE_SHIPPED_EXT "Use shipped CL extension header" $ENV{PYOPENCL_USE_SHIPPED_EXT})

set(CL_INC_DIR CACHE STRING "OpenCL include directory")
set(CL_LIB_DIR CACHE STRING "OpenCL library directory")
set(CL_LIBNAME CACHE STRING "OpenCL library name")

set(PYOPENCL_PRETEND_CL_VERSION CACHE STRING "Pretend to be a different OpenCL version")

if(NOT CL_INC_DIR)
  message(STATUS "CL_INC_DIR not set, trying to guess it from environment variables.")
  if(DEFINED ENV{CL_INC_DIR})
    message(STATUS "Using OpenCL include directory from environment '$ENV{CL_INC_DIR}'")
    set(CL_INC_DIR $ENV{CL_INC_DIR})
  endif()

  if(DEFINED ENV{CL_LIB_DIR})
    message(STATUS "Using OpenCL library directory from environment '$ENV{CL_INC_DIR}'")
    set(CL_LIB_DIR $ENV{CL_LIB_DIR})
  endif()

  if(DEFINED ENV{CL_LIBNAME})
    message(STATUS "Using OpenCL library name from environment '$ENV{CL_LIBNAME}'")
    set(CL_LIBNAME $ENV{CL_LIBNAME})
  endif()
endif(NOT CL_INC_DIR)

if(NOT CL_INC_DIR)
  message(STATUS "CL_INC_DIR not set, trying to guess it from conda environment.")
  if(DEFINED ENV{CONDA_PREFIX})
    # Linux/MacOS:
    if(EXISTS $ENV{CONDA_PREFIX}/lib/libOpenCL${CMAKE_SHARED_LIBRARY_SUFFIX})
      message(STATUS "Found OpenCL in conda environment '$ENV{CONDA_PREFIX}'")
      set(CL_INC_DIR $ENV{CONDA_PREFIX}/include)
      set(CL_LIB_DIR $ENV{CONDA_PREFIX}/lib)
      set(CL_LIBNAME OpenCL)
    # Windows:
    elseif(EXISTS $ENV{CONDA_PREFIX}/Library/lib/OpenCL${CMAKE_STATIC_LIBRARY_SUFFIX})
      message(STATUS "Found OpenCL in conda environment '$ENV{CONDA_PREFIX}'")
      set(CL_INC_DIR $ENV{CONDA_PREFIX}/Library/include)
      set(CL_LIB_DIR $ENV{CONDA_PREFIX}/Library/lib)
      set(CL_LIBNAME OpenCL)
    endif()

  endif(DEFINED ENV{CONDA_PREFIX})
endif(NOT CL_INC_DIR)

if(NOT PYOPENCL_PRETEND_CL_VERSION)
  if(DEFINED ENV{PYOPENCL_PRETEND_CL_VERSION})
    set(PYOPENCL_PRETEND_CL_VERSION $ENV{PYOPENCL_PRETEND_CL_VERSION})
  endif()
endif()

if(PYOPENCL_PRETEND_CL_VERSION)
  # Split the version string into a list
  string(REPLACE "." ";" VERSION_LIST ${PYOPENCL_PRETEND_CL_VERSION})

  # Get the major and minor version numbers
  list(GET VERSION_LIST 0 MAJOR)
  list(GET VERSION_LIST 1 MINOR)

  # Calculate the numerical value
  math(EXPR ARG "0x1000*${MAJOR} + 0x10*${MINOR}")
  message(STATUS "Pretending to use OpenCL version ${PYOPENCL_PRETEND_CL_VERSION} (${ARG})")
  set(PYOPENCL_PRETEND_CL_VERSION ${ARG})
endif()

message(STATUS "CL_INC_DIR ${CL_INC_DIR}")
message(STATUS "CL_LIB_DIR ${CL_LIB_DIR}")
message(STATUS "CL_LIBNAME ${CL_LIBNAME}")

# }}}

# {{{ Get version information

find_program(GIT git)

if(GIT AND EXISTS ${CMAKE_SOURCE_DIR}/.git)
  # Exact tag match => released version
  execute_process(COMMAND git describe --exact-match --dirty=*
             OUTPUT_VARIABLE PYOPENCL_VERSION_GIT
             RESULT_VARIABLE git_result
             OUTPUT_STRIP_TRAILING_WHITESPACE
             ERROR_QUIET
             )
  if(NOT ${git_result} EQUAL 0)
    # No exact tag match => development version
    execute_process(COMMAND git describe --long --always --dirty=*
             OUTPUT_VARIABLE PYOPENCL_VERSION_GIT
             OUTPUT_STRIP_TRAILING_WHITESPACE
             )
    set(PYOPENCL_REL "(dev)")
  else()
    set(PYOPENCL_REL "(release)")
  endif()
else()
  set(PYOPENCL_VERSION_GIT "v${PROJECT_VERSION}")
  set(PYOPENCL_REL "(non-git)")
endif()

# }}}

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# {{{ Detect nanobind and import it

execute_process(
  COMMAND
  "${PYTHON_EXECUTABLE}" -c "import nanobind; print(nanobind.__version__)"
  OUTPUT_VARIABLE NANOBIND_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)

execute_process(
  COMMAND
  "${PYTHON_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE NANOBIND_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_DIR}")

# }}}

link_directories(${CL_LIB_DIR})
include_directories(${CL_INC_DIR} ${Python_NumPy_INCLUDE_DIRS})

find_package(nanobind CONFIG REQUIRED)

set(OpenCL_ROOT ${CL_LIB_DIR})
set(OpenCL_INCLUDE_DIR ${CL_INC_DIR})
set(OpenCL_LIBRARY ${CL_LIBNAME})
find_package(OpenCL REQUIRED)

nanobind_add_module(
  _cl
  NB_STATIC # Build static libnanobind (the extension module itself remains a shared library)
  LTO
  NOMINSIZE
  src/wrap_constants.cpp
  src/wrap_cl.cpp
  src/wrap_cl_part_1.cpp
  src/wrap_cl_part_2.cpp
  src/wrap_mempool.cpp
  src/bitlog.cpp
)

target_link_libraries(_cl PRIVATE ${OpenCL_LIBRARY})

target_compile_definitions(_cl
  PRIVATE
  PYGPU_PACKAGE=pyopencl
  PYGPU_PYOPENCL
)

if (PYOPENCL_PRETEND_CL_VERSION)
  target_compile_definitions(
    _cl PRIVATE PYOPENCL_PRETEND_CL_VERSION=${PYOPENCL_PRETEND_CL_VERSION})
endif()

if (PYOPENCL_ENABLE_GL)
  target_compile_definitions(_cl PRIVATE HAVE_GL=1)
endif()

if (PYOPENCL_TRACE)
  target_compile_definitions(_cl PRIVATE PYOPENCL_TRACE=1)
endif()

if (PYOPENCL_USE_SHIPPED_EXT)
  target_compile_definitions(_cl PRIVATE PYOPENCL_USE_SHIPPED_EXT=1)
endif()

install(TARGETS _cl LIBRARY DESTINATION pyopencl)


# {{{ Print configuration

message("==============================")
message("PyOpenCL ${PYOPENCL_VERSION_GIT} ${PYOPENCL_REL} configuration: ")
message("  PyOpenCL options: PYOPENCL_TRACE=${PYOPENCL_TRACE} PYOPENCL_ENABLE_GL=${PYOPENCL_ENABLE_GL} PYOPENCL_USE_SHIPPED_EXT=${PYOPENCL_USE_SHIPPED_EXT} PYOPENCL_PRETEND_CL_VERSION=${PYOPENCL_PRETEND_CL_VERSION}")
message("  OpenCL:           ${OpenCL_LIBRARIES} [${OpenCL_VERSION_STRING}]")
message("  Python:           ${Python_EXECUTABLE} [${Python_VERSION}]")
message("  Build type:       ${CMAKE_BUILD_TYPE}")
message("  C++ compiler:     ${CMAKE_CXX_COMPILER} [${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}]")
message("  CMake:            ${CMAKE_COMMAND} [${CMAKE_VERSION}]")
message("  Nanobind:         ${NANOBIND_DIR} [${NANOBIND_VERSION}]")
message("  Build tool:       ${CMAKE_MAKE_PROGRAM}")
message("==============================")

# }}}

# vim: foldmethod=marker:sw=2
