cmake_minimum_required(VERSION 3.17...3.22)

project(pyopencl)

if (NOT SKBUILD)
  message(FATAL_ERROR "This CMake file should be executed via scikit-build. "
                      "Please run\n$ pip install .")
endif()

# {{{ Options

option(PYOPENCL_TRACE "Enable OpenCL tracing" OFF)
option(PYOPENCL_ENABLE_GL "Enable OpenGL interoperability" OFF)
option(PYOPENCL_USE_SHIPPED_EXT "Use shipped CL extension header" OFF)
set(CL_INC_DIR CACHE STRING "OpenCL include directory")
set(CL_LIB_DIR CACHE STRING "OpenCL library directory")
set(CL_LIBNAME CACHE STRING "OpenCL library name")

if(DEFINED ENV{CONDA_PREFIX})
  if(EXISTS $ENV{CONDA_PREFIX}/lib/libOpenCL${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(CL_INC_DIR $ENV{CONDA_PREFIX}/include)
    set(CL_LIB_DIR $ENV{CONDA_PREFIX}/lib)
    set(CL_LIBNAME OpenCL)
  endif()
endif()

if(DEFINED ENV{CL_INC_DIR})
  set(CL_INC_DIR $ENV{CL_INC_DIR})
endif()

if(DEFINED ENV{CL_LIB_DIR})
  set(CL_LIB_DIR $ENV{CL_LIB_DIR})
endif()

if(DEFINED ENV{CL_LIBNAME})
  set(CL_LIBNAME $ENV{CL_LIBNAME})
endif()

# }}}

# {{{ Constrain FindPython to find the Python version used by scikit-build

message(STATUS "Python_VERSION ${PYTHON_VERSION_STRING}")
message(STATUS "Python_EXECUTABLE ${PYTHON_EXECUTABLE}")
message(STATUS "Python_INCLUDE_DIR ${PYTHON_INCLUDE_DIR}")
message(STATUS "Python_LIBRARIES ${PYTHON_LIBRARY}")
set(Python_VERSION "${PYTHON_VERSION_STRING}")
set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
set(Python_LIBRARIES "${PYTHON_LIBRARY}")

message(STATUS "==============================")
message(STATUS "CL_INC_DIR ${CL_INC_DIR}")
message(STATUS "CL_LIB_DIR ${CL_LIB_DIR}")
message(STATUS "CL_LIBNAME ${CL_LIBNAME}")

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

# }}}

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# {{{ Detect nanobind and import it

execute_process(
  COMMAND
  "${PYTHON_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

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

# vim: foldmethod=marker:sw=2
