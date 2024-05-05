cmake_minimum_required(VERSION 3.17...3.22)

project(pyopencl)

if (NOT SKBUILD)
  message(FATAL_ERROR "This CMake file should be executed via scikit-build. "
                      "Please run\n$ pip install .")
endif()

# {{{ Constrain FindPython to find the Python version used by scikit-build

if (SKBUILD)
  message(STATUS "Python_VERSION ${PYTHON_VERSION_STRING}")
  message(STATUS "Python_EXECUTABLE ${PYTHON_EXECUTABLE}")
  message(STATUS "Python_INCLUDE_DIR ${PYTHON_INCLUDE_DIR}")
  message(STATUS "Python_LIBRARIES ${PYTHON_LIBRARY}")
  set(Python_VERSION "${PYTHON_VERSION_STRING}")
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
  set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
  set(Python_LIBRARIES "${PYTHON_LIBRARY}")
endif()
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# }}}

# {{{ Detect nanobind and import it

execute_process(
  COMMAND
  "${PYTHON_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

# }}}

link_directories(${PYOPENCL_CL_LIB_DIRS})

find_package(nanobind CONFIG REQUIRED)
find_package(NumPy REQUIRED)
find_package(OpenCL REQUIRED)

include_directories(${OpenCL_INCLUDE_DIR} ${NumPy_INCLUDE_DIRS})

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

install(TARGETS _cl LIBRARY DESTINATION .)

# vim: foldmethod=marker:sw=2
