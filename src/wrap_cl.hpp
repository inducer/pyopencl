// PyOpenCL-flavored C++ wrapper of the CL API
//
// Copyright (C) 2009 Andreas Kloeckner
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.


#ifndef _AFJHAYYTA_PYOPENCL_HEADER_SEEN_WRAP_CL_HPP
#define _AFJHAYYTA_PYOPENCL_HEADER_SEEN_WRAP_CL_HPP

// CL 1.2 undecided:
// clSetPrintfCallback

// CL 2.0 complete

// CL 2.1 complete

// CL 2.2 complete

// CL 3.0 missing:
// clCreateBufferWithProperties
// clCreateImageWithProperties
// (no wrappers for now: OpenCL 3.0 does not define any optional properties for
// buffers or images, no implementations to test with.)


// {{{ includes

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#ifdef __APPLE__

// Mac ------------------------------------------------------------------------
#include <OpenCL/opencl.h>
#include "pyopencl_ext.h"
#ifdef HAVE_GL

#define PYOPENCL_GL_SHARING_VERSION 1

#include <OpenGL/OpenGL.h>
#include <OpenCL/cl_gl.h>
#include <OpenCL/cl_gl_ext.h>
#endif

#else

// elsewhere ------------------------------------------------------------------
#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>
#include "pyopencl_ext.h"

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

#ifdef HAVE_GL
#include <GL/gl.h>
#include <CL/cl_gl.h>
#endif

#if defined(cl_khr_gl_sharing) && (cl_khr_gl_sharing >= 1)
#define PYOPENCL_GL_SHARING_VERSION cl_khr_gl_sharing
#endif

#endif

#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <cstdio>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <utility>
#include <numeric>
#include "wrap_helpers.hpp"
#include <numpy/arrayobject.h>
#include "tools.hpp"

#ifdef PYOPENCL_PRETEND_CL_VERSION
#define PYOPENCL_CL_VERSION PYOPENCL_PRETEND_CL_VERSION
#else

#if defined(CL_VERSION_3_0)
#define PYOPENCL_CL_VERSION 0x3000
#elif defined(CL_VERSION_2_2)
#define PYOPENCL_CL_VERSION 0x2020
#elif defined(CL_VERSION_2_1)
#define PYOPENCL_CL_VERSION 0x2010
#elif defined(CL_VERSION_2_0)
#define PYOPENCL_CL_VERSION 0x2000
#elif defined(CL_VERSION_1_2)
#define PYOPENCL_CL_VERSION 0x1020
#elif defined(CL_VERSION_1_1)
#define PYOPENCL_CL_VERSION 0x1010
#else
#define PYOPENCL_CL_VERSION 0x1000
#endif

#endif


#if defined(_WIN32)
// MSVC does not understand variable-length arrays
#define PYOPENCL_STACK_CONTAINER(TYPE, NAME, COUNT) std::vector<TYPE> NAME(COUNT)
#define PYOPENCL_STACK_CONTAINER_GET_PTR(NAME) (NAME.size() ? NAME.data() : nullptr)
#else
// gcc et al complain about stripping attributes in template arguments
#define PYOPENCL_STACK_CONTAINER(TYPE, NAME, COUNT) TYPE NAME[COUNT]
#define PYOPENCL_STACK_CONTAINER_GET_PTR(NAME) NAME
#endif

// }}}





// {{{ tools
#if PY_VERSION_HEX >= 0x02050000
  typedef Py_ssize_t PYOPENCL_BUFFER_SIZE_T;
#else
  typedef int PYOPENCL_BUFFER_SIZE_T;
#endif

#define PYOPENCL_CAST_BOOL(B) ((B) ? CL_TRUE : CL_FALSE)





#define PYOPENCL_DEPRECATED(WHAT, KILL_VERSION, EXTRA_MSG) \
  { \
    PyErr_Warn( \
        PyExc_DeprecationWarning, \
        WHAT " is deprecated and will stop working in PyOpenCL " KILL_VERSION". " \
        EXTRA_MSG); \
  }

#if PYOPENCL_CL_VERSION >= 0x1020

#define PYOPENCL_GET_EXT_FUN(PLATFORM, NAME, VAR) \
    NAME##_fn VAR \
      = (NAME##_fn) \
      clGetExtensionFunctionAddressForPlatform(PLATFORM, #NAME); \
    \
    if (!VAR) \
      throw error(#NAME, CL_INVALID_VALUE, #NAME \
          "not available");

#else

#define PYOPENCL_GET_EXT_FUN(PLATFORM, NAME, VAR) \
    NAME##_fn VAR \
      = (NAME##_fn) \
      clGetExtensionFunctionAddress(#NAME); \
    \
    if (!VAR) \
      throw error(#NAME, CL_INVALID_VALUE, #NAME \
          "not available");

#endif


#define PYOPENCL_PARSE_PY_DEVICES \
    std::vector<cl_device_id> devices_vec; \
    cl_uint num_devices; \
    cl_device_id *devices; \
    \
    if (py_devices.ptr() == Py_None) \
    { \
      num_devices = 0; \
      devices = 0; \
    } \
    else \
    { \
      for (py::handle py_dev: py_devices) \
        devices_vec.push_back( \
            (py_dev).cast<device &>().data()); \
      num_devices = devices_vec.size(); \
      devices = devices_vec.empty( ) ? nullptr : &devices_vec.front(); \
    } \


#define PYOPENCL_RETRY_RETURN_IF_MEM_ERROR(OPERATION) \
    try \
    { \
      OPERATION \
    } \
    catch (pyopencl::error &e) \
    { \
      if (!e.is_out_of_memory()) \
        throw; \
    } \
    \
    /* If we get here, we got an error from CL.
     * We should run the Python GC to try and free up
     * some memory references. */ \
    run_python_gc(); \
    \
    /* Now retry the allocation. If it fails again,
     * let it fail. */ \
    { \
      OPERATION \
    }




#define PYOPENCL_RETRY_IF_MEM_ERROR(OPERATION) \
  { \
    bool failed_with_mem_error = false; \
    try \
    { \
      OPERATION \
    } \
    catch (pyopencl::error &e) \
    { \
      failed_with_mem_error = true; \
      if (!e.is_out_of_memory()) \
        throw; \
    } \
    \
    if (failed_with_mem_error) \
    { \
      /* If we get here, we got an error from CL.
       * We should run the Python GC to try and free up
       * some memory references. */ \
      run_python_gc(); \
      \
      /* Now retry the allocation. If it fails again,
       * let it fail. */ \
      { \
        OPERATION \
      } \
    } \
  }

// }}}

// {{{ tracing and error reporting
#ifdef PYOPENCL_TRACE
  #define PYOPENCL_PRINT_CALL_TRACE(NAME) \
    std::cerr << NAME << std::endl;
  #define PYOPENCL_PRINT_CALL_TRACE_INFO(NAME, EXTRA_INFO) \
    std::cerr << NAME << " (" << EXTRA_INFO << ')' << std::endl;
#else
  #define PYOPENCL_PRINT_CALL_TRACE(NAME) /*nothing*/
  #define PYOPENCL_PRINT_CALL_TRACE_INFO(NAME, EXTRA_INFO) /*nothing*/
#endif

#define PYOPENCL_CALL_GUARDED_THREADED_WITH_TRACE_INFO(NAME, ARGLIST, TRACE_INFO) \
  { \
    PYOPENCL_PRINT_CALL_TRACE_INFO(#NAME, TRACE_INFO); \
    cl_int status_code; \
    { \
      py::gil_scoped_release release; \
      status_code = NAME ARGLIST; \
    } \
    if (status_code != CL_SUCCESS) \
      throw pyopencl::error(#NAME, status_code);\
  }

#define PYOPENCL_CALL_GUARDED_WITH_TRACE_INFO(NAME, ARGLIST, TRACE_INFO) \
  { \
    PYOPENCL_PRINT_CALL_TRACE_INFO(#NAME, TRACE_INFO); \
    cl_int status_code; \
    status_code = NAME ARGLIST; \
    if (status_code != CL_SUCCESS) \
      throw pyopencl::error(#NAME, status_code);\
  }

#define PYOPENCL_CALL_GUARDED_THREADED(NAME, ARGLIST) \
  { \
    PYOPENCL_PRINT_CALL_TRACE(#NAME); \
    cl_int status_code; \
    { \
      py::gil_scoped_release release; \
      status_code = NAME ARGLIST; \
    } \
    if (status_code != CL_SUCCESS) \
      throw pyopencl::error(#NAME, status_code);\
  }

#define PYOPENCL_CALL_GUARDED(NAME, ARGLIST) \
  { \
    PYOPENCL_PRINT_CALL_TRACE(#NAME); \
    cl_int status_code; \
    status_code = NAME ARGLIST; \
    if (status_code != CL_SUCCESS) \
      throw pyopencl::error(#NAME, status_code);\
  }
#define PYOPENCL_CALL_GUARDED_CLEANUP(NAME, ARGLIST) \
  { \
    PYOPENCL_PRINT_CALL_TRACE(#NAME); \
    cl_int status_code; \
    status_code = NAME ARGLIST; \
    if (status_code != CL_SUCCESS) \
      std::cerr \
        << "PyOpenCL WARNING: a clean-up operation failed (dead context maybe?)" \
        << std::endl \
        << #NAME " failed with code " << status_code \
        << std::endl; \
  }

// }}}

// {{{ get_info helpers
#define PYOPENCL_GET_OPAQUE_INFO(WHAT, FIRST_ARG, SECOND_ARG, CL_TYPE, TYPE) \
  { \
    CL_TYPE param_value; \
    PYOPENCL_CALL_GUARDED(clGet##WHAT##Info, \
          (FIRST_ARG, SECOND_ARG, sizeof(param_value), &param_value, 0)); \
    if (param_value) \
      return py::object(handle_from_new_ptr( \
            new TYPE(param_value, /*retain*/ true))); \
    else \
      return py::none(); \
  }

#define PYOPENCL_GET_VEC_INFO(WHAT, FIRST_ARG, SECOND_ARG, RES_VEC) \
  { \
    size_t size; \
    PYOPENCL_CALL_GUARDED(clGet##WHAT##Info, \
        (FIRST_ARG, SECOND_ARG, 0, 0, &size)); \
    \
    RES_VEC.resize(size / sizeof(RES_VEC.front())); \
    \
    PYOPENCL_CALL_GUARDED(clGet##WHAT##Info, \
        (FIRST_ARG, SECOND_ARG, size, \
         RES_VEC.empty( ) ? nullptr : &RES_VEC.front(), &size)); \
  }

#define PYOPENCL_GET_STR_INFO(WHAT, FIRST_ARG, SECOND_ARG) \
  { \
    size_t param_value_size; \
    PYOPENCL_CALL_GUARDED(clGet##WHAT##Info, \
        (FIRST_ARG, SECOND_ARG, 0, 0, &param_value_size)); \
    \
    std::vector<char> param_value(param_value_size); \
    PYOPENCL_CALL_GUARDED(clGet##WHAT##Info, \
        (FIRST_ARG, SECOND_ARG, param_value_size,  \
         param_value.empty( ) ? nullptr : &param_value.front(), &param_value_size)); \
    \
    return py::cast( \
        param_value.empty( ) ? "" : std::string(&param_value.front(), param_value_size-1)); \
  }




#define PYOPENCL_GET_TYPED_INFO(WHAT, FIRST_ARG, SECOND_ARG, TYPE) \
  { \
    TYPE param_value; \
    PYOPENCL_CALL_GUARDED(clGet##WHAT##Info, \
        (FIRST_ARG, SECOND_ARG, sizeof(param_value), &param_value, 0)); \
    return py::cast(param_value); \
  }

// }}}

// {{{ event helpers --------------------------------------------------------------
#define PYOPENCL_PARSE_WAIT_FOR \
    cl_uint num_events_in_wait_list = 0; \
    std::vector<cl_event> event_wait_list; \
    \
    if (py_wait_for.ptr() != Py_None) \
    { \
      for (py::handle evt: py_wait_for) \
      { \
        event_wait_list.push_back(evt.cast<const event &>().data()); \
        ++num_events_in_wait_list; \
      } \
    }

#define PYOPENCL_WAITLIST_ARGS \
    num_events_in_wait_list, (num_events_in_wait_list == 0) ? nullptr : &event_wait_list.front()

#define PYOPENCL_RETURN_NEW_NANNY_EVENT(evt, obj) \
    try \
    { \
      return new nanny_event(evt, false, obj); \
    } \
    catch (...) \
    { \
      clReleaseEvent(evt); \
      throw; \
    }

#define PYOPENCL_RETURN_NEW_EVENT(evt) \
    try \
    { \
      return new event(evt, false); \
    } \
    catch (...) \
    { \
      clReleaseEvent(evt); \
      throw; \
    }

// }}}

// {{{ equality testing
#define PYOPENCL_EQUALITY_TESTS(cls) \
    bool operator==(cls const &other) const \
    { return data() == other.data(); } \
    bool operator!=(cls const &other) const \
    { return data() != other.data(); } \
    long hash() const \
    { return (long) (intptr_t) data(); }
// }}}



namespace pyopencl
{
  class program;
  class command_queue;

  // {{{ error
  class error : public std::runtime_error
  {
    private:
      std::string m_routine;
      cl_int m_code;

      // This is here because clLinkProgram returns a program
      // object *just* so that there is somewhere for it to
      // stuff the linker logs. :/
      bool m_program_initialized;
      cl_program m_program;

    public:
      error(const char *routine, cl_int c, const char *msg="")
        : std::runtime_error(msg), m_routine(routine), m_code(c),
        m_program_initialized(false), m_program(nullptr)
      { }

      error(const char *routine, cl_program prg, cl_int c,
          const char *msg="")
        : std::runtime_error(msg), m_routine(routine), m_code(c),
        m_program_initialized(true), m_program(prg)
      { }

      virtual ~error()
      {
        if (m_program_initialized)
          clReleaseProgram(m_program);
      }

      const std::string &routine() const
      {
        return m_routine;
      }

      cl_int code() const
      {
        return m_code;
      }

      bool is_out_of_memory() const
      {
        return (code() == CL_MEM_OBJECT_ALLOCATION_FAILURE
            || code() == CL_OUT_OF_RESOURCES
            || code() == CL_OUT_OF_HOST_MEMORY);
      }

      program *get_program() const;

  };

  // }}}


  // {{{ buffer interface helper


  class py_buffer_wrapper : public noncopyable
  {
    private:
      bool m_initialized;

    public:
      Py_buffer m_buf;

    py_buffer_wrapper()
      : m_initialized(false)
    {}

    void get(PyObject *obj, int flags)
    {
#ifdef PYPY_VERSION
      // work around https://bitbucket.org/pypy/pypy/issues/2873
      if (flags & PyBUF_ANY_CONTIGUOUS)
      {
        int flags_wo_cont = flags & ~PyBUF_ANY_CONTIGUOUS;
        if (PyObject_GetBuffer(obj, &m_buf, flags_wo_cont | PyBUF_C_CONTIGUOUS))
        {
          PyErr_Clear();
          if (PyObject_GetBuffer(obj, &m_buf, flags_wo_cont | PyBUF_F_CONTIGUOUS))
            throw py::error_already_set();
        }
      }
      else
#endif
      if (PyObject_GetBuffer(obj, &m_buf, flags))
        throw py::error_already_set();

      m_initialized = true;
    }

    virtual ~py_buffer_wrapper()
    {
      if (m_initialized)
        PyBuffer_Release(&m_buf);
    }
  };


  // }}}

  inline
  py::tuple get_cl_header_version()
  {
    return py::make_tuple(
        PYOPENCL_CL_VERSION >> (3*4),
        (PYOPENCL_CL_VERSION >> (1*4)) & 0xff
        );
  }


  // {{{ platform

  class platform : noncopyable
  {
    private:
      cl_platform_id m_platform;

    public:
      platform(cl_platform_id pid)
      : m_platform(pid)
      { }

      platform(cl_platform_id pid, bool /*retain (ignored)*/)
      : m_platform(pid)
      { }

      cl_platform_id data() const
      {
        return m_platform;
      }

      PYOPENCL_EQUALITY_TESTS(platform);

      py::object get_info(cl_platform_info param_name) const
      {
        switch (param_name)
        {
          case CL_PLATFORM_PROFILE:
          case CL_PLATFORM_VERSION:
          case CL_PLATFORM_NAME:
          case CL_PLATFORM_VENDOR:
#if !(defined(CL_PLATFORM_NVIDIA) && CL_PLATFORM_NVIDIA == 0x3001)
          case CL_PLATFORM_EXTENSIONS:
#endif
            PYOPENCL_GET_STR_INFO(Platform, m_platform, param_name);

#if PYOPENCL_CL_VERSION >= 0x2010
          case CL_PLATFORM_HOST_TIMER_RESOLUTION:
            PYOPENCL_GET_TYPED_INFO(Platform, m_platform, param_name, cl_ulong);
#endif
#if PYOPENCL_CL_VERSION >= 0x3000
          case CL_PLATFORM_NUMERIC_VERSION:
            PYOPENCL_GET_TYPED_INFO(Platform, m_platform, param_name, cl_version);
          case CL_PLATFORM_EXTENSIONS_WITH_VERSION:
            {
              std::vector<cl_name_version> result;
              PYOPENCL_GET_VEC_INFO(Platform, m_platform, param_name, result);
              PYOPENCL_RETURN_VECTOR(cl_name_version, result);
            }
#endif
          default:
            throw error("Platform.get_info", CL_INVALID_VALUE);
        }
      }

      py::list get_devices(cl_device_type devtype);
  };




  inline
  py::list get_platforms()
  {
    cl_uint num_platforms = 0;
    PYOPENCL_CALL_GUARDED(clGetPlatformIDs, (0, 0, &num_platforms));

    std::vector<cl_platform_id> platforms(num_platforms);
    PYOPENCL_CALL_GUARDED(clGetPlatformIDs,
        (num_platforms, platforms.empty( ) ? nullptr : &platforms.front(), &num_platforms));

    py::list result;
    for (cl_platform_id pid: platforms)
      result.append(handle_from_new_ptr(
            new platform(pid)));

    return result;
  }

  // }}}


  // {{{ device

  class device : noncopyable
  {
    public:
      enum reference_type_t {
        REF_NOT_OWNABLE,
#if PYOPENCL_CL_VERSION >= 0x1020
        REF_CL_1_2,
#endif
      };
    private:
      cl_device_id m_device;
      reference_type_t m_ref_type;

    public:
      device(cl_device_id did)
      : m_device(did), m_ref_type(REF_NOT_OWNABLE)
      { }

      device(cl_device_id did, bool retain, reference_type_t ref_type=REF_NOT_OWNABLE)
      : m_device(did), m_ref_type(ref_type)
      {
        if (retain && ref_type != REF_NOT_OWNABLE)
        {
          if (false)
          { }

#if PYOPENCL_CL_VERSION >= 0x1020
          else if (ref_type == REF_CL_1_2)
          {
            PYOPENCL_CALL_GUARDED(clRetainDevice, (did));
          }
#endif

          else
            throw error("Device", CL_INVALID_VALUE,
                "cannot own references to devices when device fission or CL 1.2 is not available");
        }
      }

      ~device()
      {
#if PYOPENCL_CL_VERSION >= 0x1020
        if (m_ref_type == REF_CL_1_2)
          PYOPENCL_CALL_GUARDED_CLEANUP(clReleaseDevice, (m_device));
#endif
      }

      cl_device_id data() const
      {
        return m_device;
      }

      PYOPENCL_EQUALITY_TESTS(device);

      py::object get_info(cl_device_info param_name) const
      {
#define DEV_GET_INT_INF(TYPE) \
        PYOPENCL_GET_TYPED_INFO(Device, m_device, param_name, TYPE);

        switch (param_name)
        {
          case CL_DEVICE_TYPE: DEV_GET_INT_INF(cl_device_type);
          case CL_DEVICE_VENDOR_ID: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_MAX_COMPUTE_UNITS: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_MAX_WORK_GROUP_SIZE: DEV_GET_INT_INF(size_t);

          case CL_DEVICE_MAX_WORK_ITEM_SIZES:
            {
              std::vector<size_t> result;
              PYOPENCL_GET_VEC_INFO(Device, m_device, param_name, result);
              PYOPENCL_RETURN_VECTOR(size_t, result);
            }

          case CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE: DEV_GET_INT_INF(cl_uint);

          case CL_DEVICE_MAX_CLOCK_FREQUENCY: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_ADDRESS_BITS: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_MAX_READ_IMAGE_ARGS: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_MAX_WRITE_IMAGE_ARGS: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_MAX_MEM_ALLOC_SIZE: DEV_GET_INT_INF(cl_ulong);
          case CL_DEVICE_IMAGE2D_MAX_WIDTH: DEV_GET_INT_INF(size_t);
          case CL_DEVICE_IMAGE2D_MAX_HEIGHT: DEV_GET_INT_INF(size_t);
          case CL_DEVICE_IMAGE3D_MAX_WIDTH: DEV_GET_INT_INF(size_t);
          case CL_DEVICE_IMAGE3D_MAX_HEIGHT: DEV_GET_INT_INF(size_t);
          case CL_DEVICE_IMAGE3D_MAX_DEPTH: DEV_GET_INT_INF(size_t);
          case CL_DEVICE_IMAGE_SUPPORT: DEV_GET_INT_INF(cl_bool);
          case CL_DEVICE_MAX_PARAMETER_SIZE: DEV_GET_INT_INF(size_t);
          case CL_DEVICE_MAX_SAMPLERS: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_MEM_BASE_ADDR_ALIGN: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_SINGLE_FP_CONFIG: DEV_GET_INT_INF(cl_device_fp_config);
#ifdef CL_DEVICE_DOUBLE_FP_CONFIG
          case CL_DEVICE_DOUBLE_FP_CONFIG: DEV_GET_INT_INF(cl_device_fp_config);
#endif
#ifdef CL_DEVICE_HALF_FP_CONFIG
          case CL_DEVICE_HALF_FP_CONFIG: DEV_GET_INT_INF(cl_device_fp_config);
#endif

          case CL_DEVICE_GLOBAL_MEM_CACHE_TYPE: DEV_GET_INT_INF(cl_device_mem_cache_type);
          case CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_GLOBAL_MEM_CACHE_SIZE: DEV_GET_INT_INF(cl_ulong);
          case CL_DEVICE_GLOBAL_MEM_SIZE: DEV_GET_INT_INF(cl_ulong);

          case CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE: DEV_GET_INT_INF(cl_ulong);
          case CL_DEVICE_MAX_CONSTANT_ARGS: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_LOCAL_MEM_TYPE: DEV_GET_INT_INF(cl_device_local_mem_type);
          case CL_DEVICE_LOCAL_MEM_SIZE: DEV_GET_INT_INF(cl_ulong);
          case CL_DEVICE_ERROR_CORRECTION_SUPPORT: DEV_GET_INT_INF(cl_bool);
          case CL_DEVICE_PROFILING_TIMER_RESOLUTION: DEV_GET_INT_INF(size_t);
          case CL_DEVICE_ENDIAN_LITTLE: DEV_GET_INT_INF(cl_bool);
          case CL_DEVICE_AVAILABLE: DEV_GET_INT_INF(cl_bool);
          case CL_DEVICE_COMPILER_AVAILABLE: DEV_GET_INT_INF(cl_bool);
          case CL_DEVICE_EXECUTION_CAPABILITIES: DEV_GET_INT_INF(cl_device_exec_capabilities);
#if PYOPENCL_CL_VERSION >= 0x2000
          case CL_DEVICE_QUEUE_ON_HOST_PROPERTIES: DEV_GET_INT_INF(cl_command_queue_properties);
#else
          case CL_DEVICE_QUEUE_PROPERTIES: DEV_GET_INT_INF(cl_command_queue_properties);
#endif

          case CL_DEVICE_NAME:
          case CL_DEVICE_VENDOR:
          case CL_DRIVER_VERSION:
          case CL_DEVICE_PROFILE:
          case CL_DEVICE_VERSION:
          case CL_DEVICE_EXTENSIONS:
            PYOPENCL_GET_STR_INFO(Device, m_device, param_name);

          case CL_DEVICE_PLATFORM:
            PYOPENCL_GET_OPAQUE_INFO(Device, m_device, param_name, cl_platform_id, platform);

#if PYOPENCL_CL_VERSION >= 0x1010
          case CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF: DEV_GET_INT_INF(cl_uint);

          case CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_NATIVE_VECTOR_WIDTH_INT: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF: DEV_GET_INT_INF(cl_uint);

          case CL_DEVICE_HOST_UNIFIED_MEMORY: DEV_GET_INT_INF(cl_bool);
          case CL_DEVICE_OPENCL_C_VERSION:
            PYOPENCL_GET_STR_INFO(Device, m_device, param_name);
#endif
#ifdef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
          case CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV:
          case CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV:
          case CL_DEVICE_REGISTERS_PER_BLOCK_NV:
          case CL_DEVICE_WARP_SIZE_NV:
            DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_GPU_OVERLAP_NV:
          case CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV:
          case CL_DEVICE_INTEGRATED_MEMORY_NV:
            DEV_GET_INT_INF(cl_bool);
#endif
#ifdef CL_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT_NV
          case CL_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT_NV:
            DEV_GET_INT_INF(cl_uint);
#endif
#ifdef CL_DEVICE_PCI_BUS_ID_NV
          case CL_DEVICE_PCI_BUS_ID_NV:
            DEV_GET_INT_INF(cl_uint);
#endif
#ifdef CL_DEVICE_PCI_SLOT_ID_NV
          case CL_DEVICE_PCI_SLOT_ID_NV:
            DEV_GET_INT_INF(cl_uint);
#endif
#ifdef CL_DEVICE_THREAD_TRACE_SUPPORTED_AMD
          case CL_DEVICE_THREAD_TRACE_SUPPORTED_AMD: DEV_GET_INT_INF(cl_bool);
#endif
#ifdef CL_DEVICE_GFXIP_MAJOR_AMD
          case CL_DEVICE_GFXIP_MAJOR_AMD: DEV_GET_INT_INF(cl_uint);
#endif
#ifdef CL_DEVICE_GFXIP_MINOR_AMD
          case CL_DEVICE_GFXIP_MINOR_AMD: DEV_GET_INT_INF(cl_uint);
#endif
#ifdef CL_DEVICE_AVAILABLE_ASYNC_QUEUES_AMD
          case CL_DEVICE_AVAILABLE_ASYNC_QUEUES_AMD: DEV_GET_INT_INF(cl_uint);
#endif
#if PYOPENCL_CL_VERSION >= 0x1020
          case CL_DEVICE_LINKER_AVAILABLE: DEV_GET_INT_INF(cl_bool);
          case CL_DEVICE_BUILT_IN_KERNELS:
            PYOPENCL_GET_STR_INFO(Device, m_device, param_name);
          case CL_DEVICE_IMAGE_MAX_BUFFER_SIZE: DEV_GET_INT_INF(size_t);
          case CL_DEVICE_IMAGE_MAX_ARRAY_SIZE: DEV_GET_INT_INF(size_t);
          case CL_DEVICE_PARENT_DEVICE:
            PYOPENCL_GET_OPAQUE_INFO(Device, m_device, param_name, cl_device_id, device);
          case CL_DEVICE_PARTITION_MAX_SUB_DEVICES: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_PARTITION_TYPE:
          case CL_DEVICE_PARTITION_PROPERTIES:
            {
              std::vector<cl_device_partition_property> result;
              PYOPENCL_GET_VEC_INFO(Device, m_device, param_name, result);
              PYOPENCL_RETURN_VECTOR(cl_device_partition_property, result);
            }
          case CL_DEVICE_PARTITION_AFFINITY_DOMAIN:
            {
#if defined(__GNUG__) && !defined(__clang__)
#pragma GCC diagnostic push
// what's being ignored here is an alignment attribute to native size, which
// shouldn't matter on the relevant ABIs that I'm aware of.
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
              std::vector<cl_device_affinity_domain> result;
#if defined(__GNUG__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
              PYOPENCL_GET_VEC_INFO(Device, m_device, param_name, result);
              PYOPENCL_RETURN_VECTOR(cl_device_affinity_domain, result);
            }
          case CL_DEVICE_REFERENCE_COUNT: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_PREFERRED_INTEROP_USER_SYNC: DEV_GET_INT_INF(cl_bool);
          case CL_DEVICE_PRINTF_BUFFER_SIZE: DEV_GET_INT_INF(cl_bool);
#endif
// {{{ AMD dev attrs cl_amd_device_attribute_query
//
// types of AMD dev attrs divined from
// https://github.com/KhronosGroup/OpenCL-CLHPP/blob/3b03738fef487378b188d21cc5f2bae276aa8721/include/CL/opencl.hpp#L1471-L1500
#ifdef CL_DEVICE_PROFILING_TIMER_OFFSET_AMD
          case CL_DEVICE_PROFILING_TIMER_OFFSET_AMD: DEV_GET_INT_INF(cl_ulong);
#endif
#ifdef CL_DEVICE_TOPOLOGY_AMD
          case CL_DEVICE_TOPOLOGY_AMD:
            PYOPENCL_GET_TYPED_INFO(
                Device, m_device, param_name, cl_device_topology_amd);
#endif
#ifdef CL_DEVICE_BOARD_NAME_AMD
          case CL_DEVICE_BOARD_NAME_AMD: ;
            PYOPENCL_GET_STR_INFO(Device, m_device, param_name);
#endif
#ifdef CL_DEVICE_GLOBAL_FREE_MEMORY_AMD
          case CL_DEVICE_GLOBAL_FREE_MEMORY_AMD:
            {
              std::vector<size_t> result;
              PYOPENCL_GET_VEC_INFO(Device, m_device, param_name, result);
              PYOPENCL_RETURN_VECTOR(size_t, result);
            }
#endif
#ifdef CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD
          case CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD: DEV_GET_INT_INF(cl_uint);
#endif
#ifdef CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD
          case CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD: DEV_GET_INT_INF(cl_uint);
#endif
#ifdef CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD
          case CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD: DEV_GET_INT_INF(cl_uint);
#endif
#ifdef CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD
          case CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD: DEV_GET_INT_INF(cl_uint);
#endif
#ifdef CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD
          case CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD: DEV_GET_INT_INF(cl_uint);
#endif
#ifdef CL_DEVICE_LOCAL_MEM_BANKS_AMD
          case CL_DEVICE_LOCAL_MEM_BANKS_AMD: DEV_GET_INT_INF(cl_uint);
#endif
// FIXME: MISSING:
//
// CL_DEVICE_THREAD_TRACE_SUPPORTED_AMD
// CL_DEVICE_GFXIP_MAJOR_AMD
// CL_DEVICE_GFXIP_MINOR_AMD
// CL_DEVICE_AVAILABLE_ASYNC_QUEUES_AMD
// CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_AMD
// CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD
// CL_DEVICE_PREFERRED_CONSTANT_BUFFER_SIZE_AMD
// CL_DEVICE_PCIE_ID_AMD

// }}}

#ifdef CL_DEVICE_MAX_ATOMIC_COUNTERS_EXT
          case CL_DEVICE_MAX_ATOMIC_COUNTERS_EXT: DEV_GET_INT_INF(cl_uint);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
          case CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE: DEV_GET_INT_INF(size_t);
          case CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES: DEV_GET_INT_INF(cl_command_queue_properties);
          case CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_MAX_ON_DEVICE_QUEUES: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_MAX_ON_DEVICE_EVENTS: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_SVM_CAPABILITIES: DEV_GET_INT_INF(cl_device_svm_capabilities);
          case CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE: DEV_GET_INT_INF(size_t);
          case CL_DEVICE_MAX_PIPE_ARGS: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_PIPE_MAX_PACKET_SIZE: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT: DEV_GET_INT_INF(cl_uint);
#endif
#if PYOPENCL_CL_VERSION >= 0x2010
          case CL_DEVICE_IL_VERSION:
            PYOPENCL_GET_STR_INFO(Device, m_device, param_name);
          case CL_DEVICE_MAX_NUM_SUB_GROUPS: DEV_GET_INT_INF(cl_uint);
          case CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS: DEV_GET_INT_INF(cl_bool);
#endif
#if PYOPENCL_CL_VERSION >= 0x3000
          case CL_DEVICE_NUMERIC_VERSION: DEV_GET_INT_INF(cl_version);
          case CL_DEVICE_EXTENSIONS_WITH_VERSION:
          case CL_DEVICE_ILS_WITH_VERSION:
          case CL_DEVICE_BUILT_IN_KERNELS_WITH_VERSION:
          case CL_DEVICE_OPENCL_C_ALL_VERSIONS:
          case CL_DEVICE_OPENCL_C_FEATURES:
            {
              std::vector<cl_name_version> result;
              PYOPENCL_GET_VEC_INFO(Device, m_device, param_name, result);
              PYOPENCL_RETURN_VECTOR(cl_name_version, result);
            }
          case CL_DEVICE_ATOMIC_MEMORY_CAPABILITIES: DEV_GET_INT_INF(cl_device_atomic_capabilities);
          case CL_DEVICE_ATOMIC_FENCE_CAPABILITIES: DEV_GET_INT_INF(cl_device_atomic_capabilities);
          case CL_DEVICE_NON_UNIFORM_WORK_GROUP_SUPPORT: DEV_GET_INT_INF(cl_bool);
          case CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: DEV_GET_INT_INF(size_t);
          case CL_DEVICE_WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT: DEV_GET_INT_INF(cl_bool);
          case CL_DEVICE_GENERIC_ADDRESS_SPACE_SUPPORT: DEV_GET_INT_INF(cl_bool);

#ifdef CL_DEVICE_DEVICE_ENQUEUE_SUPPORT
          case CL_DEVICE_DEVICE_ENQUEUE_SUPPORT: DEV_GET_INT_INF(cl_bool);
#endif
#ifdef CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES
          case CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES: DEV_GET_INT_INF(cl_device_device_enqueue_capabilities);
#endif

          case CL_DEVICE_PIPE_SUPPORT: DEV_GET_INT_INF(cl_bool);
#endif

#ifdef CL_DEVICE_ME_VERSION_INTEL
          case CL_DEVICE_ME_VERSION_INTEL: DEV_GET_INT_INF(cl_uint);
#endif
#ifdef CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM
          case CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM: DEV_GET_INT_INF(cl_uint);
#endif
#ifdef CL_DEVICE_PAGE_SIZE_QCOM
          case CL_DEVICE_PAGE_SIZE_QCOM: DEV_GET_INT_INF(cl_uint);
#endif
#ifdef CL_DEVICE_SPIR_VERSIONS
          case CL_DEVICE_SPIR_VERSIONS:
            PYOPENCL_GET_STR_INFO(Device, m_device, param_name);
#endif
#ifdef CL_DEVICE_CORE_TEMPERATURE_ALTERA
          case CL_DEVICE_CORE_TEMPERATURE_ALTERA: DEV_GET_INT_INF(cl_int);
#endif

#ifdef CL_DEVICE_SIMULTANEOUS_INTEROPS_INTEL
          case CL_DEVICE_SIMULTANEOUS_INTEROPS_INTEL:
            {
              std::vector<cl_uint> result;
              PYOPENCL_GET_VEC_INFO(Device, m_device, param_name, result);
              PYOPENCL_RETURN_VECTOR(cl_uint, result);
            }
#endif
#ifdef CL_DEVICE_NUM_SIMULTANEOUS_INTEROPS_INTEL
          case CL_DEVICE_NUM_SIMULTANEOUS_INTEROPS_INTEL: DEV_GET_INT_INF(cl_uint);
#endif

          default:
            throw error("Device.get_info", CL_INVALID_VALUE);
        }
      }

#if PYOPENCL_CL_VERSION >= 0x1020
      py::list create_sub_devices(py::object py_properties)
      {
        std::vector<cl_device_partition_property> properties;

        COPY_PY_LIST(cl_device_partition_property, properties);
        properties.push_back(0);

        cl_device_partition_property *props_ptr
          = properties.empty( ) ? nullptr : &properties.front();

        cl_uint num_entries;
        PYOPENCL_CALL_GUARDED(clCreateSubDevices,
            (m_device, props_ptr, 0, nullptr, &num_entries));

        std::vector<cl_device_id> result;
        result.resize(num_entries);

        PYOPENCL_CALL_GUARDED(clCreateSubDevices,
            (m_device, props_ptr, num_entries, &result.front(), nullptr));

        py::list py_result;
        for (cl_device_id did: result)
          py_result.append(handle_from_new_ptr(
                new pyopencl::device(did, /*retain*/true,
                  device::REF_CL_1_2)));
        return py_result;
      }
#endif

#if PYOPENCL_CL_VERSION >= 0x2010
      py::tuple device_and_host_timer() const
      {
        cl_ulong device_timestamp, host_timestamp;
        PYOPENCL_CALL_GUARDED(clGetDeviceAndHostTimer,
            (m_device, &device_timestamp, &host_timestamp));
        return py::make_tuple(device_timestamp, host_timestamp);
      }

      cl_ulong host_timer() const
      {
        cl_ulong host_timestamp;
        PYOPENCL_CALL_GUARDED(clGetHostTimer,
            (m_device, &host_timestamp));
        return host_timestamp;
      }
#endif
  };




  inline py::list platform::get_devices(cl_device_type devtype)
  {
    cl_uint num_devices = 0;
    PYOPENCL_PRINT_CALL_TRACE("clGetDeviceIDs");
    {
      cl_int status_code;
      status_code = clGetDeviceIDs(m_platform, devtype, 0, 0, &num_devices);
      if (status_code == CL_DEVICE_NOT_FOUND)
        num_devices = 0;
      else if (status_code != CL_SUCCESS) \
        throw pyopencl::error("clGetDeviceIDs", status_code);
    }

    if (num_devices == 0)
      return py::list();

    std::vector<cl_device_id> devices(num_devices);
    PYOPENCL_CALL_GUARDED(clGetDeviceIDs,
        (m_platform, devtype,
         num_devices, devices.empty( ) ? nullptr : &devices.front(), &num_devices));

    py::list result;
    for (cl_device_id did: devices)
      result.append(handle_from_new_ptr(
            new device(did)));

    return result;
  }

  // }}}


  // {{{ context

  class context : public noncopyable
  {
    private:
      cl_context m_context;

    public:
      context(cl_context ctx, bool retain)
        : m_context(ctx)
      {
        if (retain)
          PYOPENCL_CALL_GUARDED(clRetainContext, (ctx));
      }

      ~context()
      {
        PYOPENCL_CALL_GUARDED_CLEANUP(clReleaseContext,
            (m_context));
      }

      cl_context data() const
      {
        return m_context;
      }

      PYOPENCL_EQUALITY_TESTS(context);

      py::object get_info(cl_context_info param_name) const
      {
        switch (param_name)
        {
          case CL_CONTEXT_REFERENCE_COUNT:
            PYOPENCL_GET_TYPED_INFO(
                Context, m_context, param_name, cl_uint);

          case CL_CONTEXT_DEVICES:
            {
              std::vector<cl_device_id> result;
              PYOPENCL_GET_VEC_INFO(Context, m_context, param_name, result);

              py::list py_result;
              for (cl_device_id did: result)
                py_result.append(handle_from_new_ptr(
                      new pyopencl::device(did)));
              return py_result;
            }

          case CL_CONTEXT_PROPERTIES:
            {
              std::vector<cl_context_properties> result;
              PYOPENCL_GET_VEC_INFO(Context, m_context, param_name, result);

              py::list py_result;
              for (size_t i = 0; i < result.size(); i+=2)
              {
                cl_context_properties key = result[i];
                py::object value;
                switch (key)
                {
                  case CL_CONTEXT_PLATFORM:
                    {
                      value = py::object(
                          handle_from_new_ptr(new platform(
                            reinterpret_cast<cl_platform_id>(result[i+1]))));
                      break;
                    }

#if defined(PYOPENCL_GL_SHARING_VERSION) && (PYOPENCL_GL_SHARING_VERSION >= 1)
#if defined(__APPLE__) && defined(HAVE_GL)
                  case CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE:
#else
                  case CL_GL_CONTEXT_KHR:
                  case CL_EGL_DISPLAY_KHR:
                  case CL_GLX_DISPLAY_KHR:
                  case CL_WGL_HDC_KHR:
                  case CL_CGL_SHAREGROUP_KHR:
#endif
                    value = py::cast(result[i+1]);
                    break;

#endif
                  case 0:
                    break;

                  default:
                    throw error("Context.get_info", CL_INVALID_VALUE,
                        "unknown context_property key encountered");
                }

                py_result.append(py::make_tuple(result[i], value));
              }
              return py_result;
            }

#if PYOPENCL_CL_VERSION >= 0x1010
          case CL_CONTEXT_NUM_DEVICES:
            PYOPENCL_GET_TYPED_INFO(
                Context, m_context, param_name, cl_uint);
#endif

          default:
            throw error("Context.get_info", CL_INVALID_VALUE);
        }
      }


      // not exposed to python
      int get_hex_platform_version() const
      {
        std::vector<cl_device_id> devices;
        PYOPENCL_GET_VEC_INFO(Context, m_context, CL_CONTEXT_DEVICES, devices);

        if (devices.size() == 0)
          throw error("Context._get_hex_version", CL_INVALID_VALUE,
              "platform has no devices");

        cl_platform_id plat;

        PYOPENCL_CALL_GUARDED(clGetDeviceInfo,
            (devices[0], CL_DEVICE_PLATFORM, sizeof(plat), &plat, nullptr));

        std::string plat_version;
        {
          size_t param_value_size;
          PYOPENCL_CALL_GUARDED(clGetPlatformInfo,
              (plat, CL_PLATFORM_VERSION, 0, 0, &param_value_size));

          std::vector<char> param_value(param_value_size);
          PYOPENCL_CALL_GUARDED(clGetPlatformInfo,
              (plat, CL_PLATFORM_VERSION, param_value_size,
               param_value.empty( ) ? nullptr : &param_value.front(), &param_value_size));

          plat_version =
              param_value.empty( ) ? "" : std::string(&param_value.front(), param_value_size-1);
        }

        int major_ver, minor_ver;
        errno = 0;
        int match_count = sscanf(plat_version.c_str(), "OpenCL %d.%d ", &major_ver, &minor_ver);
        if (errno || match_count != 2)
          throw error("Context._get_hex_platform_version", CL_INVALID_VALUE,
              "Platform version string did not have expected format");

        return major_ver << 12 | minor_ver << 4;
      }

#if PYOPENCL_CL_VERSION >= 0x2010
      void set_default_device_command_queue(device const &dev, command_queue const &queue);
#endif
  };


  inline
  std::vector<cl_context_properties> parse_context_properties(
      py::object py_properties)
  {
    std::vector<cl_context_properties> props;

    if (py_properties.ptr() != Py_None)
    {
      for (py::handle prop_tuple_py: py_properties)
      {
        py::tuple prop_tuple(prop_tuple_py.cast<py::tuple>());

        if (len(prop_tuple) != 2)
          throw error("Context", CL_INVALID_VALUE, "property tuple must have length 2");
        cl_context_properties prop = prop_tuple[0].cast<cl_context_properties>();
        props.push_back(prop);

        if (prop == CL_CONTEXT_PLATFORM)
        {
          props.push_back(
              reinterpret_cast<cl_context_properties>(
                prop_tuple[1].cast<const platform &>().data()));
        }
#if defined(PYOPENCL_GL_SHARING_VERSION) && (PYOPENCL_GL_SHARING_VERSION >= 1)
#if defined(_WIN32)
       else if (prop == CL_WGL_HDC_KHR)
       {
         // size_t is a stand-in for HANDLE, hopefully has the same size.
         size_t hnd = (prop_tuple[1]).cast<size_t>();
         props.push_back(hnd);
       }
#endif
       else if (
#if defined(__APPLE__) && defined(HAVE_GL)
            prop == CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE
#else
            prop == CL_GL_CONTEXT_KHR
            || prop == CL_EGL_DISPLAY_KHR
            || prop == CL_GLX_DISPLAY_KHR
            || prop == CL_CGL_SHAREGROUP_KHR
#endif
           )
       {
          py::object ctypes = py::module::import("ctypes");
          py::object prop = prop_tuple[1], c_void_p = ctypes.attr("c_void_p");
          py::object ptr = ctypes.attr("cast")(prop, c_void_p);
          props.push_back(ptr.attr("value").cast<cl_context_properties>());
       }
#endif
        else
          throw error("Context", CL_INVALID_VALUE, "invalid context property");
      }
      props.push_back(0);
    }

    return props;
  }


  inline
  context *create_context_inner(py::object py_devices, py::object py_properties,
      py::object py_dev_type)
  {
    std::vector<cl_context_properties> props
      = parse_context_properties(py_properties);

    cl_context_properties *props_ptr
      = props.empty( ) ? nullptr : &props.front();

    cl_int status_code;

    cl_context ctx;

    // from device list
    if (py_devices.ptr() != Py_None)
    {
      if (py_dev_type.ptr() != Py_None)
        throw error("Context", CL_INVALID_VALUE,
            "one of 'devices' or 'dev_type' must be None");

      std::vector<cl_device_id> devices;
      for (py::handle py_dev: py_devices)
        devices.push_back(py_dev.cast<const device &>().data());

      PYOPENCL_PRINT_CALL_TRACE("clCreateContext");
      ctx = clCreateContext(
          props_ptr,
          devices.size(),
          devices.empty( ) ? nullptr : &devices.front(),
          0, 0, &status_code);
    }
    // from dev_type
    else
    {
      cl_device_type dev_type = CL_DEVICE_TYPE_DEFAULT;
      if (py_dev_type.ptr() != Py_None)
        dev_type = py_dev_type.cast<cl_device_type>();

      PYOPENCL_PRINT_CALL_TRACE("clCreateContextFromType");
      ctx = clCreateContextFromType(props_ptr, dev_type, 0, 0, &status_code);
    }

    if (status_code != CL_SUCCESS)
      throw pyopencl::error("Context", status_code);

    try
    {
      return new context(ctx, false);
    }
    catch (...)
    {
      PYOPENCL_CALL_GUARDED(clReleaseContext, (ctx));
      throw;
    }
  }


  inline
  context *create_context(py::object py_devices, py::object py_properties,
      py::object py_dev_type)
  {
    PYOPENCL_RETRY_RETURN_IF_MEM_ERROR(
      return create_context_inner(py_devices, py_properties, py_dev_type);
    )
  }

  // }}}


  // {{{ command_queue

  class command_queue
  {
    private:
      cl_command_queue m_queue;

    public:
      command_queue(cl_command_queue q, bool retain)
        : m_queue(q)
      {
        if (retain)
          PYOPENCL_CALL_GUARDED(clRetainCommandQueue, (q));
      }

      command_queue(command_queue const &src)
        : m_queue(src.m_queue)
      {
        PYOPENCL_CALL_GUARDED(clRetainCommandQueue, (m_queue));
      }

      command_queue(
          const context &ctx,
          const device *py_dev=nullptr,
          py::object py_props=py::none())
      {
        cl_device_id dev;
        if (py_dev)
          dev = py_dev->data();
        else
        {
          std::vector<cl_device_id> devs;
          PYOPENCL_GET_VEC_INFO(Context, ctx.data(), CL_CONTEXT_DEVICES, devs);
          if (devs.size() == 0)
            throw pyopencl::error("CommandQueue", CL_INVALID_VALUE,
                "context doesn't have any devices? -- don't know which one to default to");
          dev = devs[0];
        }

        int hex_plat_version = ctx.get_hex_platform_version();

        bool props_given_as_numeric;
        cl_command_queue_properties num_props;
        if (py_props.is_none())
        {
          num_props = 0;
          props_given_as_numeric = true;
        }
        else
        {
          try
          {
            num_props = py::cast<cl_command_queue_properties>(py_props);
            props_given_as_numeric = true;
          }
          catch (py::cast_error &)
          {
            props_given_as_numeric = false;
          }
        }

        if (props_given_as_numeric)
        {
#if PYOPENCL_CL_VERSION >= 0x2000
          if (hex_plat_version  >= 0x2000)
          {
            cl_queue_properties props_list[] = { CL_QUEUE_PROPERTIES, num_props, 0 };

            cl_int status_code;

            PYOPENCL_PRINT_CALL_TRACE("clCreateCommandQueueWithProperties");
            m_queue = clCreateCommandQueueWithProperties(
                ctx.data(), dev, props_list, &status_code);

            if (status_code != CL_SUCCESS)
              throw pyopencl::error("CommandQueue", status_code);
          }
          else
#endif
          {
            cl_int status_code;

            PYOPENCL_PRINT_CALL_TRACE("clCreateCommandQueue");
#if defined(__GNUG__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
            m_queue = clCreateCommandQueue(
                ctx.data(), dev, num_props, &status_code);
#if defined(__GNUG__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
            if (status_code != CL_SUCCESS)
              throw pyopencl::error("CommandQueue", status_code);
          }
        }
        else
        {
#if PYOPENCL_CL_VERSION < 0x2000
            throw error("CommandQueue", CL_INVALID_VALUE,
                "queue properties given as an iterable, "
                "which is only allowed when PyOpenCL was built "
                "against an OpenCL 2+ header");
#else
          if (hex_plat_version  < 0x2000)
          {
            std::cerr <<
                "queue properties given as an iterable, "
                "which uses an OpenCL 2+-only interface, "
                "but the context's platform does not "
                "declare OpenCL 2 support. Proceeding "
                "as requested, but the next thing you see "
                "may be a crash." << std:: endl;
          }

          PYOPENCL_STACK_CONTAINER(cl_queue_properties, props, py::len(py_props) + 1);
          {
            size_t i = 0;
            for (auto prop: py_props)
              props[i++] = py::cast<cl_queue_properties>(prop);
            props[i++] = 0;
          }

          cl_int status_code;
          PYOPENCL_PRINT_CALL_TRACE("clCreateCommandQueueWithProperties");
          m_queue = clCreateCommandQueueWithProperties(
              ctx.data(), dev, PYOPENCL_STACK_CONTAINER_GET_PTR(props), &status_code);

          if (status_code != CL_SUCCESS)
            throw pyopencl::error("CommandQueue", status_code);
#endif
        }
      }

      ~command_queue()
      {
        PYOPENCL_CALL_GUARDED_CLEANUP(clReleaseCommandQueue,
            (m_queue));
      }

      const cl_command_queue data() const
      { return m_queue; }

      PYOPENCL_EQUALITY_TESTS(command_queue);

      py::object get_info(cl_command_queue_info param_name) const
      {
        switch (param_name)
        {
          case CL_QUEUE_CONTEXT:
            PYOPENCL_GET_OPAQUE_INFO(CommandQueue, m_queue, param_name,
                cl_context, context);
          case CL_QUEUE_DEVICE:
            PYOPENCL_GET_OPAQUE_INFO(CommandQueue, m_queue, param_name,
                cl_device_id, device);
          case CL_QUEUE_REFERENCE_COUNT:
            PYOPENCL_GET_TYPED_INFO(CommandQueue, m_queue, param_name,
                cl_uint);
          case CL_QUEUE_PROPERTIES:
            PYOPENCL_GET_TYPED_INFO(CommandQueue, m_queue, param_name,
                cl_command_queue_properties);
#if PYOPENCL_CL_VERSION >= 0x2000
          case CL_QUEUE_SIZE:
            PYOPENCL_GET_TYPED_INFO(CommandQueue, m_queue, param_name,
                cl_uint);
#endif
#if PYOPENCL_CL_VERSION >= 0x2010
          case CL_QUEUE_DEVICE_DEFAULT:
            PYOPENCL_GET_OPAQUE_INFO(
                CommandQueue, m_queue, param_name, cl_command_queue, command_queue);
#endif
#if PYOPENCL_CL_VERSION >= 0x3000
          case CL_QUEUE_PROPERTIES_ARRAY:
            {
              std::vector<cl_queue_properties> result;
              PYOPENCL_GET_VEC_INFO(CommandQueue, m_queue, param_name, result);
              PYOPENCL_RETURN_VECTOR(cl_queue_properties, result);
            }
#endif

          default:
            throw error("CommandQueue.get_info", CL_INVALID_VALUE);
        }
      }

      std::unique_ptr<context> get_context() const
      {
        cl_context param_value;
        PYOPENCL_CALL_GUARDED(clGetCommandQueueInfo,
            (m_queue, CL_QUEUE_CONTEXT, sizeof(param_value), &param_value, 0));
        return std::unique_ptr<context>(
            new context(param_value, /*retain*/ true));
      }

#if PYOPENCL_CL_VERSION < 0x1010
      cl_command_queue_properties set_property(
          cl_command_queue_properties prop,
          bool enable)
      {
        cl_command_queue_properties old_prop;
        PYOPENCL_CALL_GUARDED(clSetCommandQueueProperty,
            (m_queue, prop, PYOPENCL_CAST_BOOL(enable), &old_prop));
        return old_prop;
      }
#endif

      void flush()
      { PYOPENCL_CALL_GUARDED(clFlush, (m_queue)); }
      void finish()
      { PYOPENCL_CALL_GUARDED_THREADED(clFinish, (m_queue)); }

      // not exposed to python
      int get_hex_device_version() const
      {
        cl_device_id dev;

        PYOPENCL_CALL_GUARDED(clGetCommandQueueInfo,
            (m_queue, CL_QUEUE_DEVICE, sizeof(dev), &dev, nullptr));

        std::string dev_version;
        {
          size_t param_value_size;
          PYOPENCL_CALL_GUARDED(clGetDeviceInfo,
              (dev, CL_DEVICE_VERSION, 0, 0, &param_value_size));

          std::vector<char> param_value(param_value_size);
          PYOPENCL_CALL_GUARDED(clGetDeviceInfo,
              (dev, CL_DEVICE_VERSION, param_value_size,
               param_value.empty( ) ? nullptr : &param_value.front(), &param_value_size));

          dev_version =
              param_value.empty( ) ? "" : std::string(&param_value.front(), param_value_size-1);
        }

        int major_ver, minor_ver;
        errno = 0;
        int match_count = sscanf(dev_version.c_str(), "OpenCL %d.%d ", &major_ver, &minor_ver);
        if (errno || match_count != 2)
          throw error("CommandQueue._get_hex_device_version", CL_INVALID_VALUE,
              "Platform version string did not have expected format");

        return major_ver << 12 | minor_ver << 4;
      }
  };

  // }}}


  // {{{ event/synchronization

  class event : noncopyable
  {
    private:
      cl_event m_event;

    public:
      event(cl_event event, bool retain)
        : m_event(event)
      {
        if (retain)
          PYOPENCL_CALL_GUARDED(clRetainEvent, (event));
      }

      event(event const &src)
        : m_event(src.m_event)
      { PYOPENCL_CALL_GUARDED(clRetainEvent, (m_event)); }

      virtual ~event()
      {
        PYOPENCL_CALL_GUARDED_CLEANUP(clReleaseEvent,
            (m_event));
      }

      const cl_event data() const
      { return m_event; }

      PYOPENCL_EQUALITY_TESTS(event);

      py::object get_info(cl_event_info param_name) const
      {
        switch (param_name)
        {
          case CL_EVENT_COMMAND_QUEUE:
            PYOPENCL_GET_OPAQUE_INFO(Event, m_event, param_name,
                cl_command_queue, command_queue);
          case CL_EVENT_COMMAND_TYPE:
            PYOPENCL_GET_TYPED_INFO(Event, m_event, param_name,
                cl_command_type);
          case CL_EVENT_COMMAND_EXECUTION_STATUS:
            PYOPENCL_GET_TYPED_INFO(Event, m_event, param_name,
                cl_int);
          case CL_EVENT_REFERENCE_COUNT:
            PYOPENCL_GET_TYPED_INFO(Event, m_event, param_name,
                cl_uint);
#if PYOPENCL_CL_VERSION >= 0x1010
          case CL_EVENT_CONTEXT:
            PYOPENCL_GET_OPAQUE_INFO(Event, m_event, param_name,
                cl_context, context);
#endif

          default:
            throw error("Event.get_info", CL_INVALID_VALUE);
        }
      }

      py::object get_profiling_info(cl_profiling_info param_name) const
      {
        switch (param_name)
        {
          case CL_PROFILING_COMMAND_QUEUED:
          case CL_PROFILING_COMMAND_SUBMIT:
          case CL_PROFILING_COMMAND_START:
          case CL_PROFILING_COMMAND_END:
#if PYOPENCL_CL_VERSION >= 0x2000
          case CL_PROFILING_COMMAND_COMPLETE:
#endif
            PYOPENCL_GET_TYPED_INFO(EventProfiling, m_event, param_name,
                cl_ulong);
          default:
            throw error("Event.get_profiling_info", CL_INVALID_VALUE);
        }
      }

      virtual void wait()
      {
        PYOPENCL_CALL_GUARDED_THREADED(clWaitForEvents, (1, &m_event));
      }

      // Called from a destructor context below:
      // - Should not release the GIL
      // - Should fail gracefully in the face of errors
      virtual void wait_during_cleanup_without_releasing_the_gil()
      {
        PYOPENCL_CALL_GUARDED_CLEANUP(clWaitForEvents, (1, &m_event));
      }

#if PYOPENCL_CL_VERSION >= 0x1010
    // {{{ set_callback, by way of a a thread-based construction

    private:
      struct event_callback_info_t
      {
        std::mutex m_mutex;
        std::condition_variable m_condvar;

        py::object m_py_event;
        py::object m_py_callback;

        bool m_set_callback_suceeded;

        bool m_notify_thread_wakeup_is_genuine;

        cl_event m_event;
        cl_int m_command_exec_status;

        event_callback_info_t(py::object py_event, py::object py_callback)
        : m_py_event(py_event), m_py_callback(py_callback), m_set_callback_suceeded(true),
        m_notify_thread_wakeup_is_genuine(false)
        {}
      };

      static void CL_CALLBACK evt_callback(cl_event evt, cl_int command_exec_status, void *user_data)
      {
        event_callback_info_t *cb_info = reinterpret_cast<event_callback_info_t *>(user_data);
        {
          std::lock_guard<std::mutex> lg(cb_info->m_mutex);
          cb_info->m_event = evt;
          cb_info->m_command_exec_status = command_exec_status;
          cb_info->m_notify_thread_wakeup_is_genuine = true;
        }

        cb_info->m_condvar.notify_one();
      }

    public:
      void set_callback(cl_int command_exec_callback_type, py::object pfn_event_notify)
      {
        // The reason for doing this via a thread is that we're able to wait on
        // acquiring the GIL. (which we can't in the callback)

        std::unique_ptr<event_callback_info_t> cb_info_holder(
            new event_callback_info_t(
              handle_from_new_ptr(new event(*this)),
              pfn_event_notify));
        event_callback_info_t *cb_info = cb_info_holder.get();

        std::thread notif_thread([cb_info]()
            {
              {
                std::unique_lock<std::mutex> ulk(cb_info->m_mutex);
                cb_info->m_condvar.wait(
                    ulk,
                    [&](){ return cb_info->m_notify_thread_wakeup_is_genuine; });

                // ulk no longer held here, cb_info ready for deletion
              }

              {
                py::gil_scoped_acquire acquire;

                if (cb_info->m_set_callback_suceeded)
                {
                  try {
                    cb_info->m_py_callback(
                        // cb_info->m_py_event,
                        cb_info->m_command_exec_status);
                  }
                  catch (std::exception &exc)
                  {
                    std::cerr
                    << "[pyopencl] event callback handler threw an exception, ignoring: "
                    << exc.what()
                    << std::endl;
                  }
                }

                // Need to hold GIL to delete py::object instances in
                // event_callback_info_t
                delete cb_info;
              }
            });
        // Thread is away--it is now its responsibility to free cb_info.
        cb_info_holder.release();

        // notif_thread should no longer be coupled to the lifetime of the thread.
        notif_thread.detach();

        try
        {
          PYOPENCL_CALL_GUARDED(clSetEventCallback, (
                data(), command_exec_callback_type, &event::evt_callback, cb_info));
        }
        catch (...) {
          // Setting the callback did not succeed. The thread would never
          // be woken up. Wake it up to let it know that it can stop.
          {
            std::lock_guard<std::mutex> lg(cb_info->m_mutex);
            cb_info->m_set_callback_suceeded = false;
            cb_info->m_notify_thread_wakeup_is_genuine = true;
          }
          cb_info->m_condvar.notify_one();
          throw;
        }
      }
      // }}}
#endif
  };

  class nanny_event : public event
  {
    // In addition to everything an event does, the nanny event holds a reference
    // to a Python object and waits for its own completion upon destruction.

    protected:
      std::unique_ptr<py_buffer_wrapper> m_ward;

    public:

      nanny_event(cl_event evt, bool retain, std::unique_ptr<py_buffer_wrapper> &ward)
        : event(evt, retain), m_ward(std::move(ward))
      { }

      ~nanny_event()
      {
        // It appears that Pybind can get very confused if we release the GIL here:
        // https://github.com/inducer/pyopencl/issues/296
        wait_during_cleanup_without_releasing_the_gil();
      }

      py::object get_ward() const
      {
        if (m_ward.get())
        {
          return py::reinterpret_borrow<py::object>(m_ward->m_buf.obj);
        }
        else
          return py::none();
      }

      virtual void wait()
      {
        event::wait();
        m_ward.reset();
      }

      virtual void wait_during_cleanup_without_releasing_the_gil()
      {
        event::wait_during_cleanup_without_releasing_the_gil();
        m_ward.reset();
      }
  };




  inline
  void wait_for_events(py::object events)
  {
    cl_uint num_events_in_wait_list = 0;
    std::vector<cl_event> event_wait_list(len(events));

    for (py::handle evt: events)
      event_wait_list[num_events_in_wait_list++] =
        evt.cast<event &>().data();

    PYOPENCL_CALL_GUARDED_THREADED(clWaitForEvents, (
          PYOPENCL_WAITLIST_ARGS));
  }




#if PYOPENCL_CL_VERSION >= 0x1020
  inline
  event *enqueue_marker_with_wait_list(command_queue &cq,
      py::object py_wait_for)
  {
    PYOPENCL_PARSE_WAIT_FOR;
    cl_event evt;

    PYOPENCL_CALL_GUARDED(clEnqueueMarkerWithWaitList, (
          cq.data(), PYOPENCL_WAITLIST_ARGS, &evt));

    PYOPENCL_RETURN_NEW_EVENT(evt);
  }

  inline
  event *enqueue_barrier_with_wait_list(command_queue &cq,
      py::object py_wait_for)
  {
    PYOPENCL_PARSE_WAIT_FOR;
    cl_event evt;

    PYOPENCL_CALL_GUARDED(clEnqueueBarrierWithWaitList,
        (cq.data(), PYOPENCL_WAITLIST_ARGS, &evt));

    PYOPENCL_RETURN_NEW_EVENT(evt);
  }
#endif


  // {{{ used internally for pre-OpenCL-1.2 contexts

  inline
  event *enqueue_marker(command_queue &cq)
  {
    cl_event evt;

    PYOPENCL_CALL_GUARDED(clEnqueueMarker, (
          cq.data(), &evt));

    PYOPENCL_RETURN_NEW_EVENT(evt);
  }

  inline
  void enqueue_wait_for_events(command_queue &cq, py::object py_events)
  {
    cl_uint num_events = 0;
    std::vector<cl_event> event_list(len(py_events));

    for (py::handle py_evt: py_events)
      event_list[num_events++] = py_evt.cast<event &>().data();

    PYOPENCL_CALL_GUARDED(clEnqueueWaitForEvents, (
          cq.data(), num_events, event_list.empty( ) ? nullptr : &event_list.front()));
  }

  inline
  void enqueue_barrier(command_queue &cq)
  {
    PYOPENCL_CALL_GUARDED(clEnqueueBarrier, (cq.data()));
  }

  // }}}


#if PYOPENCL_CL_VERSION >= 0x1010
  class user_event : public event
  {
    public:
      user_event(cl_event evt, bool retain)
        : event(evt, retain)
      { }

      void set_status(cl_int execution_status)
      {
        PYOPENCL_CALL_GUARDED(clSetUserEventStatus, (data(), execution_status));
      }
  };




  inline
  user_event *create_user_event(context &ctx)
  {
    cl_int status_code;
    PYOPENCL_PRINT_CALL_TRACE("clCreateUserEvent");
    cl_event evt = clCreateUserEvent(ctx.data(), &status_code);

    if (status_code != CL_SUCCESS)
      throw pyopencl::error("UserEvent", status_code);

    try
    {
      return new user_event(evt, false);
    }
    catch (...)
    {
      clReleaseEvent(evt);
      throw;
    }
  }

#endif

  // }}}


  // {{{ memory_object

  py::object create_mem_object_wrapper(cl_mem mem, bool retain);

  class memory_object_holder
  {
    public:
      virtual const cl_mem data() const = 0;

      PYOPENCL_EQUALITY_TESTS(memory_object_holder);

      size_t size() const
      {
        size_t param_value;
        PYOPENCL_CALL_GUARDED(clGetMemObjectInfo,
            (data(), CL_MEM_SIZE, sizeof(param_value), &param_value, 0));
        return param_value;
      }

      py::object get_info(cl_mem_info param_name) const;
  };




  class memory_object : noncopyable, public memory_object_holder
  {
    public:
      typedef std::unique_ptr<py_buffer_wrapper> hostbuf_t;

    private:
      bool m_valid;
      cl_mem m_mem;
      hostbuf_t m_hostbuf;

    public:
      memory_object(cl_mem mem, bool retain, hostbuf_t hostbuf=hostbuf_t())
        : m_valid(true), m_mem(mem)
      {
        if (retain)
          PYOPENCL_CALL_GUARDED(clRetainMemObject, (mem));

        m_hostbuf = std::move(hostbuf);
      }

      memory_object(memory_object &src)
        : m_valid(true), m_mem(src.m_mem),
        m_hostbuf(std::move(src.m_hostbuf))
      {
        PYOPENCL_CALL_GUARDED(clRetainMemObject, (m_mem));
      }

      memory_object(memory_object_holder const &src)
        : m_valid(true), m_mem(src.data())
      {
        PYOPENCL_CALL_GUARDED(clRetainMemObject, (m_mem));
      }

      void release()
      {
        if (!m_valid)
            throw error("MemoryObject.free", CL_INVALID_VALUE,
                "trying to double-unref mem object");
        PYOPENCL_CALL_GUARDED_CLEANUP(clReleaseMemObject, (m_mem));
        m_valid = false;
      }

      virtual ~memory_object()
      {
        if (m_valid)
          release();
      }

      py::object hostbuf()
      {
        if (m_hostbuf.get())
          return py::reinterpret_borrow<py::object>(m_hostbuf->m_buf.obj);
        else
          return py::none();
      }

      const cl_mem data() const
      { return m_mem; }

  };

#if PYOPENCL_CL_VERSION >= 0x1020
  inline
  event *enqueue_migrate_mem_objects(
      command_queue &cq,
      py::object py_mem_objects,
      cl_mem_migration_flags flags,
      py::object py_wait_for)
  {
    PYOPENCL_PARSE_WAIT_FOR;

    std::vector<cl_mem> mem_objects;
    for (py::handle mo: py_mem_objects)
      mem_objects.push_back(mo.cast<const memory_object &>().data());

    cl_event evt;
    PYOPENCL_RETRY_IF_MEM_ERROR(
      PYOPENCL_CALL_GUARDED(clEnqueueMigrateMemObjects, (
            cq.data(),
            mem_objects.size(), mem_objects.empty( ) ? nullptr : &mem_objects.front(),
            flags,
            PYOPENCL_WAITLIST_ARGS, &evt
            ));
      );
    PYOPENCL_RETURN_NEW_EVENT(evt);
  }
#endif

  // }}}


  // {{{ buffer

  inline cl_mem create_buffer(
      cl_context ctx,
      cl_mem_flags flags,
      size_t size,
      void *host_ptr)
  {
    cl_int status_code;
    PYOPENCL_PRINT_CALL_TRACE("clCreateBuffer");
    cl_mem mem = clCreateBuffer(ctx, flags, size, host_ptr, &status_code);

    if (status_code != CL_SUCCESS)
      throw pyopencl::error("create_buffer", status_code);

    return mem;
  }




  inline cl_mem create_buffer_gc(
      cl_context ctx,
      cl_mem_flags flags,
      size_t size,
      void *host_ptr)
  {
    PYOPENCL_RETRY_RETURN_IF_MEM_ERROR(
      return create_buffer(ctx, flags, size, host_ptr);
    );
  }



#if PYOPENCL_CL_VERSION >= 0x1010
  inline cl_mem create_sub_buffer(
      cl_mem buffer, cl_mem_flags flags, cl_buffer_create_type bct,
      const void *buffer_create_info)
  {
    cl_int status_code;
    PYOPENCL_PRINT_CALL_TRACE("clCreateSubBuffer");
    cl_mem mem = clCreateSubBuffer(buffer, flags,
        bct, buffer_create_info, &status_code);

    if (status_code != CL_SUCCESS)
      throw pyopencl::error("clCreateSubBuffer", status_code);

    return mem;
  }




  inline cl_mem create_sub_buffer_gc(
      cl_mem buffer, cl_mem_flags flags, cl_buffer_create_type bct,
      const void *buffer_create_info)
  {
    PYOPENCL_RETRY_RETURN_IF_MEM_ERROR(
      return create_sub_buffer(buffer, flags, bct, buffer_create_info);
    );
  }
#endif



  class buffer : public memory_object
  {
    public:
      buffer(cl_mem mem, bool retain, hostbuf_t hostbuf=hostbuf_t())
        : memory_object(mem, retain, std::move(hostbuf))
      { }

#if PYOPENCL_CL_VERSION >= 0x1010
      buffer *get_sub_region(
          size_t origin, size_t size, cl_mem_flags flags) const
      {
        cl_buffer_region region = { origin, size};

        cl_mem mem = create_sub_buffer_gc(
            data(), flags, CL_BUFFER_CREATE_TYPE_REGION, &region);

        try
        {
          return new buffer(mem, false);
        }
        catch (...)
        {
          PYOPENCL_CALL_GUARDED(clReleaseMemObject, (mem));
          throw;
        }
      }

      buffer *getitem(py::slice slc) const
      {
        PYOPENCL_BUFFER_SIZE_T start, end, stride, length;

        size_t my_length;
        PYOPENCL_CALL_GUARDED(clGetMemObjectInfo,
            (data(), CL_MEM_SIZE, sizeof(my_length), &my_length, 0));

#if PY_VERSION_HEX >= 0x03020000
        if (PySlice_GetIndicesEx(slc.ptr(),
              my_length, &start, &end, &stride, &length) != 0)
          throw py::error_already_set();
#else
        if (PySlice_GetIndicesEx(reinterpret_cast<PySliceObject *>(slc.ptr()),
              my_length, &start, &end, &stride, &length) != 0)
          throw py::error_already_set();
#endif

        if (stride != 1)
          throw pyopencl::error("Buffer.__getitem__", CL_INVALID_VALUE,
              "Buffer slice must have stride 1");

        cl_mem_flags my_flags;
        PYOPENCL_CALL_GUARDED(clGetMemObjectInfo,
            (data(), CL_MEM_FLAGS, sizeof(my_flags), &my_flags, 0));

        my_flags &= ~CL_MEM_COPY_HOST_PTR;

        if (end <= start)
          throw pyopencl::error("Buffer.__getitem__", CL_INVALID_VALUE,
              "Buffer slice have end > start");

        return get_sub_region(start, end-start, my_flags);
      }
#endif
  };

  // {{{ buffer creation

  inline
  buffer *create_buffer_py(
      context &ctx,
      cl_mem_flags flags,
      size_t size,
      py::object py_hostbuf
      )
  {
    if (py_hostbuf.ptr() != Py_None &&
        !(flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR)))
      PyErr_Warn(PyExc_UserWarning, "'hostbuf' was passed, "
          "but no memory flags to make use of it.");

    void *buf = 0;

    std::unique_ptr<py_buffer_wrapper> retained_buf_obj;
    if (py_hostbuf.ptr() != Py_None)
    {
      retained_buf_obj = std::unique_ptr<py_buffer_wrapper>(new py_buffer_wrapper);

      int py_buf_flags = PyBUF_ANY_CONTIGUOUS;
      if ((flags & CL_MEM_USE_HOST_PTR)
          && ((flags & CL_MEM_READ_WRITE)
            || (flags & CL_MEM_WRITE_ONLY)))
        py_buf_flags |= PyBUF_WRITABLE;

      retained_buf_obj->get(py_hostbuf.ptr(), py_buf_flags);

      buf = retained_buf_obj->m_buf.buf;

      if (size > size_t(retained_buf_obj->m_buf.len))
        throw pyopencl::error("Buffer", CL_INVALID_VALUE,
            "specified size is greater than host buffer size");
      if (size == 0)
        size = retained_buf_obj->m_buf.len;
    }

    cl_mem mem = create_buffer_gc(ctx.data(), flags, size, buf);

    if (!(flags & CL_MEM_USE_HOST_PTR))
      retained_buf_obj.reset();

    try
    {
      return new buffer(mem, false, std::move(retained_buf_obj));
    }
    catch (...)
    {
      PYOPENCL_CALL_GUARDED(clReleaseMemObject, (mem));
      throw;
    }
  }

  // }}}

  // {{{ buffer transfers

  // {{{ byte-for-byte transfers

  inline
  event *enqueue_read_buffer(
      command_queue &cq,
      memory_object_holder &mem,
      py::object buffer,
      size_t device_offset,
      py::object py_wait_for,
      bool is_blocking)
  {
    PYOPENCL_PARSE_WAIT_FOR;

    void *buf;
    PYOPENCL_BUFFER_SIZE_T len;

    std::unique_ptr<py_buffer_wrapper> ward(new py_buffer_wrapper);

    ward->get(buffer.ptr(), PyBUF_ANY_CONTIGUOUS | PyBUF_WRITABLE);

    buf = ward->m_buf.buf;
    len = ward->m_buf.len;

    cl_event evt;
    PYOPENCL_RETRY_IF_MEM_ERROR(
      PYOPENCL_CALL_GUARDED_THREADED(clEnqueueReadBuffer, (
            cq.data(),
            mem.data(),
            PYOPENCL_CAST_BOOL(is_blocking),
            device_offset, len, buf,
            PYOPENCL_WAITLIST_ARGS, &evt
            ))
      );
    PYOPENCL_RETURN_NEW_NANNY_EVENT(evt, ward);
  }




  inline
  event *enqueue_write_buffer(
      command_queue &cq,
      memory_object_holder &mem,
      py::object buffer,
      size_t device_offset,
      py::object py_wait_for,
      bool is_blocking)
  {
    PYOPENCL_PARSE_WAIT_FOR;

    const void *buf;
    PYOPENCL_BUFFER_SIZE_T len;

    std::unique_ptr<py_buffer_wrapper> ward(new py_buffer_wrapper);

    ward->get(buffer.ptr(), PyBUF_ANY_CONTIGUOUS);

    buf = ward->m_buf.buf;
    len = ward->m_buf.len;

    cl_event evt;
    PYOPENCL_RETRY_IF_MEM_ERROR(
      PYOPENCL_CALL_GUARDED_THREADED(clEnqueueWriteBuffer, (
            cq.data(),
            mem.data(),
            PYOPENCL_CAST_BOOL(is_blocking),
            device_offset, len, buf,
            PYOPENCL_WAITLIST_ARGS, &evt
            ))
      );
    PYOPENCL_RETURN_NEW_NANNY_EVENT(evt, ward);
  }




  inline
  event *enqueue_copy_buffer(
      command_queue &cq,
      memory_object_holder &src,
      memory_object_holder &dst,
      ptrdiff_t byte_count,
      size_t src_offset,
      size_t dst_offset,
      py::object py_wait_for)
  {
    PYOPENCL_PARSE_WAIT_FOR;

    if (byte_count < 0)
    {
      size_t byte_count_src = 0;
      size_t byte_count_dst = 0;
      PYOPENCL_CALL_GUARDED(clGetMemObjectInfo,
          (src.data(), CL_MEM_SIZE, sizeof(byte_count), &byte_count_src, 0));
      PYOPENCL_CALL_GUARDED(clGetMemObjectInfo,
          (src.data(), CL_MEM_SIZE, sizeof(byte_count), &byte_count_dst, 0));
      byte_count = std::min(byte_count_src, byte_count_dst);
    }

    cl_event evt;
    PYOPENCL_RETRY_IF_MEM_ERROR(
      PYOPENCL_CALL_GUARDED(clEnqueueCopyBuffer, (
            cq.data(),
            src.data(), dst.data(),
            src_offset, dst_offset,
            byte_count,
            PYOPENCL_WAITLIST_ARGS,
            &evt
            ))
      );

    PYOPENCL_RETURN_NEW_EVENT(evt);
  }

  // }}}

  // {{{ rectangular transfers
#if PYOPENCL_CL_VERSION >= 0x1010
  inline
  event *enqueue_read_buffer_rect(
      command_queue &cq,
      memory_object_holder &mem,
      py::object buffer,
      py::object py_buffer_origin,
      py::object py_host_origin,
      py::object py_region,
      py::object py_buffer_pitches,
      py::object py_host_pitches,
      py::object py_wait_for,
      bool is_blocking
      )
  {
    PYOPENCL_PARSE_WAIT_FOR;
    COPY_PY_COORD_TRIPLE(buffer_origin);
    COPY_PY_COORD_TRIPLE(host_origin);
    COPY_PY_REGION_TRIPLE(region);
    COPY_PY_PITCH_TUPLE(buffer_pitches);
    COPY_PY_PITCH_TUPLE(host_pitches);

    void *buf;

    std::unique_ptr<py_buffer_wrapper> ward(new py_buffer_wrapper);

    ward->get(buffer.ptr(), PyBUF_ANY_CONTIGUOUS | PyBUF_WRITABLE);

    buf = ward->m_buf.buf;

    cl_event evt;
    PYOPENCL_RETRY_IF_MEM_ERROR(
      PYOPENCL_CALL_GUARDED_THREADED(clEnqueueReadBufferRect, (
            cq.data(),
            mem.data(),
            PYOPENCL_CAST_BOOL(is_blocking),
            buffer_origin, host_origin, region,
            buffer_pitches[0], buffer_pitches[1],
            host_pitches[0], host_pitches[1],
            buf,
            PYOPENCL_WAITLIST_ARGS, &evt
            ))
      );
    PYOPENCL_RETURN_NEW_NANNY_EVENT(evt, ward);
  }




  inline
  event *enqueue_write_buffer_rect(
      command_queue &cq,
      memory_object_holder &mem,
      py::object buffer,
      py::object py_buffer_origin,
      py::object py_host_origin,
      py::object py_region,
      py::object py_buffer_pitches,
      py::object py_host_pitches,
      py::object py_wait_for,
      bool is_blocking
      )
  {
    PYOPENCL_PARSE_WAIT_FOR;
    COPY_PY_COORD_TRIPLE(buffer_origin);
    COPY_PY_COORD_TRIPLE(host_origin);
    COPY_PY_REGION_TRIPLE(region);
    COPY_PY_PITCH_TUPLE(buffer_pitches);
    COPY_PY_PITCH_TUPLE(host_pitches);

    const void *buf;

    std::unique_ptr<py_buffer_wrapper> ward(new py_buffer_wrapper);

    ward->get(buffer.ptr(), PyBUF_ANY_CONTIGUOUS);

    buf = ward->m_buf.buf;

    cl_event evt;
    PYOPENCL_RETRY_IF_MEM_ERROR(
      PYOPENCL_CALL_GUARDED_THREADED(clEnqueueWriteBufferRect, (
            cq.data(),
            mem.data(),
            PYOPENCL_CAST_BOOL(is_blocking),
            buffer_origin, host_origin, region,
            buffer_pitches[0], buffer_pitches[1],
            host_pitches[0], host_pitches[1],
            buf,
            PYOPENCL_WAITLIST_ARGS, &evt
            ))
      );
    PYOPENCL_RETURN_NEW_NANNY_EVENT(evt, ward);
  }




  inline
  event *enqueue_copy_buffer_rect(
      command_queue &cq,
      memory_object_holder &src,
      memory_object_holder &dst,
      py::object py_src_origin,
      py::object py_dst_origin,
      py::object py_region,
      py::object py_src_pitches,
      py::object py_dst_pitches,
      py::object py_wait_for)
  {
    PYOPENCL_PARSE_WAIT_FOR;
    COPY_PY_COORD_TRIPLE(src_origin);
    COPY_PY_COORD_TRIPLE(dst_origin);
    COPY_PY_REGION_TRIPLE(region);
    COPY_PY_PITCH_TUPLE(src_pitches);
    COPY_PY_PITCH_TUPLE(dst_pitches);

    cl_event evt;
    PYOPENCL_RETRY_IF_MEM_ERROR(
      PYOPENCL_CALL_GUARDED(clEnqueueCopyBufferRect, (
            cq.data(),
            src.data(), dst.data(),
            src_origin, dst_origin, region,
            src_pitches[0], src_pitches[1],
            dst_pitches[0], dst_pitches[1],
            PYOPENCL_WAITLIST_ARGS,
            &evt
            ))
      );

    PYOPENCL_RETURN_NEW_EVENT(evt);
  }

#endif

  // }}}

  // }}}

#if PYOPENCL_CL_VERSION >= 0x1020
  inline
  event *enqueue_fill_buffer(
      command_queue &cq,
      memory_object_holder &mem,
      py::object pattern,
      size_t offset,
      size_t size,
      py::object py_wait_for
      )
  {
    PYOPENCL_PARSE_WAIT_FOR;

    const void *pattern_buf;
    PYOPENCL_BUFFER_SIZE_T pattern_len;

    std::unique_ptr<py_buffer_wrapper> ward(new py_buffer_wrapper);

    ward->get(pattern.ptr(), PyBUF_ANY_CONTIGUOUS);

    pattern_buf = ward->m_buf.buf;
    pattern_len = ward->m_buf.len;

    cl_event evt;
    PYOPENCL_RETRY_IF_MEM_ERROR(
      PYOPENCL_CALL_GUARDED(clEnqueueFillBuffer, (
            cq.data(),
            mem.data(),
            pattern_buf, pattern_len, offset, size,
            PYOPENCL_WAITLIST_ARGS, &evt
            ))
      );
    PYOPENCL_RETURN_NEW_EVENT(evt);
  }
#endif

  // }}}


  // {{{ image

  class image : public memory_object
  {
    public:
      image(cl_mem mem, bool retain, hostbuf_t hostbuf=hostbuf_t())
        : memory_object(mem, retain, std::move(hostbuf))
      { }

      py::object get_image_info(cl_image_info param_name) const
      {
        switch (param_name)
        {
          case CL_IMAGE_FORMAT:
            PYOPENCL_GET_TYPED_INFO(Image, data(), param_name,
                cl_image_format);
          case CL_IMAGE_ELEMENT_SIZE:
          case CL_IMAGE_ROW_PITCH:
          case CL_IMAGE_SLICE_PITCH:
          case CL_IMAGE_WIDTH:
          case CL_IMAGE_HEIGHT:
          case CL_IMAGE_DEPTH:
#if PYOPENCL_CL_VERSION >= 0x1020
          case CL_IMAGE_ARRAY_SIZE:
#endif
            PYOPENCL_GET_TYPED_INFO(Image, data(), param_name, size_t);

#if PYOPENCL_CL_VERSION >= 0x1020
          case CL_IMAGE_BUFFER:
            {
              cl_mem param_value;
              PYOPENCL_CALL_GUARDED(clGetImageInfo, \
                  (data(), param_name, sizeof(param_value), &param_value, 0));
              if (param_value == 0)
              {
                // no associated memory object? no problem.
                return py::none();
              }

              return create_mem_object_wrapper(param_value, /* retain */ true);
            }

          case CL_IMAGE_NUM_MIP_LEVELS:
          case CL_IMAGE_NUM_SAMPLES:
            PYOPENCL_GET_TYPED_INFO(Image, data(), param_name, cl_uint);
#endif

          default:
            throw error("Image.get_image_info", CL_INVALID_VALUE);
        }
      }
  };




  // {{{ image formats

  inline
  cl_image_format *make_image_format(cl_channel_order ord, cl_channel_type tp)
  {
    std::unique_ptr<cl_image_format> result(new cl_image_format);
    result->image_channel_order = ord;
    result->image_channel_data_type = tp;
    return result.release();
  }

  inline
  py::list get_supported_image_formats(
      context const &ctx,
      cl_mem_flags flags,
      cl_mem_object_type image_type)
  {
    cl_uint num_image_formats;
    PYOPENCL_CALL_GUARDED(clGetSupportedImageFormats, (
          ctx.data(), flags, image_type,
          0, nullptr, &num_image_formats));

    std::vector<cl_image_format> formats(num_image_formats);
    PYOPENCL_CALL_GUARDED(clGetSupportedImageFormats, (
          ctx.data(), flags, image_type,
          formats.size(), formats.empty( ) ? nullptr : &formats.front(), nullptr));

    PYOPENCL_RETURN_VECTOR(cl_image_format, formats);
  }

  inline
  cl_uint get_image_format_channel_count(cl_image_format const &fmt)
  {
    switch (fmt.image_channel_order)
    {
      case CL_R: return 1;
      case CL_A: return 1;
      case CL_RG: return 2;
      case CL_RA: return 2;
      case CL_RGB: return 3;
      case CL_RGBA: return 4;
      case CL_BGRA: return 4;
      case CL_INTENSITY: return 1;
      case CL_LUMINANCE: return 1;
      default:
        throw pyopencl::error("ImageFormat.channel_dtype_size",
            CL_INVALID_VALUE,
            "unrecognized channel order");
    }
  }

  inline
  cl_uint get_image_format_channel_dtype_size(cl_image_format const &fmt)
  {
    switch (fmt.image_channel_data_type)
    {
      case CL_SNORM_INT8: return 1;
      case CL_SNORM_INT16: return 2;
      case CL_UNORM_INT8: return 1;
      case CL_UNORM_INT16: return 2;
      case CL_UNORM_SHORT_565: return 2;
      case CL_UNORM_SHORT_555: return 2;
      case CL_UNORM_INT_101010: return 4;
      case CL_SIGNED_INT8: return 1;
      case CL_SIGNED_INT16: return 2;
      case CL_SIGNED_INT32: return 4;
      case CL_UNSIGNED_INT8: return 1;
      case CL_UNSIGNED_INT16: return 2;
      case CL_UNSIGNED_INT32: return 4;
      case CL_HALF_FLOAT: return 2;
      case CL_FLOAT: return 4;
      default:
        throw pyopencl::error("ImageFormat.channel_dtype_size",
            CL_INVALID_VALUE,
            "unrecognized channel data type");
    }
  }

  inline
  cl_uint get_image_format_item_size(cl_image_format const &fmt)
  {
    return get_image_format_channel_count(fmt)
      * get_image_format_channel_dtype_size(fmt);
  }

  // }}}

  // {{{ image creation

  inline
  image *create_image(
      context const &ctx,
      cl_mem_flags flags,
      cl_image_format const &fmt,
      py::sequence shape,
      py::sequence pitches,
      py::object buffer)
  {
    if (shape.ptr() == Py_None)
      throw pyopencl::error("Image", CL_INVALID_VALUE,
          "'shape' must be given");

    void *buf = 0;
    PYOPENCL_BUFFER_SIZE_T len = 0;

    std::unique_ptr<py_buffer_wrapper> retained_buf_obj;
    if (buffer.ptr() != Py_None)
    {
      retained_buf_obj = std::unique_ptr<py_buffer_wrapper>(new py_buffer_wrapper);

      int py_buf_flags = PyBUF_ANY_CONTIGUOUS;
      if ((flags & CL_MEM_USE_HOST_PTR)
          && ((flags & CL_MEM_READ_WRITE)
            || (flags & CL_MEM_WRITE_ONLY)))
        py_buf_flags |= PyBUF_WRITABLE;

      retained_buf_obj->get(buffer.ptr(), py_buf_flags);

      buf = retained_buf_obj->m_buf.buf;
      len = retained_buf_obj->m_buf.len;
    }

    unsigned dims = py::len(shape);
    cl_int status_code;
    cl_mem mem;
    if (dims == 2)
    {
      size_t width = (shape[0]).cast<size_t>();
      size_t height = (shape[1]).cast<size_t>();

      size_t pitch = 0;
      if (pitches.ptr() != Py_None)
      {
        if (py::len(pitches) != 1)
          throw pyopencl::error("Image", CL_INVALID_VALUE,
              "invalid length of pitch tuple");
        pitch = (pitches[0]).cast<size_t>();
      }

      // check buffer size
      cl_int itemsize = get_image_format_item_size(fmt);
      if (buf && std::max(pitch, width*itemsize)*height > cl_uint(len))
          throw pyopencl::error("Image", CL_INVALID_VALUE,
              "buffer too small");

      PYOPENCL_PRINT_CALL_TRACE("clCreateImage2D");
      PYOPENCL_RETRY_IF_MEM_ERROR(
          {
            mem = clCreateImage2D(ctx.data(), flags, &fmt,
                width, height, pitch, buf, &status_code);
            if (status_code != CL_SUCCESS)
              throw pyopencl::error("clCreateImage2D", status_code);
          } );

    }
    else if (dims == 3)
    {
      size_t width = (shape[0]).cast<size_t>();
      size_t height = (shape[1]).cast<size_t>();
      size_t depth = (shape[2]).cast<size_t>();

      size_t pitch_x = 0;
      size_t pitch_y = 0;

      if (pitches.ptr() != Py_None)
      {
        if (py::len(pitches) != 2)
          throw pyopencl::error("Image", CL_INVALID_VALUE,
              "invalid length of pitch tuple");

        pitch_x = (pitches[0]).cast<size_t>();
        pitch_y = (pitches[1]).cast<size_t>();
      }

      // check buffer size
      cl_int itemsize = get_image_format_item_size(fmt);
      if (buf &&
          std::max(std::max(pitch_x, width*itemsize)*height, pitch_y)
          * depth > cl_uint(len))
        throw pyopencl::error("Image", CL_INVALID_VALUE,
            "buffer too small");

      PYOPENCL_PRINT_CALL_TRACE("clCreateImage3D");
      PYOPENCL_RETRY_IF_MEM_ERROR(
          {
            mem = clCreateImage3D(ctx.data(), flags, &fmt,
              width, height, depth, pitch_x, pitch_y, buf, &status_code);
            if (status_code != CL_SUCCESS)
              throw pyopencl::error("clCreateImage3D", status_code);
          } );
    }
    else
      throw pyopencl::error("Image", CL_INVALID_VALUE,
          "invalid dimension");

    if (!(flags & CL_MEM_USE_HOST_PTR))
      retained_buf_obj.reset();

    try
    {
      return new image(mem, false, std::move(retained_buf_obj));
    }
    catch (...)
    {
      PYOPENCL_CALL_GUARDED(clReleaseMemObject, (mem));
      throw;
    }
  }

#if PYOPENCL_CL_VERSION >= 0x1020

  inline
  image *create_image_from_desc(
      context const &ctx,
      cl_mem_flags flags,
      cl_image_format const &fmt,
      cl_image_desc &desc,
      py::object buffer)
  {
    if (buffer.ptr() != Py_None &&
        !(flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR)))
      PyErr_Warn(PyExc_UserWarning, "'hostbuf' was passed, "
          "but no memory flags to make use of it.");

    void *buf = 0;

    std::unique_ptr<py_buffer_wrapper> retained_buf_obj;
    if (buffer.ptr() != Py_None)
    {
      retained_buf_obj = std::unique_ptr<py_buffer_wrapper>(new py_buffer_wrapper);

      int py_buf_flags = PyBUF_ANY_CONTIGUOUS;
      if ((flags & CL_MEM_USE_HOST_PTR)
          && ((flags & CL_MEM_READ_WRITE)
            || (flags & CL_MEM_WRITE_ONLY)))
        py_buf_flags |= PyBUF_WRITABLE;

      retained_buf_obj->get(buffer.ptr(), py_buf_flags);

      buf = retained_buf_obj->m_buf.buf;
    }

    PYOPENCL_PRINT_CALL_TRACE("clCreateImage");
    cl_int status_code;
    cl_mem mem = clCreateImage(ctx.data(), flags, &fmt, &desc, buf, &status_code);
    if (status_code != CL_SUCCESS)
      throw pyopencl::error("clCreateImage", status_code);

    if (!(flags & CL_MEM_USE_HOST_PTR))
      retained_buf_obj.reset();

    try
    {
      return new image(mem, false, std::move(retained_buf_obj));
    }
    catch (...)
    {
      PYOPENCL_CALL_GUARDED(clReleaseMemObject, (mem));
      throw;
    }
  }

#endif

  // }}}

  // {{{ image transfers

  inline
  event *enqueue_read_image(
      command_queue &cq,
      image &img,
      py::object py_origin, py::object py_region,
      py::object buffer,
      size_t row_pitch, size_t slice_pitch,
      py::object py_wait_for,
      bool is_blocking)
  {
    PYOPENCL_PARSE_WAIT_FOR;
    COPY_PY_COORD_TRIPLE(origin);
    COPY_PY_REGION_TRIPLE(region);

    void *buf;

    std::unique_ptr<py_buffer_wrapper> ward(new py_buffer_wrapper);

    ward->get(buffer.ptr(), PyBUF_ANY_CONTIGUOUS | PyBUF_WRITABLE);

    buf = ward->m_buf.buf;

    cl_event evt;

    PYOPENCL_RETRY_IF_MEM_ERROR(
      PYOPENCL_CALL_GUARDED(clEnqueueReadImage, (
            cq.data(),
            img.data(),
            PYOPENCL_CAST_BOOL(is_blocking),
            origin, region, row_pitch, slice_pitch, buf,
            PYOPENCL_WAITLIST_ARGS, &evt
            ));
      );
    PYOPENCL_RETURN_NEW_NANNY_EVENT(evt, ward);
  }




  inline
  event *enqueue_write_image(
      command_queue &cq,
      image &img,
      py::object py_origin, py::object py_region,
      py::object buffer,
      size_t row_pitch, size_t slice_pitch,
      py::object py_wait_for,
      bool is_blocking)
  {
    PYOPENCL_PARSE_WAIT_FOR;
    COPY_PY_COORD_TRIPLE(origin);
    COPY_PY_REGION_TRIPLE(region);

    const void *buf;

    std::unique_ptr<py_buffer_wrapper> ward(new py_buffer_wrapper);

    ward->get(buffer.ptr(), PyBUF_ANY_CONTIGUOUS);

    buf = ward->m_buf.buf;

    cl_event evt;
    PYOPENCL_RETRY_IF_MEM_ERROR(
      PYOPENCL_CALL_GUARDED(clEnqueueWriteImage, (
            cq.data(),
            img.data(),
            PYOPENCL_CAST_BOOL(is_blocking),
            origin, region, row_pitch, slice_pitch, buf,
            PYOPENCL_WAITLIST_ARGS, &evt
            ));
      );
    PYOPENCL_RETURN_NEW_NANNY_EVENT(evt, ward);
  }




  inline
  event *enqueue_copy_image(
      command_queue &cq,
      memory_object_holder &src,
      memory_object_holder &dest,
      py::object py_src_origin,
      py::object py_dest_origin,
      py::object py_region,
      py::object py_wait_for
      )
  {
    PYOPENCL_PARSE_WAIT_FOR;
    COPY_PY_COORD_TRIPLE(src_origin);
    COPY_PY_COORD_TRIPLE(dest_origin);
    COPY_PY_REGION_TRIPLE(region);

    cl_event evt;
    PYOPENCL_RETRY_IF_MEM_ERROR(
      PYOPENCL_CALL_GUARDED(clEnqueueCopyImage, (
            cq.data(), src.data(), dest.data(),
            src_origin, dest_origin, region,
            PYOPENCL_WAITLIST_ARGS, &evt
            ));
      );
    PYOPENCL_RETURN_NEW_EVENT(evt);
  }




  inline
  event *enqueue_copy_image_to_buffer(
      command_queue &cq,
      memory_object_holder &src,
      memory_object_holder &dest,
      py::object py_origin,
      py::object py_region,
      size_t offset,
      py::object py_wait_for
      )
  {
    PYOPENCL_PARSE_WAIT_FOR;
    COPY_PY_COORD_TRIPLE(origin);
    COPY_PY_REGION_TRIPLE(region);

    cl_event evt;
    PYOPENCL_RETRY_IF_MEM_ERROR(
      PYOPENCL_CALL_GUARDED(clEnqueueCopyImageToBuffer, (
            cq.data(), src.data(), dest.data(),
            origin, region, offset,
            PYOPENCL_WAITLIST_ARGS, &evt
            ));
      );
    PYOPENCL_RETURN_NEW_EVENT(evt);
  }




  inline
  event *enqueue_copy_buffer_to_image(
      command_queue &cq,
      memory_object_holder &src,
      memory_object_holder &dest,
      size_t offset,
      py::object py_origin,
      py::object py_region,
      py::object py_wait_for
      )
  {
    PYOPENCL_PARSE_WAIT_FOR;
    COPY_PY_COORD_TRIPLE(origin);
    COPY_PY_REGION_TRIPLE(region);

    cl_event evt;
    PYOPENCL_RETRY_IF_MEM_ERROR(
      PYOPENCL_CALL_GUARDED(clEnqueueCopyBufferToImage, (
            cq.data(), src.data(), dest.data(),
            offset, origin, region,
            PYOPENCL_WAITLIST_ARGS, &evt
            ));
      );
    PYOPENCL_RETURN_NEW_EVENT(evt);
  }

  // }}}

#if PYOPENCL_CL_VERSION >= 0x1020
  inline
  event *enqueue_fill_image(
      command_queue &cq,
      memory_object_holder &mem,
      py::object color,
      py::object py_origin, py::object py_region,
      py::object py_wait_for
      )
  {
    PYOPENCL_PARSE_WAIT_FOR;

    COPY_PY_COORD_TRIPLE(origin);
    COPY_PY_REGION_TRIPLE(region);

    const void *color_buf;

    std::unique_ptr<py_buffer_wrapper> ward(new py_buffer_wrapper);

    ward->get(color.ptr(), PyBUF_ANY_CONTIGUOUS);

    color_buf = ward->m_buf.buf;

    cl_event evt;
    PYOPENCL_RETRY_IF_MEM_ERROR(
      PYOPENCL_CALL_GUARDED(clEnqueueFillImage, (
            cq.data(),
            mem.data(),
            color_buf, origin, region,
            PYOPENCL_WAITLIST_ARGS, &evt
            ));
      );
    PYOPENCL_RETURN_NEW_EVENT(evt);
  }
#endif

  // }}}


  // {{{ pipe

  class pipe : public memory_object
  {
    public:
      pipe(cl_mem mem, bool retain)
        : memory_object(mem, retain)
      { }

#if PYOPENCL_CL_VERSION < 0x2000
      typedef void* cl_pipe_info;
#endif

      py::object get_pipe_info(cl_pipe_info param_name) const
      {
#if PYOPENCL_CL_VERSION >= 0x2000
        switch (param_name)
        {
          case CL_PIPE_PACKET_SIZE:
          case CL_PIPE_MAX_PACKETS:
            PYOPENCL_GET_TYPED_INFO(Pipe, data(), param_name, cl_uint);

          default:
            throw error("Pipe.get_pipe_info", CL_INVALID_VALUE);
        }
#else
        throw error("Pipes not available. PyOpenCL was not compiled against a CL2+ header.",
            CL_INVALID_VALUE);
#endif
      }
  };

#if PYOPENCL_CL_VERSION >= 0x2000
  inline
  pipe *create_pipe(
      context const &ctx,
      cl_mem_flags flags,
      cl_uint pipe_packet_size,
      cl_uint pipe_max_packets,
      py::sequence py_props)
  {
    PYOPENCL_STACK_CONTAINER(cl_pipe_properties, props, py::len(py_props) + 1);
    {
      size_t i = 0;
      for (auto prop: py_props)
        props[i++] = py::cast<cl_pipe_properties>(prop);
      props[i++] = 0;
    }

    cl_int status_code;
    PYOPENCL_PRINT_CALL_TRACE("clCreatePipe");

    cl_mem mem = clCreatePipe(
        ctx.data(),
        flags,
        pipe_packet_size,
        pipe_max_packets,
        PYOPENCL_STACK_CONTAINER_GET_PTR(props),
        &status_code);

    if (status_code != CL_SUCCESS)
      throw pyopencl::error("Pipe", status_code);

    try
    {
      return new pipe(mem, false);
    }
    catch (...)
    {
      PYOPENCL_CALL_GUARDED(clReleaseMemObject, (mem));
      throw;
    }
}
#endif

  // }}}


  // {{{ maps
  class memory_map
  {
    private:
      bool m_valid;
      std::shared_ptr<command_queue> m_queue;
      memory_object m_mem;
      void *m_ptr;

    public:
      memory_map(std::shared_ptr<command_queue> cq, memory_object const &mem, void *ptr)
        : m_valid(true), m_queue(cq), m_mem(mem), m_ptr(ptr)
      {
      }

      ~memory_map()
      {
        if (m_valid)
          delete release(0, py::none());
      }

      event *release(command_queue *cq, py::object py_wait_for)
      {
        PYOPENCL_PARSE_WAIT_FOR;

        if (cq == 0)
          cq = m_queue.get();

        cl_event evt;
        PYOPENCL_CALL_GUARDED(clEnqueueUnmapMemObject, (
              cq->data(), m_mem.data(), m_ptr,
              PYOPENCL_WAITLIST_ARGS, &evt
              ));

        m_valid = false;

        PYOPENCL_RETURN_NEW_EVENT(evt);
      }
  };




  // FIXME: Reenable in pypy
#ifndef PYPY_VERSION
  inline
  py::object enqueue_map_buffer(
      std::shared_ptr<command_queue> cq,
      memory_object_holder &buf,
      cl_map_flags flags,
      size_t offset,
      py::object py_shape, py::object dtype,
      py::object py_order, py::object py_strides,
      py::object py_wait_for,
      bool is_blocking
      )
  {
    PYOPENCL_PARSE_WAIT_FOR;
    PYOPENCL_PARSE_NUMPY_ARRAY_SPEC;

    npy_uintp size_in_bytes = tp_descr->elsize;
    for (npy_intp sdim: shape)
      size_in_bytes *= sdim;

    py::object result;

    cl_event evt;
    cl_int status_code;
    PYOPENCL_PRINT_CALL_TRACE("clEnqueueMapBuffer");
    void *mapped;

    PYOPENCL_RETRY_IF_MEM_ERROR(
        {
          {
            py::gil_scoped_release release;
            mapped = clEnqueueMapBuffer(
                  cq->data(), buf.data(),
                  PYOPENCL_CAST_BOOL(is_blocking), flags,
                  offset, size_in_bytes,
                  PYOPENCL_WAITLIST_ARGS, &evt,
                  &status_code);
          }
          if (status_code != CL_SUCCESS)
            throw pyopencl::error("clEnqueueMapBuffer", status_code);
        } );

    event evt_handle(evt, false);

    std::unique_ptr<memory_map> map;
    try
    {
      result = py::object(py::reinterpret_steal<py::object>(PyArray_NewFromDescr(
          &PyArray_Type, tp_descr,
          shape.size(),
          shape.empty() ? nullptr : &shape.front(),
          strides.empty() ? nullptr : &strides.front(),
          mapped, ary_flags, /*obj*/nullptr)));

      if (size_in_bytes != (npy_uintp) PyArray_NBYTES(result.ptr()))
        throw pyopencl::error("enqueue_map_buffer", CL_INVALID_VALUE,
            "miscalculated numpy array size (not contiguous?)");

       map = std::unique_ptr<memory_map>(new memory_map(cq, buf, mapped));
    }
    catch (...)
    {
      PYOPENCL_CALL_GUARDED_CLEANUP(clEnqueueUnmapMemObject, (
            cq->data(), buf.data(), mapped, 0, 0, 0));
      throw;
    }

    py::object map_py(handle_from_new_ptr(map.release()));
    PyArray_BASE(result.ptr()) = map_py.ptr();
    Py_INCREF(map_py.ptr());

    return py::make_tuple(
        result,
        handle_from_new_ptr(new event(evt_handle)));
  }
#endif




  // FIXME: Reenable in pypy
#ifndef PYPY_VERSION
  inline
  py::object enqueue_map_image(
      std::shared_ptr<command_queue> cq,
      memory_object_holder &img,
      cl_map_flags flags,
      py::object py_origin,
      py::object py_region,
      py::object py_shape, py::object dtype,
      py::object py_order, py::object py_strides,
      py::object py_wait_for,
      bool is_blocking
      )
  {
    PYOPENCL_PARSE_WAIT_FOR;
    PYOPENCL_PARSE_NUMPY_ARRAY_SPEC;
    COPY_PY_COORD_TRIPLE(origin);
    COPY_PY_REGION_TRIPLE(region);

    cl_event evt;
    cl_int status_code;
    PYOPENCL_PRINT_CALL_TRACE("clEnqueueMapImage");
    size_t row_pitch, slice_pitch;
    void *mapped;
    PYOPENCL_RETRY_IF_MEM_ERROR(
      {
        {
          py::gil_scoped_release release;
          mapped = clEnqueueMapImage(
                cq->data(), img.data(),
                PYOPENCL_CAST_BOOL(is_blocking), flags,
                origin, region, &row_pitch, &slice_pitch,
                PYOPENCL_WAITLIST_ARGS, &evt,
                &status_code);
        }
        if (status_code != CL_SUCCESS)
          throw pyopencl::error("clEnqueueMapImage", status_code);
      } );

    event evt_handle(evt, false);

    std::unique_ptr<memory_map> map;
    try
    {
       map = std::unique_ptr<memory_map>(new memory_map(cq, img, mapped));
    }
    catch (...)
    {
      PYOPENCL_CALL_GUARDED_CLEANUP(clEnqueueUnmapMemObject, (
            cq->data(), img.data(), mapped, 0, 0, 0));
      throw;
    }

    py::object result = py::reinterpret_steal<py::object>(PyArray_NewFromDescr(
        &PyArray_Type, tp_descr,
        shape.size(),
        shape.empty() ? nullptr : &shape.front(),
        strides.empty() ? nullptr : &strides.front(),
        mapped, ary_flags, /*obj*/nullptr));

    py::object map_py(handle_from_new_ptr(map.release()));
    PyArray_BASE(result.ptr()) = map_py.ptr();
    Py_INCREF(map_py.ptr());

    return py::make_tuple(
        result,
        handle_from_new_ptr(new event(evt_handle)),
        row_pitch, slice_pitch);
  }
#endif

  // }}}


  // {{{ svm

#if PYOPENCL_CL_VERSION >= 0x2000

  class svm_arg_wrapper
  {
    private:
      void *m_ptr;
      PYOPENCL_BUFFER_SIZE_T m_size;
      std::unique_ptr<py_buffer_wrapper> ward;

    public:
      svm_arg_wrapper(py::object holder)
      {
        ward = std::unique_ptr<py_buffer_wrapper>(new py_buffer_wrapper);
#ifdef PYPY_VERSION
        // FIXME: get a read-only buffer
        // Not quite honest, but Pypy doesn't consider numpy arrays
        // created from objects with the __aray_interface__ writeable.
        ward->get(holder.ptr(), PyBUF_ANY_CONTIGUOUS);
#else
        ward->get(holder.ptr(), PyBUF_ANY_CONTIGUOUS | PyBUF_WRITABLE);
#endif
        m_ptr = ward->m_buf.buf;
        m_size = ward->m_buf.len;
      }

      void *ptr() const
      {
        return m_ptr;
      }
      size_t size() const
      {
        return m_size;
      }
  };


  class svm_allocation : noncopyable
  {
    private:
      std::shared_ptr<context> m_context;
      void *m_allocation;

    public:
      svm_allocation(std::shared_ptr<context> const &ctx, size_t size, cl_uint alignment, cl_svm_mem_flags flags)
        : m_context(ctx)
      {
        PYOPENCL_PRINT_CALL_TRACE("clSVMalloc");
        m_allocation = clSVMAlloc(
            ctx->data(),
            flags, size, alignment);

        if (!m_allocation)
          throw pyopencl::error("clSVMAlloc", CL_OUT_OF_RESOURCES);
      }

      ~svm_allocation()
      {
        if (m_allocation)
          release();
      }

      void release()
      {
        if (!m_allocation)
          throw error("SVMAllocation.release", CL_INVALID_VALUE,
              "trying to double-unref svm allocation");

        clSVMFree(m_context->data(), m_allocation);
        m_allocation = nullptr;
      }

      void enqueue_release(command_queue &queue, py::object py_wait_for)
      {
        PYOPENCL_PARSE_WAIT_FOR;

        if (!m_allocation)
          throw error("SVMAllocation.release", CL_INVALID_VALUE,
              "trying to double-unref svm allocation");

        cl_event evt;

        PYOPENCL_CALL_GUARDED_CLEANUP(clEnqueueSVMFree, (
              queue.data(), 1, &m_allocation,
              nullptr, nullptr,
              PYOPENCL_WAITLIST_ARGS, &evt));

        m_allocation = nullptr;
      }

      void *ptr() const
      {
        return m_allocation;
      }

      intptr_t ptr_as_int() const
      {
        return (intptr_t) m_allocation;
      }

      bool operator==(svm_allocation const &other) const
      {
        return m_allocation == other.m_allocation;
      }

      bool operator!=(svm_allocation const &other) const
      {
        return m_allocation != other.m_allocation;
      }
  };


  inline
  event *enqueue_svm_memcpy(
      command_queue &cq,
      cl_bool is_blocking,
      svm_arg_wrapper &dst, svm_arg_wrapper &src,
      py::object py_wait_for
      )
  {
    PYOPENCL_PARSE_WAIT_FOR;

    if (src.size() != dst.size())
      throw error("_enqueue_svm_memcpy", CL_INVALID_VALUE,
          "sizes of source and destination buffer do not match");

    cl_event evt;
    PYOPENCL_CALL_GUARDED(
        clEnqueueSVMMemcpy,
        (
          cq.data(),
          is_blocking,
          dst.ptr(), src.ptr(),
          dst.size(),
          PYOPENCL_WAITLIST_ARGS,
          &evt
        ));

    PYOPENCL_RETURN_NEW_EVENT(evt);
  }


  inline
  event *enqueue_svm_memfill(
      command_queue &cq,
      svm_arg_wrapper &dst, py::object py_pattern,
      py::object byte_count,
      py::object py_wait_for
      )
  {
    PYOPENCL_PARSE_WAIT_FOR;

    const void *pattern_ptr;
    PYOPENCL_BUFFER_SIZE_T pattern_len;

    std::unique_ptr<py_buffer_wrapper> pattern_ward(new py_buffer_wrapper);

    pattern_ward->get(py_pattern.ptr(), PyBUF_ANY_CONTIGUOUS);

    pattern_ptr = pattern_ward->m_buf.buf;
    pattern_len = pattern_ward->m_buf.len;

    size_t fill_size = dst.size();
    if (!byte_count.is_none())
      fill_size = py::cast<size_t>(byte_count);

    cl_event evt;
    PYOPENCL_CALL_GUARDED(
        clEnqueueSVMMemFill,
        (
          cq.data(),
          dst.ptr(), pattern_ptr,
          pattern_len,
          fill_size,
          PYOPENCL_WAITLIST_ARGS,
          &evt
        ));

    PYOPENCL_RETURN_NEW_EVENT(evt);
  }


  inline
  event *enqueue_svm_map(
      command_queue &cq,
      cl_bool is_blocking,
      cl_map_flags flags,
      svm_arg_wrapper &svm,
      py::object py_wait_for
      )
  {
    PYOPENCL_PARSE_WAIT_FOR;

    cl_event evt;
    PYOPENCL_CALL_GUARDED(
        clEnqueueSVMMap,
        (
          cq.data(),
          is_blocking,
          flags,
          svm.ptr(), svm.size(),
          PYOPENCL_WAITLIST_ARGS,
          &evt
        ));

    PYOPENCL_RETURN_NEW_EVENT(evt);
  }


  inline
  event *enqueue_svm_unmap(
      command_queue &cq,
      svm_arg_wrapper &svm,
      py::object py_wait_for
      )
  {
    PYOPENCL_PARSE_WAIT_FOR;

    cl_event evt;
    PYOPENCL_CALL_GUARDED(
        clEnqueueSVMUnmap,
        (
          cq.data(),
          svm.ptr(),
          PYOPENCL_WAITLIST_ARGS,
          &evt
        ));

    PYOPENCL_RETURN_NEW_EVENT(evt);
  }
#endif


#if PYOPENCL_CL_VERSION >= 0x2010
  inline
  event *enqueue_svm_migratemem(
      command_queue &cq,
      py::sequence svms,
      cl_mem_migration_flags flags,
      py::object py_wait_for
      )
  {
    PYOPENCL_PARSE_WAIT_FOR;

    std::vector<const void *> svm_pointers;
    std::vector<size_t> sizes;

    for (py::handle py_svm: svms)
    {
      svm_arg_wrapper &svm(py::cast<svm_arg_wrapper &>(py_svm));

      svm_pointers.push_back(svm.ptr());
      sizes.push_back(svm.size());
    }

    cl_event evt;
    PYOPENCL_CALL_GUARDED(
        clEnqueueSVMMigrateMem,
        (
         cq.data(),
         svm_pointers.size(),
         svm_pointers.empty() ? nullptr : &svm_pointers.front(),
         sizes.empty() ? nullptr : &sizes.front(),
         flags,
         PYOPENCL_WAITLIST_ARGS,
         &evt
        ));

    PYOPENCL_RETURN_NEW_EVENT(evt);
  }
#endif

  // }}}


  // {{{ sampler

  class sampler : noncopyable
  {
    private:
      cl_sampler m_sampler;

    public:
#if PYOPENCL_CL_VERSION >= 0x2000
      sampler(context const &ctx, py::sequence py_props)
      {
        int hex_plat_version = ctx.get_hex_platform_version();

        if (hex_plat_version  < 0x2000)
        {
          std::cerr <<
            "sampler properties given as an iterable, "
            "which uses an OpenCL 2+-only interface, "
            "but the context's platform does not "
            "declare OpenCL 2 support. Proceeding "
            "as requested, but the next thing you see "
            "may be a crash." << std:: endl;
        }

        PYOPENCL_STACK_CONTAINER(cl_sampler_properties, props, py::len(py_props) + 1);
        {
          size_t i = 0;
          for (auto prop: py_props)
            props[i++] = py::cast<cl_sampler_properties>(prop);
          props[i++] = 0;
        }

        cl_int status_code;
        PYOPENCL_PRINT_CALL_TRACE("clCreateSamplerWithProperties");

        m_sampler = clCreateSamplerWithProperties(
            ctx.data(),
            PYOPENCL_STACK_CONTAINER_GET_PTR(props),
            &status_code);

        if (status_code != CL_SUCCESS)
          throw pyopencl::error("Sampler", status_code);
      }
#endif

      sampler(context const &ctx, bool normalized_coordinates,
          cl_addressing_mode am, cl_filter_mode fm)
      {
        PYOPENCL_PRINT_CALL_TRACE("clCreateSampler");

        int hex_plat_version = ctx.get_hex_platform_version();
#if PYOPENCL_CL_VERSION >= 0x2000
        if (hex_plat_version  >= 0x2000)
        {
            cl_sampler_properties props_list[] = {
              CL_SAMPLER_NORMALIZED_COORDS, normalized_coordinates,
              CL_SAMPLER_ADDRESSING_MODE, am,
              CL_SAMPLER_FILTER_MODE, fm,
              0,
            };

            cl_int status_code;

            PYOPENCL_PRINT_CALL_TRACE("clCreateSamplerWithProperties");
            m_sampler = clCreateSamplerWithProperties(
                ctx.data(), props_list, &status_code);

            if (status_code != CL_SUCCESS)
              throw pyopencl::error("Sampler", status_code);
        }
        else
#endif
        {
          cl_int status_code;

#if defined(__GNUG__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
          m_sampler = clCreateSampler(
              ctx.data(),
              normalized_coordinates,
              am, fm, &status_code);
#if defined(__GNUG__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

          if (status_code != CL_SUCCESS)
            throw pyopencl::error("Sampler", status_code);
        }
      }

      sampler(cl_sampler samp, bool retain)
        : m_sampler(samp)
      {
        if (retain)
          PYOPENCL_CALL_GUARDED(clRetainSampler, (samp));
      }

      ~sampler()
      {
        PYOPENCL_CALL_GUARDED_CLEANUP(clReleaseSampler, (m_sampler));
      }

      cl_sampler data() const
      {
        return m_sampler;
      }

      PYOPENCL_EQUALITY_TESTS(sampler);

      py::object get_info(cl_sampler_info param_name) const
      {
        switch (param_name)
        {
          case CL_SAMPLER_REFERENCE_COUNT:
            PYOPENCL_GET_TYPED_INFO(Sampler, m_sampler, param_name,
                cl_uint);
          case CL_SAMPLER_CONTEXT:
            PYOPENCL_GET_OPAQUE_INFO(Sampler, m_sampler, param_name,
                cl_context, context);
          case CL_SAMPLER_ADDRESSING_MODE:
            PYOPENCL_GET_TYPED_INFO(Sampler, m_sampler, param_name,
                cl_addressing_mode);
          case CL_SAMPLER_FILTER_MODE:
            PYOPENCL_GET_TYPED_INFO(Sampler, m_sampler, param_name,
                cl_filter_mode);
          case CL_SAMPLER_NORMALIZED_COORDS:
            PYOPENCL_GET_TYPED_INFO(Sampler, m_sampler, param_name,
                cl_bool);
#if PYOPENCL_CL_VERSION >= 0x3000
          case CL_SAMPLER_PROPERTIES:
            {
              std::vector<cl_sampler_properties> result;
              PYOPENCL_GET_VEC_INFO(Sampler, m_sampler, param_name, result);
              PYOPENCL_RETURN_VECTOR(cl_sampler_properties, result);
            }
#endif

#ifdef CL_SAMPLER_MIP_FILTER_MODE_KHR
          case CL_SAMPLER_MIP_FILTER_MODE_KHR:
            PYOPENCL_GET_TYPED_INFO(Sampler, m_sampler, param_name,
                cl_filter_mode);
          case CL_SAMPLER_LOD_MIN_KHR:
          case CL_SAMPLER_LOD_MAX_KHR:
            PYOPENCL_GET_TYPED_INFO(Sampler, m_sampler, param_name, float);
#endif

          default:
            throw error("Sampler.get_info", CL_INVALID_VALUE);
        }
      }
  };

  // }}}


  // {{{ program

  class program : noncopyable
  {
    public:
      enum program_kind_type { KND_UNKNOWN, KND_SOURCE, KND_BINARY, KND_IL };

    private:
      cl_program m_program;
      program_kind_type m_program_kind;

    public:
      program(cl_program prog, bool retain, program_kind_type progkind=KND_UNKNOWN)
        : m_program(prog), m_program_kind(progkind)
      {
        if (retain)
          PYOPENCL_CALL_GUARDED(clRetainProgram, (prog));
      }

      ~program()
      {
        PYOPENCL_CALL_GUARDED_CLEANUP(clReleaseProgram, (m_program));
      }

      cl_program data() const
      {
        return m_program;
      }

      program_kind_type kind() const
      {
        return m_program_kind;
      }

      PYOPENCL_EQUALITY_TESTS(program);

      py::object get_info(cl_program_info param_name) const
      {
        switch (param_name)
        {
          case CL_PROGRAM_REFERENCE_COUNT:
            PYOPENCL_GET_TYPED_INFO(Program, m_program, param_name,
                cl_uint);
          case CL_PROGRAM_CONTEXT:
            PYOPENCL_GET_OPAQUE_INFO(Program, m_program, param_name,
                cl_context, context);
          case CL_PROGRAM_NUM_DEVICES:
            PYOPENCL_GET_TYPED_INFO(Program, m_program, param_name, cl_uint);
          case CL_PROGRAM_DEVICES:
            {
              std::vector<cl_device_id> result;
              PYOPENCL_GET_VEC_INFO(Program, m_program, param_name, result);

              py::list py_result;
              for (cl_device_id did: result)
                py_result.append(handle_from_new_ptr(
                      new pyopencl::device(did)));
              return py_result;
            }
          case CL_PROGRAM_SOURCE:
            PYOPENCL_GET_STR_INFO(Program, m_program, param_name);
          case CL_PROGRAM_BINARY_SIZES:
            {
              std::vector<size_t> result;
              PYOPENCL_GET_VEC_INFO(Program, m_program, param_name, result);
              PYOPENCL_RETURN_VECTOR(size_t, result);
            }
          case CL_PROGRAM_BINARIES:
            // {{{
            {
              std::vector<size_t> sizes;
              PYOPENCL_GET_VEC_INFO(Program, m_program, CL_PROGRAM_BINARY_SIZES, sizes);

              size_t total_size = std::accumulate(sizes.begin(), sizes.end(), 0);

              std::unique_ptr<unsigned char []> result(
                  new unsigned char[total_size]);
              std::vector<unsigned char *> result_ptrs;

              unsigned char *ptr = result.get();
              for (unsigned i = 0; i < sizes.size(); ++i)
              {
                result_ptrs.push_back(ptr);
                ptr += sizes[i];
              }

              PYOPENCL_CALL_GUARDED(clGetProgramInfo,
                  (m_program, param_name, sizes.size()*sizeof(unsigned char *),
                   result_ptrs.empty( ) ? nullptr : &result_ptrs.front(), 0)); \

              py::list py_result;
              ptr = result.get();
              for (unsigned i = 0; i < sizes.size(); ++i)
              {
                py::object binary_pyobj(
                    py::reinterpret_steal<py::object>(
#if PY_VERSION_HEX >= 0x03000000
                    PyBytes_FromStringAndSize(
                      reinterpret_cast<char *>(ptr), sizes[i])
#else
                    PyString_FromStringAndSize(
                      reinterpret_cast<char *>(ptr), sizes[i])
#endif
                    ));
                py_result.append(binary_pyobj);
                ptr += sizes[i];
              }
              return py_result;
            }
            // }}}
#if PYOPENCL_CL_VERSION >= 0x1020
          case CL_PROGRAM_NUM_KERNELS:
            PYOPENCL_GET_TYPED_INFO(Program, m_program, param_name,
                size_t);
          case CL_PROGRAM_KERNEL_NAMES:
            PYOPENCL_GET_STR_INFO(Program, m_program, param_name);
#endif
#if PYOPENCL_CL_VERSION >= 0x2010
          case CL_PROGRAM_IL:
            PYOPENCL_GET_STR_INFO(Program, m_program, param_name);
#endif
#if PYOPENCL_CL_VERSION >= 0x2020
          case CL_PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT:
          case CL_PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT:
            PYOPENCL_GET_TYPED_INFO(Program, m_program, param_name, cl_bool);
#endif

          default:
            throw error("Program.get_info", CL_INVALID_VALUE);
        }
      }

      py::object get_build_info(
          device const &dev,
          cl_program_build_info param_name) const
      {
        switch (param_name)
        {
#define PYOPENCL_FIRST_ARG m_program, dev.data() // hackety hack
          case CL_PROGRAM_BUILD_STATUS:
            PYOPENCL_GET_TYPED_INFO(ProgramBuild,
                PYOPENCL_FIRST_ARG, param_name,
                cl_build_status);
          case CL_PROGRAM_BUILD_OPTIONS:
          case CL_PROGRAM_BUILD_LOG:
            PYOPENCL_GET_STR_INFO(ProgramBuild,
                PYOPENCL_FIRST_ARG, param_name);
#if PYOPENCL_CL_VERSION >= 0x1020
          case CL_PROGRAM_BINARY_TYPE:
            PYOPENCL_GET_TYPED_INFO(ProgramBuild,
                PYOPENCL_FIRST_ARG, param_name,
                cl_program_binary_type);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
          case CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE:
            PYOPENCL_GET_TYPED_INFO(ProgramBuild,
                PYOPENCL_FIRST_ARG, param_name,
                size_t);
#endif
#undef PYOPENCL_FIRST_ARG

          default:
            throw error("Program.get_build_info", CL_INVALID_VALUE);
        }
      }

      void build(std::string options, py::object py_devices)
      {
        PYOPENCL_PARSE_PY_DEVICES;

        PYOPENCL_CALL_GUARDED_THREADED(clBuildProgram,
            (m_program, num_devices, devices,
             options.c_str(), 0 ,0));
      }

#if PYOPENCL_CL_VERSION >= 0x1020
      void compile(std::string options, py::object py_devices,
          py::object py_headers)
      {
        PYOPENCL_PARSE_PY_DEVICES;

        // {{{ pick apart py_headers
        // py_headers is a list of tuples *(name, program)*

        std::vector<std::string> header_names;
        std::vector<cl_program> programs;
        for (py::handle name_hdr_tup_py: py_headers)
        {
          py::tuple name_hdr_tup = py::reinterpret_borrow<py::tuple>(name_hdr_tup_py);
          if (py::len(name_hdr_tup) != 2)
            throw error("Program.compile", CL_INVALID_VALUE,
                "epxected (name, header) tuple in headers list");
          std::string name = (name_hdr_tup[0]).cast<std::string>();
          program &prg = (name_hdr_tup[1]).cast<program &>();

          header_names.push_back(name);
          programs.push_back(prg.data());
        }

        std::vector<const char *> header_name_ptrs;
        for (std::string const &name: header_names)
          header_name_ptrs.push_back(name.c_str());

        // }}}

        PYOPENCL_CALL_GUARDED_THREADED(clCompileProgram,
            (m_program, num_devices, devices,
             options.c_str(), header_names.size(),
             programs.empty() ? nullptr : &programs.front(),
             header_name_ptrs.empty() ? nullptr : &header_name_ptrs.front(),
             0, 0));
      }
#endif

#if PYOPENCL_CL_VERSION >= 0x2020
      void set_specialization_constant(cl_uint spec_id, py::object py_buffer)
      {
        py_buffer_wrapper bufwrap;
        bufwrap.get(py_buffer.ptr(), PyBUF_ANY_CONTIGUOUS);
        PYOPENCL_CALL_GUARDED(clSetProgramSpecializationConstant,
            (m_program, spec_id, bufwrap.m_buf.len, bufwrap.m_buf.buf));
      }
#endif
  };




  inline
  program *create_program_with_source(
      context &ctx,
      std::string const &src)
  {
    const char *string = src.c_str();
    size_t length = src.size();

    cl_int status_code;
    PYOPENCL_PRINT_CALL_TRACE("clCreateProgramWithSource");
    cl_program result = clCreateProgramWithSource(
        ctx.data(), 1, &string, &length, &status_code);
    if (status_code != CL_SUCCESS)
      throw pyopencl::error("clCreateProgramWithSource", status_code);

    try
    {
      return new program(result, false, program::KND_SOURCE);
    }
    catch (...)
    {
      clReleaseProgram(result);
      throw;
    }
  }





  inline
  program *create_program_with_binary(
      context &ctx,
      py::sequence py_devices,
      py::sequence py_binaries)
  {
    std::vector<cl_device_id> devices;
    std::vector<const unsigned char *> binaries;
    std::vector<size_t> sizes;

    size_t num_devices = len(py_devices);
    if (len(py_binaries) != num_devices)
      throw error("create_program_with_binary", CL_INVALID_VALUE,
          "device and binary counts don't match");

    for (size_t i = 0; i < num_devices; ++i)
    {
      devices.push_back(
          (py_devices[i]).cast<device const &>().data());
      const void *buf;
      PYOPENCL_BUFFER_SIZE_T len;

      py_buffer_wrapper buf_wrapper;

      buf_wrapper.get(py::object(py_binaries[i]).ptr(), PyBUF_ANY_CONTIGUOUS);

      buf = buf_wrapper.m_buf.buf;
      len = buf_wrapper.m_buf.len;

      binaries.push_back(reinterpret_cast<const unsigned char *>(buf));
      sizes.push_back(len);
    }

    PYOPENCL_STACK_CONTAINER(cl_int, binary_statuses, num_devices);

    cl_int status_code;
    PYOPENCL_PRINT_CALL_TRACE("clCreateProgramWithBinary");
    cl_program result = clCreateProgramWithBinary(
        ctx.data(), num_devices,
        devices.empty( ) ? nullptr : &devices.front(),
        sizes.empty( ) ? nullptr : &sizes.front(),
        binaries.empty( ) ? nullptr : &binaries.front(),
        PYOPENCL_STACK_CONTAINER_GET_PTR(binary_statuses),
        &status_code);
    if (status_code != CL_SUCCESS)
      throw pyopencl::error("clCreateProgramWithBinary", status_code);

    /*
    for (int i = 0; i < num_devices; ++i)
      printf("%d:%d\n", i, binary_statuses[i]);
      */

    try
    {
      return new program(result, false, program::KND_BINARY);
    }
    catch (...)
    {
      clReleaseProgram(result);
      throw;
    }
  }



#if (PYOPENCL_CL_VERSION >= 0x1020) || \
      ((PYOPENCL_CL_VERSION >= 0x1030) && defined(__APPLE__))
  inline
  program *create_program_with_built_in_kernels(
      context &ctx,
      py::object py_devices,
      std::string const &kernel_names)
  {
    PYOPENCL_PARSE_PY_DEVICES;

    cl_int status_code;
    PYOPENCL_PRINT_CALL_TRACE("clCreateProgramWithBuiltInKernels");
    cl_program result = clCreateProgramWithBuiltInKernels(
        ctx.data(), num_devices, devices,
        kernel_names.c_str(), &status_code);
    if (status_code != CL_SUCCESS)
      throw pyopencl::error("clCreateProgramWithBuiltInKernels", status_code);

    try
    {
      return new program(result, false);
    }
    catch (...)
    {
      clReleaseProgram(result);
      throw;
    }
  }
#endif



#if (PYOPENCL_CL_VERSION >= 0x2010)
  inline
  program *create_program_with_il(
      context &ctx,
      std::string const &src)
  {
    cl_int status_code;
    PYOPENCL_PRINT_CALL_TRACE("clCreateProgramWithIL");
    cl_program result = clCreateProgramWithIL(
        ctx.data(), src.c_str(), src.size(), &status_code);
    if (status_code != CL_SUCCESS)
      throw pyopencl::error("clCreateProgramWithIL", status_code);

    try
    {
      return new program(result, false, program::KND_IL);
    }
    catch (...)
    {
      clReleaseProgram(result);
      throw;
    }
  }
#endif





#if PYOPENCL_CL_VERSION >= 0x1020
  inline
  program *link_program(
      context &ctx,
      py::object py_programs,
      std::string const &options,
      py::object py_devices
      )
  {
    PYOPENCL_PARSE_PY_DEVICES;

    std::vector<cl_program> programs;
    for (py::handle py_prg: py_programs)
    {
      program &prg = (py_prg).cast<program &>();
      programs.push_back(prg.data());
    }

    cl_int status_code;
    PYOPENCL_PRINT_CALL_TRACE("clLinkProgram");
    cl_program result = clLinkProgram(
        ctx.data(), num_devices, devices,
        options.c_str(),
        programs.size(),
        programs.empty() ? nullptr : &programs.front(),
        0, 0,
        &status_code);

    if (status_code != CL_SUCCESS)
      throw pyopencl::error("clLinkProgram", result, status_code);

    try
    {
      return new program(result, false);
    }
    catch (...)
    {
      clReleaseProgram(result);
      throw;
    }
  }

#endif


#if PYOPENCL_CL_VERSION >= 0x1020
  inline
  void unload_platform_compiler(platform &plat)
  {
    PYOPENCL_CALL_GUARDED(clUnloadPlatformCompiler, (plat.data()));
  }
#endif

  // }}}


  // {{{ kernel
  class local_memory
  {
    private:
      size_t m_size;

    public:
      local_memory(size_t size)
        : m_size(size)
      { }

      size_t size() const
      { return m_size; }
  };




  class kernel : noncopyable
  {
    private:
      cl_kernel m_kernel;

    public:
      kernel(cl_kernel knl, bool retain)
        : m_kernel(knl)
      {
        if (retain)
          PYOPENCL_CALL_GUARDED(clRetainKernel, (knl));
      }

      kernel(program const &prg, std::string const &kernel_name)
      {
        cl_int status_code;

        PYOPENCL_PRINT_CALL_TRACE("clCreateKernel");
        m_kernel = clCreateKernel(prg.data(), kernel_name.c_str(),
            &status_code);
        if (status_code != CL_SUCCESS)
          throw pyopencl::error("clCreateKernel", status_code);
      }

      ~kernel()
      {
        PYOPENCL_CALL_GUARDED_CLEANUP(clReleaseKernel, (m_kernel));
      }

      cl_kernel data() const
      {
        return m_kernel;
      }

      PYOPENCL_EQUALITY_TESTS(kernel);

#if PYOPENCL_CL_VERSION >= 0x2010
      kernel *clone()
      {
        cl_int status_code;

        PYOPENCL_PRINT_CALL_TRACE("clCloneKernel");
        cl_kernel result = clCloneKernel(m_kernel, &status_code);
        if (status_code != CL_SUCCESS)
          throw pyopencl::error("clCloneKernel", status_code);

        try
        {
          return new kernel(result, /* retain */ false);
        }
        catch (...)
        {
          PYOPENCL_CALL_GUARDED_CLEANUP(clReleaseKernel, (result));
          throw;
        }
      }
#endif

      void set_arg_null(cl_uint arg_index)
      {
        cl_mem m = 0;
        PYOPENCL_CALL_GUARDED(clSetKernelArg, (m_kernel, arg_index,
              sizeof(cl_mem), &m));
      }

      void set_arg_mem(cl_uint arg_index, memory_object_holder &moh)
      {
        cl_mem m = moh.data();
        PYOPENCL_CALL_GUARDED(clSetKernelArg,
            (m_kernel, arg_index, sizeof(cl_mem), &m));
      }

      void set_arg_local(cl_uint arg_index, local_memory const &loc)
      {
        PYOPENCL_CALL_GUARDED(clSetKernelArg,
            (m_kernel, arg_index, loc.size(), 0));
      }

      void set_arg_sampler(cl_uint arg_index, sampler const &smp)
      {
        cl_sampler s = smp.data();
        PYOPENCL_CALL_GUARDED(clSetKernelArg,
            (m_kernel, arg_index, sizeof(cl_sampler), &s));
      }

      void set_arg_command_queue(cl_uint arg_index, command_queue const &queue)
      {
        cl_command_queue q = queue.data();
        PYOPENCL_CALL_GUARDED(clSetKernelArg,
            (m_kernel, arg_index, sizeof(cl_command_queue), &q));
      }

      void set_arg_buf_pack(cl_uint arg_index, py::handle py_typechar, py::handle obj)
      {
        std::string typechar_str(py::cast<std::string>(py_typechar));
        if (typechar_str.size() != 1)
          throw error("Kernel.set_arg_buf_pack", CL_INVALID_VALUE,
              "type char argument must have exactly one character");

        char typechar = typechar_str[0];

#define PYOPENCL_KERNEL_PACK_AND_SET_ARG(TYPECH_VAL, TYPE) \
        case TYPECH_VAL: \
          { \
            TYPE val = py::cast<TYPE>(obj); \
            PYOPENCL_CALL_GUARDED(clSetKernelArg, (m_kernel, arg_index, sizeof(val), &val)); \
            break; \
          }
        switch (typechar)
        {
          PYOPENCL_KERNEL_PACK_AND_SET_ARG('c', char)
          PYOPENCL_KERNEL_PACK_AND_SET_ARG('b', signed char)
          PYOPENCL_KERNEL_PACK_AND_SET_ARG('B', unsigned char)
          PYOPENCL_KERNEL_PACK_AND_SET_ARG('h', short)
          PYOPENCL_KERNEL_PACK_AND_SET_ARG('H', unsigned short)
          PYOPENCL_KERNEL_PACK_AND_SET_ARG('i', int)
          PYOPENCL_KERNEL_PACK_AND_SET_ARG('I', unsigned int)
          PYOPENCL_KERNEL_PACK_AND_SET_ARG('l', long)
          PYOPENCL_KERNEL_PACK_AND_SET_ARG('L', unsigned long)
          PYOPENCL_KERNEL_PACK_AND_SET_ARG('f', float)
          PYOPENCL_KERNEL_PACK_AND_SET_ARG('d', double)
          default:
            throw error("Kernel.set_arg_buf_pack", CL_INVALID_VALUE,
                "invalid type char");
        }
#undef PYOPENCL_KERNEL_PACK_AND_SET_ARG
      }

      void set_arg_buf(cl_uint arg_index, py::handle py_buffer)
      {
        const void *buf;
        PYOPENCL_BUFFER_SIZE_T len;

        py_buffer_wrapper buf_wrapper;

        try
        {
          buf_wrapper.get(py_buffer.ptr(), PyBUF_ANY_CONTIGUOUS);
        }
        catch (py::error_already_set &)
        {
          PyErr_Clear();
          throw error("Kernel.set_arg", CL_INVALID_VALUE,
              "invalid kernel argument");
        }

        buf = buf_wrapper.m_buf.buf;
        len = buf_wrapper.m_buf.len;

        PYOPENCL_CALL_GUARDED(clSetKernelArg,
            (m_kernel, arg_index, len, buf));
      }

#if PYOPENCL_CL_VERSION >= 0x2000
      void set_arg_svm(cl_uint arg_index, svm_arg_wrapper const &wrp)
      {
        PYOPENCL_CALL_GUARDED(clSetKernelArgSVMPointer,
            (m_kernel, arg_index, wrp.ptr()));
      }
#endif

      void set_arg(cl_uint arg_index, py::handle arg)
      {
        if (arg.ptr() == Py_None)
        {
          set_arg_null(arg_index);
          return;
        }

        try
        {
          set_arg_mem(arg_index, arg.cast<memory_object_holder &>());
          return;
        }
        catch (py::cast_error &) { }

#if PYOPENCL_CL_VERSION >= 0x2000
        try
        {
          set_arg_svm(arg_index, arg.cast<svm_arg_wrapper const &>());
          return;
        }
        catch (py::cast_error &) { }
#endif

        try
        {
          set_arg_local(arg_index, arg.cast<local_memory>());
          return;
        }
        catch (py::cast_error &) { }

        try
        {
          set_arg_sampler(arg_index, arg.cast<const sampler &>());
          return;
        }
        catch (py::cast_error &) { }

        try
        {
          set_arg_command_queue(arg_index, arg.cast<const command_queue &>());
          return;
        }
        catch (py::cast_error &) { }

        set_arg_buf(arg_index, arg);
      }

      py::object get_info(cl_kernel_info param_name) const
      {
        switch (param_name)
        {
          case CL_KERNEL_FUNCTION_NAME:
            PYOPENCL_GET_STR_INFO(Kernel, m_kernel, param_name);
          case CL_KERNEL_NUM_ARGS:
          case CL_KERNEL_REFERENCE_COUNT:
            PYOPENCL_GET_TYPED_INFO(Kernel, m_kernel, param_name,
                cl_uint);
          case CL_KERNEL_CONTEXT:
            PYOPENCL_GET_OPAQUE_INFO(Kernel, m_kernel, param_name,
                cl_context, context);
          case CL_KERNEL_PROGRAM:
            PYOPENCL_GET_OPAQUE_INFO(Kernel, m_kernel, param_name,
                cl_program, program);
#if PYOPENCL_CL_VERSION >= 0x1020
          case CL_KERNEL_ATTRIBUTES:
            PYOPENCL_GET_STR_INFO(Kernel, m_kernel, param_name);
#endif
          default:
            throw error("Kernel.get_info", CL_INVALID_VALUE);
        }
      }

      py::object get_work_group_info(
          cl_kernel_work_group_info param_name,
          device const &dev
          ) const
      {
        switch (param_name)
        {
#define PYOPENCL_FIRST_ARG m_kernel, dev.data() // hackety hack
          case CL_KERNEL_WORK_GROUP_SIZE:
            PYOPENCL_GET_TYPED_INFO(KernelWorkGroup,
                PYOPENCL_FIRST_ARG, param_name,
                size_t);
          case CL_KERNEL_COMPILE_WORK_GROUP_SIZE:
            {
              std::vector<size_t> result;
              PYOPENCL_GET_VEC_INFO(KernelWorkGroup,
                  PYOPENCL_FIRST_ARG, param_name, result);

              PYOPENCL_RETURN_VECTOR(size_t, result);
            }
          case CL_KERNEL_LOCAL_MEM_SIZE:
#if PYOPENCL_CL_VERSION >= 0x1010
          case CL_KERNEL_PRIVATE_MEM_SIZE:
#endif
            PYOPENCL_GET_TYPED_INFO(KernelWorkGroup,
                PYOPENCL_FIRST_ARG, param_name,
                cl_ulong);

#if PYOPENCL_CL_VERSION >= 0x1010
          case CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE:
            PYOPENCL_GET_TYPED_INFO(KernelWorkGroup,
                PYOPENCL_FIRST_ARG, param_name,
                size_t);
#endif
          default:
            throw error("Kernel.get_work_group_info", CL_INVALID_VALUE);
#undef PYOPENCL_FIRST_ARG
        }
      }

#if PYOPENCL_CL_VERSION >= 0x1020
      py::object get_arg_info(
          cl_uint arg_index,
          cl_kernel_arg_info param_name
          ) const
      {
        switch (param_name)
        {
#define PYOPENCL_FIRST_ARG m_kernel, arg_index // hackety hack
          case CL_KERNEL_ARG_ADDRESS_QUALIFIER:
            PYOPENCL_GET_TYPED_INFO(KernelArg,
                PYOPENCL_FIRST_ARG, param_name,
                cl_kernel_arg_address_qualifier);

          case CL_KERNEL_ARG_ACCESS_QUALIFIER:
            PYOPENCL_GET_TYPED_INFO(KernelArg,
                PYOPENCL_FIRST_ARG, param_name,
                cl_kernel_arg_access_qualifier);

          case CL_KERNEL_ARG_TYPE_NAME:
          case CL_KERNEL_ARG_NAME:
            PYOPENCL_GET_STR_INFO(KernelArg, PYOPENCL_FIRST_ARG, param_name);

          case CL_KERNEL_ARG_TYPE_QUALIFIER:
            PYOPENCL_GET_TYPED_INFO(KernelArg,
                PYOPENCL_FIRST_ARG, param_name,
                cl_kernel_arg_type_qualifier);
#undef PYOPENCL_FIRST_ARG
          default:
            throw error("Kernel.get_arg_info", CL_INVALID_VALUE);
        }
      }
#endif

#if PYOPENCL_CL_VERSION >= 0x2010
    py::object get_sub_group_info(
        device const &dev,
        cl_kernel_sub_group_info param_name,
        py::object py_input_value)
    {
      switch (param_name)
      {
        // size_t * -> size_t
        case CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE:
        case CL_KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE:
          {
            std::vector<size_t> input_value;
            COPY_PY_LIST(size_t, input_value);

            size_t param_value;
            PYOPENCL_CALL_GUARDED(clGetKernelSubGroupInfo,
                (m_kernel, dev.data(), param_name,
                 input_value.size()*sizeof(input_value.front()),
                 input_value.empty() ? nullptr : &input_value.front(),
                 sizeof(param_value), &param_value, 0));

            return py::cast(param_value);
          }

        // size_t -> size_t[]
        case CL_KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT:
          {
            size_t input_value = py::cast<size_t>(py_input_value);

            std::vector<size_t> result;
            size_t size;
            PYOPENCL_CALL_GUARDED(clGetKernelSubGroupInfo,
                (m_kernel, dev.data(), param_name,
                 sizeof(input_value), &input_value,
                 0, nullptr, &size));
            result.resize(size / sizeof(result.front()));
            PYOPENCL_CALL_GUARDED(clGetKernelSubGroupInfo,
                (m_kernel, dev.data(), param_name,
                 sizeof(input_value), &input_value,
                 size, result.empty() ? nullptr : &result.front(), 0));

            PYOPENCL_RETURN_VECTOR(size_t, result);
          }

        // () -> size_t
        case CL_KERNEL_MAX_NUM_SUB_GROUPS:
        case CL_KERNEL_COMPILE_NUM_SUB_GROUPS:
          {
            size_t param_value;
            PYOPENCL_CALL_GUARDED(clGetKernelSubGroupInfo,
                (m_kernel, dev.data(), param_name,
                 0, nullptr,
                 sizeof(param_value), &param_value, 0));

            return py::cast(param_value);
          }

        default:
          throw error("Kernel.get_sub_group_info", CL_INVALID_VALUE);
      }
  }
#endif
  };

#define PYOPENCL_KERNEL_SET_ARG_MULTI_ERROR_HANDLER \
    catch (error &err) \
    { \
      std::string msg( \
          std::string("when processing arg#") + std::to_string(arg_index+1) \
          + std::string(" (1-based): ") + std::string(err.what())); \
      auto mod_cl_ary(py::module::import("pyopencl.array")); \
      auto cls_array(mod_cl_ary.attr("Array")); \
      if (arg_value.ptr() && py::isinstance(arg_value, cls_array)) \
        msg.append( \
            " (perhaps you meant to pass 'array.data' instead of the array itself?)"); \
      throw error(err.routine().c_str(), err.code(), msg.c_str()); \
    } \
    catch (std::exception &err) \
    { \
      std::string msg( \
          std::string("when processing arg#") + std::to_string(arg_index+1) \
          + std::string(" (1-based): ") + std::string(err.what())); \
      throw std::runtime_error(msg.c_str()); \
    }

  inline
  void set_arg_multi(
      std::function<void(cl_uint, py::handle)> set_arg_func,
      py::tuple args_and_indices)
  {
    cl_uint arg_index;
    py::handle arg_value;

    auto it = args_and_indices.begin(), end = args_and_indices.end();
    try
    {
      /* This is an internal interface that assumes it gets fed well-formed
       * data.  No meaningful error checking is being performed on
       * off-interval exhaustion of the iterator, on purpose.
       */
      while (it != end)
      {
        // special value in case integer cast fails
        arg_index = 9999 - 1;

        arg_index = py::cast<cl_uint>(*it++);
        arg_value = *it++;
        set_arg_func(arg_index, arg_value);
      }
    }
    PYOPENCL_KERNEL_SET_ARG_MULTI_ERROR_HANDLER
  }


  inline
  void set_arg_multi(
      std::function<void(cl_uint, py::handle, py::handle)> set_arg_func,
      py::tuple args_and_indices)
  {
    cl_uint arg_index;
    py::handle arg_descr, arg_value;

    auto it = args_and_indices.begin(), end = args_and_indices.end();
    try
    {
      /* This is an internal interface that assumes it gets fed well-formed
       * data.  No meaningful error checking is being performed on
       * off-interval exhaustion of the iterator, on purpose.
       */
      while (it != end)
      {
        // special value in case integer cast fails
        arg_index = 9999 - 1;

        arg_index = py::cast<cl_uint>(*it++);
        arg_descr = *it++;
        arg_value = *it++;
        set_arg_func(arg_index, arg_descr, arg_value);
      }
    }
    PYOPENCL_KERNEL_SET_ARG_MULTI_ERROR_HANDLER
  }


  inline
  py::list create_kernels_in_program(program &pgm)
  {
    cl_uint num_kernels;
    PYOPENCL_CALL_GUARDED(clCreateKernelsInProgram, (
          pgm.data(), 0, 0, &num_kernels));

    std::vector<cl_kernel> kernels(num_kernels);
    PYOPENCL_CALL_GUARDED(clCreateKernelsInProgram, (
          pgm.data(), num_kernels,
          kernels.empty( ) ? nullptr : &kernels.front(), &num_kernels));

    py::list result;
    for (cl_kernel knl: kernels)
      result.append(handle_from_new_ptr(new kernel(knl, true)));

    return result;
  }

#define MAX_WS_DIM_COUNT 10

  inline
  event *enqueue_nd_range_kernel(
      command_queue &cq,
      kernel &knl,
      py::handle py_global_work_size,
      py::handle py_local_work_size,
      py::handle py_global_work_offset,
      py::handle py_wait_for,
      bool g_times_l,
      bool allow_empty_ndrange)
  {
    PYOPENCL_PARSE_WAIT_FOR;

    std::array<size_t, MAX_WS_DIM_COUNT> global_work_size;
    unsigned gws_size = 0;
    COPY_PY_ARRAY("enqueue_nd_range_kernel", size_t, global_work_size, gws_size);
    cl_uint work_dim = gws_size;

    std::array<size_t, MAX_WS_DIM_COUNT> local_work_size;
    unsigned lws_size = 0;
    size_t *local_work_size_ptr = nullptr;

    if (py_local_work_size.ptr() != Py_None)
    {
      COPY_PY_ARRAY("enqueue_nd_range_kernel", size_t, local_work_size, lws_size);

      if (g_times_l)
        work_dim = std::max(work_dim, lws_size);
      else
        if (work_dim != lws_size)
          throw error("enqueue_nd_range_kernel", CL_INVALID_VALUE,
              "global/local work sizes have differing dimensions");

      while (lws_size < work_dim)
        local_work_size[lws_size++] = 1;
      while (gws_size < work_dim)
        global_work_size[gws_size++] = 1;

      local_work_size_ptr = &local_work_size.front();
    }

    if (g_times_l && lws_size)
    {
      for (cl_uint work_axis = 0; work_axis < work_dim; ++work_axis)
        global_work_size[work_axis] *= local_work_size[work_axis];
    }

    size_t *global_work_offset_ptr = nullptr;
    std::array<size_t, MAX_WS_DIM_COUNT> global_work_offset;
    if (py_global_work_offset.ptr() != Py_None)
    {
      unsigned gwo_size = 0;
      COPY_PY_ARRAY("enqueue_nd_range_kernel", size_t, global_work_offset, gwo_size);

      if (work_dim != gwo_size)
        throw error("enqueue_nd_range_kernel", CL_INVALID_VALUE,
            "global work size and offset have differing dimensions");

      if (g_times_l && local_work_size_ptr)
      {
        for (cl_uint work_axis = 0; work_axis < work_dim; ++work_axis)
          global_work_offset[work_axis] *= local_work_size[work_axis];
      }

      global_work_offset_ptr = &global_work_offset.front();
    }

    if (allow_empty_ndrange)
    {
#if PYOPENCL_CL_VERSION >= 0x1020
      bool is_empty = false;
      for (cl_uint work_axis = 0; work_axis < work_dim; ++work_axis)
        if (global_work_size[work_axis] == 0)
          is_empty = true;
      if (local_work_size_ptr)
        for (cl_uint work_axis = 0; work_axis < work_dim; ++work_axis)
          if (local_work_size_ptr[work_axis] == 0)
            is_empty = true;

      if (is_empty)
      {
        cl_event evt;
        PYOPENCL_CALL_GUARDED(clEnqueueMarkerWithWaitList, (
              cq.data(), PYOPENCL_WAITLIST_ARGS, &evt));
        PYOPENCL_RETURN_NEW_EVENT(evt);
      }
#else
      // clEnqueueWaitForEvents + clEnqueueMarker is not equivalent
      // in the case of an out-of-order queue.
      throw error("enqueue_nd_range_kernel", CL_INVALID_VALUE,
          "allow_empty_ndrange requires OpenCL 1.2");
#endif
    }

    PYOPENCL_RETRY_RETURN_IF_MEM_ERROR( {
          cl_event evt;
          PYOPENCL_CALL_GUARDED(clEnqueueNDRangeKernel, (
                cq.data(),
                knl.data(),
                work_dim,
                global_work_offset_ptr,
                &global_work_size.front(),
                local_work_size_ptr,
                PYOPENCL_WAITLIST_ARGS, &evt
                ));
          PYOPENCL_RETURN_NEW_EVENT(evt);
        } );
  }

  // }}}


  // {{{ gl interop
  inline
  bool have_gl()
  {
#ifdef HAVE_GL
    return true;
#else
    return false;
#endif
  }




#ifdef HAVE_GL

#ifdef __APPLE__
  inline
  cl_context_properties get_apple_cgl_share_group()
  {
    CGLContextObj kCGLContext = CGLGetCurrentContext();
    CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);

    return (cl_context_properties) kCGLShareGroup;
  }
#endif /* __APPLE__ */




  class gl_buffer : public memory_object
  {
    public:
      gl_buffer(cl_mem mem, bool retain, hostbuf_t hostbuf=hostbuf_t())
        : memory_object(mem, retain, std::move(hostbuf))
      { }
  };




  class gl_renderbuffer : public memory_object
  {
    public:
      gl_renderbuffer(cl_mem mem, bool retain, hostbuf_t hostbuf=hostbuf_t())
        : memory_object(mem, retain, std::move(hostbuf))
      { }
  };




  class gl_texture : public image
  {
    public:
      gl_texture(cl_mem mem, bool retain, hostbuf_t hostbuf=hostbuf_t())
        : image(mem, retain, std::move(hostbuf))
      { }

      py::object get_gl_texture_info(cl_gl_texture_info param_name)
      {
        switch (param_name)
        {
          case CL_GL_TEXTURE_TARGET:
            PYOPENCL_GET_TYPED_INFO(GLTexture, data(), param_name, GLenum);
          case CL_GL_MIPMAP_LEVEL:
            PYOPENCL_GET_TYPED_INFO(GLTexture, data(), param_name, GLint);

          default:
            throw error("MemoryObject.get_gl_texture_info", CL_INVALID_VALUE);
        }
      }
  };




#define PYOPENCL_WRAP_BUFFER_CREATOR(TYPE, NAME, CL_NAME, ARGS, CL_ARGS) \
  inline \
  TYPE *NAME ARGS \
  { \
    cl_int status_code; \
    PYOPENCL_PRINT_CALL_TRACE(#CL_NAME); \
    cl_mem mem = CL_NAME CL_ARGS; \
    \
    if (status_code != CL_SUCCESS) \
      throw pyopencl::error(#CL_NAME, status_code); \
    \
    try \
    { \
      return new TYPE(mem, false); \
    } \
    catch (...) \
    { \
      PYOPENCL_CALL_GUARDED(clReleaseMemObject, (mem)); \
      throw; \
    } \
  }




  PYOPENCL_WRAP_BUFFER_CREATOR(gl_buffer,
      create_from_gl_buffer, clCreateFromGLBuffer,
      (context &ctx, cl_mem_flags flags, GLuint bufobj),
      (ctx.data(), flags, bufobj, &status_code));
  PYOPENCL_WRAP_BUFFER_CREATOR(gl_texture,
      create_from_gl_texture_2d, clCreateFromGLTexture2D,
      (context &ctx, cl_mem_flags flags,
         GLenum texture_target, GLint miplevel, GLuint texture),
      (ctx.data(), flags, texture_target, miplevel, texture, &status_code));
  PYOPENCL_WRAP_BUFFER_CREATOR(gl_texture,
      create_from_gl_texture_3d, clCreateFromGLTexture3D,
      (context &ctx, cl_mem_flags flags,
         GLenum texture_target, GLint miplevel, GLuint texture),
      (ctx.data(), flags, texture_target, miplevel, texture, &status_code));
  PYOPENCL_WRAP_BUFFER_CREATOR(gl_renderbuffer,
      create_from_gl_renderbuffer, clCreateFromGLRenderbuffer,
      (context &ctx, cl_mem_flags flags, GLuint renderbuffer),
      (ctx.data(), flags, renderbuffer, &status_code));

  inline
  gl_texture *create_from_gl_texture(
      context &ctx, cl_mem_flags flags,
      GLenum texture_target, GLint miplevel,
      GLuint texture, unsigned dims)
  {
    if (dims == 2)
      return create_from_gl_texture_2d(ctx, flags, texture_target, miplevel, texture);
    else if (dims == 3)
      return create_from_gl_texture_3d(ctx, flags, texture_target, miplevel, texture);
    else
      throw pyopencl::error("Image", CL_INVALID_VALUE,
          "invalid dimension");
  }





  inline
  py::tuple get_gl_object_info(memory_object_holder const &mem)
  {
    cl_gl_object_type otype;
    GLuint gl_name;
    PYOPENCL_CALL_GUARDED(clGetGLObjectInfo, (mem.data(), &otype, &gl_name));
    return py::make_tuple(otype, gl_name);
  }

#define WRAP_GL_ENQUEUE(what, What) \
  inline \
  event *enqueue_##what##_gl_objects( \
      command_queue &cq, \
      py::object py_mem_objects, \
      py::object py_wait_for) \
  { \
    PYOPENCL_PARSE_WAIT_FOR; \
    \
    std::vector<cl_mem> mem_objects; \
    for (py::handle mo: py_mem_objects) \
      mem_objects.push_back((mo).cast<memory_object_holder &>().data()); \
    \
    cl_event evt; \
    PYOPENCL_CALL_GUARDED(clEnqueue##What##GLObjects, ( \
          cq.data(), \
          mem_objects.size(), mem_objects.empty( ) ? nullptr : &mem_objects.front(), \
          PYOPENCL_WAITLIST_ARGS, &evt \
          )); \
    \
    PYOPENCL_RETURN_NEW_EVENT(evt); \
  }

  WRAP_GL_ENQUEUE(acquire, Acquire);
  WRAP_GL_ENQUEUE(release, Release);
#endif




#if defined(cl_khr_gl_sharing) && (cl_khr_gl_sharing >= 1)
  inline
  py::object get_gl_context_info_khr(
      py::object py_properties,
      cl_gl_context_info param_name,
      py::object py_platform
      )
  {
    std::vector<cl_context_properties> props
      = parse_context_properties(py_properties);

    typedef CL_API_ENTRY cl_int (CL_API_CALL
      *func_ptr_type)(const cl_context_properties * /* properties */,
          cl_gl_context_info            /* param_name */,
          size_t                        /* param_value_size */,
          void *                        /* param_value */,
          size_t *                      /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

    func_ptr_type func_ptr;

#if PYOPENCL_CL_VERSION >= 0x1020
    if (py_platform.ptr() != Py_None)
    {
      platform &plat = (py_platform).cast<platform &>();

      func_ptr = (func_ptr_type) clGetExtensionFunctionAddressForPlatform(
            plat.data(), "clGetGLContextInfoKHR");
    }
    else
    {
      PYOPENCL_DEPRECATED("get_gl_context_info_khr with platform=None", "2013.1", );

      func_ptr = (func_ptr_type) clGetExtensionFunctionAddress(
            "clGetGLContextInfoKHR");
    }
#else
    func_ptr = (func_ptr_type) clGetExtensionFunctionAddress(
          "clGetGLContextInfoKHR");
#endif


    if (!func_ptr)
      throw error("Context.get_info", CL_INVALID_PLATFORM,
          "clGetGLContextInfoKHR extension function not present");

    cl_context_properties *props_ptr
      = props.empty( ) ? nullptr : &props.front();

    switch (param_name)
    {
      case CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR:
        {
          cl_device_id param_value;
          PYOPENCL_CALL_GUARDED(func_ptr,
              (props_ptr, param_name, sizeof(param_value), &param_value, 0));
          return py::object(handle_from_new_ptr( \
                new device(param_value, /*retain*/ true)));
        }

      case CL_DEVICES_FOR_GL_CONTEXT_KHR:
        {
          size_t size;
          PYOPENCL_CALL_GUARDED(func_ptr,
              (props_ptr, param_name, 0, 0, &size));

          std::vector<cl_device_id> devices;

          devices.resize(size / sizeof(devices.front()));

          PYOPENCL_CALL_GUARDED(func_ptr,
              (props_ptr, param_name, size,
               devices.empty( ) ? nullptr : &devices.front(), &size));

          py::list result;
          for (cl_device_id did: devices)
            result.append(handle_from_new_ptr(
                  new device(did)));

          return result;
        }

      default:
        throw error("get_gl_context_info_khr", CL_INVALID_VALUE);
    }
  }

#endif

  // }}}


  // {{{ deferred implementation bits

#if PYOPENCL_CL_VERSION >= 0x2010
  inline void context::set_default_device_command_queue(device const &dev, command_queue const &queue)
  {
    PYOPENCL_CALL_GUARDED(clSetDefaultDeviceCommandQueue,
        (m_context, dev.data(), queue.data()));
  }
#endif


  inline program *error::get_program() const
  {
    return new program(m_program, /* retain */ true);
  }

  inline py::object create_mem_object_wrapper(cl_mem mem, bool retain=true)
  {
    cl_mem_object_type mem_obj_type;
    PYOPENCL_CALL_GUARDED(clGetMemObjectInfo, \
        (mem, CL_MEM_TYPE, sizeof(mem_obj_type), &mem_obj_type, 0));

    switch (mem_obj_type)
    {
      case CL_MEM_OBJECT_BUFFER:
        return py::object(handle_from_new_ptr(
              new buffer(mem, retain)));
      case CL_MEM_OBJECT_IMAGE2D:
      case CL_MEM_OBJECT_IMAGE3D:
#if PYOPENCL_CL_VERSION >= 0x1020
      case CL_MEM_OBJECT_IMAGE2D_ARRAY:
      case CL_MEM_OBJECT_IMAGE1D:
      case CL_MEM_OBJECT_IMAGE1D_ARRAY:
      case CL_MEM_OBJECT_IMAGE1D_BUFFER:
#endif
        return py::object(handle_from_new_ptr(
              new image(mem, retain)));
      default:
        return py::object(handle_from_new_ptr(
              new memory_object(mem, retain)));
    }
  }

  inline
  py::object memory_object_from_int(intptr_t cl_mem_as_int, bool retain)
  {
    return create_mem_object_wrapper((cl_mem) cl_mem_as_int, retain);
  }


  inline
  py::object memory_object_holder::get_info(cl_mem_info param_name) const
  {
    switch (param_name)
    {
      case CL_MEM_TYPE:
        PYOPENCL_GET_TYPED_INFO(MemObject, data(), param_name,
            cl_mem_object_type);
      case CL_MEM_FLAGS:
        PYOPENCL_GET_TYPED_INFO(MemObject, data(), param_name,
            cl_mem_flags);
      case CL_MEM_SIZE:
        PYOPENCL_GET_TYPED_INFO(MemObject, data(), param_name,
            size_t);
      case CL_MEM_HOST_PTR:
        throw pyopencl::error("MemoryObject.get_info", CL_INVALID_VALUE,
            "Use MemoryObject.get_host_array to get host pointer.");
      case CL_MEM_MAP_COUNT:
        PYOPENCL_GET_TYPED_INFO(MemObject, data(), param_name,
            cl_uint);
      case CL_MEM_REFERENCE_COUNT:
        PYOPENCL_GET_TYPED_INFO(MemObject, data(), param_name,
            cl_uint);
      case CL_MEM_CONTEXT:
        PYOPENCL_GET_OPAQUE_INFO(MemObject, data(), param_name,
            cl_context, context);

#if PYOPENCL_CL_VERSION >= 0x1010
      case CL_MEM_ASSOCIATED_MEMOBJECT:
        {
          cl_mem param_value;
          PYOPENCL_CALL_GUARDED(clGetMemObjectInfo, \
              (data(), param_name, sizeof(param_value), &param_value, 0));
          if (param_value == 0)
          {
            // no associated memory object? no problem.
            return py::none();
          }

          return create_mem_object_wrapper(param_value);
        }
      case CL_MEM_OFFSET:
        PYOPENCL_GET_TYPED_INFO(MemObject, data(), param_name,
            size_t);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
      case CL_MEM_USES_SVM_POINTER:
        PYOPENCL_GET_TYPED_INFO(MemObject, data(), param_name,
            cl_bool);
#endif
#if PYOPENCL_CL_VERSION >= 0x3000
      case CL_MEM_PROPERTIES:
            {
              std::vector<cl_mem_properties> result;
              PYOPENCL_GET_VEC_INFO(MemObject, data(), param_name, result);
              PYOPENCL_RETURN_VECTOR(cl_mem_properties, result);
            }
#endif

      default:
        throw error("MemoryObjectHolder.get_info", CL_INVALID_VALUE);
    }
  }

  // FIXME: Reenable in pypy
#ifndef PYPY_VERSION
  inline
  py::object get_mem_obj_host_array(
      py::object mem_obj_py,
      py::object shape, py::object dtype,
      py::object order_py)
  {
    memory_object_holder const &mem_obj =
      (mem_obj_py).cast<memory_object_holder const &>();
    PyArray_Descr *tp_descr;
    if (PyArray_DescrConverter(dtype.ptr(), &tp_descr) != NPY_SUCCEED)
      throw py::error_already_set();
    cl_mem_flags mem_flags;
    PYOPENCL_CALL_GUARDED(clGetMemObjectInfo,
            (mem_obj.data(), CL_MEM_FLAGS, sizeof(mem_flags), &mem_flags, 0));
    if (!(mem_flags & CL_MEM_USE_HOST_PTR))
      throw pyopencl::error("MemoryObject.get_host_array", CL_INVALID_VALUE,
                            "Only MemoryObject with USE_HOST_PTR "
                            "is supported.");

    std::vector<npy_intp> dims;
    try
    {
      dims.push_back(py::cast<npy_intp>(shape));
    }
    catch (py::cast_error &)
    {
      for (auto it: shape)
        dims.push_back(it.cast<npy_intp>());
    }

    NPY_ORDER order = PyArray_CORDER;
    PyArray_OrderConverter(order_py.ptr(), &order);

    int ary_flags = 0;
    if (order == PyArray_FORTRANORDER)
      ary_flags |= NPY_FARRAY;
    else if (order == PyArray_CORDER)
      ary_flags |= NPY_CARRAY;
    else
      throw std::runtime_error("unrecognized order specifier");

    void *host_ptr;
    size_t mem_obj_size;
    PYOPENCL_CALL_GUARDED(clGetMemObjectInfo,
        (mem_obj.data(), CL_MEM_HOST_PTR, sizeof(host_ptr),
         &host_ptr, 0));
    PYOPENCL_CALL_GUARDED(clGetMemObjectInfo,
        (mem_obj.data(), CL_MEM_SIZE, sizeof(mem_obj_size),
         &mem_obj_size, 0));

    py::object result = py::reinterpret_steal<py::object>(PyArray_NewFromDescr(
        &PyArray_Type, tp_descr,
        dims.size(), &dims.front(), /*strides*/ nullptr,
        host_ptr, ary_flags, /*obj*/nullptr));

    if ((size_t) PyArray_NBYTES(result.ptr()) > mem_obj_size)
      throw pyopencl::error("MemoryObject.get_host_array",
          CL_INVALID_VALUE,
          "Resulting array is larger than memory object.");

    PyArray_BASE(result.ptr()) = mem_obj_py.ptr();
    Py_INCREF(mem_obj_py.ptr());

    return result;
  }
#endif

  // }}}
}

#endif

// vim: foldmethod=marker
