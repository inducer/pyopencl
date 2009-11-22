#ifndef _AFJHAYYTA_PYOPENCL_HEADER_SEEN_CL_HPP
#define _AFJHAYYTA_PYOPENCL_HEADER_SEEN_CL_HPP




// TODO: GL Interop




#ifdef __APPLE__

// Mac ------------------------------------------------------------------------
#include <OpenCL/opencl.h>
#ifdef HAVE_GL
#include <OpenCL/opencl_gl.h>
#endif

#else

// elsewhere ------------------------------------------------------------------
#include <CL/cl.h>
#ifdef HAVE_GL
#include <GL/gl.h>
#include <CL/cl_gl.h>
#endif

#endif

#include <stdexcept>
#include <iostream>
#include <vector>
#include <utility>
#include <numeric>
#include <boost/foreach.hpp>
#include <boost/scoped_array.hpp>
#include "wrap_helpers.hpp"
#include "numpy_init.hpp"




// #define PYOPENCL_TRACE




// tools ----------------------------------------------------------------------
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




// tracing and error reporting ------------------------------------------------
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
    Py_BEGIN_ALLOW_THREADS \
      status_code = NAME ARGLIST; \
    Py_END_ALLOW_THREADS \
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
    Py_BEGIN_ALLOW_THREADS \
      status_code = NAME ARGLIST; \
    Py_END_ALLOW_THREADS \
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
        << pyopencl::error::make_message(#NAME, status_code) \
        << std::endl; \
  }





// get_info helpers -----------------------------------------------------------
#define PYOPENCL_GET_OPAQUE_INFO(WHAT, FIRST_ARG, SECOND_ARG, CL_TYPE, TYPE) \
  { \
    CL_TYPE param_value; \
    PYOPENCL_CALL_GUARDED(clGet##WHAT##Info, \
          (FIRST_ARG, SECOND_ARG, sizeof(param_value), &param_value, 0)); \
    return py::object(handle_from_new_ptr( \
          new TYPE(param_value, /*retain*/ true))); \
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
         RES_VEC.empty( ) ? NULL : &RES_VEC.front(), &size)); \
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
         param_value.empty( ) ? NULL : &param_value.front(), &param_value_size)); \
    \
    return py::object( \
        param_value.empty( ) ? "" : std::string(&param_value.front(), param_value_size-1)); \
  }




#define PYOPENCL_GET_INTEGRAL_INFO(WHAT, FIRST_ARG, SECOND_ARG, TYPE) \
  { \
    TYPE param_value; \
    PYOPENCL_CALL_GUARDED(clGet##WHAT##Info, \
        (FIRST_ARG, SECOND_ARG, sizeof(param_value), &param_value, 0)); \
    return py::object(param_value); \
  }




// event helpers --------------------------------------------------------------
#define PYOPENCL_PARSE_WAIT_FOR \
    cl_uint num_events_in_wait_list = 0; \
    std::vector<cl_event> event_wait_list; \
    \
    if (py_wait_for.ptr() != Py_None) \
    { \
      event_wait_list.resize(len(py_wait_for)); \
      PYTHON_FOREACH(evt, py_wait_for) \
        event_wait_list[num_events_in_wait_list++] = \
        py::extract<event &>(evt)().data(); \
    }

#define PYOPENCL_RETURN_NEW_EVENT(EVT) \
    try \
    { \
      return new event(EVT, false); \
    } \
    catch (...) \
    { \
      clReleaseEvent(EVT); \
      throw; \
    }




// equality testing -----------------------------------------------------------
#define PYOPENCL_EQUALITY_TESTS(cls) \
    bool operator==(cls const &other) const \
    { return data() == other.data(); } \
    bool operator!=(cls const &other) const \
    { return data() != other.data(); }




namespace pyopencl
{
  class error : public std::runtime_error
  {
    private:
      const char *m_routine;
      cl_int m_code;

    public:
      static std::string make_message(const char *rout, cl_int c, const char *msg=0)
      {
        std::string result = rout;
        result += " failed: ";
        result += cl_error_to_str(c);
        if (msg)
        {
          result += " - ";
          result += msg;
        }
        return result;
      }

      error(const char *rout, cl_int c, const char *msg=0)
        : std::runtime_error(make_message(rout, c, msg)),
        m_routine(rout), m_code(c)
      { }

      const char *routine() const
      {
        return m_routine;
      }

      cl_int code() const
      {
        return m_code;
      }

      static const char *cl_error_to_str(cl_int e)
      {
        switch (e)
        {
          case CL_SUCCESS: return "success";
          case CL_DEVICE_NOT_FOUND: return "device not found";
          case CL_DEVICE_NOT_AVAILABLE: return "device not available";
#if !(defined(CL_PLATFORM_NVIDIA) && CL_PLATFORM_NVIDIA == 0x3001)
          case CL_COMPILER_NOT_AVAILABLE: return "device compiler not available";
#endif
          case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "mem object allocation failure";
          case CL_OUT_OF_RESOURCES: return "out of resources";
          case CL_OUT_OF_HOST_MEMORY: return "out of host memory";
          case CL_PROFILING_INFO_NOT_AVAILABLE: return "profiling info not available";
          case CL_MEM_COPY_OVERLAP: return "mem copy overlap";
          case CL_IMAGE_FORMAT_MISMATCH: return "image format mismatch";
          case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "image format not supported";
          case CL_BUILD_PROGRAM_FAILURE: return "build program failure";
          case CL_MAP_FAILURE: return "map failure";

          case CL_INVALID_VALUE: return "invalid value";
          case CL_INVALID_DEVICE_TYPE: return "invalid device type";
          case CL_INVALID_PLATFORM: return "invalid platform";
          case CL_INVALID_DEVICE: return "invalid device";
          case CL_INVALID_CONTEXT: return "invalid context";
          case CL_INVALID_QUEUE_PROPERTIES: return "invalid queue properties";
          case CL_INVALID_COMMAND_QUEUE: return "invalid command queue";
          case CL_INVALID_HOST_PTR: return "invalid host ptr";
          case CL_INVALID_MEM_OBJECT: return "invalid mem object";
          case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "invalid image format descriptor";
          case CL_INVALID_IMAGE_SIZE: return "invalid image size";
          case CL_INVALID_SAMPLER: return "invalid sampler";
          case CL_INVALID_BINARY: return "invalid binary";
          case CL_INVALID_BUILD_OPTIONS: return "invalid build options";
          case CL_INVALID_PROGRAM: return "invalid program";
          case CL_INVALID_PROGRAM_EXECUTABLE: return "invalid program executable";
          case CL_INVALID_KERNEL_NAME: return "invalid kernel name";
          case CL_INVALID_KERNEL_DEFINITION: return "invalid kernel definition";
          case CL_INVALID_KERNEL: return "invalid kernel";
          case CL_INVALID_ARG_INDEX: return "invalid arg index";
          case CL_INVALID_ARG_VALUE: return "invalid arg value";
          case CL_INVALID_ARG_SIZE: return "invalid arg size";
          case CL_INVALID_KERNEL_ARGS: return "invalid kernel args";
          case CL_INVALID_WORK_DIMENSION: return "invalid work dimension";
          case CL_INVALID_WORK_GROUP_SIZE: return "invalid work group size";
          case CL_INVALID_WORK_ITEM_SIZE: return "invalid work item size";
          case CL_INVALID_GLOBAL_OFFSET: return "invalid global offset";
          case CL_INVALID_EVENT_WAIT_LIST: return "invalid event wait list";
          case CL_INVALID_EVENT: return "invalid event";
          case CL_INVALID_OPERATION: return "invalid operation";
          case CL_INVALID_GL_OBJECT: return "invalid gl object";
          case CL_INVALID_BUFFER_SIZE: return "invalid buffer size";
          case CL_INVALID_MIP_LEVEL: return "invalid mip level";

          default: return "invalid error code";
        }
      }
  };




  // platform -----------------------------------------------------------------
  class platform : boost::noncopyable
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

      npy_intp obj_ptr() const
      {
        return (npy_intp) data();
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

          default:
            throw error("Platform.get_info", CL_INVALID_VALUE);
        }
      }

      py::list get_devices(cl_device_type devtype);
  };




  py::list get_platforms()
  {
    cl_uint num_platforms = 0;
    PYOPENCL_CALL_GUARDED(clGetPlatformIDs, (0, 0, &num_platforms));

    std::vector<cl_platform_id> platforms(num_platforms);
    PYOPENCL_CALL_GUARDED(clGetPlatformIDs,
        (num_platforms, platforms.empty( ) ? NULL : &platforms.front(), &num_platforms));

    py::list result;
    BOOST_FOREACH(cl_platform_id pid, platforms)
      result.append(handle_from_new_ptr(
            new platform(pid)));

    return result;
  }




  // device -------------------------------------------------------------------
  class device : boost::noncopyable
  {
    private:
      cl_device_id m_device;

    public:
      device(cl_device_id did)
      : m_device(did)
      { }

      device(cl_device_id did, bool /*retain (ignored)*/)
      : m_device(did)
      { }

      cl_device_id data() const
      {
        return m_device;
      }

      npy_intp obj_ptr() const
      {
        return (npy_intp) data();
      }

      PYOPENCL_EQUALITY_TESTS(device);

      py::object get_info(cl_device_info param_name) const
      {
#define DEV_GET_INT_INF(TYPE) \
        PYOPENCL_GET_INTEGRAL_INFO(Device, m_device, param_name, TYPE);

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
          case CL_DEVICE_QUEUE_PROPERTIES: DEV_GET_INT_INF(cl_command_queue_properties);

          case CL_DEVICE_NAME:
          case CL_DEVICE_VENDOR:
          case CL_DRIVER_VERSION:
          case CL_DEVICE_PROFILE:
          case CL_DEVICE_VERSION:
          case CL_DEVICE_EXTENSIONS:
            PYOPENCL_GET_STR_INFO(Device, m_device, param_name);

          case CL_DEVICE_PLATFORM:
            PYOPENCL_GET_OPAQUE_INFO(Device, m_device, param_name, cl_platform_id, platform);

          default:
            throw error("Platform.get_info", CL_INVALID_VALUE);
        }
      }
  };




  inline py::list platform::get_devices(cl_device_type devtype)
  {
    cl_uint num_devices = 0;
    PYOPENCL_CALL_GUARDED(clGetDeviceIDs,
        (m_platform, devtype, 0, 0, &num_devices));

    std::vector<cl_device_id> devices(num_devices);
    PYOPENCL_CALL_GUARDED(clGetDeviceIDs,
        (m_platform, devtype,
         num_devices, devices.empty( ) ? NULL : &devices.front(), &num_devices));

    py::list result;
    BOOST_FOREACH(cl_device_id did, devices)
      result.append(handle_from_new_ptr(
            new device(did)));

    return result;
  }




  // context ------------------------------------------------------------------
  class context : public boost::noncopyable
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

      npy_intp obj_ptr() const
      {
        return (npy_intp) data();
      }

      PYOPENCL_EQUALITY_TESTS(context);

      py::object get_info(cl_context_info param_name) const
      {
        switch (param_name)
        {
          case CL_CONTEXT_REFERENCE_COUNT:
            PYOPENCL_GET_INTEGRAL_INFO(
                Context, m_context, param_name, cl_uint);

          case CL_CONTEXT_DEVICES:
            {
              std::vector<cl_device_id> result;
              PYOPENCL_GET_VEC_INFO(Context, m_context, param_name, result);

              py::list py_result;
              BOOST_FOREACH(cl_device_id did, result)
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
                    }

                  case 0:
                    break;

                  default:
                    throw error("Context.get_info", CL_INVALID_VALUE,
                        "unkown context_property key encountered");
                }

                py_result.append(py::make_tuple(result[i], value));
              }
              return py_result;
            }

          default:
            throw error("Context.get_info", CL_INVALID_VALUE);
        }
      }
  };



  context *create_context(py::object py_devices, py::object py_properties,
      py::object py_dev_type)
  {
    // parse context properties
    cl_context_properties *props_ptr = 0;
    std::vector<cl_context_properties> props;
  
    if (py_properties.ptr() != Py_None)
    {
      PYTHON_FOREACH(prop_tuple, py_properties)
      {
        if (len(prop_tuple) != 2)
          throw error("Context", CL_INVALID_VALUE, "property tuple must have length 2");
        cl_context_properties prop =
            py::extract<cl_context_properties>(prop_tuple[0]);
        props.push_back(prop);
    
        if (prop == CL_CONTEXT_PLATFORM)
        {
          py::extract<const platform &> value(prop_tuple[1]);
          props.push_back(
              reinterpret_cast<cl_context_properties>(value().data()));
        }
        else
          throw error("Context", CL_INVALID_VALUE, "invalid context property");
      }
      props.push_back(0);
      props_ptr = props.empty( ) ? NULL : &props.front();
    }

    cl_int status_code;

    cl_context ctx;

    // from device list
    if (py_devices.ptr() != Py_None)
    {
      if (py_dev_type.ptr() != Py_None)
        throw error("Context", CL_INVALID_VALUE, 
            "one of 'devices' or 'dev_type' must be None");

      std::vector<cl_device_id> devices;
      PYTHON_FOREACH(py_dev, py_devices)
      {
        py::extract<const device &> dev(py_dev);
          devices.push_back(dev().data());
      }

      ctx = clCreateContext(
          props_ptr,
          devices.size(),
          devices.empty( ) ? NULL : &devices.front(),
          0, 0, &status_code);

      PYOPENCL_PRINT_CALL_TRACE("clCreateContext");
    }
    // from dev_type
    else
    {
      cl_device_type dev_type = CL_DEVICE_TYPE_DEFAULT;
      if (py_dev_type.ptr() != Py_None)
        dev_type = py::extract<cl_device_type>(py_dev_type)();

      ctx = clCreateContextFromType(props_ptr, dev_type, 0, 0, &status_code);

      PYOPENCL_PRINT_CALL_TRACE("clCreateContextFromType");
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




  // command_queue ------------------------------------------------------------
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
          const device *py_dev=0,
          cl_command_queue_properties props=0)
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

        cl_int status_code;
        m_queue = clCreateCommandQueue(
            ctx.data(), dev, props, &status_code);

        PYOPENCL_PRINT_CALL_TRACE("clCreateCommandQueue");
        if (status_code != CL_SUCCESS)
          throw pyopencl::error("CommandQueue", status_code);
      }

      ~command_queue()
      {
        PYOPENCL_CALL_GUARDED_CLEANUP(clReleaseCommandQueue,
            (m_queue));
      }

      const cl_command_queue data() const
      { return m_queue; }

      npy_intp obj_ptr() const
      {
        return (npy_intp) data();
      }

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
            PYOPENCL_GET_INTEGRAL_INFO(CommandQueue, m_queue, param_name,
                cl_uint);
          case CL_QUEUE_PROPERTIES:
            PYOPENCL_GET_INTEGRAL_INFO(CommandQueue, m_queue, param_name,
                cl_command_queue_properties);

          default:
            throw error("CommandQueue.get_info", CL_INVALID_VALUE);
        }
      }

      cl_command_queue_properties set_property(
          cl_command_queue_properties prop,
          bool enable)
      {
        cl_command_queue_properties old_prop;
        PYOPENCL_CALL_GUARDED(clSetCommandQueueProperty,
            (m_queue, prop, PYOPENCL_CAST_BOOL(enable), &old_prop));
        return old_prop;
      }

      void flush()
      { PYOPENCL_CALL_GUARDED(clFlush, (m_queue)); }
      void finish()
      { PYOPENCL_CALL_GUARDED_THREADED(clFlush, (m_queue)); }
  };




  // events -------------------------------------------------------------------
  class event : boost::noncopyable
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

      ~event()
      {
        PYOPENCL_CALL_GUARDED_CLEANUP(clReleaseEvent,
            (m_event));
      }

      const cl_event data() const
      { return m_event; }

      npy_intp obj_ptr() const
      {
        return (npy_intp) data();
      }

      PYOPENCL_EQUALITY_TESTS(event);

      py::object get_info(cl_event_info param_name) const
      {
        switch (param_name)
        {
          case CL_EVENT_COMMAND_QUEUE:
            PYOPENCL_GET_OPAQUE_INFO(Event, m_event, param_name,
                cl_command_queue, command_queue);
          case CL_EVENT_COMMAND_TYPE:
            PYOPENCL_GET_INTEGRAL_INFO(Event, m_event, param_name,
                cl_command_type);
          case CL_EVENT_COMMAND_EXECUTION_STATUS:
            PYOPENCL_GET_INTEGRAL_INFO(Event, m_event, param_name,
                cl_int);
          case CL_EVENT_REFERENCE_COUNT:
            PYOPENCL_GET_INTEGRAL_INFO(Event, m_event, param_name,
                cl_uint);

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
            PYOPENCL_GET_INTEGRAL_INFO(EventProfiling, m_event, param_name,
                cl_ulong);
          default:
            throw error("Event.get_profiling_info", CL_INVALID_VALUE);
        }
      }
  };




  void wait_for_events(py::object events)
  {
    cl_uint num_events_in_wait_list = 0;
    std::vector<cl_event> event_wait_list(len(events));

    PYTHON_FOREACH(evt, events)
      event_wait_list[num_events_in_wait_list++] =
      py::extract<event &>(evt)().data();

    PYOPENCL_CALL_GUARDED_THREADED(clWaitForEvents, (
          num_events_in_wait_list, event_wait_list.empty( ) ? NULL : &event_wait_list.front()));
  }




  event *enqueue_marker(command_queue &cq)
  {
    cl_event evt;
    PYOPENCL_CALL_GUARDED(clEnqueueMarker, (
          cq.data(), &evt));

    PYOPENCL_RETURN_NEW_EVENT(evt);
  }





  void enqueue_wait_for_events(command_queue &cq, py::object py_events)
  {
    cl_uint num_events = 0;
    std::vector<cl_event> event_list(len(py_events));

    PYTHON_FOREACH(py_evt, py_events)
      event_list[num_events++] =
      py::extract<event &>(py_evt)().data();

    PYOPENCL_CALL_GUARDED(clEnqueueWaitForEvents, (
          cq.data(), num_events, event_list.empty( ) ? NULL : &event_list.front()));
  }




  void enqueue_barrier(command_queue &cq)
  {
    PYOPENCL_CALL_GUARDED(clEnqueueBarrier, (cq.data()));
  }




  // memory -------------------------------------------------------------------
  class memory_object : boost::noncopyable
  {
    private:
      bool m_valid;
      cl_mem m_mem;
      py::object m_hostbuf;

    public:
      memory_object(cl_mem mem, bool retain, py::object *hostbuf=0)
        : m_valid(true), m_mem(mem)
      {
        if (retain)
          PYOPENCL_CALL_GUARDED(clRetainMemObject, (mem));

        if (hostbuf)
          m_hostbuf = *hostbuf;
      }

      memory_object(memory_object &src)
        : m_valid(true), m_mem(src.m_mem), m_hostbuf(src.m_hostbuf)
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
      { return m_hostbuf; }

      const cl_mem data() const
      { return m_mem; }

      npy_intp obj_ptr() const
      {
        return (npy_intp) data();
      }

      PYOPENCL_EQUALITY_TESTS(memory_object);

      size_t size() const
      {
        size_t param_value;
        PYOPENCL_CALL_GUARDED(clGetMemObjectInfo,
            (m_mem, CL_MEM_SIZE, sizeof(param_value), &param_value, 0));
        return param_value;
      }

      py::object get_info(cl_mem_info param_name) const
      {
        switch (param_name)
        {
          case CL_MEM_TYPE:
            PYOPENCL_GET_INTEGRAL_INFO(MemObject, m_mem, param_name,
                cl_mem_object_type);
          case CL_MEM_FLAGS:
            PYOPENCL_GET_INTEGRAL_INFO(MemObject, m_mem, param_name,
                cl_mem_flags);
          case CL_MEM_SIZE:
            PYOPENCL_GET_INTEGRAL_INFO(MemObject, m_mem, param_name,
                size_t);
          case CL_MEM_HOST_PTR:
            return m_hostbuf;
          case CL_MEM_MAP_COUNT:
            PYOPENCL_GET_INTEGRAL_INFO(MemObject, m_mem, param_name,
                cl_uint);
          case CL_MEM_REFERENCE_COUNT:
            PYOPENCL_GET_INTEGRAL_INFO(MemObject, m_mem, param_name,
                cl_uint);
          case CL_MEM_CONTEXT:
            PYOPENCL_GET_OPAQUE_INFO(MemObject, m_mem, param_name,
                cl_context, context);

          default:
            throw error("MemoryObject.get_info", CL_INVALID_VALUE);
        }
      }
  };




  class buffer : public memory_object
  {
    public:
      buffer(cl_mem mem, bool retain, py::object *hostbuf=0)
        : memory_object(mem, retain, hostbuf)
      { }
  };




  memory_object *create_buffer(
      context &ctx,
      cl_mem_flags flags,
      size_t size,
      py::object buffer
      )
  {
    if (buffer.ptr() != Py_None && 
        !(flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR)))
      PyErr_Warn(PyExc_UserWarning, "'hostbuf' was passed, "
          "but no memory flags to make use of it.");

    void *buf = 0;
    py::object *retained_buf_obj = 0;
    if (buffer.ptr() != Py_None)
    {
      PYOPENCL_BUFFER_SIZE_T len;
      if (flags & CL_MEM_USE_HOST_PTR)
      {
        if (PyObject_AsWriteBuffer(buffer.ptr(), &buf, &len))
          throw py::error_already_set();
      }
      else
      {
        if (PyObject_AsReadBuffer(
              buffer.ptr(), const_cast<const void **>(&buf), &len))
          throw py::error_already_set();
      }

      if (flags & CL_MEM_USE_HOST_PTR)
        retained_buf_obj = &buffer;

      if (size > size_t(len))
        throw pyopencl::error("Buffer", CL_INVALID_VALUE,
            "specified size is greater than host buffer size");
      if (size == 0)
        size = len;
    }

    cl_int status_code;
    cl_mem mem = clCreateBuffer(ctx.data(), flags, size, buf, &status_code);

    PYOPENCL_PRINT_CALL_TRACE("clCreateBuffer");
    if (status_code != CL_SUCCESS)
      throw pyopencl::error("create_host_buffer", status_code);

    try
    {
      return new memory_object(mem, false, retained_buf_obj);
    }
    catch (...)
    {
      PYOPENCL_CALL_GUARDED(clReleaseMemObject, (mem));
      throw;
    }
  }




  event *enqueue_read_buffer(
      command_queue &cq,
      memory_object &mem,
      py::object buffer,
      size_t device_offset,
      py::object py_wait_for,
      bool is_blocking,
      py::object host_buffer_deprecated
      )
  {
    if (host_buffer_deprecated.ptr() != Py_None)
    {
      PYOPENCL_DEPRECATED("The 'host_buffer' keyword argument", "0.93", );
      buffer = host_buffer_deprecated;
    }

    PYOPENCL_PARSE_WAIT_FOR;

    void *buf;
    PYOPENCL_BUFFER_SIZE_T len;

    if (PyObject_AsWriteBuffer(buffer.ptr(), &buf, &len))
      throw py::error_already_set();

    cl_event evt;
    PYOPENCL_CALL_GUARDED(clEnqueueReadBuffer, (
          cq.data(),
          mem.data(),
          PYOPENCL_CAST_BOOL(is_blocking),
          device_offset, len, buf,
          num_events_in_wait_list, event_wait_list.empty( ) ? NULL : &event_wait_list.front(), &evt
          ));
    PYOPENCL_RETURN_NEW_EVENT(evt);
  }




  event *enqueue_write_buffer(
      command_queue &cq,
      memory_object &mem,
      py::object buffer,
      size_t device_offset,
      py::object py_wait_for,
      bool is_blocking,
      py::object host_buffer_deprecated
      )
  {
    if (host_buffer_deprecated.ptr() != Py_None)
    {
      PYOPENCL_DEPRECATED("The 'host_buffer' keyword argument", "0.93", );
      buffer = host_buffer_deprecated;
    }

    PYOPENCL_PARSE_WAIT_FOR;

    const void *buf;
    PYOPENCL_BUFFER_SIZE_T len;

    if (PyObject_AsReadBuffer(buffer.ptr(), &buf, &len))
      throw py::error_already_set();

    cl_event evt;
    PYOPENCL_CALL_GUARDED(clEnqueueWriteBuffer, (
          cq.data(),
          mem.data(),
          PYOPENCL_CAST_BOOL(is_blocking),
          device_offset, len, buf,
          num_events_in_wait_list, event_wait_list.empty( ) ? NULL : &event_wait_list.front(), &evt
          ));
    PYOPENCL_RETURN_NEW_EVENT(evt);
  }




  // images -------------------------------------------------------------------
  class image : public memory_object
  {
    public:
      image(cl_mem mem, bool retain, py::object *hostbuf=0)
        : memory_object(mem, retain, hostbuf)
      { }

      py::object get_image_info(cl_image_info param_name) const
      {
        switch (param_name)
        {
          case CL_IMAGE_FORMAT:
            PYOPENCL_GET_INTEGRAL_INFO(Image, data(), param_name,
                cl_image_format);
          case CL_IMAGE_ELEMENT_SIZE:
          case CL_IMAGE_ROW_PITCH:
          case CL_IMAGE_SLICE_PITCH:
          case CL_IMAGE_WIDTH:
          case CL_IMAGE_HEIGHT:
          case CL_IMAGE_DEPTH:
            PYOPENCL_GET_INTEGRAL_INFO(Image, data(), param_name, size_t);

          default:
            throw error("MemoryObject.get_image_info", CL_INVALID_VALUE);
        }
      }
  };




  cl_image_format *make_image_format(cl_channel_order ord, cl_channel_type tp)
  {
    std::auto_ptr<cl_image_format> result(new cl_image_format);
    result->image_channel_order = ord;
    result->image_channel_data_type = tp;
    return result.release();
  }

  py::list get_supported_image_formats(
      context const &ctx,
      cl_mem_flags flags,
      cl_mem_object_type image_type)
  {
    cl_uint num_image_formats = 1024;
    std::vector<cl_image_format> formats(num_image_formats);
    do
    {
      if (formats.size() < num_image_formats)
        formats.resize(num_image_formats);

      PYOPENCL_CALL_GUARDED(clGetSupportedImageFormats, (
            ctx.data(), flags, image_type,
            formats.size(), formats.empty( ) ? NULL : &formats.front(), &num_image_formats));
    } while (formats.size() < num_image_formats);

    PYOPENCL_RETURN_VECTOR(cl_image_format, formats);
  }




  inline image *create_image(
      context const &ctx,
      cl_mem_flags flags,
      cl_image_format const &fmt,
      py::object shape,
      py::object pitches,
      py::object buffer,
      py::object host_buffer_deprecated
      )
  {
    if (host_buffer_deprecated.ptr() != Py_None)
    {
      PYOPENCL_DEPRECATED("The 'host_buffer' keyword argument", "0.93", );
      buffer = host_buffer_deprecated;
    }

    if (buffer.ptr() != Py_None && 
        !(flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR)))
      PyErr_Warn(PyExc_UserWarning, "'hostbuf' was passed, "
          "but no memory flags to make use of it.");

    if (shape.ptr() == Py_None)
    {
      if (buffer.ptr() == Py_None)
        throw pyopencl::error("Image", CL_INVALID_VALUE, 
            "'shape' must be passed if 'hostbuf' is not given");

      shape = buffer.attr("shape");
    }

    void *buf = 0;
    PYOPENCL_BUFFER_SIZE_T len;
    py::object *retained_buf_obj = 0;

    if (buffer.ptr() != Py_None)
    {
      if (flags & CL_MEM_USE_HOST_PTR)
      {
        if (PyObject_AsWriteBuffer(buffer.ptr(), &buf, &len))
          throw py::error_already_set();
      }
      else
      {
        if (PyObject_AsReadBuffer(
              buffer.ptr(), const_cast<const void **>(&buf), &len))
          throw py::error_already_set();
      }

      if (flags & CL_MEM_USE_HOST_PTR)
        retained_buf_obj = &buffer;
    }

    unsigned dims = py::len(shape);
    cl_int status_code;
    cl_mem mem;
    if (dims == 2)
    {
      size_t width = py::extract<size_t>(shape[0]);
      size_t height = py::extract<size_t>(shape[1]);

      size_t pitch = 0;
      if (pitches.ptr() != Py_None)
      {
        if (py::len(pitches) != 1)
          throw pyopencl::error("Image", CL_INVALID_VALUE, 
              "invalid length of pitch tuple");
        pitch = py::extract<size_t>(pitches[0]);
      }

      mem = clCreateImage2D(ctx.data(), flags, &fmt,
          width, height, pitch, buf, &status_code);

      PYOPENCL_PRINT_CALL_TRACE("clCreateImage2D");
      if (status_code != CL_SUCCESS)
        throw pyopencl::error("clCreateImage2D", status_code);
    }
    else if (dims == 3)
    {
      size_t width = py::extract<size_t>(shape[0]);
      size_t height = py::extract<size_t>(shape[1]);
      size_t depth = py::extract<size_t>(shape[2]);

      size_t pitch_x = 0;
      size_t pitch_y = 0;

      if (pitches.ptr() != Py_None)
      {
        if (py::len(pitches) != 2)
          throw pyopencl::error("Image", CL_INVALID_VALUE, 
              "invalid length of pitch tuple");

        pitch_x = py::extract<size_t>(pitches[0]);
        pitch_y = py::extract<size_t>(pitches[1]);
      }

      mem = clCreateImage3D(ctx.data(), flags, &fmt,
          width, height, depth, pitch_x, pitch_y, buf, &status_code);

      PYOPENCL_PRINT_CALL_TRACE("clCreateImage3D");
      if (status_code != CL_SUCCESS)
        throw pyopencl::error("clCreateImage3D", status_code);
    }
    else
      throw pyopencl::error("Image", CL_INVALID_VALUE, 
          "invalid dimension");

    try
    {
      return new image(mem, false, retained_buf_obj);
    }
    catch (...)
    {
      PYOPENCL_CALL_GUARDED(clReleaseMemObject, (mem));
      throw;
    }
  }




  event *enqueue_read_image(
      command_queue &cq,
      image &img,
      py::object py_origin, py::object py_region,
      py::object buffer,
      size_t row_pitch, size_t slice_pitch,
      py::object py_wait_for,
      bool is_blocking,
      py::object host_buffer_deprecated
      )
  {
    if (host_buffer_deprecated.ptr() != Py_None)
    {
      PYOPENCL_DEPRECATED("The 'host_buffer' keyword argument", "0.93", );
      buffer = host_buffer_deprecated;
    }

    PYOPENCL_PARSE_WAIT_FOR;
    COPY_PY_COORD_TRIPLE(origin);
    COPY_PY_REGION_TRIPLE(region);

    void *buf;
    PYOPENCL_BUFFER_SIZE_T len;

    if (PyObject_AsWriteBuffer(buffer.ptr(), &buf, &len))
      throw py::error_already_set();

    cl_event evt;
    PYOPENCL_CALL_GUARDED(clEnqueueReadImage, (
          cq.data(),
          img.data(),
          PYOPENCL_CAST_BOOL(is_blocking),
          origin, region, row_pitch, slice_pitch, buf,
          num_events_in_wait_list, event_wait_list.empty( ) ? NULL : &event_wait_list.front(), &evt
          ));
    PYOPENCL_RETURN_NEW_EVENT(evt);
  }




  event *enqueue_write_image(
      command_queue &cq,
      image &img,
      py::object py_origin, py::object py_region,
      py::object buffer,
      size_t row_pitch, size_t slice_pitch,
      py::object py_wait_for,
      bool is_blocking,
      py::object host_buffer_deprecated
      )
  {
    if (host_buffer_deprecated.ptr() != Py_None)
    {
      PYOPENCL_DEPRECATED("The 'host_buffer' keyword argument", "0.93", );
      buffer = host_buffer_deprecated;
    }

    PYOPENCL_PARSE_WAIT_FOR;
    COPY_PY_COORD_TRIPLE(origin);
    COPY_PY_REGION_TRIPLE(region);

    const void *buf;
    PYOPENCL_BUFFER_SIZE_T len;

    if (PyObject_AsReadBuffer(buffer.ptr(), &buf, &len))
      throw py::error_already_set();

    cl_event evt;
    PYOPENCL_CALL_GUARDED(clEnqueueWriteImage, (
          cq.data(),
          img.data(),
          PYOPENCL_CAST_BOOL(is_blocking),
          origin, region, row_pitch, slice_pitch, buf,
          num_events_in_wait_list, event_wait_list.empty( ) ? NULL : &event_wait_list.front(), &evt
          ));
    PYOPENCL_RETURN_NEW_EVENT(evt);
  }




  event *enqueue_copy_image(
      command_queue &cq,
      memory_object &src,
      memory_object &dest,
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
    PYOPENCL_CALL_GUARDED(clEnqueueCopyImage, (
          cq.data(), src.data(), dest.data(),
          src_origin, dest_origin, region,
          num_events_in_wait_list, event_wait_list.empty( ) ? NULL : &event_wait_list.front(), &evt
          ));
    PYOPENCL_RETURN_NEW_EVENT(evt);
  }




  event *enqueue_copy_image_to_buffer(
      command_queue &cq,
      memory_object &src,
      memory_object &dest,
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
    PYOPENCL_CALL_GUARDED(clEnqueueCopyImageToBuffer, (
          cq.data(), src.data(), dest.data(),
          origin, region, offset,
          num_events_in_wait_list, event_wait_list.empty( ) ? NULL : &event_wait_list.front(), &evt
          ));
    PYOPENCL_RETURN_NEW_EVENT(evt);
  }




  event *enqueue_copy_buffer_to_image(
      command_queue &cq,
      memory_object &src,
      memory_object &dest,
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
    PYOPENCL_CALL_GUARDED(clEnqueueCopyBufferToImage, (
          cq.data(), src.data(), dest.data(),
          offset, origin, region,
          num_events_in_wait_list, event_wait_list.empty( ) ? NULL : &event_wait_list.front(), &evt
          ));
    PYOPENCL_RETURN_NEW_EVENT(evt);
  }




  // maps ---------------------------------------------------------------------
  class memory_map
  {
    private:
      bool m_valid;
      command_queue m_queue;
      memory_object m_mem;
      void *m_ptr;

    public:
      memory_map(command_queue &cq, memory_object &mem, void *ptr)
        : m_valid(true), m_queue(cq), m_mem(mem), m_ptr(ptr)
      {
      }

      ~memory_map()
      {
        if (m_valid)
          delete release(0, py::object());
      }

      event *release(command_queue *cq, py::object py_wait_for)
      {
        PYOPENCL_PARSE_WAIT_FOR;

        if (cq == 0)
          cq = &m_queue;

        cl_event evt;
        PYOPENCL_CALL_GUARDED(clEnqueueUnmapMemObject, (
              cq->data(), m_mem.data(), m_ptr,
              num_events_in_wait_list, event_wait_list.empty( ) ? NULL : &event_wait_list.front(), &evt
              ));

        m_valid = false;

        PYOPENCL_RETURN_NEW_EVENT(evt);
      }
  };




  py::object enqueue_map_buffer(
      command_queue &cq,
      memory_object &buf,
      cl_map_flags flags,
      size_t offset,
      py::object py_shape, py::object dtype, py::object order_py,
      py::object py_wait_for,
      bool is_blocking
      )
  {
    PYOPENCL_PARSE_WAIT_FOR;
    PYOPENCL_PARSE_NUMPY_ARRAY_SPEC;

    py::handle<> result = py::handle<>(PyArray_NewFromDescr(
        &PyArray_Type, tp_descr,
        shape.size(), shape.empty( ) ? NULL : &shape.front(), /*strides*/ NULL,
        0, ary_flags, /*obj*/NULL));

    cl_event evt;
    cl_int status_code;
    PYOPENCL_PRINT_CALL_TRACE("clEnqueueMapBuffer");
    void *mapped = clEnqueueMapBuffer(
          cq.data(), buf.data(),
          PYOPENCL_CAST_BOOL(is_blocking), flags,
          offset, PyArray_NBYTES(result.get()),
          num_events_in_wait_list, event_wait_list.empty( ) ? NULL : &event_wait_list.front(), &evt,
          &status_code);
    if (status_code != CL_SUCCESS)
      throw pyopencl::error("clEnqueueMapBuffer", status_code);

    event evt_handle(evt, false);

    std::auto_ptr<memory_map> map;
    try
    {
       map = std::auto_ptr<memory_map>(new memory_map(cq, buf, mapped));
    }
    catch (...)
    {
      PYOPENCL_CALL_GUARDED_CLEANUP(clEnqueueUnmapMemObject, (
            cq.data(), buf.data(), mapped, 0, 0, 0));
      throw;
    }

    PyArray_BYTES(result.get()) = reinterpret_cast<char *>(mapped);

    py::handle<> array_py(handle_from_new_ptr(map.release()));
    PyArray_BASE(result.get()) = array_py.get();
    Py_INCREF(array_py.get());

    return py::make_tuple(
        array_py,
        handle_from_new_ptr(new event(evt_handle)));
  }




  py::object enqueue_map_image(
      command_queue &cq,
      memory_object &img,
      cl_map_flags flags,
      py::object py_origin,
      py::object py_region,
      py::object py_shape, py::object dtype, py::object order_py,
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
    void *mapped = clEnqueueMapImage(
          cq.data(), img.data(),
          PYOPENCL_CAST_BOOL(is_blocking), flags,
          origin, region, &row_pitch, &slice_pitch,
          num_events_in_wait_list, event_wait_list.empty( ) ? NULL : &event_wait_list.front(), &evt,
          &status_code);
    if (status_code != CL_SUCCESS)
      throw pyopencl::error("clEnqueueMapImage", status_code);

    event evt_handle(evt, false);

    std::auto_ptr<memory_map> map;
    try
    {
       map = std::auto_ptr<memory_map>(new memory_map(cq, img, mapped));
    }
    catch (...)
    {
      PYOPENCL_CALL_GUARDED_CLEANUP(clEnqueueUnmapMemObject, (
            cq.data(), img.data(), mapped, 0, 0, 0));
      throw;
    }

    py::handle<> result = py::handle<>(PyArray_NewFromDescr(
        &PyArray_Type, tp_descr,
        shape.size(), shape.empty( ) ? NULL : &shape.front(), /*strides*/ NULL,
        mapped, ary_flags, /*obj*/NULL));

    py::handle<> array_py(handle_from_new_ptr(map.release()));
    PyArray_BASE(result.get()) = array_py.get();
    Py_INCREF(array_py.get());

    return py::make_tuple(
        array_py,
        handle_from_new_ptr(new event(evt_handle)),
        row_pitch, slice_pitch);
  }




  // sampler ------------------------------------------------------------------
  class sampler : boost::noncopyable
  {
    private:
      cl_sampler m_sampler;

    public:
      sampler(context const &ctx, bool normalized_coordinates,
          cl_addressing_mode am, cl_filter_mode fm)
      {
        cl_int status_code;
        m_sampler = clCreateSampler(
            ctx.data(),
            normalized_coordinates,
            am, fm, &status_code);

        PYOPENCL_PRINT_CALL_TRACE("clCreateSampler");
        if (status_code != CL_SUCCESS)
          throw pyopencl::error("Sampler", status_code);
      }

      ~sampler()
      {
        PYOPENCL_CALL_GUARDED_CLEANUP(clReleaseSampler, (m_sampler));
      }

      cl_sampler data() const
      {
        return m_sampler;
      }

      npy_intp obj_ptr() const
      {
        return (npy_intp) data();
      }

      PYOPENCL_EQUALITY_TESTS(sampler);

      py::object get_info(cl_sampler_info param_name) const
      {
        switch (param_name)
        {
          case CL_SAMPLER_REFERENCE_COUNT:
            PYOPENCL_GET_INTEGRAL_INFO(Sampler, m_sampler, param_name,
                cl_uint);
          case CL_SAMPLER_CONTEXT:
            PYOPENCL_GET_OPAQUE_INFO(Sampler, m_sampler, param_name,
                cl_context, context);
          case CL_SAMPLER_ADDRESSING_MODE:
            PYOPENCL_GET_INTEGRAL_INFO(Sampler, m_sampler, param_name,
                cl_addressing_mode);
          case CL_SAMPLER_FILTER_MODE:
            PYOPENCL_GET_INTEGRAL_INFO(Sampler, m_sampler, param_name,
                cl_filter_mode);
          case CL_SAMPLER_NORMALIZED_COORDS:
            PYOPENCL_GET_INTEGRAL_INFO(Sampler, m_sampler, param_name,
                cl_bool);

          default:
            throw error("Sampler.get_info", CL_INVALID_VALUE);
        }
      }
  };




  // program ------------------------------------------------------------------
  class program : boost::noncopyable
  {
    private:
      cl_program m_program;

    public:
      program(cl_program prog, bool retain)
        : m_program(prog)
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

      npy_intp obj_ptr() const
      {
        return (npy_intp) data();
      }

      PYOPENCL_EQUALITY_TESTS(program);

      py::object get_info(cl_program_info param_name) const
      {
        switch (param_name)
        {
          case CL_PROGRAM_REFERENCE_COUNT:
            PYOPENCL_GET_INTEGRAL_INFO(Program, m_program, param_name,
                cl_uint);
          case CL_PROGRAM_CONTEXT:
            PYOPENCL_GET_OPAQUE_INFO(Program, m_program, param_name,
                cl_context, context);
          case CL_PROGRAM_NUM_DEVICES:
            PYOPENCL_GET_INTEGRAL_INFO(Program, m_program, param_name,
                cl_uint);
          case CL_PROGRAM_DEVICES:
            {
              std::vector<cl_device_id> result;
              PYOPENCL_GET_VEC_INFO(Program, m_program, param_name, result);

              py::list py_result;
              BOOST_FOREACH(cl_device_id did, result)
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
            {
              std::vector<size_t> sizes;
              PYOPENCL_GET_VEC_INFO(Program, m_program, CL_PROGRAM_BINARY_SIZES, sizes);

              size_t total_size = std::accumulate(sizes.begin(), sizes.end(), 0);

              boost::scoped_array<unsigned char> result(
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
                   result_ptrs.empty( ) ? NULL : &result_ptrs.front(), 0)); \

              py::list py_result;
              ptr = result.get();
              for (unsigned i = 0; i < sizes.size(); ++i)
              {
                py_result.append(py::str(
                      reinterpret_cast<char *>(ptr),
                      sizes[i]));
                ptr += sizes[i];
              }
              return py_result;
            }

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
            PYOPENCL_GET_INTEGRAL_INFO(ProgramBuild,
                PYOPENCL_FIRST_ARG, param_name,
                cl_build_status);
          case CL_PROGRAM_BUILD_OPTIONS:
          case CL_PROGRAM_BUILD_LOG:
            PYOPENCL_GET_STR_INFO(ProgramBuild,
                PYOPENCL_FIRST_ARG, param_name);
#undef PYOPENCL_FIRST_ARG

          default:
            throw error("Program.get_build_info", CL_INVALID_VALUE);
        }
      }

      void build(std::string options, py::object py_devices)
      {
        if (py_devices.ptr() == Py_None)
        {
          PYOPENCL_CALL_GUARDED(clBuildProgram,
              (m_program, 0, 0, options.c_str(), 0 ,0));
        }
        else
        {
          std::vector<cl_device_id> devices;
          PYTHON_FOREACH(py_dev, py_devices)
            devices.push_back(
                py::extract<device &>(py_dev)().data());
          PYOPENCL_CALL_GUARDED(clBuildProgram,
              (m_program, devices.size(), devices.empty( ) ? NULL : &devices.front(),
               options.c_str(), 0 ,0));
        }
      }
  };




  inline program *create_program_with_source(
      context &ctx,
      std::string const &src)
  {
    const char *string = src.c_str();
    size_t length = src.size();

    cl_int status_code;
    cl_program result = clCreateProgramWithSource(
        ctx.data(), 1, &string, &length, &status_code);
    PYOPENCL_PRINT_CALL_TRACE("clCreateProrgramWithSource");
    if (status_code != CL_SUCCESS)
      throw pyopencl::error("clCreateProrgramWithSource", status_code);

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





  inline program *create_program_with_binary(
      context &ctx,
      py::object py_devices,
      py::object py_binaries)
  {
    std::vector<cl_device_id> devices;
    std::vector<const unsigned char *> binaries;
    std::vector<size_t> sizes;

    int num_devices = len(py_devices);
    if (len(py_binaries) != num_devices)
      throw error("create_program_with_binary", CL_INVALID_VALUE,
          "device and binary counts don't match");

    for (int i = 0; i < num_devices; ++i)
    {
      devices.push_back(
          py::extract<device const &>(py_devices[i])().data());
      const void *buf;
      PYOPENCL_BUFFER_SIZE_T len;

      if (PyObject_AsReadBuffer(
            py::object(py_binaries[i]).ptr(), &buf, &len))
        throw py::error_already_set();

      binaries.push_back(reinterpret_cast<const unsigned char *>(buf));
      sizes.push_back(len);
    }

    cl_int status_code;
    cl_program result = clCreateProgramWithBinary(
        ctx.data(), num_devices,
        devices.empty( ) ? NULL : &devices.front(), sizes.empty( ) ? NULL : &sizes.front(), binaries.empty( ) ? NULL : &binaries.front(),
        /*binary_status*/ 0, &status_code);
    PYOPENCL_PRINT_CALL_TRACE("clCreateProrgramWithBinary");
    if (status_code != CL_SUCCESS)
      throw pyopencl::error("clCreateProrgramWithBinary", status_code);

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



  void unload_compiler()
  {
    PYOPENCL_CALL_GUARDED(clUnloadCompiler, ());
  }




  // kernel -------------------------------------------------------------------
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




  class kernel : boost::noncopyable
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

        m_kernel = clCreateKernel(prg.data(), kernel_name.c_str(),
            &status_code);
        PYOPENCL_PRINT_CALL_TRACE("clCreateKernel");
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

      npy_intp obj_ptr() const
      {
        return (npy_intp) data();
      }

      PYOPENCL_EQUALITY_TESTS(kernel);

      void set_arg_null(cl_uint arg_index)
      {
        PYOPENCL_CALL_GUARDED(clSetKernelArg, (m_kernel, arg_index, 0, 0));
      }

      void set_arg_mem(cl_uint arg_index, memory_object &mo)
      {
        cl_mem m = mo.data();
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

      void set_arg_buf(cl_uint arg_index, py::object py_buffer)
      {
        const void *buf;
        PYOPENCL_BUFFER_SIZE_T len;

        if (PyObject_AsReadBuffer(py_buffer.ptr(), &buf, &len))
          throw py::error_already_set();

        PYOPENCL_CALL_GUARDED(clSetKernelArg,
            (m_kernel, arg_index, len, buf));
      }

      void set_arg(cl_uint arg_index, py::object arg)
      {
        if (arg.ptr() == Py_None)
        {
          set_arg_null(arg_index);
          return;
        }

        py::extract<memory_object &> ex_mo(arg);
        if (ex_mo.check())
        {
          set_arg_mem(arg_index, ex_mo());
          return;
        }

        py::extract<local_memory const &> ex_loc(arg);
        if (ex_loc.check())
        {
          set_arg_local(arg_index, ex_loc());
          return;
        }

        py::extract<sampler const &> ex_smp(arg);
        if (ex_smp.check())
        {
          set_arg_sampler(arg_index, ex_smp());
          return;
        }

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
            PYOPENCL_GET_INTEGRAL_INFO(Kernel, m_kernel, param_name,
                cl_uint);
          case CL_KERNEL_CONTEXT:
            PYOPENCL_GET_OPAQUE_INFO(Kernel, m_kernel, param_name,
                cl_context, context);
          case CL_KERNEL_PROGRAM:
            PYOPENCL_GET_OPAQUE_INFO(Kernel, m_kernel, param_name,
                cl_program, program);
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
            PYOPENCL_GET_INTEGRAL_INFO(KernelWorkGroup,
                PYOPENCL_FIRST_ARG, param_name,
                size_t);
          case CL_KERNEL_COMPILE_WORK_GROUP_SIZE:
            {
              std::vector<size_t> result;
              PYOPENCL_GET_VEC_INFO(KernelWorkGroup,
                  PYOPENCL_FIRST_ARG, param_name, result);
              return py::list(result);
            }
          case CL_KERNEL_LOCAL_MEM_SIZE:
            PYOPENCL_GET_INTEGRAL_INFO(KernelWorkGroup,
                PYOPENCL_FIRST_ARG, param_name,
                cl_ulong);
          default:
            throw error("Kernel.get_work_group_info", CL_INVALID_VALUE);
#undef PYOPENCL_FIRST_ARG
        }
      }
  };



  py::list create_kernels_in_program(program &pgm)
  {
    py::list result;
    cl_uint num_kernels;
    PYOPENCL_CALL_GUARDED(clCreateKernelsInProgram, (
          pgm.data(), 0, 0, &num_kernels));

    std::vector<cl_kernel> kernels;
    PYOPENCL_CALL_GUARDED(clCreateKernelsInProgram, (
          pgm.data(), num_kernels, kernels.empty( ) ? NULL : &kernels.front(), &num_kernels));

    BOOST_FOREACH(cl_kernel knl, kernels)
      result.append(handle_from_new_ptr(new kernel(knl, true)));

    return result;
  }



  event *enqueue_nd_range_kernel(
      command_queue &cq,
      kernel &knl,
      py::object py_global_work_size,
      py::object py_local_work_size,
      py::object py_global_work_offset,
      py::object py_wait_for)
  {
    PYOPENCL_PARSE_WAIT_FOR;

    cl_uint work_dim = len(py_global_work_size);

    std::vector<size_t> global_work_size;
    COPY_PY_LIST(size_t, global_work_size);

    size_t *local_work_size_ptr = 0;
    std::vector<size_t> local_work_size;
    if (py_local_work_size.ptr() != Py_None)
    {
      if (work_dim != len(py_local_work_size))
        throw error("enqueue_nd_range_kernel", CL_INVALID_VALUE,
            "global/work work sizes have differing dimensions");

      COPY_PY_LIST(size_t, local_work_size);

      local_work_size_ptr = local_work_size.empty( ) ? NULL : &local_work_size.front();
    }

    size_t *global_work_offset_ptr = 0;
    std::vector<size_t> global_work_offset;
    if (py_global_work_offset.ptr() != Py_None)
    {
      if (work_dim != len(py_local_work_size))
        throw error("enqueue_nd_range_kernel", CL_INVALID_VALUE,
            "global work size and offset have differing dimensions");

      COPY_PY_LIST(size_t, global_work_offset);

      global_work_offset_ptr = global_work_offset.empty( ) ? NULL :  &global_work_offset.front();
    }

    cl_event evt;
    PYOPENCL_CALL_GUARDED(clEnqueueNDRangeKernel, (
          cq.data(),
          knl.data(),
          work_dim,
          global_work_offset_ptr,
          global_work_size.empty( ) ? NULL : &global_work_size.front(),
          local_work_size_ptr,
          num_events_in_wait_list, event_wait_list.empty( ) ? NULL : &event_wait_list.front(), &evt
          ));

    PYOPENCL_RETURN_NEW_EVENT(evt);
  }






  event *enqueue_task(
      command_queue &cq,
      kernel &knl,
      py::object py_wait_for)
  {
    PYOPENCL_PARSE_WAIT_FOR;

    cl_event evt;
    PYOPENCL_CALL_GUARDED(clEnqueueTask, (
          cq.data(),
          knl.data(),
          num_events_in_wait_list, event_wait_list.empty( ) ? NULL : &event_wait_list.front(), &evt
          ));

    PYOPENCL_RETURN_NEW_EVENT(evt);
  }




  // gl interop ---------------------------------------------------------------
  bool have_gl()
  {
#ifdef HAVE_GL
    return true;
#else
    return false;
#endif
  }




#ifdef HAVE_GL
  class gl_buffer : public memory_object
  {
    public:
      gl_buffer(cl_mem mem, bool retain, py::object *hostbuf=0)
        : memory_object(mem, retain, hostbuf)
      { }
  };




  class gl_renderbuffer : public memory_object
  {
    public:
      gl_renderbuffer(cl_mem mem, bool retain, py::object *hostbuf=0)
        : memory_object(mem, retain, hostbuf)
      { }
  };




  class gl_texture : public image
  {
    public:
      gl_texture(cl_mem mem, bool retain, py::object *hostbuf=0)
        : image(mem, retain, hostbuf)
      { }

      py::object get_gl_texture_info(cl_gl_texture_info param_name)
      {
        switch (param_name)
        {
          case CL_GL_TEXTURE_TARGET:
            PYOPENCL_GET_INTEGRAL_INFO(GLTexture, data(), param_name, GLenum);
          case CL_GL_MIPMAP_LEVEL:
            PYOPENCL_GET_INTEGRAL_INFO(GLTexture, data(), param_name, GLint);

          default:
            throw error("MemoryObject.get_gl_texture_info", CL_INVALID_VALUE);
        }
      }
  };




#define PYOPENCL_WRAP_BUFFER_CREATOR(TYPE, NAME, CL_NAME, ARGS, CL_ARGS) \
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





  py::tuple get_gl_object_info(memory_object const &mem)
  {
    cl_gl_object_type otype;
    GLuint gl_name;
    PYOPENCL_CALL_GUARDED(clGetGLObjectInfo, (mem.data(), &otype, &gl_name));
    return py::make_tuple(otype, gl_name);
  }

#define WRAP_GL_ENQUEUE(what, What) \
  event *enqueue_##what##_gl_objects( \
      command_queue &cq, \
      py::object py_mem_objects, \
      py::object py_wait_for) \
  { \
    PYOPENCL_PARSE_WAIT_FOR; \
    \
    std::vector<cl_mem> mem_objects; \
    PYTHON_FOREACH(mo, py_mem_objects) \
      mem_objects.push_back(py::extract<memory_object &>(mo)().data()); \
    \
    cl_event evt; \
    PYOPENCL_CALL_GUARDED(clEnqueue##What##GLObjects, ( \
          cq.data(), \
          mem_objects.size(), mem_objects.empty( ) ? NULL : &mem_objects.front(), \
          num_events_in_wait_list, event_wait_list.empty( ) ? NULL : &event_wait_list.front(), &evt \
          )); \
    \
    PYOPENCL_RETURN_NEW_EVENT(evt); \
  }

  WRAP_GL_ENQUEUE(acquire, Acquire);
  WRAP_GL_ENQUEUE(release, Release);
#endif
}




#endif
