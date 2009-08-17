#ifndef _AFJHAYYTA_PYOPENCL_HEADER_SEEN_CL_HPP
#define _AFJHAYYTA_PYOPENCL_HEADER_SEEN_CL_HPP




// TODO: Images and samplers
// TODO: Memory mapping
// TODO: Profiling
// TODO: GL Interop




#include <CL/cl.h>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <utility>
#include <boost/foreach.hpp>
#include "wrap_helpers.hpp"




// #define PYOPENCL_TRACE




// tools ----------------------------------------------------------------------
#if PY_VERSION_HEX >= 0x02050000
  typedef Py_ssize_t PYOPENCL_BUFFER_SIZE_T;
#else
  typedef int PYOPENCL_BUFFER_SIZE_T;
#endif

#define PYOPENCL_CAST_BOOL(B) ((B) ? CL_TRUE : CL_FALSE)





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
    cl_uint size; \
    PYOPENCL_CALL_GUARDED(clGet##WHAT##Info, \
        (FIRST_ARG, SECOND_ARG, 0, 0, &size)); \
    \
    RES_VEC.resize(size / sizeof(*RES_VEC.data())); \
    \
    PYOPENCL_CALL_GUARDED(clGet##WHAT##Info, \
        (FIRST_ARG, SECOND_ARG, size, \
         RES_VEC.data(), &size)); \
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
         0, &param_value_size)); \
    \
    return py::object( \
        std::string(param_value.data(), param_value_size)); \
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
    std::vector<cl_event> event_wait_list(len(py_wait_for)); \
    \
    { \
      if (py_wait_for.ptr() != Py_None) \
      { \
        PYTHON_FOREACH(evt, py_wait_for) \
          event_wait_list[num_events_in_wait_list++] = \
          py::extract<event &>(evt)().data(); \
      } \
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
          case CL_COMPILER_NOT_AVAILABLE: return "device compiler not available";
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

      py::object get_info(cl_platform_info param_name) const
      {
        switch (param_name)
        {
          case CL_PLATFORM_PROFILE:
          case CL_PLATFORM_VERSION:
          case CL_PLATFORM_NAME:
          case CL_PLATFORM_VENDOR:
          case CL_PLATFORM_EXTENSIONS:
            PYOPENCL_GET_STR_INFO(Platform, m_platform, param_name);

          default:
            throw error("Platform.get_info", CL_INVALID_VALUE);
        }
      }

      py::list get_devices(cl_device_type devtype);

      cl_platform_id data() const
      {
        return m_platform;
      }
  };




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
          case CL_DEVICE_MAX_WORK_GROUP_SIZE: DEV_GET_INT_INF(cl_uint);

          case CL_DEVICE_MAX_WORK_ITEM_SIZES: 
            {
              std::vector<size_t> result;
              PYOPENCL_GET_VEC_INFO(Device, m_device, param_name, result);
              return py::list(result);
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
          case CL_DEVICE_IMAGE_SUPPORT: DEV_GET_INT_INF(bool);
          case CL_DEVICE_MAX_PARAMETER_SIZE: DEV_GET_INT_INF(bool);
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
          case CL_DEVICE_ENDIAN_LITTLE: DEV_GET_INT_INF(bool);
          case CL_DEVICE_AVAILABLE: DEV_GET_INT_INF(bool);
          case CL_DEVICE_COMPILER_AVAILABLE: DEV_GET_INT_INF(bool);
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
      
      cl_device_id data() const
      {
        return m_device;
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
         num_devices, devices.data(), &num_devices));

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

      context(
          py::list py_devices,
          py::list py_properties)
      {
        std::vector<cl_context_properties> props;

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

        std::vector<cl_device_id> devices;
        PYTHON_FOREACH(py_dev, py_devices)
        {
          py::extract<const device &> dev(py_dev);
            devices.push_back(dev().data());
        }

        cl_int status_code;
        m_context = clCreateContext(
            props.data(),
            devices.size(),
            devices.data(),
            0, 0, &status_code);

        if (status_code != CL_SUCCESS)
          throw pyopencl::error("Context", status_code);
      }

      ~context()
      {
        PYOPENCL_CALL_GUARDED_CLEANUP(clReleaseContext, 
            (m_context));
      }

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

          case CL_CONTEXT_PROPERTIES: // FIXME: complicated

          default:
            throw error("Context.get_info", CL_INVALID_VALUE);
        }
      }

      cl_context data() const
      {
        return m_context;
      }
  };




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

      command_queue(
          const context &ctx,
          const device &dev,
          cl_command_queue_properties props=0)
      {
        cl_int status_code;
        m_queue = clCreateCommandQueue(
            ctx.data(),
            dev.data(),
            props,
            &status_code);

        if (status_code != CL_SUCCESS)
          throw pyopencl::error("CommandQueue", status_code);
      }

      ~command_queue()
      {
        PYOPENCL_CALL_GUARDED_CLEANUP(clReleaseCommandQueue, 
            (m_queue));
      }

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

      const cl_command_queue data() const
      { return m_queue; }
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

      ~event()
      {
        PYOPENCL_CALL_GUARDED_CLEANUP(clReleaseEvent, 
            (m_event));
      }

      py::object get_info(cl_command_queue_info param_name) const
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

      const cl_event data() const
      { return m_event; }
  };




  void wait_for_events(py::object events)
  {
    cl_uint num_events_in_wait_list = 0;
    std::vector<cl_event> event_wait_list(len(events));

    PYTHON_FOREACH(evt, events)
      event_wait_list[num_events_in_wait_list++] = 
      py::extract<event &>(evt)().data();

    PYOPENCL_CALL_GUARDED_THREADED(clWaitForEvents, (
          num_events_in_wait_list, event_wait_list.data()));
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
          cq.data(), num_events, event_list.data()));
  }




  void enqueue_barrier(command_queue &cq)
  {
    PYOPENCL_CALL_GUARDED(clEnqueueBarrier, (cq.data()));
  }




  // memory -------------------------------------------------------------------
  class memory_object : boost::noncopyable
  {
    private:
      cl_mem    m_mem;
      py::object m_hostbuf;

    public:
      memory_object(cl_mem mem, bool retain, py::object *hostbuf=0)
        : m_mem(mem)
      { 
        if (retain)
          PYOPENCL_CALL_GUARDED(clRetainMemObject, (mem));

        if (hostbuf)
          m_hostbuf = *hostbuf;
      }

      ~memory_object()
      {
        PYOPENCL_CALL_GUARDED_CLEANUP(clReleaseMemObject, 
            (m_mem));
      }

      const cl_mem data() const
      { return m_mem; }

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

      py::object get_image_info(cl_image_info param_name) const
      {
        switch (param_name) 
        {
          case CL_IMAGE_FORMAT:
            PYOPENCL_GET_INTEGRAL_INFO(Image, m_mem, param_name, 
                cl_image_format);
          case CL_IMAGE_ELEMENT_SIZE:
          case CL_IMAGE_ROW_PITCH:
          case CL_IMAGE_SLICE_PITCH:
          case CL_IMAGE_WIDTH:
          case CL_IMAGE_HEIGHT:
          case CL_IMAGE_DEPTH:
            PYOPENCL_GET_INTEGRAL_INFO(Image, m_mem, param_name, size_t);

          default:
            throw error("MemoryObject.get_image_info", CL_INVALID_VALUE);
        }
      }
  };




  memory_object *create_buffer(
      context &ctx,
      cl_mem_flags flags,
      size_t size)
  {
    cl_int status_code;
    cl_mem mem = clCreateBuffer(ctx.data(), flags, size, 0, &status_code);

    if (status_code != CL_SUCCESS)
      throw pyopencl::error("create_buffer", status_code);

    try
    {
      return new memory_object(mem, false);
    }
    catch (...)
    {
      PYOPENCL_CALL_GUARDED(clReleaseMemObject, (mem));
      throw;
    }
  }




  memory_object *create_host_buffer(
      context &ctx,
      cl_mem_flags flags,
      py::object buffer)
  {
    void *buf;
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

    cl_int status_code;
    cl_mem mem = clCreateBuffer(ctx.data(), flags, len, buf, &status_code);

    if (status_code != CL_SUCCESS)
      throw pyopencl::error("create_host_buffer", status_code);

    py::object *retained_buf_obj = 0;
    if (flags & CL_MEM_USE_HOST_PTR)
      retained_buf_obj = &buffer;

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
      bool is_blocking
      )
  {
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
          num_events_in_wait_list, event_wait_list.data(), &evt
          ));
    PYOPENCL_RETURN_NEW_EVENT(evt);
  }




  event *enqueue_write_buffer(
      command_queue &cq,
      memory_object &mem,
      py::object buffer,
      size_t device_offset,
      py::object py_wait_for,
      bool is_blocking
      )
  {
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
          num_events_in_wait_list, event_wait_list.data(), &evt
          ));
    PYOPENCL_RETURN_NEW_EVENT(evt);
  }
}





#endif
