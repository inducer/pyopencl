#include "wrap_cl.h"
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <string.h>
#include <memory>
#include <sstream>
#include "bitlog.hpp"

#define MALLOC(TYPE, VAR, N) TYPE *VAR = reinterpret_cast<TYPE*>(malloc(sizeof(TYPE)*(N)));

// {{{ tracing and error reporting
#ifdef PYOPENCL_TRACE
#define PYOPENCL_PRINT_CALL_TRACE(NAME)		\
  std::cerr << NAME << std::endl;
#define PYOPENCL_PRINT_CALL_TRACE_INFO(NAME, EXTRA_INFO)	\
  std::cerr << NAME << " (" << EXTRA_INFO << ')' << std::endl;
#else
#define PYOPENCL_PRINT_CALL_TRACE(NAME) /*nothing*/
#define PYOPENCL_PRINT_CALL_TRACE_INFO(NAME, EXTRA_INFO) /*nothing*/
#endif

#define C_HANDLE_ERROR(OPERATION)			\
  try {							\
    OPERATION						\
      } catch(const pyopencl::error& e) {		\
    MALLOC(::error, error, 1);				\
    error->routine = pyopencl::_copy_str(e.routine());	\
    error->msg = pyopencl::_copy_str(e.what());		\
    error->code = e.code();				\
    error->other = 0;					\
    return error;					\
  } catch(const std::exception& e) {			\
    /* non-pyopencl exceptions shall be */		\
    /* converted as well */				\
    MALLOC(::error, error, 1);				\
    error->other = 1;					\
    error->msg = pyopencl::_copy_str(e.what());		\
    return error;					\
  }							\
  

// TODO Py_BEGIN_ALLOW_THREADS \ Py_END_ALLOW_THREADS below
#define PYOPENCL_CALL_GUARDED_THREADED(NAME, ARGLIST)	\
  {							\
    PYOPENCL_PRINT_CALL_TRACE(#NAME);			\
    cl_int status_code;					\
    status_code = NAME ARGLIST;				\
    if (status_code != CL_SUCCESS)			\
      throw pyopencl::error(#NAME, status_code);	\
  }


#define PYOPENCL_CALL_GUARDED(NAME, ARGLIST)		\
  {							\
    PYOPENCL_PRINT_CALL_TRACE(#NAME);			\
    cl_int status_code;					\
    status_code = NAME ARGLIST;				\
    if (status_code != CL_SUCCESS)			\
      throw pyopencl::error(#NAME, status_code);	\
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



#define PYOPENCL_GET_VEC_INFO(WHAT, FIRST_ARG, SECOND_ARG, RES_VEC)	\
  {									\
    size_t size;							\
    PYOPENCL_CALL_GUARDED(clGet##WHAT##Info,				\
			  (FIRST_ARG, SECOND_ARG, 0, 0, &size));	\
									\
    RES_VEC.resize(size / sizeof(RES_VEC.front()));			\
									\
    PYOPENCL_CALL_GUARDED(clGet##WHAT##Info,				\
			  (FIRST_ARG, SECOND_ARG, size,			\
			   RES_VEC.empty( ) ? NULL : &RES_VEC.front(), &size)); \
  }

#define PYOPENCL_GET_OPAQUE_INFO(WHAT, FIRST_ARG, SECOND_ARG, CL_TYPE, TYPE, TYPEU) \
  {									\
    CL_TYPE param_value;						\
    PYOPENCL_CALL_GUARDED(clGet##WHAT##Info,				\
			  (FIRST_ARG, SECOND_ARG, sizeof(param_value), &param_value, 0)); \
    generic_info info;							\
    info.dontfree = 0;							\
    info.opaque_class = CLASS_##TYPEU;					\
      info.type = "void *";						\
      if (param_value)							\
	info.value = (void*)(new TYPE(param_value, /*retain*/ true));	\
      else								\
	info.value = NULL;						\
      return info;							\
  }


#define PYOPENCL_GET_OPAQUE_ARRAY_INFO(TYPE, CLS, CLSU, VEC)		\
  {									\
    MALLOC(void*, ar, VEC.size());					\
    for(uint32_t i = 0; i < VEC.size(); ++i) {				\
      ar[i] = new pyopencl::CLS(VEC[i]);				\
    }									\
    generic_info info;							\
    info.dontfree = 0;							\
    info.opaque_class = CLASS_##CLSU;					\
      info.type = _copy_str(std::string("void*[") + tostring(VEC.size()) + "]"); \
      info.value = (void**)ar;						\
      return info;							\
  }
  

#define PYOPENCL_GET_STR_INFO(WHAT, FIRST_ARG, SECOND_ARG)		\
  {									\
    size_t param_value_size;						\
    PYOPENCL_CALL_GUARDED(clGet##WHAT##Info,				\
			  (FIRST_ARG, SECOND_ARG, 0, 0, &param_value_size)); \
									\
    MALLOC(char, param_value, param_value_size);			\
    PYOPENCL_CALL_GUARDED(clGet##WHAT##Info,				\
			  (FIRST_ARG, SECOND_ARG, param_value_size,	\
			   param_value, &param_value_size));		\
    generic_info info;							\
    info.dontfree = 0;							\
    info.opaque_class = CLASS_NONE;					\
    info.type = "char*";						\
    info.value = (void*)param_value;					\
    return info;							\
  }

#define PYOPENCL_GET_INTEGRAL_INFO(WHAT, FIRST_ARG, SECOND_ARG, TYPE)	\
  {									\
    MALLOC(TYPE, param_value, 1);					\
    PYOPENCL_CALL_GUARDED(clGet##WHAT##Info,				\
			  (FIRST_ARG, SECOND_ARG, sizeof(param_value), param_value, 0)); \
    generic_info info;							\
    info.dontfree = 0;							\
    info.opaque_class = CLASS_NONE;					\
    info.type = #TYPE"*";						\
      info.value = (void*)param_value;					\
      return info;							\
  }

#define PYOPENCL_GET_ARRAY_INFO(TYPE, VEC)				\
  {									\
    MALLOC(TYPE, ar, VEC.size());					\
    for(uint32_t i = 0; i < VEC.size(); ++i) {				\
      ar[i] = VEC[i];							\
    }									\
    generic_info info;							\
    info.dontfree = 0;							\
    info.opaque_class = CLASS_NONE;					\
    info.type = _copy_str(std::string(#TYPE"[") + tostring(VEC.size()) + "]"); \
    info.value = (void*)ar;						\
    return info;							\
  }

// {{{ event helpers --------------------------------------------------------------
#define PYOPENCL_PARSE_WAIT_FOR						\
  std::vector<cl_event> event_wait_list(num_wait_for);			\
  {									\
    for(unsigned i = 0; i < num_wait_for; ++i) {			\
      event_wait_list[i] = static_cast<pyopencl::event*>(wait_for[i])->data(); \
    }									\
  }

#define PYOPENCL_WAITLIST_ARGS						\
  num_wait_for, event_wait_list.empty( ) ? NULL : &event_wait_list.front()


#define PYOPENCL_RETURN_NEW_EVENT(evt)		\
  try						\
    {						\
      return new event(evt, false);		\
    }						\
  catch (...)					\
    {						\
      clReleaseEvent(evt);			\
      throw;					\
    }

// }}}

// {{{ equality testing
#define PYOPENCL_EQUALITY_TESTS(cls)		\
  bool operator==(cls const &other) const	\
  { return data() == other.data(); }		\
  bool operator!=(cls const &other) const	\
  { return data() != other.data(); }		\
  long hash() const				\
  { return (long) (intptr_t) data(); }
// }}}



// {{{ tools
#define PYOPENCL_CAST_BOOL(B) ((B) ? CL_TRUE : CL_FALSE)

#define PYOPENCL_PARSE_PY_DEVICES					\
  std::vector<cl_device_id> devices_vec;				\
  cl_uint num_devices;							\
  cl_device_id *devices;						\
									\
  if (py_devices.ptr() == Py_None)					\
    {									\
      num_devices = 0;							\
      devices = 0;							\
    }									\
  else									\
    {									\
      PYTHON_FOREACH(py_dev, py_devices)				\
        devices_vec.push_back(						\
			      py::extract<device &>(py_dev)().data());	\
      num_devices = devices_vec.size();					\
      devices = devices_vec.empty( ) ? NULL : &devices_vec.front();	\
    }									\



#define PYOPENCL_RETRY_IF_MEM_ERROR(OPERATION)	\
  {						\
  bool failed_with_mem_error = false;		\
  try						\
    {						\
      OPERATION					\
	}					\
  catch (pyopencl::error &e)			\
    {						\
      failed_with_mem_error = true;		\
      if (!e.is_out_of_memory())		\
	throw;					\
    }						\
						\
  if (failed_with_mem_error)			\
    {						\
  /* If we get here, we got an error from CL.
   * We should run the Python GC to try and free up
   * some memory references. */			\
run_python_gc();				\
\
/* Now retry the allocation. If it fails again,
 * let it fail. */				\
{						\
  OPERATION					\
    }						\
}						\
}

// }}}


#define SWITCHCLASS(OPERATION)						\
  switch(class_) {							\
  case ::CLASS_PLATFORM: OPERATION(PLATFORM, platform); break;		\
  case ::CLASS_DEVICE: OPERATION(DEVICE, device); break;		\
  case ::CLASS_KERNEL: OPERATION(KERNEL, kernel); break;		\
  case ::CLASS_CONTEXT: OPERATION(CONTEXT, context); break;		\
  case ::CLASS_COMMAND_QUEUE: OPERATION(COMMAND_QUEUE, command_queue); break; \
  case ::CLASS_BUFFER: OPERATION(BUFFER, buffer); break;		\
  case ::CLASS_PROGRAM: OPERATION(PROGRAM, program); break;		\
  case ::CLASS_EVENT: OPERATION(EVENT, event); break;			\
  case ::CLASS_GL_BUFFER: OPERATION(GL_BUFFER, gl_buffer); break;			\
  default: throw pyopencl::error("unknown class", CL_INVALID_VALUE);	\
  }

#define PYOPENCL_CL_PLATFORM cl_platform_id
#define PYOPENCL_CL_DEVICE cl_device_id
#define PYOPENCL_CL_KERNEL cl_kernel
#define PYOPENCL_CL_CONTEXT cl_context
#define PYOPENCL_CL_COMMAND_QUEUE cl_command_queue
#define PYOPENCL_CL_BUFFER cl_mem
#define PYOPENCL_CL_PROGRAM cl_program
#define PYOPENCL_CL_EVENT cl_event
#define PYOPENCL_CL_GL_BUFFER cl_mem

template<class T>
std::string tostring(const T& v) {
  std::ostringstream ostr;
  ostr << v;
  return ostr.str();
}


namespace pyopencl
{
  char *_copy_str(const std::string& str) {
    MALLOC(char, cstr, str.size() + 1);
    std::size_t len = str.copy(cstr, str.size());
    cstr[len] = '\0';
    return cstr;
  }
  
  // {{{ error
  class error : public std::runtime_error
  {
  private:
    const char *m_routine;
    cl_int m_code;

  public:
    error(const char *rout, cl_int c, const char *msg="")
      : std::runtime_error(msg), m_routine(rout), m_code(c)
    { std::cout << rout <<";" << msg<< ";" << c << std::endl; }
    const char *routine() const
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

  };

  // }}}


  //#define MAKE_INFO(name, type, value) {  }

  class _common {
  };
  

  // {{{ platform
  class platform : public _common
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

    generic_info get_info(cl_platform_info param_name) const
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

    std::vector<cl_device_id> get_devices(cl_device_type devtype);
  }; 

  
  inline std::vector<cl_device_id> platform::get_devices(cl_device_type devtype)
  {
    cl_uint num_devices = 0;
    PYOPENCL_CALL_GUARDED(clGetDeviceIDs,
			  (m_platform, devtype, 0, 0, &num_devices));

    std::vector<cl_device_id> devices(num_devices);
    PYOPENCL_CALL_GUARDED(clGetDeviceIDs,
			  (m_platform, devtype,
			   num_devices, devices.empty( ) ? NULL : &devices.front(), &num_devices));
    
    return devices;
  }

  // }}}

  // {{{ device
  class device : public _common // : boost::noncopyable
  {
  public:
    enum reference_type_t {
      REF_NOT_OWNABLE,
      REF_FISSION_EXT,
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
#if (defined(cl_ext_device_fission) && defined(PYOPENCL_USE_DEVICE_FISSION))
          else if (ref_type == REF_FISSION_EXT)
	    {
#if PYOPENCL_CL_VERSION >= 0x1020
	      cl_platform_id plat;
	      PYOPENCL_CALL_GUARDED(clGetDeviceInfo, (m_device, CL_DEVICE_PLATFORM,
						      sizeof(plat), &plat, NULL));
#endif

	      PYOPENCL_GET_EXT_FUN(plat,
				   clRetainDeviceEXT, retain_func);

	      PYOPENCL_CALL_GUARDED(retain_func, (did));
	    }
#endif

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
      if (false)
        { }
#if defined(cl_ext_device_fission) && defined(PYOPENCL_USE_DEVICE_FISSION)
      else if (m_ref_type == REF_FISSION_EXT)
        {
#if PYOPENCL_CL_VERSION >= 0x1020
          cl_platform_id plat;
          PYOPENCL_CALL_GUARDED(clGetDeviceInfo, (m_device, CL_DEVICE_PLATFORM,
						  sizeof(plat), &plat, NULL));
#endif

          PYOPENCL_GET_EXT_FUN(plat,
			       clReleaseDeviceEXT, release_func);

          PYOPENCL_CALL_GUARDED_CLEANUP(release_func, (m_device));
        }
#endif

#if PYOPENCL_CL_VERSION >= 0x1020
      else if (m_ref_type == REF_CL_1_2)
	PYOPENCL_CALL_GUARDED(clReleaseDevice, (m_device));
#endif
    }

    cl_device_id data() const
    {
      return m_device;
    }

    PYOPENCL_EQUALITY_TESTS(device);

    generic_info get_info(cl_device_info param_name) const
    {
#define DEV_GET_INT_INF(TYPE) PYOPENCL_GET_INTEGRAL_INFO(Device, m_device, param_name, TYPE);

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
	    PYOPENCL_GET_ARRAY_INFO(size_t, result);
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
	case CL_DEVICE_QUEUE_PROPERTIES: DEV_GET_INT_INF(cl_command_queue_properties);

	case CL_DEVICE_NAME:
	case CL_DEVICE_VENDOR:
	case CL_DRIVER_VERSION:
	case CL_DEVICE_PROFILE:
	case CL_DEVICE_VERSION:
	case CL_DEVICE_EXTENSIONS:
	  PYOPENCL_GET_STR_INFO(Device, m_device, param_name);
	  
	case CL_DEVICE_PLATFORM:
	  PYOPENCL_GET_OPAQUE_INFO(Device, m_device, param_name, cl_platform_id, platform, PLATFORM);

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
#if defined(cl_ext_device_fission) && defined(PYOPENCL_USE_DEVICE_FISSION)
	case CL_DEVICE_PARENT_DEVICE_EXT:
	  PYOPENCL_GET_OPAQUE_INFO(Device, m_device, param_name, cl_device_id, device, DEVICE);
	case CL_DEVICE_PARTITION_TYPES_EXT:
	case CL_DEVICE_AFFINITY_DOMAINS_EXT:
	case CL_DEVICE_PARTITION_STYLE_EXT:
	  {
	    std::vector<cl_device_partition_property_ext> result;
	    PYOPENCL_GET_VEC_INFO(Device, m_device, param_name, result);
	    PYOPENCL_GET_ARRAY_INFO(cl_device_partition_property_ext, result);
	  }
	case CL_DEVICE_REFERENCE_COUNT_EXT: DEV_GET_INT_INF(cl_uint);
#endif
#if PYOPENCL_CL_VERSION >= 0x1020
	case CL_DEVICE_LINKER_AVAILABLE: DEV_GET_INT_INF(cl_bool);
	case CL_DEVICE_BUILT_IN_KERNELS:
	  PYOPENCL_GET_STR_INFO(Device, m_device, param_name);
	case CL_DEVICE_IMAGE_MAX_BUFFER_SIZE: DEV_GET_INT_INF(size_t);
	case CL_DEVICE_IMAGE_MAX_ARRAY_SIZE: DEV_GET_INT_INF(size_t);
	case CL_DEVICE_PARENT_DEVICE:
	  PYOPENCL_GET_OPAQUE_INFO(Device, m_device, param_name, cl_device_id, device, DEVICE);
	case CL_DEVICE_PARTITION_MAX_SUB_DEVICES: DEV_GET_INT_INF(cl_uint);
	case CL_DEVICE_PARTITION_TYPE:
	case CL_DEVICE_PARTITION_PROPERTIES:
	  {
	    std::vector<cl_device_partition_property> result;
	    PYOPENCL_GET_VEC_INFO(Device, m_device, param_name, result);
	    PYOPENCL_GET_ARRAY_INFO(cl_device_partition_property, result);
	  }
	case CL_DEVICE_PARTITION_AFFINITY_DOMAIN:
	  {
	    std::vector<cl_device_affinity_domain> result;
	    PYOPENCL_GET_VEC_INFO(Device, m_device, param_name, result);
	    PYOPENCL_GET_ARRAY_INFO(cl_device_affinity_domain, result);
	  }
	case CL_DEVICE_REFERENCE_COUNT: DEV_GET_INT_INF(cl_uint);
	case CL_DEVICE_PREFERRED_INTEROP_USER_SYNC: DEV_GET_INT_INF(cl_bool);
	case CL_DEVICE_PRINTF_BUFFER_SIZE: DEV_GET_INT_INF(cl_bool);
#endif
	  // {{{ AMD dev attrs
	  //
	  // types of AMD dev attrs divined from
	  // https://www.khronos.org/registry/cl/api/1.2/cl.hpp
#ifdef CL_DEVICE_PROFILING_TIMER_OFFSET_AMD
	case CL_DEVICE_PROFILING_TIMER_OFFSET_AMD: DEV_GET_INT_INF(cl_ulong);
#endif
	  /* FIXME
	     #ifdef CL_DEVICE_TOPOLOGY_AMD
	     case CL_DEVICE_TOPOLOGY_AMD:
	     #endif
	  */
#ifdef CL_DEVICE_BOARD_NAME_AMD
	case CL_DEVICE_BOARD_NAME_AMD: ;
	  PYOPENCL_GET_STR_INFO(Device, m_device, param_name);
#endif
#ifdef CL_DEVICE_GLOBAL_FREE_MEMORY_AMD
	case CL_DEVICE_GLOBAL_FREE_MEMORY_AMD:
	  {
	    std::vector<size_t> result;
	    PYOPENCL_GET_VEC_INFO(Device, m_device, param_name, result);
	    PYOPENCL_GET_ARRAY_INFO(size_t, result);
	  }
#endif
#ifdef CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD
	case CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD: DEV_GET_INT_INF(cl_uint);
#endif
#ifdef CL_DEVICE_SIMD_WIDTH_AMD
	case CL_DEVICE_SIMD_WIDTH_AMD: DEV_GET_INT_INF(cl_uint);
#endif
#ifdef CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD
	case CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD: DEV_GET_INT_INF(cl_uint);
#endif
#ifdef CL_DEVICE_WAVEFRONT_WIDTH_AMD
	case CL_DEVICE_WAVEFRONT_WIDTH_AMD: DEV_GET_INT_INF(cl_uint);
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
	  // }}}

#ifdef CL_DEVICE_MAX_ATOMIC_COUNTERS_EXT
	case CL_DEVICE_MAX_ATOMIC_COUNTERS_EXT: DEV_GET_INT_INF(cl_uint);
#endif

	default:
	  throw error("Device.get_info", CL_INVALID_VALUE);
        }
    }

    // #if PYOPENCL_CL_VERSION >= 0x1020
    //       py::list create_sub_devices(py::object py_properties)
    //       {
    //         std::vector<cl_device_partition_property> properties;

    //         COPY_PY_LIST(cl_device_partition_property, properties);
    //         properties.push_back(0);

    //         cl_device_partition_property *props_ptr
    //           = properties.empty( ) ? NULL : &properties.front();

    //         cl_uint num_entries;
    //         PYOPENCL_CALL_GUARDED(clCreateSubDevices,
    //             (m_device, props_ptr, 0, NULL, &num_entries));

    //         std::vector<cl_device_id> result;
    //         result.resize(num_entries);

    //         PYOPENCL_CALL_GUARDED(clCreateSubDevices,
    //             (m_device, props_ptr, num_entries, &result.front(), NULL));

    //         py::list py_result;
    //         BOOST_FOREACH(cl_device_id did, result)
    //           py_result.append(handle_from_new_ptr(
    //                 new pyopencl::device(did, /*retain*/true,
    //                   device::REF_CL_1_2)));
    //         return py_result;
    //       }
    // #endif

    // #if defined(cl_ext_device_fission) && defined(PYOPENCL_USE_DEVICE_FISSION)
    //       py::list create_sub_devices_ext(py::object py_properties)
    //       {
    //         std::vector<cl_device_partition_property_ext> properties;

    // #if PYOPENCL_CL_VERSION >= 0x1020
    //         cl_platform_id plat;
    //         PYOPENCL_CALL_GUARDED(clGetDeviceInfo, (m_device, CL_DEVICE_PLATFORM,
    //               sizeof(plat), &plat, NULL));
    // #endif

    //         PYOPENCL_GET_EXT_FUN(plat, clCreateSubDevicesEXT, create_sub_dev);

    //         COPY_PY_LIST(cl_device_partition_property_ext, properties);
    //         properties.push_back(CL_PROPERTIES_LIST_END_EXT);

    //         cl_device_partition_property_ext *props_ptr
    //           = properties.empty( ) ? NULL : &properties.front();

    //         cl_uint num_entries;
    //         PYOPENCL_CALL_GUARDED(create_sub_dev,
    //             (m_device, props_ptr, 0, NULL, &num_entries));

    //         std::vector<cl_device_id> result;
    //         result.resize(num_entries);

    //         PYOPENCL_CALL_GUARDED(create_sub_dev,
    //             (m_device, props_ptr, num_entries, &result.front(), NULL));

    //         py::list py_result;
    //         BOOST_FOREACH(cl_device_id did, result)
    //           py_result.append(handle_from_new_ptr(
    //                 new pyopencl::device(did, /*retain*/true,
    //                   device::REF_FISSION_EXT)));
    //         return py_result;
    //       }
    // #endif

    
  };

  // }}}

  

  // {{{ context
  class context : public _common // : public boost::noncopyable
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

    generic_info get_info(cl_context_info param_name) const
    {
      switch (param_name)
	{
	case CL_CONTEXT_REFERENCE_COUNT:
	  PYOPENCL_GET_INTEGRAL_INFO(Context, m_context, param_name, cl_uint);

	case CL_CONTEXT_DEVICES:
	  {
	    std::vector<cl_device_id> result;
	    PYOPENCL_GET_VEC_INFO(Context, m_context, param_name, result);
	    PYOPENCL_GET_OPAQUE_ARRAY_INFO(cl_device_id, device, DEVICE, result);
	    
	  }

	case CL_CONTEXT_PROPERTIES:
	  {
	    
	    std::vector<cl_context_properties> result;
	    PYOPENCL_GET_VEC_INFO(Context, m_context, param_name, result);
	    std::vector<generic_info> py_result;
	    for (size_t i = 0; i < result.size(); i+=2)
	      {
		cl_context_properties key = result[i];
		if(key == 0)
		  break;
		generic_info info;
		info.dontfree = 0;
		info.opaque_class = CLASS_NONE;
		switch (key)
		  {
		  case CL_CONTEXT_PLATFORM:
		    {
		      info.opaque_class = CLASS_PLATFORM;
		      info.type = "void *";
		      info.value = new platform(reinterpret_cast<cl_platform_id>(result[i+1]));
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
		    info.type = "intptr_t *";
		    info.value = (void*)result[i+1];
		    // we do not own this object
		    info.dontfree = 1;
		    break;

#endif
		  default:
		    throw error("Context.get_info", CL_INVALID_VALUE,
				"unknown context_property key encountered");
		  }

		py_result.push_back(info);
	      }
	    PYOPENCL_GET_ARRAY_INFO(generic_info, py_result);
	  }

#if PYOPENCL_CL_VERSION >= 0x1010
	case CL_CONTEXT_NUM_DEVICES:
	  PYOPENCL_GET_INTEGRAL_INFO(Context, m_context, param_name, cl_uint);
#endif

	default:
	  throw error("Context.get_info", CL_INVALID_VALUE);
	}
    }

    // }}}

  };

  // {{{ command_queue
  class command_queue : public _common
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
      PYOPENCL_PRINT_CALL_TRACE("clCreateCommandQueue");
      m_queue = clCreateCommandQueue(
				     ctx.data(), dev, props, &status_code);

      if (status_code != CL_SUCCESS) {
	throw pyopencl::error("CommandQueue", status_code);
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

    generic_info get_info(cl_command_queue_info param_name) const
    {
      switch (param_name)
	{
        case CL_QUEUE_CONTEXT:
          PYOPENCL_GET_OPAQUE_INFO(CommandQueue, m_queue, param_name,
				   cl_context, context, CONTEXT);
        case CL_QUEUE_DEVICE:
          PYOPENCL_GET_OPAQUE_INFO(CommandQueue, m_queue, param_name,
				   cl_device_id, device, DEVICE);
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

    std::auto_ptr<context> get_context() const
    {
      cl_context param_value;
      PYOPENCL_CALL_GUARDED(clGetCommandQueueInfo,
			    (m_queue, CL_QUEUE_CONTEXT, sizeof(param_value), &param_value, 0));
      return std::auto_ptr<context>(
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
    {
      // TODO
      // PYOPENCL_CALL_GUARDED_THREADED(clFinish, (m_queue));
    }
  };

  // }}}


  // {{{ event/synchronization
  class event : public _common // : boost::noncopyable
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

    generic_info get_info(cl_event_info param_name) const
    {
      switch (param_name)
	{
	case CL_EVENT_COMMAND_QUEUE:
	  PYOPENCL_GET_OPAQUE_INFO(Event, m_event, param_name,
				   cl_command_queue, command_queue, COMMAND_QUEUE);
	case CL_EVENT_COMMAND_TYPE:
	  PYOPENCL_GET_INTEGRAL_INFO(Event, m_event, param_name,
				     cl_command_type);
	case CL_EVENT_COMMAND_EXECUTION_STATUS:
	  PYOPENCL_GET_INTEGRAL_INFO(Event, m_event, param_name,
				     cl_int);
	case CL_EVENT_REFERENCE_COUNT:
	  PYOPENCL_GET_INTEGRAL_INFO(Event, m_event, param_name,
				     cl_uint);
#if PYOPENCL_CL_VERSION >= 0x1010
	case CL_EVENT_CONTEXT:
	  PYOPENCL_GET_OPAQUE_INFO(Event, m_event, param_name,
				   cl_context, context, CONTEXT);
#endif

	default:
	  throw error("Event.get_info", CL_INVALID_VALUE);
	}
    }

    generic_info get_profiling_info(cl_profiling_info param_name) const
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

    virtual void wait()
    {
      PYOPENCL_CALL_GUARDED_THREADED(clWaitForEvents, (1, &m_event));
    }
  };

  // }}}


  // {{{ memory_object

  //py::object create_mem_object_wrapper(cl_mem mem);

  class memory_object_holder : public _common
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

    generic_info get_info(cl_mem_info param_name) {
      switch (param_name){
      case CL_MEM_TYPE:
	PYOPENCL_GET_INTEGRAL_INFO(MemObject, data(), param_name,
				   cl_mem_object_type);
      case CL_MEM_FLAGS:
	PYOPENCL_GET_INTEGRAL_INFO(MemObject, data(), param_name,
				   cl_mem_flags);
      case CL_MEM_SIZE:
	PYOPENCL_GET_INTEGRAL_INFO(MemObject, data(), param_name,
				   size_t);
      case CL_MEM_HOST_PTR:
	throw pyopencl::error("MemoryObject.get_info", CL_INVALID_VALUE,
			      "Use MemoryObject.get_host_array to get host pointer.");
      case CL_MEM_MAP_COUNT:
	PYOPENCL_GET_INTEGRAL_INFO(MemObject, data(), param_name,
				   cl_uint);
      case CL_MEM_REFERENCE_COUNT:
	PYOPENCL_GET_INTEGRAL_INFO(MemObject, data(), param_name,
				   cl_uint);
      case CL_MEM_CONTEXT:
	PYOPENCL_GET_OPAQUE_INFO(MemObject, data(), param_name,
				 cl_context, context, CONTEXT);

#if PYOPENCL_CL_VERSION >= 0x1010
	//       case CL_MEM_ASSOCIATED_MEMOBJECT:
	// 	{
	// 	  cl_mem param_value;
	// 	  PYOPENCL_CALL_GUARDED(clGetMemObjectInfo, (data(), param_name, sizeof(param_value), &param_value, 0));
	// 	  if (param_value == 0)
	// 	    {
	// 	      // no associated memory object? no problem.
	// 	      return py::object();
	// 	    }

	// 	  return create_mem_object_wrapper(param_value);
	// 	}
      case CL_MEM_OFFSET:
	PYOPENCL_GET_INTEGRAL_INFO(MemObject, data(), param_name,
				   size_t);
#endif

      default:
	throw error("MemoryObjectHolder.get_info", CL_INVALID_VALUE);
      }
    }
  };


  class memory_object : /*boost::noncopyable, */ public memory_object_holder
  {
  private:
    bool m_valid;
    cl_mem m_mem;
    void *m_hostbuf;

  public:
    memory_object(cl_mem mem, bool retain, void *hostbuf=0)
      : m_valid(true), m_mem(mem)
    {
      if (retain)
	PYOPENCL_CALL_GUARDED(clRetainMemObject, (mem));

      if (hostbuf)
	m_hostbuf = hostbuf;
    }

    memory_object(memory_object const &src)
      : m_valid(true), m_mem(src.m_mem), m_hostbuf(src.m_hostbuf)
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

    void *hostbuf()
    { return m_hostbuf; }

    const cl_mem data() const
    { return m_mem; }

  };

  // #if PYOPENCL_CL_VERSION >= 0x1020
  //   inline
  //   event *enqueue_migrate_mem_objects(
  //       command_queue &cq,
  //       py::object py_mem_objects,
  //       cl_mem_migration_flags flags,
  //       py::object py_wait_for)
  //   {
  //     PYOPENCL_PARSE_WAIT_FOR;

  //     std::vector<cl_mem> mem_objects;
  //     PYTHON_FOREACH(mo, py_mem_objects)
  //       mem_objects.push_back(py::extract<memory_object &>(mo)().data());

  //     cl_event evt;
  //     PYOPENCL_RETRY_IF_MEM_ERROR(
  //       PYOPENCL_CALL_GUARDED(clEnqueueMigrateMemObjects, (
  //             cq.data(),
  //             mem_objects.size(), mem_objects.empty( ) ? NULL : &mem_objects.front(),
  //             flags,
  //             PYOPENCL_WAITLIST_ARGS, &evt
  //             ));
  //       );
  //     PYOPENCL_RETURN_NEW_EVENT(evt);
  //   }
  // #endif

  // #ifdef cl_ext_migrate_memobject
  //   inline
  //   event *enqueue_migrate_mem_object_ext(
  //       command_queue &cq,
  //       py::object py_mem_objects,
  //       cl_mem_migration_flags_ext flags,
  //       py::object py_wait_for)
  //   {
  //     PYOPENCL_PARSE_WAIT_FOR;

  // #if PYOPENCL_CL_VERSION >= 0x1020
  //     // {{{ get platform
  //     cl_device_id dev;
  //     PYOPENCL_CALL_GUARDED(clGetCommandQueueInfo, (cq.data(), CL_QUEUE_DEVICE,
  //           sizeof(dev), &dev, NULL));
  //     cl_platform_id plat;
  //     PYOPENCL_CALL_GUARDED(clGetDeviceInfo, (cq.data(), CL_DEVICE_PLATFORM,
  //           sizeof(plat), &plat, NULL));
  //     // }}}
  // #endif

  //     PYOPENCL_GET_EXT_FUN(plat,
  //         clEnqueueMigrateMemObjectEXT, enqueue_migrate_fn);

  //     std::vector<cl_mem> mem_objects;
  //     PYTHON_FOREACH(mo, py_mem_objects)
  //       mem_objects.push_back(py::extract<memory_object &>(mo)().data());

  //     cl_event evt;
  //     PYOPENCL_RETRY_IF_MEM_ERROR(
  //       PYOPENCL_CALL_GUARDED(enqueue_migrate_fn, (
  //             cq.data(),
  //             mem_objects.size(), mem_objects.empty( ) ? NULL : &mem_objects.front(),
  //             flags,
  //             PYOPENCL_WAITLIST_ARGS, &evt
  //             ));
  //       );
  //     PYOPENCL_RETURN_NEW_EVENT(evt);
  //   }
  // #endif

  // }}}


  // {{{ image

  class image : public memory_object
  {
  public:
    image(cl_mem mem, bool retain, void *hostbuf=0)
      : memory_object(mem, retain, hostbuf)
    { }

    generic_info get_image_info(cl_image_info param_name) const
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
#if PYOPENCL_CL_VERSION >= 0x1020
	case CL_IMAGE_ARRAY_SIZE:
#endif
	  PYOPENCL_GET_INTEGRAL_INFO(Image, data(), param_name, size_t);

// #if PYOPENCL_CL_VERSION >= 0x1020
// 	case CL_IMAGE_BUFFER:
// 	  {
// 	    cl_mem param_value;
// 	    PYOPENCL_CALL_GUARDED(clGetImageInfo, (data(), param_name, sizeof(param_value), &param_value, 0));
// 	    if (param_value == 0)
//               {
//                 // no associated memory object? no problem.
//                 return py::object();
//               }

// 	    return create_mem_object_wrapper(param_value);
// 	  }

// 	case CL_IMAGE_NUM_MIP_LEVELS:
// 	case CL_IMAGE_NUM_SAMPLES:
// 	  PYOPENCL_GET_INTEGRAL_INFO(Image, data(), param_name, cl_uint);
// #endif

	default:
	  throw error("MemoryObject.get_image_info", CL_INVALID_VALUE);
        }
    }
  };




//   // {{{ image formats

//   inline
//   cl_image_format *make_image_format(cl_channel_order ord, cl_channel_type tp)
//   {
//     std::auto_ptr<cl_image_format> result(new cl_image_format);
//     result->image_channel_order = ord;
//     result->image_channel_data_type = tp;
//     return result.release();
//   }

//   inline
//   py::list get_supported_image_formats(
//       context const &ctx,
//       cl_mem_flags flags,
//       cl_mem_object_type image_type)
//   {
//     cl_uint num_image_formats;
//     PYOPENCL_CALL_GUARDED(clGetSupportedImageFormats, (
//           ctx.data(), flags, image_type,
//           0, NULL, &num_image_formats));

//     std::vector<cl_image_format> formats(num_image_formats);
//     PYOPENCL_CALL_GUARDED(clGetSupportedImageFormats, (
//           ctx.data(), flags, image_type,
//           formats.size(), formats.empty( ) ? NULL : &formats.front(), NULL));

//     PYOPENCL_RETURN_VECTOR(cl_image_format, formats);
//   }

//   inline
//   cl_uint get_image_format_channel_count(cl_image_format const &fmt)
//   {
//     switch (fmt.image_channel_order)
//     {
//       case CL_R: return 1;
//       case CL_A: return 1;
//       case CL_RG: return 2;
//       case CL_RA: return 2;
//       case CL_RGB: return 3;
//       case CL_RGBA: return 4;
//       case CL_BGRA: return 4;
//       case CL_INTENSITY: return 1;
//       case CL_LUMINANCE: return 1;
//       default:
//         throw pyopencl::error("ImageFormat.channel_dtype_size",
//             CL_INVALID_VALUE,
//             "unrecognized channel order");
//     }
//   }

//   inline
//   cl_uint get_image_format_channel_dtype_size(cl_image_format const &fmt)
//   {
//     switch (fmt.image_channel_data_type)
//     {
//       case CL_SNORM_INT8: return 1;
//       case CL_SNORM_INT16: return 2;
//       case CL_UNORM_INT8: return 1;
//       case CL_UNORM_INT16: return 2;
//       case CL_UNORM_SHORT_565: return 2;
//       case CL_UNORM_SHORT_555: return 2;
//       case CL_UNORM_INT_101010: return 4;
//       case CL_SIGNED_INT8: return 1;
//       case CL_SIGNED_INT16: return 2;
//       case CL_SIGNED_INT32: return 4;
//       case CL_UNSIGNED_INT8: return 1;
//       case CL_UNSIGNED_INT16: return 2;
//       case CL_UNSIGNED_INT32: return 4;
//       case CL_HALF_FLOAT: return 2;
//       case CL_FLOAT: return 4;
//       default:
//         throw pyopencl::error("ImageFormat.channel_dtype_size",
//             CL_INVALID_VALUE,
//             "unrecognized channel data type");
//     }
//   }

//   inline
//   cl_uint get_image_format_item_size(cl_image_format const &fmt)
//   {
//     return get_image_format_channel_count(fmt)
//       * get_image_format_channel_dtype_size(fmt);
//   }

//   // }}}

//   // {{{ image creation

//   inline
//   image *create_image(
//       context const &ctx,
//       cl_mem_flags flags,
//       cl_image_format const &fmt,
//       py::object shape,
//       py::object pitches,
//       py::object buffer)
//   {
//     if (shape.ptr() == Py_None)
//       throw pyopencl::error("Image", CL_INVALID_VALUE,
//           "'shape' must be given");

//     void *buf = 0;
//     PYOPENCL_BUFFER_SIZE_T len;
//     py::object *retained_buf_obj = 0;

//     if (buffer.ptr() != Py_None)
//     {
//       if (flags & CL_MEM_USE_HOST_PTR)
//       {
//         if (PyObject_AsWriteBuffer(buffer.ptr(), &buf, &len))
//           throw py::error_already_set();
//       }
//       else
//       {
//         if (PyObject_AsReadBuffer(
//               buffer.ptr(), const_cast<const void **>(&buf), &len))
//           throw py::error_already_set();
//       }

//       if (flags & CL_MEM_USE_HOST_PTR)
//         retained_buf_obj = &buffer;
//     }

//     unsigned dims = py::len(shape);
//     cl_int status_code;
//     cl_mem mem;
//     if (dims == 2)
//     {
//       size_t width = py::extract<size_t>(shape[0]);
//       size_t height = py::extract<size_t>(shape[1]);

//       size_t pitch = 0;
//       if (pitches.ptr() != Py_None)
//       {
//         if (py::len(pitches) != 1)
//           throw pyopencl::error("Image", CL_INVALID_VALUE,
//               "invalid length of pitch tuple");
//         pitch = py::extract<size_t>(pitches[0]);
//       }

//       // check buffer size
//       cl_int itemsize = get_image_format_item_size(fmt);
//       if (buf && std::max(pitch, width*itemsize)*height > cl_uint(len))
//           throw pyopencl::error("Image", CL_INVALID_VALUE,
//               "buffer too small");

//       PYOPENCL_PRINT_CALL_TRACE("clCreateImage2D");
//       PYOPENCL_RETRY_IF_MEM_ERROR(
//           {
//             mem = clCreateImage2D(ctx.data(), flags, &fmt,
//                 width, height, pitch, buf, &status_code);
//             if (status_code != CL_SUCCESS)
//               throw pyopencl::error("clCreateImage2D", status_code);
//           } );

//     }
//     else if (dims == 3)
//     {
//       size_t width = py::extract<size_t>(shape[0]);
//       size_t height = py::extract<size_t>(shape[1]);
//       size_t depth = py::extract<size_t>(shape[2]);

//       size_t pitch_x = 0;
//       size_t pitch_y = 0;

//       if (pitches.ptr() != Py_None)
//       {
//         if (py::len(pitches) != 2)
//           throw pyopencl::error("Image", CL_INVALID_VALUE,
//               "invalid length of pitch tuple");

//         pitch_x = py::extract<size_t>(pitches[0]);
//         pitch_y = py::extract<size_t>(pitches[1]);
//       }

//       // check buffer size
//       cl_int itemsize = get_image_format_item_size(fmt);
//       if (buf &&
//           std::max(std::max(pitch_x, width*itemsize)*height, pitch_y)
//           * depth > cl_uint(len))
//         throw pyopencl::error("Image", CL_INVALID_VALUE,
//             "buffer too small");

//       PYOPENCL_PRINT_CALL_TRACE("clCreateImage3D");
//       PYOPENCL_RETRY_IF_MEM_ERROR(
//           {
//             mem = clCreateImage3D(ctx.data(), flags, &fmt,
//               width, height, depth, pitch_x, pitch_y, buf, &status_code);
//             if (status_code != CL_SUCCESS)
//               throw pyopencl::error("clCreateImage3D", status_code);
//           } );
//     }
//     else
//       throw pyopencl::error("Image", CL_INVALID_VALUE,
//           "invalid dimension");

//     try
//     {
//       return new image(mem, false, retained_buf_obj);
//     }
//     catch (...)
//     {
//       PYOPENCL_CALL_GUARDED(clReleaseMemObject, (mem));
//       throw;
//     }
//   }

// #if PYOPENCL_CL_VERSION >= 0x1020

//   inline
//   image *create_image_from_desc(
//       context const &ctx,
//       cl_mem_flags flags,
//       cl_image_format const &fmt,
//       cl_image_desc &desc,
//       py::object buffer)
//   {
//     if (buffer.ptr() != Py_None &&
//         !(flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR)))
//       PyErr_Warn(PyExc_UserWarning, "'hostbuf' was passed, "
//           "but no memory flags to make use of it.");

//     void *buf = 0;
//     PYOPENCL_BUFFER_SIZE_T len;
//     py::object *retained_buf_obj = 0;

//     if (buffer.ptr() != Py_None)
//     {
//       if (flags & CL_MEM_USE_HOST_PTR)
//       {
//         if (PyObject_AsWriteBuffer(buffer.ptr(), &buf, &len))
//           throw py::error_already_set();
//       }
//       else
//       {
//         if (PyObject_AsReadBuffer(
//               buffer.ptr(), const_cast<const void **>(&buf), &len))
//           throw py::error_already_set();
//       }

//       if (flags & CL_MEM_USE_HOST_PTR)
//         retained_buf_obj = &buffer;
//     }

//     PYOPENCL_PRINT_CALL_TRACE("clCreateImage");
//     cl_int status_code;
//     cl_mem mem = clCreateImage(ctx.data(), flags, &fmt, &desc, buf, &status_code);
//     if (status_code != CL_SUCCESS)
//       throw pyopencl::error("clCreateImage", status_code);

//     try
//     {
//       return new image(mem, false, retained_buf_obj);
//     }
//     catch (...)
//     {
//       PYOPENCL_CALL_GUARDED(clReleaseMemObject, (mem));
//       throw;
//     }
//   }

// #endif

//   // }}}

//   // {{{ image transfers

//   inline
//   event *enqueue_read_image(
//       command_queue &cq,
//       image &img,
//       py::object py_origin, py::object py_region,
//       py::object buffer,
//       size_t row_pitch, size_t slice_pitch,
//       py::object py_wait_for,
//       bool is_blocking)
//   {
//     PYOPENCL_PARSE_WAIT_FOR;
//     COPY_PY_COORD_TRIPLE(origin);
//     COPY_PY_REGION_TRIPLE(region);

//     void *buf;
//     PYOPENCL_BUFFER_SIZE_T len;

//     if (PyObject_AsWriteBuffer(buffer.ptr(), &buf, &len))
//       throw py::error_already_set();

//     cl_event evt;

//     PYOPENCL_RETRY_IF_MEM_ERROR(
//       PYOPENCL_CALL_GUARDED(clEnqueueReadImage, (
//             cq.data(),
//             img.data(),
//             PYOPENCL_CAST_BOOL(is_blocking),
//             origin, region, row_pitch, slice_pitch, buf,
//             PYOPENCL_WAITLIST_ARGS, &evt
//             ));
//       );
//     PYOPENCL_RETURN_NEW_NANNY_EVENT(evt, buffer);
//   }




//   inline
//   event *enqueue_write_image(
//       command_queue &cq,
//       image &img,
//       py::object py_origin, py::object py_region,
//       py::object buffer,
//       size_t row_pitch, size_t slice_pitch,
//       py::object py_wait_for,
//       bool is_blocking)
//   {
//     PYOPENCL_PARSE_WAIT_FOR;
//     COPY_PY_COORD_TRIPLE(origin);
//     COPY_PY_REGION_TRIPLE(region);

//     const void *buf;
//     PYOPENCL_BUFFER_SIZE_T len;

//     if (PyObject_AsReadBuffer(buffer.ptr(), &buf, &len))
//       throw py::error_already_set();

//     cl_event evt;
//     PYOPENCL_RETRY_IF_MEM_ERROR(
//       PYOPENCL_CALL_GUARDED(clEnqueueWriteImage, (
//             cq.data(),
//             img.data(),
//             PYOPENCL_CAST_BOOL(is_blocking),
//             origin, region, row_pitch, slice_pitch, buf,
//             PYOPENCL_WAITLIST_ARGS, &evt
//             ));
//       );
//     PYOPENCL_RETURN_NEW_NANNY_EVENT(evt, buffer);
//   }




//   inline
//   event *enqueue_copy_image(
//       command_queue &cq,
//       memory_object_holder &src,
//       memory_object_holder &dest,
//       py::object py_src_origin,
//       py::object py_dest_origin,
//       py::object py_region,
//       py::object py_wait_for
//       )
//   {
//     PYOPENCL_PARSE_WAIT_FOR;
//     COPY_PY_COORD_TRIPLE(src_origin);
//     COPY_PY_COORD_TRIPLE(dest_origin);
//     COPY_PY_REGION_TRIPLE(region);

//     cl_event evt;
//     PYOPENCL_RETRY_IF_MEM_ERROR(
//       PYOPENCL_CALL_GUARDED(clEnqueueCopyImage, (
//             cq.data(), src.data(), dest.data(),
//             src_origin, dest_origin, region,
//             PYOPENCL_WAITLIST_ARGS, &evt
//             ));
//       );
//     PYOPENCL_RETURN_NEW_EVENT(evt);
//   }




//   inline
//   event *enqueue_copy_image_to_buffer(
//       command_queue &cq,
//       memory_object_holder &src,
//       memory_object_holder &dest,
//       py::object py_origin,
//       py::object py_region,
//       size_t offset,
//       py::object py_wait_for
//       )
//   {
//     PYOPENCL_PARSE_WAIT_FOR;
//     COPY_PY_COORD_TRIPLE(origin);
//     COPY_PY_REGION_TRIPLE(region);

//     cl_event evt;
//     PYOPENCL_RETRY_IF_MEM_ERROR(
//       PYOPENCL_CALL_GUARDED(clEnqueueCopyImageToBuffer, (
//             cq.data(), src.data(), dest.data(),
//             origin, region, offset,
//             PYOPENCL_WAITLIST_ARGS, &evt
//             ));
//       );
//     PYOPENCL_RETURN_NEW_EVENT(evt);
//   }




//   inline
//   event *enqueue_copy_buffer_to_image(
//       command_queue &cq,
//       memory_object_holder &src,
//       memory_object_holder &dest,
//       size_t offset,
//       py::object py_origin,
//       py::object py_region,
//       py::object py_wait_for
//       )
//   {
//     PYOPENCL_PARSE_WAIT_FOR;
//     COPY_PY_COORD_TRIPLE(origin);
//     COPY_PY_REGION_TRIPLE(region);

//     cl_event evt;
//     PYOPENCL_RETRY_IF_MEM_ERROR(
//       PYOPENCL_CALL_GUARDED(clEnqueueCopyBufferToImage, (
//             cq.data(), src.data(), dest.data(),
//             offset, origin, region,
//             PYOPENCL_WAITLIST_ARGS, &evt
//             ));
//       );
//     PYOPENCL_RETURN_NEW_EVENT(evt);
//   }

//   // }}}

// #if PYOPENCL_CL_VERSION >= 0x1020
//   inline
//   event *enqueue_fill_image(
//       command_queue &cq,
//       memory_object_holder &mem,
//       py::object color,
//       py::object py_origin, py::object py_region,
//       py::object py_wait_for
//       )
//   {
//     PYOPENCL_PARSE_WAIT_FOR;

//     COPY_PY_COORD_TRIPLE(origin);
//     COPY_PY_REGION_TRIPLE(region);

//     const void *color_buf;
//     PYOPENCL_BUFFER_SIZE_T color_len;

//     if (PyObject_AsReadBuffer(color.ptr(), &color_buf, &color_len))
//       throw py::error_already_set();

//     cl_event evt;
//     PYOPENCL_RETRY_IF_MEM_ERROR(
//       PYOPENCL_CALL_GUARDED(clEnqueueFillImage, (
//             cq.data(),
//             mem.data(),
//             color_buf, origin, region,
//             PYOPENCL_WAITLIST_ARGS, &evt
//             ));
//       );
//     PYOPENCL_RETURN_NEW_EVENT(evt);
//   }
// #endif

  // }}}

  
  
  // {{{ gl interop


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
    gl_buffer(cl_mem mem, bool retain, void *hostbuf=0)
      : memory_object(mem, retain, hostbuf)
    { }
  };




  class gl_renderbuffer : public memory_object
  {
  public:
    gl_renderbuffer(cl_mem mem, bool retain, void *hostbuf=0)
      : memory_object(mem, retain, hostbuf)
    { }
  };




  class gl_texture : public image
  {
  public:
    gl_texture(cl_mem mem, bool retain, void *hostbuf=0)
      : image(mem, retain, hostbuf)
    { }

    generic_info get_gl_texture_info(cl_gl_texture_info param_name)
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
  inline								\
  TYPE *NAME ARGS							\
  {									\
    cl_int status_code;							\
    PYOPENCL_PRINT_CALL_TRACE(#CL_NAME);				\
    cl_mem mem = CL_NAME CL_ARGS;					\
									\
    if (status_code != CL_SUCCESS)					\
      throw pyopencl::error(#CL_NAME, status_code);			\
									\
    try									\
      {									\
	return new TYPE(mem, false);					\
      }									\
    catch (...)								\
      {									\
	PYOPENCL_CALL_GUARDED(clReleaseMemObject, (mem));		\
	throw;								\
      }									\
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





  // inline
  // py::tuple get_gl_object_info(memory_object_holder const &mem)
  // {
  //   cl_gl_object_type otype;
  //   GLuint gl_name;
  //   PYOPENCL_CALL_GUARDED(clGetGLObjectInfo, (mem.data(), &otype, &gl_name));
  //   return py::make_tuple(otype, gl_name);
  // }
  /*
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
    PYTHON_FOREACH(mo, py_mem_objects) \
      mem_objects.push_back(py::extract<memory_object_holder &>(mo)().data()); \
    \
    cl_event evt; \
    PYOPENCL_CALL_GUARDED(clEnqueue##What##GLObjects, ( \
          cq.data(), \
          mem_objects.size(), mem_objects.empty( ) ? NULL : &mem_objects.front(), \
          PYOPENCL_WAITLIST_ARGS, &evt \
          )); \
    \
    PYOPENCL_RETURN_NEW_EVENT(evt); \
  }

  WRAP_GL_ENQUEUE(acquire, Acquire);
  WRAP_GL_ENQUEUE(release, Release);
#endif
  */



// #if defined(cl_khr_gl_sharing) && (cl_khr_gl_sharing >= 1)
//   inline
//   py::object get_gl_context_info_khr(
//       py::object py_properties,
//       cl_gl_context_info param_name,
//       py::object py_platform
//       )
//   {
//     std::vector<cl_context_properties> props
//       = parse_context_properties(py_properties);

//     typedef CL_API_ENTRY cl_int (CL_API_CALL
//       *func_ptr_type)(const cl_context_properties * /* properties */,
//           cl_gl_context_info            /* param_name */,
//           size_t                        /* param_value_size */,
//           void *                        /* param_value */,
//           size_t *                      /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

//     func_ptr_type func_ptr;

// #if PYOPENCL_CL_VERSION >= 0x1020
//     if (py_platform.ptr() != Py_None)
//     {
//       platform &plat = py::extract<platform &>(py_platform);

//       func_ptr = (func_ptr_type) clGetExtensionFunctionAddressForPlatform(
//             plat.data(), "clGetGLContextInfoKHR");
//     }
//     else
//     {
//       PYOPENCL_DEPRECATED("get_gl_context_info_khr with platform=None", "2013.1", );

//       func_ptr = (func_ptr_type) clGetExtensionFunctionAddress(
//             "clGetGLContextInfoKHR");
//     }
// #else
//     func_ptr = (func_ptr_type) clGetExtensionFunctionAddress(
//           "clGetGLContextInfoKHR");
// #endif


//     if (!func_ptr)
//       throw error("Context.get_info", CL_INVALID_PLATFORM,
//           "clGetGLContextInfoKHR extension function not present");

//     cl_context_properties *props_ptr
//       = props.empty( ) ? NULL : &props.front();

//     switch (param_name)
//     {
//       case CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR:
//         {
//           cl_device_id param_value;
//           PYOPENCL_CALL_GUARDED(func_ptr,
//               (props_ptr, param_name, sizeof(param_value), &param_value, 0));
//           return py::object(handle_from_new_ptr( new device(param_value, /*retain*/ true)));
  //         }

  //       case CL_DEVICES_FOR_GL_CONTEXT_KHR:
  //         {
  //           size_t size;
  //           PYOPENCL_CALL_GUARDED(func_ptr,
  //               (props_ptr, param_name, 0, 0, &size));

  //           std::vector<cl_device_id> devices;

  //           devices.resize(size / sizeof(devices.front()));

  //           PYOPENCL_CALL_GUARDED(func_ptr,
  //               (props_ptr, param_name, size,
  //                devices.empty( ) ? NULL : &devices.front(), &size));

  //           py::list result;
  //           BOOST_FOREACH(cl_device_id did, devices)
  //             result.append(handle_from_new_ptr(
  //                   new device(did)));

  //           return result;
  //         }

  //       default:
  //         throw error("get_gl_context_info_khr", CL_INVALID_VALUE);
  //     }
  //   }

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


  
  inline cl_mem create_buffer_gc(cl_context ctx,
				 cl_mem_flags flags,
				 size_t size,
				 void *host_ptr)
  {
    // TODO
    //PYOPENCL_RETRY_RETURN_IF_MEM_ERROR(
    return create_buffer(ctx, flags, size, host_ptr);
    // );
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
    // TODO
    //PYOPENCL_RETRY_RETURN_IF_MEM_ERROR(
    return create_sub_buffer(buffer, flags, bct, buffer_create_info);
    //);
  }
#endif

  class buffer : public memory_object
  {
  public:
    buffer(cl_mem mem, bool retain, void *hostbuf=0)
      : memory_object(mem, retain, hostbuf)
    { }

#if PYOPENCL_CL_VERSION >= 0x1010
    buffer *get_sub_region(size_t origin, size_t size, cl_mem_flags flags) const
    {
      cl_buffer_region region = {origin, size};

      cl_mem mem = create_sub_buffer_gc(data(), flags, CL_BUFFER_CREATE_TYPE_REGION, &region);

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

    //       buffer *getitem(py::slice slc) const
    //       {
    //         PYOPENCL_BUFFER_SIZE_T start, end, stride, length;

    //         size_t my_length;
    //         PYOPENCL_CALL_GUARDED(clGetMemObjectInfo,
    //             (data(), CL_MEM_SIZE, sizeof(my_length), &my_length, 0));

    // #if PY_VERSION_HEX >= 0x03020000
    //         if (PySlice_GetIndicesEx(slc.ptr(),
    // #else
    //         if (PySlice_GetIndicesEx(reinterpret_cast<PySliceObject *>(slc.ptr()),
    // #endif
    //               my_length, &start, &end, &stride, &length) != 0)
    //           throw py::error_already_set();

    //         if (stride != 1)
    //           throw pyopencl::error("Buffer.__getitem__", CL_INVALID_VALUE,
    //               "Buffer slice must have stride 1");

    //         cl_mem_flags my_flags;
    //         PYOPENCL_CALL_GUARDED(clGetMemObjectInfo,
    //             (data(), CL_MEM_FLAGS, sizeof(my_flags), &my_flags, 0));

    //         return get_sub_region(start, end, my_flags);
    //       }
#endif
  };

  // {{{ buffer creation

  inline
  buffer *create_buffer_py(
			   context &ctx,
			   cl_mem_flags flags,
			   size_t size,
			   void *py_hostbuf
			   )
  {
    
    void *buf = py_hostbuf;
    void *retained_buf_obj = 0;
    if (py_hostbuf != NULL)
      {
	if (flags & CL_MEM_USE_HOST_PTR)
	  retained_buf_obj = py_hostbuf;

      }

    cl_mem mem = create_buffer_gc(ctx.data(), flags, size, buf);

    try
      {
	return new buffer(mem, false, retained_buf_obj);
      }
    catch (...)
      {
	PYOPENCL_CALL_GUARDED(clReleaseMemObject, (mem));
	throw;
      }
  }

  
  // }}}
  // }}}

  
  // {{{ program

  class program : public _common //: boost::noncopyable
  {   

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

    std::vector<cl_device_id> get_info__devices()
    {
      std::vector<cl_device_id> result;
      PYOPENCL_GET_VEC_INFO(Program, m_program, CL_PROGRAM_DEVICES, result);
      return result;
    }

    generic_info get_info(cl_program_info param_name) const
    {
      switch (param_name)
        {
	case CL_PROGRAM_REFERENCE_COUNT:
	  PYOPENCL_GET_INTEGRAL_INFO(Program, m_program, param_name,
				     cl_uint);
	case CL_PROGRAM_CONTEXT:
	  PYOPENCL_GET_OPAQUE_INFO(Program, m_program, param_name,
				   cl_context, context, CONTEXT);
	case CL_PROGRAM_NUM_DEVICES:
	  PYOPENCL_GET_INTEGRAL_INFO(Program, m_program, param_name,
				     cl_uint);
	case CL_PROGRAM_DEVICES:
	  {
	    std::vector<cl_device_id> result;
	    PYOPENCL_GET_VEC_INFO(Program, m_program, param_name, result);
	    PYOPENCL_GET_OPAQUE_ARRAY_INFO(cl_device_id, device, DEVICE, result);
	  }
	case CL_PROGRAM_SOURCE:
	  PYOPENCL_GET_STR_INFO(Program, m_program, param_name);
	case CL_PROGRAM_BINARY_SIZES:
	  {
	    std::vector<size_t> result;
	    PYOPENCL_GET_VEC_INFO(Program, m_program, param_name, result);
	    PYOPENCL_GET_ARRAY_INFO(size_t, result);
	  }
	case CL_PROGRAM_BINARIES:
	  {
	    std::vector<size_t> sizes;
	    PYOPENCL_GET_VEC_INFO(Program, m_program, CL_PROGRAM_BINARY_SIZES, sizes);
	    std::vector<char *> result_ptrs(sizes.size());
	    for (unsigned i = 0; i < sizes.size(); ++i) {
	      result_ptrs[i] = new char[sizes[i]];
	    }
	    std::vector<generic_info> gis(sizes.size());
	    for(unsigned i = 0; i < sizes.size(); ++i) {
	      gis[i].opaque_class = CLASS_NONE;
	      gis[i].type =  _copy_str(std::string("char[") + tostring(sizes[i]) + "]");
	      gis[i].value = result_ptrs[i];
	    }
	    PYOPENCL_CALL_GUARDED(clGetProgramInfo,
	    			  (m_program, CL_PROGRAM_BINARIES, sizes.size()*sizeof(char *),
	    			   &result_ptrs.front(), 0));
	    PYOPENCL_GET_ARRAY_INFO(generic_info, gis);
	  }
	  
#if PYOPENCL_CL_VERSION >= 0x1020
	case CL_PROGRAM_NUM_KERNELS:
	  PYOPENCL_GET_INTEGRAL_INFO(Program, m_program, param_name,
				     size_t);
	case CL_PROGRAM_KERNEL_NAMES:
	  PYOPENCL_GET_STR_INFO(Program, m_program, param_name);
#endif

	default:
	  throw error("Program.get_info", CL_INVALID_VALUE);
        }
    }

    generic_info get_build_info(device const &dev,
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
#if PYOPENCL_CL_VERSION >= 0x1020
	case CL_PROGRAM_BINARY_TYPE:
	  PYOPENCL_GET_INTEGRAL_INFO(ProgramBuild,
				     PYOPENCL_FIRST_ARG, param_name,
				     cl_program_binary_type);
#endif
#undef PYOPENCL_FIRST_ARG

	default:
	  throw error("Program.get_build_info", CL_INVALID_VALUE);
        }
    }

    void build(char *options, cl_uint num_devices, void **ptr_devices)
    { 
      // todo: this function should get a list of device instances, not raw pointers
      // pointers are for the cffi interface and should not be here
      std::vector<cl_device_id> devices(num_devices);
      for(cl_uint i = 0; i < num_devices; ++i) {
	devices[i] = static_cast<device*>(ptr_devices[i])->data();
      }
      PYOPENCL_CALL_GUARDED_THREADED(clBuildProgram,
				     (m_program, num_devices, devices.empty( ) ? NULL : &devices.front(),
				      options, 0 ,0));
    }

    // #if PYOPENCL_CL_VERSION >= 0x1020
    //       void compile(std::string options, py::object py_devices,
    //           py::object py_headers)
    //       {
    //         PYOPENCL_PARSE_PY_DEVICES;

    //         // {{{ pick apart py_headers
    //         // py_headers is a list of tuples *(name, program)*

    //         std::vector<std::string> header_names;
    //         std::vector<cl_program> programs;
    //         PYTHON_FOREACH(name_hdr_tup, py_headers)
    //         {
    //           if (py::len(name_hdr_tup) != 2)
    //             throw error("Program.compile", CL_INVALID_VALUE,
    //                 "epxected (name, header) tuple in headers list");
    //           std::string name = py::extract<std::string const &>(name_hdr_tup[0]);
    //           program &prg = py::extract<program &>(name_hdr_tup[1]);

    //           header_names.push_back(name);
    //           programs.push_back(prg.data());
    //         }

    //         std::vector<const char *> header_name_ptrs;
    //         BOOST_FOREACH(std::string const &name, header_names)
    //           header_name_ptrs.push_back(name.c_str());

    //         // }}}

    //         PYOPENCL_CALL_GUARDED_THREADED(clCompileProgram,
    //             (m_program, num_devices, devices,
    //              options.c_str(), header_names.size(),
    //              programs.empty() ? NULL : &programs.front(),
    //              header_name_ptrs.empty() ? NULL : &header_name_ptrs.front(),
    //              0, 0));
    //       }
    // #endif
  };

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




  class kernel : public _common // : boost::noncopyable
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

    // void set_arg_sampler(cl_uint arg_index, sampler const &smp)
    // {
    //   cl_sampler s = smp.data();
    //   PYOPENCL_CALL_GUARDED(clSetKernelArg,
    //       (m_kernel, arg_index, sizeof(cl_sampler), &s));
    // }

    // void set_arg_buf(cl_uint arg_index, py::object py_buffer)
    // {
    //   const void *buf;
    //   PYOPENCL_BUFFER_SIZE_T len;

    //   if (PyObject_AsReadBuffer(py_buffer.ptr(), &buf, &len))
    //   {
    //     PyErr_Clear();
    //     throw error("Kernel.set_arg", CL_INVALID_VALUE,
    //         "invalid kernel argument");
    //   }

    //   PYOPENCL_CALL_GUARDED(clSetKernelArg,
    //       (m_kernel, arg_index, len, buf));
    // }

    // void set_arg(cl_uint arg_index, py::object arg)
    // {
    //   if (arg.ptr() == Py_None)
    //   {
    //     set_arg_null(arg_index);
    //     return;
    //   }

    //   py::extract<memory_object_holder &> ex_mo(arg);
    //   if (ex_mo.check())
    //   {
    //     set_arg_mem(arg_index, ex_mo());
    //     return;
    //   }

    //   py::extract<local_memory const &> ex_loc(arg);
    //   if (ex_loc.check())
    //   {
    //     set_arg_local(arg_index, ex_loc());
    //     return;
    //   }

    //   py::extract<sampler const &> ex_smp(arg);
    //   if (ex_smp.check())
    //   {
    //     set_arg_sampler(arg_index, ex_smp());
    //     return;
    //   }

    //   set_arg_buf(arg_index, arg);
    // }
    
    generic_info get_info(cl_kernel_info param_name) const
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
				   cl_context, context, CONTEXT);
	case CL_KERNEL_PROGRAM:
	  PYOPENCL_GET_OPAQUE_INFO(Kernel, m_kernel, param_name,
				   cl_program, program, PROGRAM);
#if PYOPENCL_CL_VERSION >= 0x1020
	case CL_KERNEL_ATTRIBUTES:
	  PYOPENCL_GET_STR_INFO(Kernel, m_kernel, param_name);
#endif
	default:
	  throw error("Kernel.get_info", CL_INVALID_VALUE);
	}
    }

    generic_info get_work_group_info(cl_kernel_work_group_info param_name, device const &dev) const
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

	    PYOPENCL_GET_ARRAY_INFO(size_t, result);
	  }
	case CL_KERNEL_LOCAL_MEM_SIZE:
#if PYOPENCL_CL_VERSION >= 0x1010
	case CL_KERNEL_PRIVATE_MEM_SIZE:
#endif
	  PYOPENCL_GET_INTEGRAL_INFO(KernelWorkGroup,
				     PYOPENCL_FIRST_ARG, param_name,
				     cl_ulong);

#if PYOPENCL_CL_VERSION >= 0x1010
	case CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE:
	  PYOPENCL_GET_INTEGRAL_INFO(KernelWorkGroup,
				     PYOPENCL_FIRST_ARG, param_name,
				     size_t);
#endif
	default:
	  throw error("Kernel.get_work_group_info", CL_INVALID_VALUE);
#undef PYOPENCL_FIRST_ARG
	}
    }

    // #if PYOPENCL_CL_VERSION >= 0x1020
    //       py::object get_arg_info(
    //           cl_uint arg_index,
    //           cl_kernel_arg_info param_name
    //           ) const
    //       {
    //         switch (param_name)
    //         {
    // #define PYOPENCL_FIRST_ARG m_kernel, arg_index // hackety hack
    //           case CL_KERNEL_ARG_ADDRESS_QUALIFIER:
    //             PYOPENCL_GET_INTEGRAL_INFO(KernelArg,
    //                 PYOPENCL_FIRST_ARG, param_name,
    //                 cl_kernel_arg_address_qualifier);

    //           case CL_KERNEL_ARG_ACCESS_QUALIFIER:
    //             PYOPENCL_GET_INTEGRAL_INFO(KernelArg,
    //                 PYOPENCL_FIRST_ARG, param_name,
    //                 cl_kernel_arg_access_qualifier);

    //           case CL_KERNEL_ARG_TYPE_NAME:
    //           case CL_KERNEL_ARG_NAME:
    //             PYOPENCL_GET_STR_INFO(KernelArg, PYOPENCL_FIRST_ARG, param_name);
    // #undef PYOPENCL_FIRST_ARG
    //           default:
    //             throw error("Kernel.get_arg_info", CL_INVALID_VALUE);
    //         }
    //       }
    // #endif
  };

  // }}}
 
  // {{{ buffer transfers

  inline
  event *enqueue_read_buffer(
			     command_queue &cq,
			     memory_object_holder &mem,
			     void *buffer,
			     size_t size, 
			     size_t device_offset,
			     void **wait_for, uint32_t num_wait_for,
			     bool is_blocking)
  {
    PYOPENCL_PARSE_WAIT_FOR;

    cl_event evt;
    // TODO
    //PYOPENCL_RETRY_IF_MEM_ERROR(
    PYOPENCL_CALL_GUARDED_THREADED(clEnqueueReadBuffer,
				   (cq.data(),
				    mem.data(),
				    PYOPENCL_CAST_BOOL(is_blocking),
				    device_offset, size, buffer,
				    PYOPENCL_WAITLIST_ARGS,
				    &evt
				    ));
    //);
    PYOPENCL_RETURN_NEW_EVENT(evt);
  }



  inline
  event *enqueue_copy_buffer(command_queue &cq,
			     memory_object_holder &src,
			     memory_object_holder &dst,
			     ptrdiff_t byte_count,
			     size_t src_offset,
			     size_t dst_offset,
			     void **wait_for, uint32_t num_wait_for
			     )
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
    // TODO
    //PYOPENCL_RETRY_IF_MEM_ERROR(
    PYOPENCL_CALL_GUARDED(clEnqueueCopyBuffer, (cq.data(),
						src.data(), dst.data(),
						src_offset, dst_offset,
						byte_count,
						PYOPENCL_WAITLIST_ARGS,
						&evt
						))
      // );
      PYOPENCL_RETURN_NEW_EVENT(evt);
  }

  inline
  event *enqueue_write_buffer(
			      command_queue &cq,
			      memory_object_holder &mem,
			      void *buffer,
			      size_t size,
			      size_t device_offset,
			      void **wait_for, uint32_t num_wait_for, 
			      bool is_blocking)
  {
    PYOPENCL_PARSE_WAIT_FOR;

    cl_event evt;
    // TODO
    //PYOPENCL_RETRY_IF_MEM_ERROR(
    PYOPENCL_CALL_GUARDED_THREADED(clEnqueueWriteBuffer,
				   (cq.data(),
				    mem.data(),
				    PYOPENCL_CAST_BOOL(is_blocking),
				    device_offset, size, buffer,
				    PYOPENCL_WAITLIST_ARGS,
				    &evt
				    ));
    //);
    // TODO
    PYOPENCL_RETURN_NEW_EVENT(evt);
    //PYOPENCL_RETURN_NEW_NANNY_EVENT(evt, buffer);
  }


  
  // }}}

  inline event *enqueue_nd_range_kernel(
					command_queue &cq,
					kernel &knl,
					cl_uint work_dim,
					const size_t *global_work_offset,
					const size_t *global_work_size,
					const size_t *local_work_size //,
					//py::object py_global_work_offset,
					//py::object py_wait_for,
					)
  {
    // TODO
    // PYOPENCL_PARSE_WAIT_FOR;

    cl_event evt;
    
    // TODO: PYOPENCL_RETRY_RETURN_IF_MEM_ERROR
    PYOPENCL_CALL_GUARDED(clEnqueueNDRangeKernel,
			  (cq.data(),
			   knl.data(),
			   work_dim,
			   global_work_offset,
			   global_work_size,
			   local_work_size,
			   0, NULL,// PYOPENCL_WAITLIST_ARGS,
			   &evt
			   ));
    PYOPENCL_RETURN_NEW_EVENT(evt);
    
  }


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
	return new program(result, false, KND_SOURCE);
      }
    catch (...)
      {
	clReleaseProgram(result);
	throw;
      }
  }

  inline
  program *create_program_with_binary(context &ctx,
				      cl_uint num_devices, 
				      void **ptr_devices,
				      cl_uint num_binaries,
				      char **binaries,
				      size_t *binary_sizes)
  {
    std::vector<cl_device_id> devices;
    std::vector<cl_int> binary_statuses(num_devices);
    for (cl_uint i = 0; i < num_devices; ++i)
      {
	devices.push_back(static_cast<device*>(ptr_devices[i])->data());
      }
    cl_int status_code;
    PYOPENCL_PRINT_CALL_TRACE("clCreateProgramWithBinary");
    cl_program result = clCreateProgramWithBinary(ctx.data(), num_devices,
						  devices.empty( ) ? NULL : &devices.front(),
						  binary_sizes,
						  reinterpret_cast<const unsigned char**>(const_cast<const char**>(binaries)),
						  binary_statuses.empty( ) ? NULL : &binary_statuses.front(),
						  &status_code);

    // for (cl_uint i = 0; i < num_devices; ++i)
    //   std::cout << i << ":" << binary_statuses[i] << std::endl;
    if (status_code != CL_SUCCESS)
      throw pyopencl::error("clCreateProgramWithBinary", status_code);

    try
      {
	return new program(result, false, KND_BINARY);
      }
    catch (...)
      {
	clReleaseProgram(result);
	throw;
      }
  }



}


  

::error *get_platforms(void **ptr_platforms, uint32_t *num_platforms) {
  C_HANDLE_ERROR(
		 *num_platforms = 0;
		 PYOPENCL_CALL_GUARDED(clGetPlatformIDs, (0, 0, num_platforms));

		 typedef std::vector<cl_platform_id> vec;
		 vec platforms(*num_platforms);
		 PYOPENCL_CALL_GUARDED(clGetPlatformIDs,
				       (*num_platforms, platforms.empty( ) ? NULL : &platforms.front(), num_platforms));

		 MALLOC(pyopencl::platform*, _ptr_platforms, *num_platforms);
		 for(vec::size_type i = 0; i < platforms.size(); ++i) {
		   _ptr_platforms[i] = new pyopencl::platform(platforms[i]);
		 }
		 *ptr_platforms = _ptr_platforms;
		 );
  return 0;
}

void _free(void *p) {
  free(p);
}

void _free2(void **p, uint32_t size) {
  for(uint32_t i = 0; i < size; ++i) {
    _free(p[i]);
  }
}

 
::error *platform__get_devices(void *ptr_platform, void **ptr_devices, uint32_t *num_devices, cl_device_type devtype) {
  typedef std::vector<cl_device_id> vec;
  C_HANDLE_ERROR(vec devices = static_cast<pyopencl::platform*>(ptr_platform)->get_devices(devtype);
		 *num_devices = devices.size();
		 
		 MALLOC(pyopencl::device*, _ptr_devices, *num_devices);
		 for(vec::size_type i = 0; i < devices.size(); ++i) {
		   _ptr_devices[i] = new pyopencl::device(devices[i]);
		 }
		 *ptr_devices = _ptr_devices;
		 );
  return 0;
}


::error *_create_context(void **ptr_ctx, cl_context_properties *properties, cl_uint num_devices, void **ptr_devices) {
  C_HANDLE_ERROR(
		 cl_int status_code;
		 std::vector<cl_device_id> devices(num_devices);
		 for(cl_uint i = 0; i < num_devices; ++i) {
		   devices[i] = static_cast<pyopencl::device*>(ptr_devices[i])->data();
		 }
		 cl_context ctx = clCreateContext(properties,
						  num_devices,
						  devices.empty() ? NULL : &devices.front(),
						  0, 0, &status_code);
		 if (status_code != CL_SUCCESS) {
		   throw pyopencl::error("Context", status_code);
		 }
		 *ptr_ctx = new pyopencl::context(ctx, false);
		 );
  return 0;
}

::error *_create_command_queue(void **ptr_command_queue, void *ptr_context, void *ptr_device, cl_command_queue_properties properties) {
  pyopencl::context *ctx = static_cast<pyopencl::context*>(ptr_context);
  pyopencl::device *dev = static_cast<pyopencl::device*>(ptr_device);
  C_HANDLE_ERROR(
		 *ptr_command_queue = new pyopencl::command_queue(*ctx, dev, properties);
		 );
  return 0;
}

::error *_create_buffer(void **ptr_buffer, void *ptr_context, cl_mem_flags flags, size_t size, void *hostbuf) {
  pyopencl::context *ctx = static_cast<pyopencl::context*>(ptr_context);
  C_HANDLE_ERROR(
		 *ptr_buffer = create_buffer_py(*ctx, flags, size, hostbuf);
		 );
  return 0;
}

::error *_create_program_with_source(void **ptr_program, void *ptr_context, char *src) {
  pyopencl::context *ctx = static_cast<pyopencl::context*>(ptr_context);
  C_HANDLE_ERROR(
		 *ptr_program = create_program_with_source(*ctx, src);
		 );
  return 0;
}

::error *_create_program_with_binary(void **ptr_program, void *ptr_context, cl_uint num_devices, void **ptr_devices, cl_uint num_binaries, char **binaries, size_t *binary_sizes) {
  pyopencl::context *ctx = static_cast<pyopencl::context*>(ptr_context);
  C_HANDLE_ERROR(
		 *ptr_program = create_program_with_binary(*ctx, num_devices, ptr_devices, num_binaries, reinterpret_cast<char **>(binaries), binary_sizes);
		 );
  return 0;
}

::error *program__build(void *ptr_program, char *options, cl_uint num_devices, void **ptr_devices) {
  C_HANDLE_ERROR(
		 static_cast<pyopencl::program*>(ptr_program)->build(options, num_devices, ptr_devices);
		 );
  return 0;
}

::error *program__kind(void *ptr_program, int *kind) {
  C_HANDLE_ERROR(
		 *kind = static_cast<pyopencl::program*>(ptr_program)->kind();
		 );
  return 0;
}

  
::error *program__get_build_info(void *ptr_program, void *ptr_device, cl_program_build_info param, generic_info *out) {
  C_HANDLE_ERROR(
		 *out = static_cast<pyopencl::program*>(ptr_program)->get_build_info(*static_cast<pyopencl::device*>(ptr_device),
										     param);
		 );
  return 0;
}

::error *event__get_profiling_info(void *ptr, cl_profiling_info param, generic_info *out) {
  C_HANDLE_ERROR(*out = static_cast<pyopencl::event*>(ptr)->get_profiling_info(param););
  return 0;
}

::error *event__wait(void *ptr) {
  C_HANDLE_ERROR(static_cast<pyopencl::event*>(ptr)->wait(););
  return 0;
}
  
::error *_create_kernel(void **ptr_kernel, void *ptr_program, char *name) {
  pyopencl::program *prg = static_cast<pyopencl::program*>(ptr_program);
  C_HANDLE_ERROR(
		 *ptr_kernel = new pyopencl::kernel(*prg, name);
		 );
  return 0;
}

::error *kernel__set_arg_mem_buffer(void *ptr_kernel, cl_uint arg_index, void *ptr_buffer) {
  pyopencl::buffer *buf = static_cast<pyopencl::buffer*>(ptr_buffer);
  C_HANDLE_ERROR(
		 static_cast<pyopencl::kernel*>(ptr_kernel)->set_arg_mem(arg_index, *buf);
		 );
  return 0;
}

::error *kernel__get_work_group_info(void *ptr, cl_kernel_work_group_info param, void *ptr_device, generic_info *out) {
  C_HANDLE_ERROR(*out = static_cast<pyopencl::kernel*>(ptr)->get_work_group_info(param, *static_cast<pyopencl::device*>(ptr_device)););
  return 0;
}


::error *_enqueue_nd_range_kernel(void **ptr_event, void *ptr_command_queue, void *ptr_kernel, cl_uint work_dim, const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size) {
  C_HANDLE_ERROR(
		 *ptr_event = enqueue_nd_range_kernel(*static_cast<pyopencl::command_queue*>(ptr_command_queue),
						      *static_cast<pyopencl::kernel*>(ptr_kernel),
						      work_dim,
						      global_work_offset,
						      global_work_size,
						      local_work_size);
		 );
  return 0;
}


::error *_enqueue_read_buffer(void **ptr_event, void *ptr_command_queue, void *ptr_memory_object_holder, void *buffer, size_t size, size_t device_offset, void **wait_for, uint32_t num_wait_for, int is_blocking) {
  C_HANDLE_ERROR(
		 *ptr_event = enqueue_read_buffer(*static_cast<pyopencl::command_queue*>(ptr_command_queue),
						  *static_cast<pyopencl::memory_object_holder*>(ptr_memory_object_holder),
						  buffer, size, device_offset, wait_for, num_wait_for, (bool)is_blocking);
		 );
  return 0;
}

::error *_enqueue_write_buffer(void **ptr_event, void *ptr_command_queue, void *ptr_memory_object_holder, void *buffer, size_t size, size_t device_offset, void **wait_for, uint32_t num_wait_for, int is_blocking) {
  C_HANDLE_ERROR(
		 *ptr_event = enqueue_write_buffer(*static_cast<pyopencl::command_queue*>(ptr_command_queue),
						   *static_cast<pyopencl::memory_object_holder*>(ptr_memory_object_holder),
						   buffer, size, device_offset, wait_for, num_wait_for, (bool)is_blocking);
		 );
  return 0;
}

::error *_enqueue_copy_buffer(void **ptr_event, void *ptr_command_queue, void *ptr_src, void *ptr_dst, ptrdiff_t byte_count, size_t src_offset, size_t dst_offset, void **wait_for, uint32_t num_wait_for) {
  C_HANDLE_ERROR(
		 *ptr_event = enqueue_copy_buffer(*static_cast<pyopencl::command_queue*>(ptr_command_queue),
						  *static_cast<pyopencl::memory_object_holder*>(ptr_src),
						  *static_cast<pyopencl::memory_object_holder*>(ptr_dst),
						  byte_count, src_offset, dst_offset, wait_for, num_wait_for);
		 );
  return 0;
}
  
intptr_t _int_ptr(void* ptr, class_t class_) {
#define INT_PTR(CLSU, CLS) return (intptr_t)(static_cast<pyopencl::CLS*>(ptr)->data());
  SWITCHCLASS(INT_PTR);
}

void* _from_int_ptr(void **ptr_out, intptr_t int_ptr_value, class_t class_) {
#define FROM_INT_PTR(CLSU, CLS) C_HANDLE_ERROR(*ptr_out = new pyopencl::CLS((PYOPENCL_CL_##CLSU)int_ptr_value, /* retain */ true);)
  SWITCHCLASS(FROM_INT_PTR);
  return 0;								
}
  
long _hash(void *ptr, class_t class_) {	
#define HASH(CLSU, CLS)	return static_cast<pyopencl::CLS*>(ptr)->hash();
  SWITCHCLASS(HASH);
}

  
::error *_get_info(void *ptr, class_t class_, cl_uint param, generic_info *out) {
#define GET_INFO(CLSU, CLS) C_HANDLE_ERROR(*out = static_cast<pyopencl::CLS*>(ptr)->get_info(param);)
  SWITCHCLASS(GET_INFO);
  return 0;
}

void _delete(void *ptr, class_t class_) {
#define DELETE(CLSU, CLS) delete static_cast<pyopencl::CLS*>(ptr);
  SWITCHCLASS(DELETE);
}


int get_cl_version(void) {
  return PYOPENCL_CL_VERSION;
}  

unsigned bitlog2(unsigned long v) {
  return pyopencl::bitlog2(v);
}

// {{{ gl interop
int have_gl() {
#ifdef HAVE_GL
  return 1;
#else
  return 0;
#endif
}

error *_create_gl_buffer(void **ptr_buffer, void *ptr_context, cl_mem_flags flags, GLuint bufobj) {
  pyopencl::context *ctx = static_cast<pyopencl::context*>(ptr_context);
  C_HANDLE_ERROR(*ptr_buffer = create_from_gl_buffer(*ctx, flags, bufobj););
  return 0;  
}

// }}}
