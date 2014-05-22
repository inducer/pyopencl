#include "error.h"
#include "utils.h"

#include <stdlib.h>
#include <vector>

// {{{ extension function pointers

#if PYOPENCL_CL_VERSION >= 0x1020

#define PYOPENCL_GET_EXT_FUN(PLATFORM, NAME, VAR) \
  NAME##_fn VAR  = (NAME##_fn) \
      clGetExtensionFunctionAddressForPlatform(PLATFORM, #NAME); \
  \
  if (!VAR) \
    throw error(#NAME, CL_INVALID_VALUE, #NAME " not available");

#else

#define PYOPENCL_GET_EXT_FUN(PLATFORM, NAME, VAR) \
  NAME##_fn VAR = (NAME##_fn) clGetExtensionFunctionAddress(#NAME); \
  \
  if (!VAR) \
    throw error(#NAME, CL_INVALID_VALUE, #NAME " not available");

#endif

// }}}


// {{{ event helpers

#define PYOPENCL_PARSE_OBJECT_LIST(CLS, TYPE, OUT, NAME, NUM)   \
  std::vector<TYPE> OUT((NUM));                                 \
  {                                                             \
    for(unsigned i = 0; i < (NUM); ++i) {                       \
      OUT[i] = static_cast<pyopencl::CLS*>(NAME[i])->data();    \
    }                                                           \
  }

#define PYOPENCL_PARSE_WAIT_FOR PYOPENCL_PARSE_OBJECT_LIST(event, cl_event, event_wait_list, wait_for, num_wait_for)

#define PYOPENCL_WAITLIST_ARGS                                          \
  num_wait_for, event_wait_list.empty( ) ? NULL : &event_wait_list.front()

// }}}


// {{{ equality testing

#define PYOPENCL_EQUALITY_TESTS(cls) \
  \
  bool operator==(cls const &other) const \
  { return data() == other.data(); } \
  \
  bool operator!=(cls const &other) const \
  { return data() != other.data(); } \
  \
  long hash() const \
  { return (long) (intptr_t) data(); }

// }}}


// {{{ tools

#define PYOPENCL_PARSE_PY_DEVICES \
  std::vector<cl_device_id> devices_vec; \
  cl_uint num_devices; \
  cl_device_id *devices; \
  \
  if (py_devices.ptr() == Py_None)                                      \
    {                                                                   \
      num_devices = 0; \
      devices = 0; \
    }                                                                   \
  else                                                                  \
    {                                                                   \
      PYTHON_FOREACH(py_dev, py_devices)                                \
        devices_vec.push_back(                                          \
                              py::extract<device &>(py_dev)().data()); \
      num_devices = devices_vec.size(); \
      devices = devices_vec.empty( ) ? NULL : &devices_vec.front(); \
    }                                                                   \

// }}}


// {{{ more odds and ends

#ifdef HAVE_GL
#define GL_SWITCHCLASS(OPERATION) \
  case ::CLASS_GL_BUFFER: OPERATION(GL_BUFFER, gl_buffer); break; \
  case ::CLASS_GL_RENDERBUFFER: OPERATION(GL_RENDERBUFFER, gl_renderbuffer); break;
#else
#define GL_SWITCHCLASS(OPERATION) \
  case ::CLASS_GL_BUFFER: \
  case ::CLASS_GL_RENDERBUFFER:
#endif

#define SWITCHCLASS(OPERATION)                                          \
  switch(class_) {                                                      \
  case ::CLASS_PLATFORM: OPERATION(PLATFORM, platform); break;          \
  case ::CLASS_DEVICE: OPERATION(DEVICE, device); break;                \
  case ::CLASS_KERNEL: OPERATION(KERNEL, kernel); break;                \
  case ::CLASS_CONTEXT: OPERATION(CONTEXT, context); break;             \
  case ::CLASS_COMMAND_QUEUE: OPERATION(COMMAND_QUEUE, command_queue); break; \
  case ::CLASS_BUFFER: OPERATION(BUFFER, buffer); break;                \
  case ::CLASS_PROGRAM: OPERATION(PROGRAM, program); break;             \
  case ::CLASS_EVENT: OPERATION(EVENT, event); break;                   \
  case ::CLASS_IMAGE: OPERATION(IMAGE, image); break; \
  case ::CLASS_SAMPLER: OPERATION(SAMPLER, sampler); break; \
  GL_SWITCHCLASS(OPERATION) \
  default: throw pyopencl::error("unknown class", CL_INVALID_VALUE);    \
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
#define PYOPENCL_CL_GL_RENDERBUFFER cl_mem
#define PYOPENCL_CL_IMAGE cl_mem
#define PYOPENCL_CL_SAMPLER cl_sampler

// }}}

namespace pyopencl
{
static int
dummy_python_gc()
{
    return 0;
}

int (*python_gc)() = dummy_python_gc;

  // {{{ platform

  class platform : public noncopyable
  {
    private:
      cl_platform_id m_platform;

    public:
      PYOPENCL_DEF_GET_CLASS_T(PLATFORM);
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
        switch (param_name) {
        case CL_PLATFORM_PROFILE:
        case CL_PLATFORM_VERSION:
        case CL_PLATFORM_NAME:
        case CL_PLATFORM_VENDOR:
#if !(defined(CL_PLATFORM_NVIDIA) && CL_PLATFORM_NVIDIA == 0x3001)
        case CL_PLATFORM_EXTENSIONS:
#endif
            return pyopencl_get_str_info(Platform, m_platform, param_name);

        default:
            throw error("Platform.get_info", CL_INVALID_VALUE);
        }
      }

      pyopencl_buf<cl_device_id> get_devices(cl_device_type devtype);
  };


inline pyopencl_buf<cl_device_id>
platform::get_devices(cl_device_type devtype)
{
    cl_uint num_devices = 0;
    print_call_trace("clGetDeviceIDs");
    cl_int status_code;
    status_code = clGetDeviceIDs(m_platform, devtype, 0, 0, &num_devices);
    if (status_code == CL_DEVICE_NOT_FOUND) {
        num_devices = 0;
    } else if (status_code != CL_SUCCESS) {
        throw pyopencl::error("clGetDeviceIDs", status_code);
    }

    pyopencl_buf<cl_device_id> devices(num_devices);
    if (num_devices == 0)
        return devices;
    pyopencl_call_guarded(clGetDeviceIDs, m_platform, devtype, num_devices,
                          devices.get(), &num_devices);
    return devices;
}

// }}}


  // {{{ device

  class device : public noncopyable
  {
    public:
      PYOPENCL_DEF_GET_CLASS_T(DEVICE);
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
            pyopencl_call_guarded(clGetDeviceInfo, m_device, CL_DEVICE_PLATFORM,
                                  sizeof(plat), &plat, NULL);
#endif

            PYOPENCL_GET_EXT_FUN(plat,
                clRetainDeviceEXT, retain_func);

            pyopencl_call_guarded(retain_func, did);
          }
#endif

#if PYOPENCL_CL_VERSION >= 0x1020
          else if (ref_type == REF_CL_1_2)
            {
              pyopencl_call_guarded(clRetainDevice, did);
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
          pyopencl_call_guarded(clGetDeviceInfo, m_device, CL_DEVICE_PLATFORM,
                                sizeof(plat), &plat, NULL);
#endif

          PYOPENCL_GET_EXT_FUN(plat,
              clReleaseDeviceEXT, release_func);

          pyopencl_call_guarded_cleanup(release_func, m_device);
        }
#endif

#if PYOPENCL_CL_VERSION >= 0x1020
        else if (m_ref_type == REF_CL_1_2)
          pyopencl_call_guarded(clReleaseDevice, m_device);
#endif
      }

      cl_device_id data() const
      {
        return m_device;
      }

      PYOPENCL_EQUALITY_TESTS(device);

      generic_info get_info(cl_device_info param_name) const
      {
#define DEV_GET_INT_INF(TYPE) pyopencl_get_int_info(TYPE, Device, m_device, param_name)
        switch (param_name) {
        case CL_DEVICE_TYPE:
            return DEV_GET_INT_INF(cl_device_type);
        case CL_DEVICE_MAX_WORK_GROUP_SIZE:
            return DEV_GET_INT_INF(size_t);
        case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:
        case CL_DEVICE_MAX_COMPUTE_UNITS:
        case CL_DEVICE_VENDOR_ID:
            return DEV_GET_INT_INF(cl_uint);

        case CL_DEVICE_MAX_WORK_ITEM_SIZES:
            return pyopencl_get_array_info(size_t, Device, m_device, param_name);

        case CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR:
        case CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT:
        case CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT:
        case CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG:
        case CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT:
        case CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE:

        case CL_DEVICE_MAX_CLOCK_FREQUENCY:
        case CL_DEVICE_ADDRESS_BITS:
        case CL_DEVICE_MAX_READ_IMAGE_ARGS:
        case CL_DEVICE_MAX_WRITE_IMAGE_ARGS:
        case CL_DEVICE_MAX_SAMPLERS:
        case CL_DEVICE_MEM_BASE_ADDR_ALIGN:
        case CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE:
            return DEV_GET_INT_INF(cl_uint);

        case CL_DEVICE_MAX_MEM_ALLOC_SIZE:
            return DEV_GET_INT_INF(cl_ulong);

        case CL_DEVICE_IMAGE2D_MAX_WIDTH:
        case CL_DEVICE_IMAGE2D_MAX_HEIGHT:
        case CL_DEVICE_IMAGE3D_MAX_WIDTH:
        case CL_DEVICE_IMAGE3D_MAX_HEIGHT:
        case CL_DEVICE_IMAGE3D_MAX_DEPTH:
        case CL_DEVICE_MAX_PARAMETER_SIZE:
            return DEV_GET_INT_INF(size_t);

        case CL_DEVICE_IMAGE_SUPPORT:
            return DEV_GET_INT_INF(cl_bool);
#ifdef CL_DEVICE_DOUBLE_FP_CONFIG
        case CL_DEVICE_DOUBLE_FP_CONFIG:
#endif
#ifdef CL_DEVICE_HALF_FP_CONFIG
        case CL_DEVICE_HALF_FP_CONFIG:
#endif
        case CL_DEVICE_SINGLE_FP_CONFIG:
            return DEV_GET_INT_INF(cl_device_fp_config);

        case CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:
            return DEV_GET_INT_INF(cl_device_mem_cache_type);
        case CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE:
            return DEV_GET_INT_INF(cl_uint);
        case CL_DEVICE_GLOBAL_MEM_CACHE_SIZE:
        case CL_DEVICE_GLOBAL_MEM_SIZE:
        case CL_DEVICE_LOCAL_MEM_SIZE:
        case CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:
            return DEV_GET_INT_INF(cl_ulong);

        case CL_DEVICE_MAX_CONSTANT_ARGS:
            return DEV_GET_INT_INF(cl_uint);
        case CL_DEVICE_LOCAL_MEM_TYPE:
            return DEV_GET_INT_INF(cl_device_local_mem_type);
        case CL_DEVICE_PROFILING_TIMER_RESOLUTION:
            return DEV_GET_INT_INF(size_t);
        case CL_DEVICE_ENDIAN_LITTLE:
        case CL_DEVICE_AVAILABLE:
        case CL_DEVICE_COMPILER_AVAILABLE:
        case CL_DEVICE_ERROR_CORRECTION_SUPPORT:
            return DEV_GET_INT_INF(cl_bool);
        case CL_DEVICE_EXECUTION_CAPABILITIES:
            return DEV_GET_INT_INF(cl_device_exec_capabilities);
        case CL_DEVICE_QUEUE_PROPERTIES:
            return DEV_GET_INT_INF(cl_command_queue_properties);

        case CL_DEVICE_NAME:
        case CL_DEVICE_VENDOR:
        case CL_DRIVER_VERSION:
        case CL_DEVICE_PROFILE:
        case CL_DEVICE_VERSION:
        case CL_DEVICE_EXTENSIONS:
            return pyopencl_get_str_info(Device, m_device, param_name);

        case CL_DEVICE_PLATFORM:
            return pyopencl_get_opaque_info(cl_platform_id, platform,
                                            Device, m_device, param_name);
#if PYOPENCL_CL_VERSION >= 0x1010
        case CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF:
        case CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR:
        case CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT:
        case CL_DEVICE_NATIVE_VECTOR_WIDTH_INT:
        case CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG:
        case CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT:
        case CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE:
        case CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF:
            return DEV_GET_INT_INF(cl_uint);

        case CL_DEVICE_HOST_UNIFIED_MEMORY:
            return DEV_GET_INT_INF(cl_bool);
        case CL_DEVICE_OPENCL_C_VERSION:
            return pyopencl_get_str_info(Device, m_device, param_name);
#endif
#ifdef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
        case CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV:
        case CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV:
        case CL_DEVICE_REGISTERS_PER_BLOCK_NV:
        case CL_DEVICE_WARP_SIZE_NV:
            return DEV_GET_INT_INF(cl_uint);
        case CL_DEVICE_GPU_OVERLAP_NV:
        case CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV:
        case CL_DEVICE_INTEGRATED_MEMORY_NV:
            return DEV_GET_INT_INF(cl_bool);
#endif
#if defined(cl_ext_device_fission) && defined(PYOPENCL_USE_DEVICE_FISSION)
        case CL_DEVICE_PARENT_DEVICE_EXT:
            return pyopencl_get_opaque_info(cl_device_id, device,
                                            Device, m_device, param_name);
        case CL_DEVICE_PARTITION_TYPES_EXT:
        case CL_DEVICE_AFFINITY_DOMAINS_EXT:
        case CL_DEVICE_PARTITION_STYLE_EXT:
            return pyopencl_get_array_info(cl_device_partition_property_ext,
                                           Device, m_device, param_name);
        case CL_DEVICE_REFERENCE_COUNT_EXT:
            return DEV_GET_INT_INF(cl_uint);
#endif
#if PYOPENCL_CL_VERSION >= 0x1020
        case CL_DEVICE_LINKER_AVAILABLE: return DEV_GET_INT_INF(cl_bool);
        case CL_DEVICE_BUILT_IN_KERNELS:
            return pyopencl_get_str_info(Device, m_device, param_name);
        case CL_DEVICE_IMAGE_MAX_BUFFER_SIZE:
        case CL_DEVICE_IMAGE_MAX_ARRAY_SIZE:
            DEV_GET_INT_INF(size_t);
        case CL_DEVICE_PARENT_DEVICE:
            return pyopencl_get_opaque_info(cl_device_id, device,
                                            Device, m_device, param_name);
        case CL_DEVICE_PARTITION_MAX_SUB_DEVICES:
            return DEV_GET_INT_INF(cl_uint);
        case CL_DEVICE_PARTITION_TYPE:
        case CL_DEVICE_PARTITION_PROPERTIES:
            return pyopencl_get_array_info(cl_device_partition_property,
                                           Device, m_device, param_name);
        case CL_DEVICE_PARTITION_AFFINITY_DOMAIN:
            return pyopencl_get_array_info(cl_device_affinity_domain,
                                           Device, m_device, param_name);
        case CL_DEVICE_REFERENCE_COUNT: DEV_GET_INT_INF(cl_uint);
        case CL_DEVICE_PREFERRED_INTEROP_USER_SYNC:
        case CL_DEVICE_PRINTF_BUFFER_SIZE:
            DEV_GET_INT_INF(cl_bool);
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
            return pyopencl_get_str_info(Device, m_device, param_name);
#endif
#ifdef CL_DEVICE_GLOBAL_FREE_MEMORY_AMD
        case CL_DEVICE_GLOBAL_FREE_MEMORY_AMD:
            return pyopencl_get_array_info(size_t, Device,
                                           m_device, param_name);
#endif
#ifdef CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD
        case CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD:
#endif
#ifdef CL_DEVICE_SIMD_WIDTH_AMD
        case CL_DEVICE_SIMD_WIDTH_AMD:
#endif
#ifdef CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD
        case CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD:
#endif
#ifdef CL_DEVICE_WAVEFRONT_WIDTH_AMD
        case CL_DEVICE_WAVEFRONT_WIDTH_AMD:
#endif
#ifdef CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD
        case CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD:
#endif
#ifdef CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD
        case CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD:
#endif
#ifdef CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD
        case CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD:
#endif
#ifdef CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD
        case CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD:
#endif
#ifdef CL_DEVICE_LOCAL_MEM_BANKS_AMD
        case CL_DEVICE_LOCAL_MEM_BANKS_AMD:
#endif

#ifdef CL_DEVICE_MAX_ATOMIC_COUNTERS_EXT
        case CL_DEVICE_MAX_ATOMIC_COUNTERS_EXT:
#endif
            return DEV_GET_INT_INF(cl_uint);
            // }}}

        default:
            throw error("Device.get_info", CL_INVALID_VALUE);
        }
      }

    // TODO: sub-devices
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

  class context : public noncopyable
  {
    private:
      cl_context m_context;

    public:
      PYOPENCL_DEF_GET_CLASS_T(CONTEXT);
      context(cl_context ctx, bool retain)
        : m_context(ctx)
      {
        if (retain)
          pyopencl_call_guarded(clRetainContext, ctx);
      }


      ~context()
      {
          pyopencl_call_guarded_cleanup(clReleaseContext, m_context);
      }

      cl_context data() const
      {
        return m_context;
      }

      PYOPENCL_EQUALITY_TESTS(context);

      generic_info get_info(cl_context_info param_name) const
      {
        switch (param_name) {
        case CL_CONTEXT_REFERENCE_COUNT:
            return pyopencl_get_int_info(cl_uint, Context,
                                         m_context, param_name);
        case CL_CONTEXT_DEVICES:
            return pyopencl_get_opaque_array_info(
                cl_device_id, device, Context, m_context, param_name);
        case CL_CONTEXT_PROPERTIES: {
            auto result = pyopencl_get_vec_info(
                cl_context_properties, Context, m_context, param_name);
            pyopencl_buf<generic_info> py_result(result.len() / 2);
            size_t i = 0;
            for (;i < py_result.len();i++) {
                cl_context_properties key = result[i * 2];
                if (key == 0)
                    break;
                cl_context_properties value = result[i * 2 + 1];
                generic_info &info = py_result[i];
                info.dontfree = 0;
                info.opaque_class = CLASS_NONE;
                switch (key) {
                case CL_CONTEXT_PLATFORM:
                    info.opaque_class = CLASS_PLATFORM;
                    info.type = "void *";
                    info.value = new platform(
                        reinterpret_cast<cl_platform_id>(value));
                    break;

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
                    info.value = (void*)value;
                    // we do not own this object
                    info.dontfree = 1;
                    break;

#endif
                default:
                    throw error("Context.get_info", CL_INVALID_VALUE,
                                "unknown context_property key encountered");
                }
            }
            py_result.resize(i);
            return pyopencl_convert_array_info(generic_info, py_result);
        }

#if PYOPENCL_CL_VERSION >= 0x1010
        case CL_CONTEXT_NUM_DEVICES:
            return pyopencl_get_int_info(cl_uint, Context,
                                         m_context, param_name);
#endif

        default:
            throw error("Context.get_info", CL_INVALID_VALUE);
        }
      }
  };

  // }}}


  // {{{ command_queue

  class command_queue : public noncopyable
  {
    private:
      cl_command_queue m_queue;

    public:
      PYOPENCL_DEF_GET_CLASS_T(COMMAND_QUEUE);
      command_queue(cl_command_queue q, bool retain)
        : m_queue(q)
      {
        if (retain)
          pyopencl_call_guarded(clRetainCommandQueue, q);
      }

      command_queue(command_queue const &src)
        : m_queue(src.m_queue)
      {
        pyopencl_call_guarded(clRetainCommandQueue, m_queue);
      }

      command_queue(
                    const context &ctx,
                    const device *py_dev=0,
                    cl_command_queue_properties props=0)
      {
        cl_device_id dev;
        if (py_dev) {
            dev = py_dev->data();
        } else {
            auto devs = pyopencl_get_vec_info(cl_device_id, Context,
                                              ctx.data(), CL_CONTEXT_DEVICES);
            if (devs.len() == 0)
              throw pyopencl::error("CommandQueue", CL_INVALID_VALUE,
                                    "context doesn't have any devices? -- "
                                    "don't know which one to default to");
            dev = devs[0];
        }
        m_queue = pyopencl_call_guarded(clCreateCommandQueue,
                                        ctx.data(), dev, props);
      }

      ~command_queue()
      {
          pyopencl_call_guarded_cleanup(clReleaseCommandQueue, m_queue);
      }

      const cl_command_queue data() const
      { return m_queue; }

      PYOPENCL_EQUALITY_TESTS(command_queue);

      generic_info get_info(cl_command_queue_info param_name) const
      {
        switch (param_name) {
        case CL_QUEUE_CONTEXT:
            return pyopencl_get_opaque_info(cl_context, context,
                                            CommandQueue, m_queue, param_name);
        case CL_QUEUE_DEVICE:
            return pyopencl_get_opaque_info(cl_device_id, device,
                                            CommandQueue, m_queue, param_name);
        case CL_QUEUE_REFERENCE_COUNT:
            return pyopencl_get_int_info(cl_uint, CommandQueue,
                                         m_queue, param_name);
        case CL_QUEUE_PROPERTIES:
            return pyopencl_get_int_info(cl_command_queue_properties,
                                         CommandQueue, m_queue, param_name);
        default:
            throw error("CommandQueue.get_info", CL_INVALID_VALUE);
        }
      }

      std::unique_ptr<context> get_context() const
      {
        cl_context param_value;
        pyopencl_call_guarded(clGetCommandQueueInfo, m_queue, CL_QUEUE_CONTEXT,
                              sizeof(param_value), &param_value, NULL);
        return std::unique_ptr<context>(
            new context(param_value, /*retain*/ true));
      }

#if PYOPENCL_CL_VERSION < 0x1010
    cl_command_queue_properties set_property(
        cl_command_queue_properties prop,
        bool enable)
    {
      cl_command_queue_properties old_prop;
      pyopencl_call_guarded(clSetCommandQueueProperty, m_queue, prop,
                            cast_bool(enable), &old_prop);
      return old_prop;
    }
#endif

    void flush()
    { pyopencl_call_guarded(clFlush, m_queue); }

    void finish()
    {
      pyopencl_call_guarded(clFinish, m_queue);
    }
  };

  // }}}


  // {{{ event

  class event : public noncopyable
  {
    private:
      cl_event m_event;

    public:
      PYOPENCL_DEF_GET_CLASS_T(EVENT);
      event(cl_event event, bool retain)
        : m_event(event)
      {
        if (retain)
          pyopencl_call_guarded(clRetainEvent, event);
      }

      event(event const &src) : m_event(src.m_event)
      {
          pyopencl_call_guarded(clRetainEvent, m_event);
      }

      virtual ~event()
      {
          pyopencl_call_guarded_cleanup(clReleaseEvent, m_event);
      }

      const cl_event data() const
      { return m_event; }

      PYOPENCL_EQUALITY_TESTS(event);

      generic_info get_info(cl_event_info param_name) const
      {
        switch (param_name) {
        case CL_EVENT_COMMAND_QUEUE:
            return pyopencl_get_opaque_info(cl_command_queue, command_queue,
                                            Event, m_event, param_name);
        case CL_EVENT_COMMAND_TYPE:
            return pyopencl_get_int_info(cl_command_type, Event,
                                         m_event, param_name);
        case CL_EVENT_COMMAND_EXECUTION_STATUS:
            return pyopencl_get_int_info(cl_int, Event, m_event, param_name);
        case CL_EVENT_REFERENCE_COUNT:
            return pyopencl_get_int_info(cl_uint, Event, m_event, param_name);
#if PYOPENCL_CL_VERSION >= 0x1010
        case CL_EVENT_CONTEXT:
            return pyopencl_get_opaque_info(cl_context, context,
                                            Event, m_event, param_name);
#endif

        default:
            throw error("Event.get_info", CL_INVALID_VALUE);
        }
      }

      generic_info get_profiling_info(cl_profiling_info param_name) const
      {
        switch (param_name) {
        case CL_PROFILING_COMMAND_QUEUED:
        case CL_PROFILING_COMMAND_SUBMIT:
        case CL_PROFILING_COMMAND_START:
        case CL_PROFILING_COMMAND_END:
            return pyopencl_get_int_info(cl_ulong, EventProfiling,
                                         m_event, param_name);
        default:
            throw error("Event.get_profiling_info", CL_INVALID_VALUE);
        }
      }

      virtual void wait()
      {
        pyopencl_call_guarded(clWaitForEvents, 1, &m_event);
      }
  };
static inline event*
new_event(cl_event evt)
{
    return pyopencl_convert_obj(event, clReleaseEvent, evt);
}

  // }}}


  // {{{ memory_object

  //py::object create_mem_object_wrapper(cl_mem mem);

  class memory_object_holder : public noncopyable
  {
    public:
      virtual const cl_mem data() const = 0;

      PYOPENCL_EQUALITY_TESTS(memory_object_holder);

      size_t size() const
      {
        size_t param_value;
        pyopencl_call_guarded(clGetMemObjectInfo, data(), CL_MEM_SIZE,
                              sizeof(param_value), &param_value, NULL);
        return param_value;
      }

      generic_info get_info(cl_mem_info param_name)
      {
          switch (param_name){
          case CL_MEM_TYPE:
              return pyopencl_get_int_info(cl_mem_object_type, MemObject,
                                           data(), param_name);
          case CL_MEM_FLAGS:
              return pyopencl_get_int_info(cl_mem_flags, MemObject,
                                           data(), param_name);
          case CL_MEM_SIZE:
              return pyopencl_get_int_info(size_t, MemObject,
                                           data(), param_name);
          case CL_MEM_HOST_PTR:
              throw pyopencl::error("MemoryObject.get_info", CL_INVALID_VALUE,
                                    "Use MemoryObject.get_host_array to get "
                                    "host pointer.");
          case CL_MEM_MAP_COUNT:
          case CL_MEM_REFERENCE_COUNT:
              return pyopencl_get_int_info(cl_uint, MemObject,
                                           data(), param_name);
          case CL_MEM_CONTEXT:
              return pyopencl_get_opaque_info(cl_context, context,
                                              MemObject, data(), param_name);

#if PYOPENCL_CL_VERSION >= 0x1010
            // TODO
            //       case CL_MEM_ASSOCIATED_MEMOBJECT:
            //      {
            //        cl_mem param_value;
            //        PYOPENCL_CALL_GUARDED(clGetMemObjectInfo, (data(), param_name, sizeof(param_value), &param_value, 0));
            //        if (param_value == 0)
            //          {
            //            // no associated memory object? no problem.
            //            return py::object();
            //          }

            //        return create_mem_object_wrapper(param_value);
            //      }
          case CL_MEM_OFFSET:
              return pyopencl_get_int_info(size_t, MemObject,
                                           data(), param_name);
#endif

          default:
              throw error("MemoryObjectHolder.get_info", CL_INVALID_VALUE);
        }
      }
  };


  class memory_object : public memory_object_holder
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
          pyopencl_call_guarded(clRetainMemObject, mem);

        if (hostbuf)
          m_hostbuf = hostbuf;
      }

      memory_object(memory_object const &src)
        : m_valid(true), m_mem(src.m_mem), m_hostbuf(src.m_hostbuf)
      {
        pyopencl_call_guarded(clRetainMemObject, m_mem);
      }

      memory_object(memory_object_holder const &src)
        : m_valid(true), m_mem(src.data())
      {
        pyopencl_call_guarded(clRetainMemObject, m_mem);
      }

      void release()
      {
          if (!m_valid)
              throw error("MemoryObject.free", CL_INVALID_VALUE,
                          "trying to double-unref mem object");
          pyopencl_call_guarded_cleanup(clReleaseMemObject, m_mem);
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
      PYOPENCL_DEF_GET_CLASS_T(IMAGE);
      image(cl_mem mem, bool retain, void *hostbuf=0)
        : memory_object(mem, retain, hostbuf)
      { }

      generic_info get_image_info(cl_image_info param_name) const
      {
        switch (param_name) {
        case CL_IMAGE_FORMAT:
            return pyopencl_get_int_info(cl_image_format, Image,
                                         data(), param_name);
        case CL_IMAGE_ELEMENT_SIZE:
        case CL_IMAGE_ROW_PITCH:
        case CL_IMAGE_SLICE_PITCH:
        case CL_IMAGE_WIDTH:
        case CL_IMAGE_HEIGHT:
        case CL_IMAGE_DEPTH:
#if PYOPENCL_CL_VERSION >= 0x1020
        case CL_IMAGE_ARRAY_SIZE:
#endif
            return pyopencl_get_int_info(size_t, Image, data(), param_name);

#if PYOPENCL_CL_VERSION >= 0x1020
            // TODO:
            //    case CL_IMAGE_BUFFER:
            //      {
            //        cl_mem param_value;
            //        PYOPENCL_CALL_GUARDED(clGetImageInfo, (data(), param_name, sizeof(param_value), &param_value, 0));
            //        if (param_value == 0)
            //               {
            //                 // no associated memory object? no problem.
            //                 return py::object();
            //               }

            //        return create_mem_object_wrapper(param_value);
            //      }

        case CL_IMAGE_NUM_MIP_LEVELS:
        case CL_IMAGE_NUM_SAMPLES:
            return pyopencl_get_int_info(cl_uint, Image, data(), param_name);
#endif

        default:
            throw error("MemoryObject.get_image_info", CL_INVALID_VALUE);
        }
      }
  };
static inline image*
new_image(cl_mem mem, void *buff=0)
{
    return pyopencl_convert_obj(image, clReleaseMemObject, mem, buff);
}


  // {{{ image formats
  inline generic_info
  get_supported_image_formats(context const &ctx, cl_mem_flags flags,
                              cl_mem_object_type image_type)
  {
    cl_uint num_image_formats;
    pyopencl_call_guarded(clGetSupportedImageFormats,
                          ctx.data(), flags, image_type,
                          0, NULL, &num_image_formats);

    pyopencl_buf<cl_image_format> formats(num_image_formats);
    pyopencl_call_guarded(clGetSupportedImageFormats,
                          ctx.data(), flags, image_type,
                          formats.len(), formats.get(), NULL);

    return pyopencl_convert_array_info(cl_image_format, formats);
  }


  // }}}

  // {{{ image creation

inline image*
create_image_2d(context const &ctx, cl_mem_flags flags,
                cl_image_format const &fmt, size_t width, size_t height,
                size_t pitch, void *buffer, size_t size)
{
    auto mem = retry_mem_error<cl_mem>([&] {
            return pyopencl_call_guarded(clCreateImage2D, ctx.data(), flags,
                                         &fmt, width, height, pitch, buffer);
        });
    return new_image(mem, flags & CL_MEM_USE_HOST_PTR ? buffer : NULL);
}

inline image*
create_image_3d(context const &ctx, cl_mem_flags flags,
                cl_image_format const &fmt, size_t width, size_t height,
                size_t depth, size_t pitch_x, size_t pitch_y,
                void *buffer, size_t size)
{
    auto mem = retry_mem_error<cl_mem>([&] {
            return pyopencl_call_guarded(clCreateImage3D, ctx.data(), flags,
                                         &fmt, width, height, depth, pitch_x,
                                         pitch_y, buffer);
        });
    return new_image(mem, flags & CL_MEM_USE_HOST_PTR ? buffer : NULL);
}


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

  // }}}

  // {{{ image transfers

inline event*
enqueue_read_image(command_queue &cq, image &img, size_t *origin,
                   size_t *region, void *buffer, size_t size, size_t row_pitch,
                   size_t slice_pitch, void **wait_for, uint32_t num_wait_for,
                   bool is_blocking)
{
    PYOPENCL_PARSE_WAIT_FOR;
    cl_event evt;
    retry_mem_error<void>([&] {
            pyopencl_call_guarded(clEnqueueReadImage, cq.data(), img.data(),
                                  cast_bool(is_blocking), origin, region,
                                  row_pitch, slice_pitch, buffer,
                                  PYOPENCL_WAITLIST_ARGS, &evt);
        });
    return new_event(evt);
    //PYOPENCL_RETURN_NEW_NANNY_EVENT(evt, buffer);
}

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
    PYOPENCL_DEF_GET_CLASS_T(GL_BUFFER);
    gl_buffer(cl_mem mem, bool retain, void *hostbuf=0)
      : memory_object(mem, retain, hostbuf)
    { }
  };




  class gl_renderbuffer : public memory_object
  {
  public:
    PYOPENCL_DEF_GET_CLASS_T(GL_RENDERBUFFER);
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
      switch (param_name) {
      case CL_GL_TEXTURE_TARGET:
          return pyopencl_get_int_info(GLenum, GLTexture, data(), param_name);
      case CL_GL_MIPMAP_LEVEL:
          return pyopencl_get_int_info(GLint, GLTexture, data(), param_name);
      default:
          throw error("MemoryObject.get_gl_texture_info", CL_INVALID_VALUE);
      }
    }
  };




#define PYOPENCL_WRAP_BUFFER_CREATOR(TYPE, NAME, CL_NAME, ARGS, CL_ARGS) \
  inline                                                                \
  TYPE *NAME ARGS                                                       \
  {                                                                     \
    cl_int status_code;                                                 \
    pyopencl::print_call_trace(#CL_NAME);                               \
    cl_mem mem = CL_NAME CL_ARGS;                                       \
                                                                        \
    if (status_code != CL_SUCCESS)                                      \
      throw pyopencl::error(#CL_NAME, status_code);                     \
                                                                        \
    try                                                                 \
      {                                                                 \
        return new TYPE(mem, false);                                    \
      }                                                                 \
    catch (...)                                                         \
      {                                                                 \
          pyopencl_call_guarded(clReleaseMemObject, mem);               \
          throw;                                                        \
      }                                                                 \
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
  gl_texture *create_from_gl_texture(context &ctx, cl_mem_flags flags,
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





  // TODO:
  // inline
  // py::tuple get_gl_object_info(memory_object_holder const &mem)
  // {
  //   cl_gl_object_type otype;
  //   GLuint gl_name;
  //   PYOPENCL_CALL_GUARDED(clGetGLObjectInfo, (mem.data(), &otype, &gl_name));
  //   return py::make_tuple(otype, gl_name);
  // }

#define WRAP_GL_ENQUEUE(what, What)                                     \
  inline                                                                \
  event *enqueue_##what##_gl_objects(command_queue &cq,                 \
                                     void **ptr_mem_objects,            \
                                     uint32_t num_mem_objects,          \
                                     void **wait_for,                   \
                                     uint32_t num_wait_for)             \
  {                                                                     \
    PYOPENCL_PARSE_WAIT_FOR;                                            \
    PYOPENCL_PARSE_OBJECT_LIST(memory_object_holder, cl_mem, mem_objects, ptr_mem_objects, num_mem_objects); \
    cl_event evt;                                                       \
    pyopencl_call_guarded(clEnqueue##What##GLObjects, cq.data(),        \
                          mem_objects.size(),                           \
                          mem_objects.empty( ) ? NULL : &mem_objects.front(), \
                          PYOPENCL_WAITLIST_ARGS, &evt);                \
                                                                        \
    return pyopencl::new_event(evt);                                    \
  }

  WRAP_GL_ENQUEUE(acquire, Acquire);
  WRAP_GL_ENQUEUE(release, Release);
#endif




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

  // #endif

  // }}}


  // {{{ buffer

inline cl_mem
create_buffer_gc(cl_context ctx, cl_mem_flags flags,
                 size_t size, void *host_ptr)
{
    return retry_mem_error<cl_mem>([&] {
            return pyopencl_call_guarded(clCreateBuffer, ctx,
                                         flags, size, host_ptr);
        });
}

#if PYOPENCL_CL_VERSION >= 0x1010
inline cl_mem
create_sub_buffer_gc(cl_mem buffer, cl_mem_flags flags,
                     cl_buffer_create_type bct, const void *buffer_create_info)
{
    return retry_mem_error<cl_mem>([&] {
            return pyopencl_call_guarded(clCreateSubBuffer, buffer, flags,
                                         bct, buffer_create_info);
        });
}
#endif

class buffer;
static inline buffer *new_buffer(cl_mem mem, void *buff=0);

class buffer : public memory_object {
public:
    PYOPENCL_DEF_GET_CLASS_T(BUFFER);
    buffer(cl_mem mem, bool retain, void *hostbuf=0)
        : memory_object(mem, retain, hostbuf)
    { }

#if PYOPENCL_CL_VERSION >= 0x1010
    buffer*
    get_sub_region(size_t origin, size_t size, cl_mem_flags flags) const
    {
        cl_buffer_region region = {origin, size};

        cl_mem mem = create_sub_buffer_gc(data(), flags,
                                          CL_BUFFER_CREATE_TYPE_REGION,
                                          &region);
        return new_buffer(mem);
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
static inline buffer*
new_buffer(cl_mem mem, void *buff)
{
    return pyopencl_convert_obj(buffer, clReleaseMemObject, mem, buff);
}

// {{{ buffer creation

inline buffer*
create_buffer_py(context &ctx, cl_mem_flags flags,
                 size_t size, void *py_hostbuf)
{
    void *buf = py_hostbuf;
    void *retained_buf_obj = 0;
    if (py_hostbuf != NULL) {
        if (flags & CL_MEM_USE_HOST_PTR) {
          retained_buf_obj = py_hostbuf;
        }
    }
    cl_mem mem = create_buffer_gc(ctx.data(), flags, size, buf);
    return new_buffer(mem, retained_buf_obj);
}

// }}}
// }}}


// {{{ sampler
class sampler : public noncopyable {
private:
    cl_sampler m_sampler;

public:
    PYOPENCL_DEF_GET_CLASS_T(SAMPLER);
    sampler(context const &ctx, bool normalized_coordinates,
            cl_addressing_mode am, cl_filter_mode fm)
        : m_sampler(pyopencl_call_guarded(clCreateSampler, ctx.data(),
                                          normalized_coordinates, am, fm)) {}

    sampler(cl_sampler samp, bool retain)
        : m_sampler(samp)
    {
        if (retain) {
            pyopencl_call_guarded(clRetainSampler, samp);
        }
    }

    ~sampler()
    {
        pyopencl_call_guarded_cleanup(clReleaseSampler, m_sampler);
    }

    cl_sampler data() const
    {
        return m_sampler;
    }

      PYOPENCL_EQUALITY_TESTS(sampler);

      generic_info get_info(cl_sampler_info param_name) const
      {
        switch (param_name) {
        case CL_SAMPLER_REFERENCE_COUNT:
            return pyopencl_get_int_info(cl_uint, Sampler,
                                         m_sampler, param_name);
        case CL_SAMPLER_CONTEXT:
            return pyopencl_get_opaque_info(cl_context, context,
                                            Sampler, m_sampler, param_name);
        case CL_SAMPLER_ADDRESSING_MODE:
            return pyopencl_get_int_info(cl_addressing_mode, Sampler,
                                         m_sampler, param_name);
        case CL_SAMPLER_FILTER_MODE:
            return pyopencl_get_int_info(cl_filter_mode, Sampler,
                                         m_sampler, param_name);
        case CL_SAMPLER_NORMALIZED_COORDS:
            return pyopencl_get_int_info(cl_bool, Sampler,
                                         m_sampler, param_name);

        default:
            throw error("Sampler.get_info", CL_INVALID_VALUE);
        }
      }
  };

  // }}}


  // {{{ program

  class program : public noncopyable
  {
    private:
      cl_program m_program;
      program_kind_type m_program_kind;

    public:
      PYOPENCL_DEF_GET_CLASS_T(PROGRAM);
      program(cl_program prog, bool retain, program_kind_type progkind=KND_UNKNOWN)
        : m_program(prog), m_program_kind(progkind)
      {
        if (retain)
          pyopencl_call_guarded(clRetainProgram, prog);
      }

      ~program()
      {
          pyopencl_call_guarded_cleanup(clReleaseProgram, m_program);
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

      pyopencl_buf<cl_device_id> get_info__devices()
      {
          return pyopencl_get_vec_info(cl_device_id, Program, m_program,
                                       CL_PROGRAM_DEVICES);
      }

      generic_info get_info(cl_program_info param_name) const
      {
        switch (param_name) {
          case CL_PROGRAM_CONTEXT:
              return pyopencl_get_opaque_info(cl_context, context,
                                              Program, m_program, param_name);
          case CL_PROGRAM_REFERENCE_COUNT:
          case CL_PROGRAM_NUM_DEVICES:
              return pyopencl_get_int_info(cl_uint, Program,
                                           m_program, param_name);
          case CL_PROGRAM_DEVICES:
              return pyopencl_get_opaque_array_info(
                  cl_device_id, device, Program, m_program, param_name);
          case CL_PROGRAM_SOURCE:
              return pyopencl_get_str_info(Program, m_program, param_name);
          case CL_PROGRAM_BINARY_SIZES:
              return pyopencl_get_array_info(size_t, Program, m_program,
                                             param_name);
          case CL_PROGRAM_BINARIES: {
              auto sizes = pyopencl_get_vec_info(size_t, Program, m_program,
                                                 CL_PROGRAM_BINARY_SIZES);
              pyopencl_buf<char*> result_ptrs(sizes.len());
              for (size_t i  = 0;i < sizes.len();i++) {
                  result_ptrs[i] = (char*)malloc(sizes[i]);
              }
              try {
                  pyopencl_call_guarded(clGetProgramInfo, m_program,
                                        CL_PROGRAM_BINARIES,
                                        sizes.len() * sizeof(char*),
                                        result_ptrs.get(), NULL);
              } catch (...) {
                  for (size_t i  = 0;i < sizes.len();i++) {
                      free(result_ptrs[i]);
                  }
              }
              pyopencl_buf<generic_info> gis(sizes.len());
              for (size_t i  = 0;i < sizes.len();i++) {
                  gis[i].value = result_ptrs[i];
                  gis[i].dontfree = 0;
                  gis[i].opaque_class = CLASS_NONE;
                  gis[i].type =  _copy_str(std::string("char[") +
                                           tostring(sizes[i]) + "]");
              }
              return pyopencl_convert_array_info(generic_info, gis);
          }

#if PYOPENCL_CL_VERSION >= 0x1020
          case CL_PROGRAM_NUM_KERNELS:
              return pyopencl_get_int_info(size_t, Program,
                                           m_program, param_name);
          case CL_PROGRAM_KERNEL_NAMES:
              return pyopencl_get_str_info(Program, m_program, param_name);
#endif

          default:
            throw error("Program.get_info", CL_INVALID_VALUE);
        }
      }

      generic_info get_build_info(device const &dev,
                                  cl_program_build_info param_name) const
      {
        switch (param_name) {
        case CL_PROGRAM_BUILD_STATUS:
            return pyopencl_get_int_info(cl_build_status, ProgramBuild,
                                         m_program, dev.data(), param_name);
        case CL_PROGRAM_BUILD_OPTIONS:
        case CL_PROGRAM_BUILD_LOG:
            return pyopencl_get_str_info(ProgramBuild, m_program,
                                         dev.data(), param_name);
#if PYOPENCL_CL_VERSION >= 0x1020
        case CL_PROGRAM_BINARY_TYPE:
            return pyopencl_get_int_info(cl_program_binary_type, ProgramBuild,
                                         m_program, dev.data(), param_name);
#endif
        default:
            throw error("Program.get_build_info", CL_INVALID_VALUE);
        }
      }

      void build(const char *options, cl_uint num_devices, void **ptr_devices)
      {
        // todo: this function should get a list of device instances, not raw pointers
        // pointers are for the cffi interface and should not be here
        std::vector<cl_device_id> devices(num_devices);
        for(cl_uint i = 0; i < num_devices; ++i) {
          devices[i] = static_cast<device*>(ptr_devices[i])->data();
        }
        pyopencl_call_guarded(clBuildProgram, m_program, num_devices,
                              devices.empty() ? NULL : &devices.front(),
                              options, NULL, NULL);
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

      //         PYOPENCL_CALL_GUARDED(clCompileProgram,
      //             (m_program, num_devices, devices,
      //              options.c_str(), header_names.size(),
      //              programs.empty() ? NULL : &programs.front(),
      //              header_name_ptrs.empty() ? NULL : &header_name_ptrs.front(),
      //              0, 0));
      //       }
      // #endif
  };
static inline program*
new_program(cl_program prog, program_kind_type progkind=KND_UNKNOWN)
{
    return pyopencl_convert_obj(program, clReleaseProgram, prog, progkind);
}

inline program*
create_program_with_source(context &ctx, const char *string)
{
    size_t length = strlen(string);
    cl_program result = pyopencl_call_guarded(clCreateProgramWithSource,
                                              ctx.data(), 1, &string, &length);
    return new_program(result, KND_SOURCE);
}


inline program*
create_program_with_binary(context &ctx, cl_uint num_devices,
                           void **ptr_devices, cl_uint num_binaries,
                           char **binaries, size_t *binary_sizes)
{
    std::vector<cl_device_id> devices;
    std::vector<cl_int> binary_statuses(num_devices);
    for (cl_uint i = 0; i < num_devices; ++i) {
        devices.push_back(static_cast<device*>(ptr_devices[i])->data());
    }
    cl_int status_code;
    print_call_trace("clCreateProgramWithBinary");
    cl_program result = clCreateProgramWithBinary(
        ctx.data(), num_devices,
        devices.empty( ) ? NULL : &devices.front(),
        binary_sizes,
        reinterpret_cast<const unsigned char**>(const_cast<const char**>(binaries)),
        binary_statuses.empty() ? NULL : &binary_statuses.front(),
        &status_code);

    // for (cl_uint i = 0; i < num_devices; ++i)
    //   std::cout << i << ":" << binary_statuses[i] << std::endl;

    if (status_code != CL_SUCCESS)
      throw pyopencl::error("clCreateProgramWithBinary", status_code);

    return new_program(result, KND_BINARY);
}

// }}}


  // {{{ kernel

class local_memory {
private:
    size_t m_size;
public:
    local_memory(size_t size) : m_size(size)
    {}
    size_t
    size() const
    {
        return m_size;
    }
};


class kernel : public noncopyable {
private:
    cl_kernel m_kernel;

public:
    PYOPENCL_DEF_GET_CLASS_T(KERNEL);
    kernel(cl_kernel knl, bool retain)
        : m_kernel(knl)
    {
        if (retain)
            pyopencl_call_guarded(clRetainKernel, knl);
    }

    kernel(program const &prg, const char *kernel_name)
        : m_kernel(pyopencl_call_guarded(clCreateKernel, prg.data(),
                                         kernel_name)) {}

    ~kernel()
    {
        pyopencl_call_guarded_cleanup(clReleaseKernel, m_kernel);
    }

    cl_kernel data() const
    {
        return m_kernel;
    }

    PYOPENCL_EQUALITY_TESTS(kernel);

    void set_arg_null(cl_uint arg_index)
    {
        cl_mem m = 0;
        pyopencl_call_guarded(clSetKernelArg, m_kernel, arg_index,
                              sizeof(cl_mem), &m);
    }

    void set_arg_mem(cl_uint arg_index, memory_object_holder &moh)
    {
      cl_mem m = moh.data();
      pyopencl_call_guarded(clSetKernelArg, m_kernel, arg_index,
                            sizeof(cl_mem), &m);
    }

    void set_arg_local(cl_uint arg_index, local_memory const &loc)
    {
      pyopencl_call_guarded(clSetKernelArg, m_kernel, arg_index,
                            loc.size(), NULL);
    }

    void set_arg_sampler(cl_uint arg_index, sampler const &smp)
    {
      cl_sampler s = smp.data();
      pyopencl_call_guarded(clSetKernelArg, m_kernel, arg_index,
                            sizeof(cl_sampler), &s);
    }

    void set_arg_buf(cl_uint arg_index, const void *buffer, size_t size)
    {
      pyopencl_call_guarded(clSetKernelArg, m_kernel, arg_index, size, buffer);
    }

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
      switch (param_name) {
      case CL_KERNEL_FUNCTION_NAME:
          return pyopencl_get_str_info(Kernel, m_kernel, param_name);
      case CL_KERNEL_NUM_ARGS:
      case CL_KERNEL_REFERENCE_COUNT:
          return pyopencl_get_int_info(cl_uint, Kernel, m_kernel, param_name);
      case CL_KERNEL_CONTEXT:
          return pyopencl_get_opaque_info(cl_context, context,
                                          Kernel, m_kernel, param_name);
      case CL_KERNEL_PROGRAM:
          return pyopencl_get_opaque_info(cl_program, program,
                                          Kernel, m_kernel, param_name);
#if PYOPENCL_CL_VERSION >= 0x1020
      case CL_KERNEL_ATTRIBUTES:
          return pyopencl_get_str_info(Kernel, m_kernel, param_name);
#endif
      default:
          throw error("Kernel.get_info", CL_INVALID_VALUE);
      }
    }

    generic_info get_work_group_info(cl_kernel_work_group_info param_name, device const &dev) const
    {
      switch (param_name) {
#if PYOPENCL_CL_VERSION >= 0x1010
      case CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE:
#endif
      case CL_KERNEL_WORK_GROUP_SIZE:
          return pyopencl_get_int_info(size_t, KernelWorkGroup,
                                       m_kernel, dev.data(), param_name);
      case CL_KERNEL_COMPILE_WORK_GROUP_SIZE:
          return pyopencl_get_array_info(size_t, KernelWorkGroup,
                                         m_kernel, dev.data(), param_name);
      case CL_KERNEL_LOCAL_MEM_SIZE:
#if PYOPENCL_CL_VERSION >= 0x1010
      case CL_KERNEL_PRIVATE_MEM_SIZE:
#endif
          return pyopencl_get_int_info(cl_ulong, KernelWorkGroup,
                                       m_kernel, dev.data(), param_name);

      default:
          throw error("Kernel.get_work_group_info", CL_INVALID_VALUE);
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

inline event*
enqueue_read_buffer(command_queue &cq, memory_object_holder &mem, void *buffer,
                    size_t size, size_t device_offset, void **wait_for,
                    uint32_t num_wait_for, bool is_blocking)
{
    PYOPENCL_PARSE_WAIT_FOR;

    cl_event evt;
    retry_mem_error<void>([&] {
            pyopencl_call_guarded(clEnqueueReadBuffer, cq.data(), mem.data(),
                                  cast_bool(is_blocking), device_offset, size,
                                  buffer, PYOPENCL_WAITLIST_ARGS, &evt);
        });
    return new_event(evt);
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
        pyopencl_call_guarded(clGetMemObjectInfo, src.data(), CL_MEM_SIZE,
                              sizeof(byte_count), &byte_count_src, NULL);
        pyopencl_call_guarded(clGetMemObjectInfo, src.data(), CL_MEM_SIZE,
                              sizeof(byte_count), &byte_count_dst, NULL);
        byte_count = std::min(byte_count_src, byte_count_dst);
      }

    cl_event evt;
    retry_mem_error<void>([&] {
            pyopencl_call_guarded(clEnqueueCopyBuffer, cq.data(), src.data(),
                                  dst.data(), src_offset, dst_offset,
                                  byte_count, PYOPENCL_WAITLIST_ARGS, &evt);
        });
    return new_event(evt);
  }

inline event*
enqueue_write_buffer(command_queue &cq,memory_object_holder &mem,
                     const void *buffer, size_t size, size_t device_offset,
                     void **wait_for, uint32_t num_wait_for, bool is_blocking)
{
    PYOPENCL_PARSE_WAIT_FOR;

    cl_event evt;
    retry_mem_error<void>([&] {
            pyopencl_call_guarded(clEnqueueWriteBuffer, cq.data(), mem.data(),
                                  cast_bool(is_blocking), device_offset,
                                  size, buffer, PYOPENCL_WAITLIST_ARGS, &evt);
        });
    return new_event(evt);
    //PYOPENCL_RETURN_NEW_NANNY_EVENT(evt, buffer);
}

// }}}

inline event*
enqueue_nd_range_kernel(command_queue &cq, kernel &knl, cl_uint work_dim,
                        const size_t *global_work_offset,
                        const size_t *global_work_size,
                        const size_t *local_work_size,
                        void **wait_for, uint32_t num_wait_for)
{
    PYOPENCL_PARSE_WAIT_FOR;

    cl_event evt;
    retry_mem_error<void>([&] {
            pyopencl_call_guarded(clEnqueueNDRangeKernel, cq.data(),
                                  knl.data(), work_dim, global_work_offset,
                                  global_work_size, local_work_size,
                                  PYOPENCL_WAITLIST_ARGS, &evt);
        });
    return new_event(evt);
}

#if PYOPENCL_CL_VERSION >= 0x1020
  inline
  event *enqueue_marker_with_wait_list(command_queue &cq,
    void **wait_for, uint32_t num_wait_for)
  {
    PYOPENCL_PARSE_WAIT_FOR;
    cl_event evt;

    pyopencl_call_guarded(clEnqueueMarkerWithWaitList, cq.data(),
                          PYOPENCL_WAITLIST_ARGS, &evt);

    return new_event(evt);
  }

  inline
  event *enqueue_barrier_with_wait_list(command_queue &cq,
    void **wait_for, uint32_t num_wait_for)
  {
    PYOPENCL_PARSE_WAIT_FOR;
    cl_event evt;

    pyopencl_call_guarded(clEnqueueBarrierWithWaitList, cq.data(),
                          PYOPENCL_WAITLIST_ARGS, &evt);

    return new_event(evt);
  }
#endif

inline event*
enqueue_marker(command_queue &cq)
{
    cl_event evt;
    pyopencl_call_guarded(clEnqueueMarker, cq.data(), &evt);
    return new_event(evt);
}

inline void
enqueue_barrier(command_queue &cq)
{
    pyopencl_call_guarded(clEnqueueBarrier, cq.data());
}
}


// {{{ c wrapper

void pyopencl_free_pointer(void *p)
{
  free(p);
}

void pyopencl_free_pointer_array(void **p, uint32_t size)
{
  for(uint32_t i = 0; i < size; ++i)
    pyopencl_free_pointer(p[i]);
}

void
pyopencl_set_gc(int (*func)())
{
    if (!func)
        func = pyopencl::dummy_python_gc;
    pyopencl::python_gc = func;
}

::error*
get_platforms(void **ptr_platforms, uint32_t *num_platforms)
{
    return pyopencl::c_handle_error([&] {
            *num_platforms = 0;
            pyopencl_call_guarded(clGetPlatformIDs, 0, NULL, num_platforms);
            pyopencl_buf<cl_platform_id> platforms(*num_platforms);
            pyopencl_call_guarded(clGetPlatformIDs, *num_platforms,
                                  platforms.get(), num_platforms);
            pyopencl_buf<pyopencl::platform*> _ptr_platforms(*num_platforms);
            for (size_t i = 0;i < platforms.len();i++) {
                _ptr_platforms[i] = new pyopencl::platform(platforms[i]);
            }
            *ptr_platforms = _ptr_platforms.release();
        });
}


::error*
platform__get_devices(void *ptr_platform, void **ptr_devices,
                      uint32_t *num_devices, cl_device_type devtype)
{
    return pyopencl::c_handle_error([&] {
            auto devices = static_cast<pyopencl::platform*>(ptr_platform)
                ->get_devices(devtype);
            *num_devices = devices.len();
            pyopencl_buf<pyopencl::device*> _ptr_devices(*num_devices);
            for(size_t i = 0; i < devices.len();i++) {
                _ptr_devices[i] = new pyopencl::device(devices[i]);
            }
            *ptr_devices = _ptr_devices.release();
        });
}


::error*
_create_context(void **ptr_ctx, cl_context_properties *properties,
                cl_uint num_devices, void **ptr_devices)
{
    return pyopencl::c_handle_error([&] {
            pyopencl_buf<cl_device_id> devices(num_devices);
            for(size_t i = 0;i < num_devices;i++) {
                devices[i] = static_cast<pyopencl::device*>(
                    ptr_devices[i])->data();
            }
            *ptr_ctx = new pyopencl::context(
                pyopencl_call_guarded(clCreateContext, properties,
                                      num_devices, devices.get(),
                                      nullptr, nullptr), false);
        });
}


::error*
_create_command_queue(void **ptr_command_queue, void *ptr_context,
                      void *ptr_device, cl_command_queue_properties properties)
{
    pyopencl::context *ctx = static_cast<pyopencl::context*>(ptr_context);
    pyopencl::device *dev = static_cast<pyopencl::device*>(ptr_device);
    return pyopencl::c_handle_error([&] {
            *ptr_command_queue = new pyopencl::command_queue(
                *ctx, dev, properties);
        });
}


::error*
_create_buffer(void **ptr_buffer, void *ptr_context, cl_mem_flags flags,
               size_t size, void *hostbuf)
{
  pyopencl::context *ctx = static_cast<pyopencl::context*>(ptr_context);
  return pyopencl::c_handle_error([&] {
          *ptr_buffer = create_buffer_py(*ctx, flags, size, hostbuf);
      });
}

// {{{ program

::error*
_create_program_with_source(void **ptr_program, void *ptr_context,
                            const char *src)
{
    pyopencl::context *ctx = static_cast<pyopencl::context*>(ptr_context);
    return pyopencl::c_handle_error([&] {
            *ptr_program = create_program_with_source(*ctx, src);
        });
}

::error*
_create_program_with_binary(
    void **ptr_program, void *ptr_context, cl_uint num_devices,
    void **ptr_devices, cl_uint num_binaries, char **binaries,
    size_t *binary_sizes)
{
    pyopencl::context *ctx = static_cast<pyopencl::context*>(ptr_context);
    return pyopencl::c_handle_error([&] {
            *ptr_program = create_program_with_binary(
                *ctx, num_devices, ptr_devices, num_binaries,
                reinterpret_cast<char**>(binaries), binary_sizes);
        });
}

::error*
program__build(void *ptr_program, const char *options, cl_uint num_devices,
               void **ptr_devices)
{
    return pyopencl::c_handle_error([&] {
            static_cast<pyopencl::program*>(ptr_program)->build(
                options, num_devices, ptr_devices);
        });
}

::error*
program__kind(void *ptr_program, int *kind)
{
    return pyopencl::c_handle_error([&] {
            *kind = static_cast<pyopencl::program*>(ptr_program)->kind();
        });
}

::error*
program__get_build_info(void *ptr_program, void *ptr_device,
                        cl_program_build_info param, generic_info *out)
{
    return pyopencl::c_handle_error([&] {
            *out = static_cast<pyopencl::program*>(ptr_program)
                ->get_build_info(
                    *static_cast<pyopencl::device*>(ptr_device), param);
        });
}

// }}}

::error*
_create_sampler(void **ptr_sampler, void *ptr_context,
                int normalized_coordinates, cl_addressing_mode am,
                cl_filter_mode fm)
{
    return pyopencl::c_handle_error([&] {
            *ptr_sampler = new pyopencl::sampler(
                *static_cast<pyopencl::context*>(ptr_context),
                (bool)normalized_coordinates, am, fm);
        });
}

// {{{ event

::error*
event__get_profiling_info(void *ptr, cl_profiling_info param, generic_info *out)
{
    return pyopencl::c_handle_error([&] {
            *out = static_cast<pyopencl::event*>(ptr)
                ->get_profiling_info(param);
        });
}

::error*
event__wait(void *ptr)
{
    return pyopencl::c_handle_error([&] {
            static_cast<pyopencl::event*>(ptr)->wait();
        });
}

// }}}


// {{{ kernel

::error*
_create_kernel(void **ptr_kernel, void *ptr_program, const char *name)
{
    pyopencl::program *prg = static_cast<pyopencl::program*>(ptr_program);
    return pyopencl::c_handle_error([&] {
            *ptr_kernel = new pyopencl::kernel(*prg, name);
        });
}

::error*
kernel__set_arg_null(void *ptr_kernel, cl_uint arg_index)
{
    return pyopencl::c_handle_error([&] {
            static_cast<pyopencl::kernel*>(ptr_kernel)->set_arg_null(arg_index);
        });
}

::error*
kernel__set_arg_mem(void *ptr_kernel, cl_uint arg_index, void *ptr_mem)
{
    pyopencl::memory_object_holder *mem =
        static_cast<pyopencl::memory_object_holder*>(ptr_mem);
    return pyopencl::c_handle_error([&] {
            static_cast<pyopencl::kernel*>(ptr_kernel)
                ->set_arg_mem(arg_index, *mem);
        });
}

::error*
kernel__set_arg_sampler(void *ptr_kernel, cl_uint arg_index, void *ptr_sampler)
{
    pyopencl::sampler *sampler = static_cast<pyopencl::sampler*>(ptr_sampler);
    return pyopencl::c_handle_error([&] {
            static_cast<pyopencl::kernel*>(ptr_kernel)
                ->set_arg_sampler(arg_index, *sampler);
        });
}

::error*
kernel__set_arg_buf(void *ptr_kernel, cl_uint arg_index,
                    const void *buffer, size_t size)
{
    return pyopencl::c_handle_error([&] {
            static_cast<pyopencl::kernel*>(ptr_kernel)
                ->set_arg_buf(arg_index, buffer, size);
        });
}


::error*
kernel__get_work_group_info(void *ptr, cl_kernel_work_group_info param,
                            void *ptr_device, generic_info *out)
{
    return pyopencl::c_handle_error([&] {
            *out = static_cast<pyopencl::kernel*>(ptr)
                ->get_work_group_info(param, *static_cast<pyopencl::device*>(
                                          ptr_device));
        });
}

// }}}


// {{{ image

::error*
_get_supported_image_formats(void *ptr_context, cl_mem_flags flags,
                             cl_mem_object_type image_type, generic_info *out)
{
    return pyopencl::c_handle_error([&] {
            *out = get_supported_image_formats(
                *static_cast<pyopencl::context*>(ptr_context),
                flags, image_type);
        });
}

error *_create_image_2d(
    void **ptr_image, void *ptr_context, cl_mem_flags flags,
    cl_image_format *fmt, size_t width, size_t height, size_t pitch,
    void *ptr_buffer, size_t size)
{
    return pyopencl::c_handle_error([&] {
            *ptr_image = create_image_2d(
                *static_cast<pyopencl::context*>(ptr_context), flags, *fmt,
                width, height, pitch, ptr_buffer, size);
        });
}

error *_create_image_3d(
    void **ptr_image, void *ptr_context, cl_mem_flags flags,
    cl_image_format *fmt, size_t width, size_t height, size_t depth,
    size_t pitch_x, size_t pitch_y, void *ptr_buffer, size_t size)
{
    return pyopencl::c_handle_error([&] {
            *ptr_image = create_image_3d(
                *static_cast<pyopencl::context*>(ptr_context), flags, *fmt,
                width, height, depth, pitch_x, pitch_y, ptr_buffer, size);
        });
}

::error*
image__get_image_info(void *ptr, cl_image_info param, generic_info *out)
{
    return pyopencl::c_handle_error([&] {
            *out = static_cast<pyopencl::image*>(ptr)->get_image_info(param);
        });
}

// }}}

::error *_enqueue_nd_range_kernel(
    void **ptr_event, void *ptr_command_queue, void *ptr_kernel,
    cl_uint work_dim, const size_t *global_work_offset,
    const size_t *global_work_size, const size_t *local_work_size,
    void **wait_for, uint32_t num_wait_for)
{
    return pyopencl::c_handle_error([&] {
            *ptr_event = enqueue_nd_range_kernel(
                *static_cast<pyopencl::command_queue*>(ptr_command_queue),
                *static_cast<pyopencl::kernel*>(ptr_kernel),
                work_dim, global_work_offset,
                global_work_size, local_work_size,
                wait_for, num_wait_for);
        });
}

#if PYOPENCL_CL_VERSION >= 0x1020
::error *_enqueue_marker_with_wait_list(
    void **ptr_event, void *ptr_command_queue, void **wait_for,
    uint32_t num_wait_for)
{
    return pyopencl::c_handle_error([&] {
            *ptr_event = enqueue_marker_with_wait_list(
                *static_cast<pyopencl::command_queue*>(ptr_command_queue),
                wait_for, num_wait_for);
        });
}

::error*
_enqueue_barrier_with_wait_list(void **ptr_event, void *ptr_command_queue,
                                void **wait_for, uint32_t num_wait_for)
{
    return pyopencl::c_handle_error([&] {
            *ptr_event = enqueue_barrier_with_wait_list(
                *static_cast<pyopencl::command_queue*>(ptr_command_queue),
                wait_for, num_wait_for);
        });
}
#endif

::error *_enqueue_marker(void **ptr_event, void *ptr_command_queue)
{
    return pyopencl::c_handle_error([&] {
            *ptr_event = enqueue_marker(
                *static_cast<pyopencl::command_queue*>(ptr_command_queue));
        });
}

::error *_enqueue_barrier(void *ptr_command_queue)
{
    return pyopencl::c_handle_error([&] {
            enqueue_barrier(*static_cast<pyopencl::command_queue*>(
                                ptr_command_queue));
        });
}

// {{{ transfer enqueues

::error *_enqueue_read_buffer(
    void **ptr_event, void *ptr_command_queue, void *ptr_memory_object_holder,
    void *buffer, size_t size, size_t device_offset, void **wait_for,
    uint32_t num_wait_for, int is_blocking)
{
    return pyopencl::c_handle_error([&] {
            *ptr_event = enqueue_read_buffer(
                *static_cast<pyopencl::command_queue*>(ptr_command_queue),
                *static_cast<pyopencl::memory_object_holder*>(
                    ptr_memory_object_holder),
                buffer, size, device_offset, wait_for,
                num_wait_for, (bool)is_blocking);
        });
}

::error *_enqueue_write_buffer(
    void **ptr_event, void *ptr_command_queue, void *ptr_mem,
    const void *buffer, size_t size, size_t device_offset, void **wait_for,
    uint32_t num_wait_for, int is_blocking)
{
    return pyopencl::c_handle_error([&] {
            *ptr_event = enqueue_write_buffer(
                *static_cast<pyopencl::command_queue*>(ptr_command_queue),
                *static_cast<pyopencl::memory_object_holder*>(ptr_mem),
                buffer, size, device_offset, wait_for,
                num_wait_for, (bool)is_blocking);
        });
}


::error*
_enqueue_copy_buffer(void **ptr_event, void *ptr_command_queue, void *ptr_src,
                     void *ptr_dst, ptrdiff_t byte_count, size_t src_offset,
                     size_t dst_offset, void **wait_for, uint32_t num_wait_for)
{
    return pyopencl::c_handle_error([&] {
            *ptr_event = enqueue_copy_buffer(
                *static_cast<pyopencl::command_queue*>(ptr_command_queue),
                *static_cast<pyopencl::memory_object_holder*>(ptr_src),
                *static_cast<pyopencl::memory_object_holder*>(ptr_dst),
                byte_count, src_offset, dst_offset, wait_for, num_wait_for);
        });
}


::error*
_enqueue_read_image(void **ptr_event, void *ptr_command_queue, void *ptr_mem,
                    size_t *origin, size_t *region, void *buffer, size_t size,
                    size_t row_pitch, size_t slice_pitch, void **wait_for,
                    uint32_t num_wait_for, int is_blocking)
{
    return pyopencl::c_handle_error([&] {
            *ptr_event = enqueue_read_image(
                *static_cast<pyopencl::command_queue*>(ptr_command_queue),
                *static_cast<pyopencl::image*>(ptr_mem),
                origin, region, buffer, size, row_pitch, slice_pitch,
                wait_for, num_wait_for, (bool)is_blocking);
        });
}

// }}}

::error*
_command_queue_finish(void *ptr_command_queue)
{
    return pyopencl::c_handle_error([&] {
            static_cast<pyopencl::command_queue*>(ptr_command_queue)->finish();
        });
}

::error*
_command_queue_flush(void *ptr_command_queue)
{
    return pyopencl::c_handle_error([&] {
            static_cast<pyopencl::command_queue*>(ptr_command_queue)->flush();
        });
}

intptr_t
_int_ptr(void* ptr, class_t class_)
{
#define INT_PTR(CLSU, CLS) return (intptr_t)(static_cast<pyopencl::CLS*>(ptr)->data());
  SWITCHCLASS(INT_PTR);
}

void* _from_int_ptr(void **ptr_out, intptr_t int_ptr_value, class_t class_)
{
#define FROM_INT_PTR(CLSU, CLS)                                         \
    *ptr_out = new pyopencl::CLS((PYOPENCL_CL_##CLSU)int_ptr_value,     \
                                 /* retain */ true);

    return pyopencl::c_handle_error([&] {
            SWITCHCLASS(FROM_INT_PTR);
        });
}

long _hash(void *ptr, class_t class_)
{
#define HASH(CLSU, CLS) return static_cast<pyopencl::CLS*>(ptr)->hash();
  SWITCHCLASS(HASH);
}

::error*
_get_info(void *ptr, class_t class_, cl_uint param, generic_info *out)
{
#define GET_INFO(CLSU, CLS)                                   \
    *out = static_cast<pyopencl::CLS*>(ptr)->get_info(param);

    return pyopencl::c_handle_error([&] {
            SWITCHCLASS(GET_INFO);
        });
}

void _delete(void *ptr, class_t class_) {
#define DELETE(CLSU, CLS) delete static_cast<pyopencl::CLS*>(ptr);
  SWITCHCLASS(DELETE);
}

::error*
_release_memobj(void *ptr)
{
    return pyopencl::c_handle_error([&] {
            static_cast<pyopencl::memory_object*>(ptr)->release();
        });
}

int pyopencl_get_cl_version(void)
{
    return PYOPENCL_CL_VERSION;
}

// {{{ gl interop

int pyopencl_have_gl()
{
#ifdef HAVE_GL
    return 1;
#else
    return 0;
#endif
}

#ifdef HAVE_GL
error *_create_from_gl_buffer(
    void **ptr, void *ptr_context, cl_mem_flags flags, GLuint bufobj)
{
    pyopencl::context *ctx = static_cast<pyopencl::context*>(ptr_context);
    return pyopencl::c_handle_error([&] {
            *ptr = create_from_gl_buffer(*ctx, flags, bufobj);
        });
}

error *_create_from_gl_renderbuffer(
    void **ptr, void *ptr_context, cl_mem_flags flags, GLuint bufobj)
{
    pyopencl::context *ctx = static_cast<pyopencl::context*>(ptr_context);
    return pyopencl::c_handle_error([&] {
            *ptr = create_from_gl_renderbuffer(*ctx, flags, bufobj);
        });
}

::error *_enqueue_acquire_gl_objects(
    void **ptr_event, void *ptr_command_queue,
    void **ptr_mem_objects, uint32_t num_mem_objects, void **wait_for,
    uint32_t num_wait_for)
{
    return pyopencl::c_handle_error([&] {
            *ptr_event = enqueue_acquire_gl_objects(
                *static_cast<pyopencl::command_queue*>(ptr_command_queue),
                ptr_mem_objects, num_mem_objects, wait_for, num_wait_for);
        });
}

::error*
_enqueue_release_gl_objects(void **ptr_event, void *ptr_command_queue,
                            void **ptr_mem_objects, uint32_t num_mem_objects,
                            void **wait_for, uint32_t num_wait_for)
{
    return pyopencl::c_handle_error([&] {
            *ptr_event = enqueue_release_gl_objects(
                *static_cast<pyopencl::command_queue*>(ptr_command_queue),
                ptr_mem_objects, num_mem_objects, wait_for, num_wait_for);
        });
}
#endif /* HAVE_GL */

// }}}

// }}}

// vim: foldmethod=marker
