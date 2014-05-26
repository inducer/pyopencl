#include "error.h"
#include "utils.h"
#include "async.h"

#include <stdlib.h>

// {{{ extension function pointers

#if PYOPENCL_CL_VERSION >= 0x1020
template<typename T>
PYOPENCL_USE_RESULT static inline T
pyopencl_get_ext_fun(cl_platform_id plat, const char *name, const char *err)
{
    T func = (T)clGetExtensionFunctionAddressForPlatform(plat, name);
    if (!func) {
        throw pyopencl::clerror(name, CL_INVALID_VALUE, err);
    }
    return func;
}
#else
template<typename T>
PYOPENCL_USE_RESULT static inline T
pyopencl_get_ext_fun(cl_platform_id, const char *name, const char *err)
{
    T func = (T)clGetExtensionFunctionAddress(name);
    if (!func) {
        throw pyopencl::clerror(name, CL_INVALID_VALUE, err);
    }
    return func;
}
#endif
#define pyopencl_get_ext_fun(plat, name)                                \
    pyopencl_get_ext_fun<name##_fn>(plat, #name, #name " not available")

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
  default: throw pyopencl::clerror("unknown class", CL_INVALID_VALUE);    \
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

namespace pyopencl {

// {{{ platform

class platform : public clobj<cl_platform_id> {
public:
    using clobj::clobj;
    PYOPENCL_DEF_GET_CLASS_T(PLATFORM);
    generic_info
    get_info(cl_uint param_name) const
    {
        switch ((cl_platform_info)param_name) {
        case CL_PLATFORM_PROFILE:
        case CL_PLATFORM_VERSION:
        case CL_PLATFORM_NAME:
        case CL_PLATFORM_VENDOR:
#if !(defined(CL_PLATFORM_NVIDIA) && CL_PLATFORM_NVIDIA == 0x3001)
        case CL_PLATFORM_EXTENSIONS:
#endif
            return pyopencl_get_str_info(Platform, data(), param_name);

        default:
            throw clerror("Platform.get_info", CL_INVALID_VALUE);
        }
    }

    pyopencl_buf<cl_device_id> get_devices(cl_device_type devtype) const;
};


PYOPENCL_USE_RESULT inline pyopencl_buf<cl_device_id>
platform::get_devices(cl_device_type devtype) const
{
    cl_uint num_devices = 0;
    try {
        pyopencl_call_guarded(clGetDeviceIDs,
                              data(), devtype, 0, NULL, &num_devices);
    } catch (const clerror &e) {
        if (e.code() != CL_DEVICE_NOT_FOUND)
            throw e;
        num_devices = 0;
    }
    pyopencl_buf<cl_device_id> devices(num_devices);
    if (num_devices == 0)
        return devices;
    pyopencl_call_guarded(clGetDeviceIDs, data(), devtype, num_devices,
                          devices.get(), &num_devices);
    return devices;
}

// }}}


// {{{ device

class device : public clobj<cl_device_id> {
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
    reference_type_t m_ref_type;

public:
    device(cl_device_id did, bool retain=false,
           reference_type_t ref_type=REF_NOT_OWNABLE)
        : clobj(did), m_ref_type(ref_type)
    {
        if (retain && ref_type != REF_NOT_OWNABLE) {
            if (false) {
            }
#if (defined(cl_ext_device_fission) && defined(PYOPENCL_USE_DEVICE_FISSION))
            else if (ref_type == REF_FISSION_EXT) {
#if PYOPENCL_CL_VERSION >= 0x1020
                cl_platform_id plat;
                pyopencl_call_guarded(clGetDeviceInfo, data(),
                                      CL_DEVICE_PLATFORM, sizeof(plat),
                                      &plat, NULL);
#endif
                pyopencl_call_guarded(
                    pyopencl_get_ext_fun(plat, clRetainDeviceEXT), did);
            }
#endif
#if PYOPENCL_CL_VERSION >= 0x1020
            else if (ref_type == REF_CL_1_2) {
                pyopencl_call_guarded(clRetainDevice, did);
            }
#endif

            else {
                throw clerror("Device", CL_INVALID_VALUE,
                              "cannot own references to devices when device "
                              "fission or CL 1.2 is not available");
            }
        }
    }

    ~device()
    {
        if (false) {
        }
#if defined(cl_ext_device_fission) && defined(PYOPENCL_USE_DEVICE_FISSION)
        else if (m_ref_type == REF_FISSION_EXT) {
#if PYOPENCL_CL_VERSION >= 0x1020
            cl_platform_id plat;
            pyopencl_call_guarded(clGetDeviceInfo, data(), CL_DEVICE_PLATFORM,
                                  sizeof(plat), &plat, NULL);
#endif
            pyopencl_call_guarded_cleanup(
                pyopencl_get_ext_fun(plat, clReleaseDeviceEXT), data());
        }
#endif
#if PYOPENCL_CL_VERSION >= 0x1020
        else if (m_ref_type == REF_CL_1_2) {
            pyopencl_call_guarded(clReleaseDevice, data());
        }
#endif
    }
    generic_info
    get_info(cl_uint param_name) const
    {
#define DEV_GET_INT_INF(TYPE)                                           \
        pyopencl_get_int_info(TYPE, Device, data(), param_name)

        switch ((cl_device_info)param_name) {
        case CL_DEVICE_TYPE:
            return DEV_GET_INT_INF(cl_device_type);
        case CL_DEVICE_MAX_WORK_GROUP_SIZE:
            return DEV_GET_INT_INF(size_t);
        case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:
        case CL_DEVICE_MAX_COMPUTE_UNITS:
        case CL_DEVICE_VENDOR_ID:
            return DEV_GET_INT_INF(cl_uint);

        case CL_DEVICE_MAX_WORK_ITEM_SIZES:
            return pyopencl_get_array_info(size_t, Device, data(), param_name);

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
            return pyopencl_get_str_info(Device, data(), param_name);

        case CL_DEVICE_PLATFORM:
            return pyopencl_get_opaque_info(cl_platform_id, platform,
                                            Device, data(), param_name);
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
            return pyopencl_get_str_info(Device, data(), param_name);
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
                                            Device, data(), param_name);
        case CL_DEVICE_PARTITION_TYPES_EXT:
        case CL_DEVICE_AFFINITY_DOMAINS_EXT:
        case CL_DEVICE_PARTITION_STYLE_EXT:
            return pyopencl_get_array_info(cl_device_partition_property_ext,
                                           Device, data(), param_name);
        case CL_DEVICE_REFERENCE_COUNT_EXT:
            return DEV_GET_INT_INF(cl_uint);
#endif
#if PYOPENCL_CL_VERSION >= 0x1020
        case CL_DEVICE_LINKER_AVAILABLE:
            return DEV_GET_INT_INF(cl_bool);
        case CL_DEVICE_BUILT_IN_KERNELS:
            return pyopencl_get_str_info(Device, data(), param_name);
        case CL_DEVICE_IMAGE_MAX_BUFFER_SIZE:
        case CL_DEVICE_IMAGE_MAX_ARRAY_SIZE:
            return DEV_GET_INT_INF(size_t);
        case CL_DEVICE_PARENT_DEVICE:
            return pyopencl_get_opaque_info(cl_device_id, device,
                                            Device, data(), param_name);
        case CL_DEVICE_PARTITION_MAX_SUB_DEVICES:
            return DEV_GET_INT_INF(cl_uint);
        case CL_DEVICE_PARTITION_TYPE:
        case CL_DEVICE_PARTITION_PROPERTIES:
            return pyopencl_get_array_info(cl_device_partition_property,
                                           Device, data(), param_name);
        case CL_DEVICE_PARTITION_AFFINITY_DOMAIN:
            return pyopencl_get_array_info(cl_device_affinity_domain,
                                           Device, data(), param_name);
        case CL_DEVICE_REFERENCE_COUNT:
            return DEV_GET_INT_INF(cl_uint);
        case CL_DEVICE_PREFERRED_INTEROP_USER_SYNC:
        case CL_DEVICE_PRINTF_BUFFER_SIZE:
            return DEV_GET_INT_INF(cl_bool);
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
            return pyopencl_get_str_info(Device, data(), param_name);
#endif
#ifdef CL_DEVICE_GLOBAL_FREE_MEMORY_AMD
        case CL_DEVICE_GLOBAL_FREE_MEMORY_AMD:
            return pyopencl_get_array_info(size_t, Device,
                                           data(), param_name);
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
            throw clerror("Device.get_info", CL_INVALID_VALUE);
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

class context : public clobj<cl_context> {
public:
    PYOPENCL_DEF_GET_CLASS_T(CONTEXT);
    context(cl_context ctx, bool retain)
        : clobj(ctx)
    {
        if (retain) {
            pyopencl_call_guarded(clRetainContext, ctx);
        }
    }
    ~context()
    {
        pyopencl_call_guarded_cleanup(clReleaseContext, data());
    }
    generic_info
    get_info(cl_uint param_name) const
    {
        switch ((cl_context_info)param_name) {
        case CL_CONTEXT_REFERENCE_COUNT:
            return pyopencl_get_int_info(cl_uint, Context,
                                         data(), param_name);
        case CL_CONTEXT_DEVICES:
            return pyopencl_get_opaque_array_info(
                cl_device_id, device, Context, data(), param_name);
        case CL_CONTEXT_PROPERTIES: {
            auto result = pyopencl_get_vec_info(
                cl_context_properties, Context, data(), param_name);
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
                    throw clerror("Context.get_info", CL_INVALID_VALUE,
                                  "unknown context_property key encountered");
                }
            }
            py_result.resize(i);
            return pyopencl_convert_array_info(generic_info, py_result);
        }

#if PYOPENCL_CL_VERSION >= 0x1010
        case CL_CONTEXT_NUM_DEVICES:
            return pyopencl_get_int_info(cl_uint, Context,
                                         data(), param_name);
#endif

        default:
            throw clerror("Context.get_info", CL_INVALID_VALUE);
        }
    }
    PYOPENCL_USE_RESULT inline generic_info
    get_supported_image_formats(cl_mem_flags flags,
                                cl_mem_object_type image_type) const
    {
        cl_uint num_image_formats;
        pyopencl_call_guarded(clGetSupportedImageFormats, data(), flags,
                              image_type, 0, NULL, &num_image_formats);
        pyopencl_buf<cl_image_format> formats(num_image_formats);
        pyopencl_call_guarded(clGetSupportedImageFormats, data(), flags,
                              image_type, formats.len(), formats.get(), NULL);
        return pyopencl_convert_array_info(cl_image_format, formats);
    }
};

// }}}


// {{{ command_queue

class command_queue : public clobj<cl_command_queue> {
private:
    static cl_command_queue
    create_cl_command_queue(const context *ctx, const device *py_dev,
                            cl_command_queue_properties props)
    {
        cl_device_id dev;
        if (py_dev) {
            dev = py_dev->data();
        } else {
            auto devs = pyopencl_get_vec_info(cl_device_id, Context,
                                              ctx->data(), CL_CONTEXT_DEVICES);
            if (devs.len() == 0) {
                throw clerror("CommandQueue", CL_INVALID_VALUE,
                              "context doesn't have any devices? -- "
                              "don't know which one to default to");
            }
            dev = devs[0];
        }
        return pyopencl_call_guarded(clCreateCommandQueue,
                                     ctx->data(), dev, props);
    }
public:
    PYOPENCL_DEF_GET_CLASS_T(COMMAND_QUEUE);
    command_queue(cl_command_queue q, bool retain)
        : clobj(q)
    {
        if (retain) {
            pyopencl_call_guarded(clRetainCommandQueue, q);
        }
    }
    command_queue(const context *ctx, const device *py_dev=0,
                  cl_command_queue_properties props=0)
        : clobj(create_cl_command_queue(ctx, py_dev, props))
    {}
    command_queue(const command_queue &queue)
        : command_queue(queue.data(), true)
    {}
    ~command_queue()
    {
        pyopencl_call_guarded_cleanup(clReleaseCommandQueue, data());
    }

    generic_info
    get_info(cl_uint param_name) const
    {
        switch ((cl_command_queue_info)param_name) {
        case CL_QUEUE_CONTEXT:
            return pyopencl_get_opaque_info(cl_context, context,
                                            CommandQueue, data(), param_name);
        case CL_QUEUE_DEVICE:
            return pyopencl_get_opaque_info(cl_device_id, device,
                                            CommandQueue, data(), param_name);
        case CL_QUEUE_REFERENCE_COUNT:
            return pyopencl_get_int_info(cl_uint, CommandQueue,
                                         data(), param_name);
        case CL_QUEUE_PROPERTIES:
            return pyopencl_get_int_info(cl_command_queue_properties,
                                         CommandQueue, data(), param_name);
        default:
            throw clerror("CommandQueue.get_info", CL_INVALID_VALUE);
        }
    }

    PYOPENCL_USE_RESULT std::unique_ptr<context>
    get_context() const
    {
        cl_context param_value;
        pyopencl_call_guarded(clGetCommandQueueInfo, data(), CL_QUEUE_CONTEXT,
                              sizeof(param_value), &param_value, NULL);
        return std::unique_ptr<context>(
            new context(param_value, /*retain*/ true));
    }

#if PYOPENCL_CL_VERSION < 0x1010
    cl_command_queue_properties
    set_property(cl_command_queue_properties prop, bool enable) const
    {
        cl_command_queue_properties old_prop;
        pyopencl_call_guarded(clSetCommandQueueProperty, data(), prop,
                              cast_bool(enable), &old_prop);
        return old_prop;
    }
#endif
    void
    flush() const
    {
        pyopencl_call_guarded(clFlush, data());
    }
    void
    finish() const
    {
        pyopencl_call_guarded(clFinish, data());
    }
};
// }}}


// {{{ event

class event : public clobj<cl_event> {
private:
#if PYOPENCL_CL_VERSION >= 0x1010
    class event_callback {
    private:
        std::function<void(cl_int)> m_func;
    public:
        event_callback(const std::function<void(cl_int)> &func)
            : m_func(func)
        {}
        static void
        cl_call_and_free(cl_event, cl_int status, void *data)
        {
            auto cb = static_cast<event_callback*>(data);
            auto func = cb->m_func;
            try {
                call_async([func, status] {func(status);});
            } catch (...) {
            }
            delete cb;
        }
    };
#endif
public:
    PYOPENCL_DEF_GET_CLASS_T(EVENT);
    event(cl_event event, bool retain)
        : clobj(event)
    {
        if (retain) {
            pyopencl_call_guarded(clRetainEvent, event);
        }
    }
    ~event()
    {
        pyopencl_call_guarded_cleanup(clReleaseEvent, data());
    }
    generic_info
    get_info(cl_uint param_name) const
    {
        switch ((cl_event_info)param_name) {
        case CL_EVENT_COMMAND_QUEUE:
            return pyopencl_get_opaque_info(cl_command_queue, command_queue,
                                            Event, data(), param_name);
        case CL_EVENT_COMMAND_TYPE:
            return pyopencl_get_int_info(cl_command_type, Event,
                                         data(), param_name);
        case CL_EVENT_COMMAND_EXECUTION_STATUS:
            return pyopencl_get_int_info(cl_int, Event, data(), param_name);
        case CL_EVENT_REFERENCE_COUNT:
            return pyopencl_get_int_info(cl_uint, Event, data(), param_name);
#if PYOPENCL_CL_VERSION >= 0x1010
        case CL_EVENT_CONTEXT:
            return pyopencl_get_opaque_info(cl_context, context,
                                            Event, data(), param_name);
#endif

        default:
            throw clerror("Event.get_info", CL_INVALID_VALUE);
        }
    }
    PYOPENCL_USE_RESULT generic_info
    get_profiling_info(cl_profiling_info param_name) const
    {
        switch (param_name) {
        case CL_PROFILING_COMMAND_QUEUED:
        case CL_PROFILING_COMMAND_SUBMIT:
        case CL_PROFILING_COMMAND_START:
        case CL_PROFILING_COMMAND_END:
            return pyopencl_get_int_info(cl_ulong, EventProfiling,
                                         data(), param_name);
        default:
            throw clerror("Event.get_profiling_info", CL_INVALID_VALUE);
        }
    }
    virtual void
    finished()
    {}
    void
    wait()
    {
        pyopencl_call_guarded(clWaitForEvents, 1, &data());
        finished();
    }
#if PYOPENCL_CL_VERSION >= 0x1010
    void
    set_callback(cl_int type, const std::function<void(cl_int)> &func)
    {
        auto cb = new event_callback(func);
        try {
            pyopencl_call_guarded(clSetEventCallback, data(), type,
                                  &event_callback::cl_call_and_free, cb);
        } catch (...) {
            delete cb;
            throw;
        }
        init_async();
    }
#endif
};
static inline event*
new_event(cl_event evt)
{
    return pyopencl_convert_obj(event, clReleaseEvent, evt);
}

class nanny_event : public event {
private:
    unsigned int m_ward;
public:
    nanny_event(cl_event evt, bool retain, void (*reffunc)(unsigned long)=0)
        : event(evt, retain), m_ward(0)
    {
        if (reffunc) {
            m_ward = next_obj_id();
            reffunc(m_ward);
        }
    }
    ~nanny_event()
    {
        if (m_ward) {
            wait();
        }
    }
    unsigned int
    get_ward() const
    {
        return m_ward;
    }
    void
    finished()
    {
        // No lock needed because multiple release is safe here.
        unsigned long ward = m_ward;
        m_ward = 0;
        python_deref(ward);
    }
};
static inline event*
new_nanny_event(cl_event evt, void (*reffunc)(unsigned long))
{
    return pyopencl_convert_obj(nanny_event, clReleaseEvent, evt, reffunc);
}

// }}}

// {{{ memory_object

//py::object create_mem_object_wrapper(cl_mem mem);

class memory_object_holder : public clobj<cl_mem> {
public:
    using clobj::clobj;
    size_t size() const
    {
        size_t param_value;
        pyopencl_call_guarded(clGetMemObjectInfo, data(), CL_MEM_SIZE,
                              sizeof(param_value), &param_value, NULL);
        return param_value;
    }
    generic_info
    get_info(cl_uint param_name) const
    {
        switch ((cl_mem_info)param_name){
        case CL_MEM_TYPE:
            return pyopencl_get_int_info(cl_mem_object_type, MemObject,
                                         data(), param_name);
        case CL_MEM_FLAGS:
            return pyopencl_get_int_info(cl_mem_flags, MemObject,
                                         data(), param_name);
        case CL_MEM_SIZE:
            return pyopencl_get_int_info(size_t, MemObject, data(), param_name);
        case CL_MEM_HOST_PTR:
            throw clerror("MemoryObject.get_info", CL_INVALID_VALUE,
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
            return pyopencl_get_int_info(size_t, MemObject, data(), param_name);
#endif

        default:
            throw clerror("MemoryObjectHolder.get_info", CL_INVALID_VALUE);
        }
    }
};

class memory_object : public memory_object_holder {
private:
    mutable volatile std::atomic_bool m_valid;
    void *m_hostbuf;
public:
    memory_object(cl_mem mem, bool retain, void *hostbuf=0)
        : memory_object_holder(mem), m_valid(true)
    {
        if (retain) {
            pyopencl_call_guarded(clRetainMemObject, mem);
        }
        if (hostbuf) {
            m_hostbuf = hostbuf;
        }
    }
    memory_object(const memory_object &mem)
        : memory_object(mem.data(), true)
    {}
    ~memory_object()
    {
        if (!m_valid.exchange(false))
            return;
        pyopencl_call_guarded_cleanup(clReleaseMemObject, data());
    }
    void
    release() const
    {
        if (!m_valid.exchange(false)) {
            throw clerror("MemoryObject.release", CL_INVALID_VALUE,
                          "trying to double-unref mem object");
        }
        pyopencl_call_guarded(clReleaseMemObject, data());
    }
    void*
    hostbuf() const
    {
        return m_hostbuf;
    }
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

class image : public memory_object {
public:
    PYOPENCL_DEF_GET_CLASS_T(IMAGE);
    image(cl_mem mem, bool retain, void *hostbuf=0)
        : memory_object(mem, retain, hostbuf)
    {}
    generic_info
    get_image_info(cl_image_info param_name) const
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
            throw clerror("Image.get_image_info", CL_INVALID_VALUE);
        }
    }
};
static inline image*
new_image(cl_mem mem, void *buff=0)
{
    return pyopencl_convert_obj(image, clReleaseMemObject, mem, buff);
}

// {{{ image creation

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
//       throw clerror("clCreateImage", status_code);

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
static inline cl_context_properties
get_apple_cgl_share_group()
{
    CGLContextObj kCGLContext = CGLGetCurrentContext();
    CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);

    return (cl_context_properties)kCGLShareGroup;
}
#endif /* __APPLE__ */

class gl_buffer : public memory_object {
public:
    PYOPENCL_DEF_GET_CLASS_T(GL_BUFFER);
    gl_buffer(cl_mem mem, bool retain, void *hostbuf=0)
        : memory_object(mem, retain, hostbuf)
    {}
};

class gl_renderbuffer : public memory_object {
public:
    PYOPENCL_DEF_GET_CLASS_T(GL_RENDERBUFFER);
    gl_renderbuffer(cl_mem mem, bool retain, void *hostbuf=0)
        : memory_object(mem, retain, hostbuf)
    {}
};

class gl_texture : public image {
  public:
    gl_texture(cl_mem mem, bool retain, void *hostbuf=0)
      : image(mem, retain, hostbuf)
    {}
    PYOPENCL_USE_RESULT generic_info
    get_gl_texture_info(cl_gl_texture_info param_name) const
    {
        switch (param_name) {
        case CL_GL_TEXTURE_TARGET:
            return pyopencl_get_int_info(GLenum, GLTexture, data(), param_name);
        case CL_GL_MIPMAP_LEVEL:
            return pyopencl_get_int_info(GLint, GLTexture, data(), param_name);
        default:
            throw clerror("MemoryObject.get_gl_texture_info", CL_INVALID_VALUE);
        }
    }
};

PYOPENCL_USE_RESULT static gl_texture*
create_from_gl_texture(context &ctx, cl_mem_flags flags, GLenum texture_target,
                       GLint miplevel, GLuint texture, unsigned dims)
{
    if (dims == 2) {
        cl_mem mem = pyopencl_call_guarded(clCreateFromGLTexture2D,
                                           ctx.data(), flags, texture_target,
                                           miplevel, texture);
        return pyopencl_convert_obj(gl_texture, clReleaseMemObject, mem);
    } else if (dims == 3) {
        cl_mem mem = pyopencl_call_guarded(clCreateFromGLTexture3D,
                                           ctx.data(), flags, texture_target,
                                           miplevel, texture);
        return pyopencl_convert_obj(gl_texture, clReleaseMemObject, mem);
    } else {
        throw clerror("Image", CL_INVALID_VALUE, "invalid dimension");
    }
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

typedef cl_int (*clEnqueueGLObjectFunc)(cl_command_queue, cl_uint,
                                        const cl_mem*, cl_uint,
                                        const cl_event*, cl_event*);

PYOPENCL_USE_RESULT static inline event*
enqueue_gl_objects(clEnqueueGLObjectFunc func, const char *name,
                   command_queue *cq, const clobj_t *mem_objects,
                   uint32_t num_mem_objects, const clobj_t *wait_for,
                   uint32_t num_wait_for)
{
    auto _wait_for = buf_from_class<event>(wait_for, num_wait_for);
    auto _mem_objs = buf_from_class<memory_object_holder>(
        mem_objects, num_mem_objects);
    cl_event evt;
    call_guarded(func, name, cq->data(), num_mem_objects, _mem_objs.get(),
                 num_wait_for, _wait_for.get(), &evt);
    return new_event(evt);
}
#define enqueue_gl_objects(what, args...)                       \
    enqueue_gl_objects(clEnqueue##what##GLObjects,              \
                       "clEnqueue" #what "GLObjects", args)

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

class buffer;
static inline buffer *new_buffer(cl_mem mem, void *buff=0);

class buffer : public memory_object {
public:
    PYOPENCL_DEF_GET_CLASS_T(BUFFER);
    buffer(cl_mem mem, bool retain, void *hostbuf=0)
        : memory_object(mem, retain, hostbuf)
    {}

#if PYOPENCL_CL_VERSION >= 0x1010
    PYOPENCL_USE_RESULT buffer*
    get_sub_region(size_t origin, size_t size, cl_mem_flags flags) const
    {
        cl_buffer_region region = {origin, size};

        auto mem = retry_mem_error<cl_mem>([&] {
                return pyopencl_call_guarded(clCreateSubBuffer, data(), flags,
                                             CL_BUFFER_CREATE_TYPE_REGION,
                                             &region);
            });
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
    //           throw clerror("Buffer.__getitem__", CL_INVALID_VALUE,
    //               "Buffer slice must have stride 1");

    //         cl_mem_flags my_flags;
    //         PYOPENCL_CALL_GUARDED(clGetMemObjectInfo,
    //             (data(), CL_MEM_FLAGS, sizeof(my_flags), &my_flags, 0));

    //         return get_sub_region(start, end, my_flags);
    //       }
#endif
};
PYOPENCL_USE_RESULT static inline buffer*
new_buffer(cl_mem mem, void *buff)
{
    return pyopencl_convert_obj(buffer, clReleaseMemObject, mem, buff);
}

// }}}

// {{{ memory_map

class memory_map : public clbase {
private:
    mutable volatile std::atomic_bool m_valid;
    command_queue m_queue;
    memory_object m_mem;
    void *m_ptr;
    size_t m_size;
public:
    memory_map(const command_queue *queue, const memory_object *mem,
               void *ptr, size_t size)
        : m_valid(true), m_queue(*queue), m_mem(*mem), m_ptr(ptr), m_size(size)
    {}
    ~memory_map()
    {
        if (!m_valid.exchange(false))
            return;
        pyopencl_call_guarded_cleanup(clEnqueueUnmapMemObject, m_queue.data(),
                                      m_mem.data(), m_ptr, 0, NULL, NULL);
    }
    event*
    release(const command_queue *queue, const clobj_t *_wait_for,
            uint32_t num_wait_for) const
    {
        if (!m_valid.exchange(false)) {
            throw clerror("MemoryMap.release", CL_INVALID_VALUE,
                          "trying to double-unref mem map");
        }
        auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
        queue = queue ? queue : &m_queue;
        cl_event evt;
        pyopencl_call_guarded(clEnqueueUnmapMemObject, queue->data(),
                              m_mem.data(), m_ptr, num_wait_for,
                              wait_for.get(), &evt);
        return new_event(evt);
    }
    intptr_t
    intptr() const
    {
        return (intptr_t)data();
    }
    generic_info
    get_info(cl_uint) const
    {
        throw clerror("MemoryMap.get_info", CL_INVALID_VALUE);
    }
    void*
    data() const
    {
        return m_valid ? m_ptr : NULL;
    }
    size_t
    size() const
    {
        return m_valid ? m_size : 0;
    }
};

// }}}

// {{{ sampler

class sampler : public clobj<cl_sampler> {
public:
    PYOPENCL_DEF_GET_CLASS_T(SAMPLER);
    sampler(const context *ctx, bool normalized_coordinates,
            cl_addressing_mode am, cl_filter_mode fm)
        : clobj(pyopencl_call_guarded(clCreateSampler, ctx->data(),
                                      normalized_coordinates, am, fm))
    {}
    sampler(cl_sampler samp, bool retain)
        : clobj(samp)
    {
        if (retain) {
            pyopencl_call_guarded(clRetainSampler, samp);
        }
    }
    ~sampler()
    {
        pyopencl_call_guarded_cleanup(clReleaseSampler, data());
    }
    generic_info
    get_info(cl_uint param_name) const
    {
        switch ((cl_sampler_info)param_name) {
        case CL_SAMPLER_REFERENCE_COUNT:
            return pyopencl_get_int_info(cl_uint, Sampler,
                                         data(), param_name);
        case CL_SAMPLER_CONTEXT:
            return pyopencl_get_opaque_info(cl_context, context,
                                            Sampler, data(), param_name);
        case CL_SAMPLER_ADDRESSING_MODE:
            return pyopencl_get_int_info(cl_addressing_mode, Sampler,
                                         data(), param_name);
        case CL_SAMPLER_FILTER_MODE:
            return pyopencl_get_int_info(cl_filter_mode, Sampler,
                                         data(), param_name);
        case CL_SAMPLER_NORMALIZED_COORDS:
            return pyopencl_get_int_info(cl_bool, Sampler,
                                         data(), param_name);

        default:
            throw clerror("Sampler.get_info", CL_INVALID_VALUE);
        }
    }
};

// }}}


// {{{ program

class program : public clobj<cl_program> {
private:
    program_kind_type m_program_kind;

public:
    PYOPENCL_DEF_GET_CLASS_T(PROGRAM);
    program(cl_program prog, bool retain,
            program_kind_type progkind=KND_UNKNOWN)
        : clobj(prog), m_program_kind(progkind)
    {
        if (retain) {
            pyopencl_call_guarded(clRetainProgram, prog);
        }
    }
    ~program()
    {
        pyopencl_call_guarded_cleanup(clReleaseProgram, data());
    }
    program_kind_type
    kind() const
    {
        return m_program_kind;
    }
    PYOPENCL_USE_RESULT pyopencl_buf<cl_device_id>
    get_info__devices() const
    {
        return pyopencl_get_vec_info(cl_device_id, Program, data(),
                                     CL_PROGRAM_DEVICES);
    }
    generic_info
    get_info(cl_uint param_name) const
    {
        switch ((cl_program_info)param_name) {
        case CL_PROGRAM_CONTEXT:
            return pyopencl_get_opaque_info(cl_context, context,
                                            Program, data(), param_name);
        case CL_PROGRAM_REFERENCE_COUNT:
        case CL_PROGRAM_NUM_DEVICES:
            return pyopencl_get_int_info(cl_uint, Program, data(), param_name);
        case CL_PROGRAM_DEVICES:
            return pyopencl_get_opaque_array_info(
                cl_device_id, device, Program, data(), param_name);
        case CL_PROGRAM_SOURCE:
            return pyopencl_get_str_info(Program, data(), param_name);
        case CL_PROGRAM_BINARY_SIZES:
            return pyopencl_get_array_info(size_t, Program, data(), param_name);
        case CL_PROGRAM_BINARIES: {
            auto sizes = pyopencl_get_vec_info(size_t, Program, data(),
                                               CL_PROGRAM_BINARY_SIZES);
            pyopencl_buf<char*> result_ptrs(sizes.len());
            for (size_t i  = 0;i < sizes.len();i++) {
                result_ptrs[i] = (char*)malloc(sizes[i]);
            }
            try {
                pyopencl_call_guarded(clGetProgramInfo, data(),
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
            return pyopencl_get_int_info(size_t, Program, data(), param_name);
        case CL_PROGRAM_KERNEL_NAMES:
            return pyopencl_get_str_info(Program, data(), param_name);
#endif
        default:
            throw clerror("Program.get_info", CL_INVALID_VALUE);
        }
    }
    PYOPENCL_USE_RESULT generic_info
    get_build_info(const device *dev, cl_program_build_info param_name) const
    {
        switch (param_name) {
        case CL_PROGRAM_BUILD_STATUS:
            return pyopencl_get_int_info(cl_build_status, ProgramBuild,
                                         data(), dev->data(), param_name);
        case CL_PROGRAM_BUILD_OPTIONS:
        case CL_PROGRAM_BUILD_LOG:
            return pyopencl_get_str_info(ProgramBuild, data(),
                                         dev->data(), param_name);
#if PYOPENCL_CL_VERSION >= 0x1020
        case CL_PROGRAM_BINARY_TYPE:
            return pyopencl_get_int_info(cl_program_binary_type, ProgramBuild,
                                         data(), dev->data(), param_name);
#endif
        default:
            throw clerror("Program.get_build_info", CL_INVALID_VALUE);
        }
    }
    void
    build(const char *options, cl_uint num_devices,
          const clobj_t *_devices) const
    {
        auto devices = buf_from_class<device>(_devices, num_devices);
        pyopencl_call_guarded(clBuildProgram, data(), num_devices,
                              devices.get(), options, NULL, NULL);
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
    //             (data(), num_devices, devices,
    //              options.c_str(), header_names.size(),
    //              programs.empty() ? NULL : &programs.front(),
    //              header_name_ptrs.empty() ? NULL : &header_name_ptrs.front(),
    //              0, 0));
    //       }
    // #endif
};
PYOPENCL_USE_RESULT static inline program*
new_program(cl_program prog, program_kind_type progkind=KND_UNKNOWN)
{
    return pyopencl_convert_obj(program, clReleaseProgram, prog, progkind);
}

// }}}

// {{{ kernel

class kernel : public clobj<cl_kernel> {
private:
    cl_kernel m_kernel;

public:
    PYOPENCL_DEF_GET_CLASS_T(KERNEL);
    kernel(cl_kernel knl, bool retain)
        : clobj(knl)
    {
        if (retain) {
            pyopencl_call_guarded(clRetainKernel, knl);
        }
    }
    kernel(const program *prg, const char *kernel_name)
        : clobj(pyopencl_call_guarded(clCreateKernel, prg->data(),
                                      kernel_name))
    {}
    ~kernel()
    {
        pyopencl_call_guarded_cleanup(clReleaseKernel, data());
    }
    void
    set_arg_null(cl_uint arg_index) const
    {
        cl_mem m = 0;
        pyopencl_call_guarded(clSetKernelArg, data(), arg_index,
                              sizeof(cl_mem), &m);
    }
    void
    set_arg_mem(cl_uint arg_index, const memory_object_holder *mem) const
    {
        pyopencl_call_guarded(clSetKernelArg, data(), arg_index,
                              sizeof(cl_mem), &mem->data());
    }
    void
    set_arg_sampler(cl_uint arg_index, const sampler *smp) const
    {
        pyopencl_call_guarded(clSetKernelArg, data(), arg_index,
                              sizeof(cl_sampler), &smp->data());
    }
    void
    set_arg_buf(cl_uint arg_index, const void *buffer, size_t size) const
    {
        pyopencl_call_guarded(clSetKernelArg, data(), arg_index, size, buffer);
    }
    generic_info
    get_info(cl_uint param_name) const
    {
        switch ((cl_kernel_info)param_name) {
        case CL_KERNEL_FUNCTION_NAME:
            return pyopencl_get_str_info(Kernel, data(), param_name);
        case CL_KERNEL_NUM_ARGS:
        case CL_KERNEL_REFERENCE_COUNT:
            return pyopencl_get_int_info(cl_uint, Kernel, data(), param_name);
        case CL_KERNEL_CONTEXT:
            return pyopencl_get_opaque_info(cl_context, context,
                                            Kernel, data(), param_name);
        case CL_KERNEL_PROGRAM:
            return pyopencl_get_opaque_info(cl_program, program,
                                            Kernel, data(), param_name);
#if PYOPENCL_CL_VERSION >= 0x1020
        case CL_KERNEL_ATTRIBUTES:
            return pyopencl_get_str_info(Kernel, data(), param_name);
#endif
        default:
            throw clerror("Kernel.get_info", CL_INVALID_VALUE);
        }
    }
    PYOPENCL_USE_RESULT generic_info
    get_work_group_info(cl_kernel_work_group_info param_name,
                        const device *dev) const
    {
        switch (param_name) {
#if PYOPENCL_CL_VERSION >= 0x1010
        case CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE:
#endif
        case CL_KERNEL_WORK_GROUP_SIZE:
            return pyopencl_get_int_info(size_t, KernelWorkGroup,
                                         data(), dev->data(), param_name);
        case CL_KERNEL_COMPILE_WORK_GROUP_SIZE:
            return pyopencl_get_array_info(size_t, KernelWorkGroup,
                                           data(), dev->data(), param_name);
        case CL_KERNEL_LOCAL_MEM_SIZE:
#if PYOPENCL_CL_VERSION >= 0x1010
        case CL_KERNEL_PRIVATE_MEM_SIZE:
#endif
            return pyopencl_get_int_info(cl_ulong, KernelWorkGroup,
                                         data(), dev->data(), param_name);
        default:
            throw clerror("Kernel.get_work_group_info", CL_INVALID_VALUE);
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
    // #define PYOPENCL_FIRST_ARG data(), arg_index // hackety hack
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

}


// {{{ c wrapper

// Import all the names in pyopencl namespace for c wrappers.
using namespace pyopencl;

// Generic functions
int
get_cl_version()
{
    return PYOPENCL_CL_VERSION;
}

void
free_pointer(void *p)
{
    free(p);
}

void
free_pointer_array(void **p, uint32_t size)
{
    for (uint32_t i = 0;i < size;i++) {
        free(p[i]);
    }
}

int
have_gl()
{
#ifdef HAVE_GL
    return 1;
#else
    return 0;
#endif
}


// Platform
error*
get_platforms(clobj_t **_platforms, uint32_t *num_platforms)
{
    return c_handle_error([&] {
            *num_platforms = 0;
            pyopencl_call_guarded(clGetPlatformIDs, 0, NULL, num_platforms);
            pyopencl_buf<cl_platform_id> platforms(*num_platforms);
            pyopencl_call_guarded(clGetPlatformIDs, *num_platforms,
                                  platforms.get(), num_platforms);
            *_platforms = buf_to_base<platform>(platforms).release();
        });
}

error*
platform__get_devices(clobj_t _plat, clobj_t **_devices,
                      uint32_t *num_devices, cl_device_type devtype)
{
    auto plat = static_cast<platform*>(_plat);
    return c_handle_error([&] {
            auto devices = plat->get_devices(devtype);
            *num_devices = devices.len();
            *_devices = buf_to_base<device>(devices).release();
        });
}


// Context
error*
create_context(clobj_t *_ctx, const cl_context_properties *properties,
               cl_uint num_devices, const clobj_t *_devices)
{
    return c_handle_error([&] {
            auto devices = buf_from_class<device>(_devices, num_devices);
            *_ctx = new context(
                pyopencl_call_guarded(
                    clCreateContext,
                    const_cast<cl_context_properties*>(properties),
                    num_devices, devices.get(), nullptr, nullptr), false);
        });
}

error*
context__get_supported_image_formats(clobj_t _ctx, cl_mem_flags flags,
                                     cl_mem_object_type image_type,
                                     generic_info *out)
{
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_error([&] {
            *out = ctx->get_supported_image_formats(flags, image_type);
        });
}


// Command Queue
error*
create_command_queue(clobj_t *queue, clobj_t _ctx,
                     clobj_t _dev, cl_command_queue_properties properties)
{
    auto ctx = static_cast<context*>(_ctx);
    auto dev = static_cast<device*>(_dev);
    return c_handle_error([&] {
            *queue = new command_queue(ctx, dev, properties);
        });
}

error*
command_queue__finish(clobj_t queue)
{
    return c_handle_error([&] {
            static_cast<command_queue*>(queue)->finish();
        });
}

error*
command_queue__flush(clobj_t queue)
{
    return c_handle_error([&] {
            static_cast<command_queue*>(queue)->flush();
        });
}


// Buffer
error*
create_buffer(clobj_t *buffer, clobj_t _ctx, cl_mem_flags flags,
              size_t size, void *hostbuf)
{
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_error([&] {
            auto mem = retry_mem_error<cl_mem>([&] {
                    return pyopencl_call_guarded(clCreateBuffer, ctx->data(),
                                                 flags, size, hostbuf);
                });
            *buffer = new_buffer(mem, (flags & CL_MEM_USE_HOST_PTR ?
                                       hostbuf : NULL));
        });
}


// Memory Object
error*
memory_object__release(clobj_t obj)
{
    return c_handle_error([&] {
            static_cast<memory_object*>(obj)->release();
        });
}


// Memory Map
error*
memory_map__release(clobj_t _map, clobj_t _queue, const clobj_t *_wait_for,
                    uint32_t num_wait_for, clobj_t *evt)
{
    auto map = static_cast<memory_map*>(_map);
    auto queue = static_cast<command_queue*>(_queue);
    return c_handle_error([&] {
            *evt = map->release(queue, _wait_for, num_wait_for);
        });
}

void*
memory_map__data(clobj_t _map)
{
    return static_cast<memory_map*>(_map)->data();
}

size_t
memory_map__size(clobj_t _map)
{
    return static_cast<memory_map*>(_map)->size();
}

// Program
error*
create_program_with_source(clobj_t *prog, clobj_t _ctx, const char *src)
{
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_error([&] {
            size_t length = strlen(src);
            cl_program result = pyopencl_call_guarded(
                clCreateProgramWithSource, ctx->data(), 1, &src, &length);
            *prog = new_program(result, KND_SOURCE);
        });
}

error*
create_program_with_binary(clobj_t *prog, clobj_t _ctx,
                           cl_uint num_devices, const clobj_t *devices,
                           char **binaries, size_t *binary_sizes)
{
    auto ctx = static_cast<context*>(_ctx);
    auto devs = buf_from_class<device>(devices, num_devices);
    pyopencl_buf<cl_int> binary_statuses(num_devices);
    return c_handle_error([&] {
            cl_program result = pyopencl_call_guarded(
                clCreateProgramWithBinary, ctx->data(), num_devices, devs.get(),
                binary_sizes, reinterpret_cast<const unsigned char**>(
                    const_cast<const char**>(binaries)), binary_statuses.get());
            // for (cl_uint i = 0; i < num_devices; ++i)
            //   std::cout << i << ":" << binary_statuses[i] << std::endl;
            *prog = new_program(result, KND_BINARY);
        });
}

error*
program__build(clobj_t _prog, const char *options,
               cl_uint num_devices, const clobj_t *devices)
{
    auto prog = static_cast<program*>(_prog);
    return c_handle_error([&] {
            prog->build(options, num_devices, devices);
        });
}

error*
program__kind(clobj_t prog, int *kind)
{
    return c_handle_error([&] {
            *kind = static_cast<program*>(prog)->kind();
        });
}

error*
program__get_build_info(clobj_t _prog, clobj_t _dev,
                        cl_program_build_info param, generic_info *out)
{
    auto prog = static_cast<program*>(_prog);
    auto dev = static_cast<device*>(_dev);
    return c_handle_error([&] {
            *out = prog->get_build_info(dev, param);
        });
}


// Sampler
error*
create_sampler(clobj_t *samp, clobj_t _ctx, int norm_coords,
               cl_addressing_mode am, cl_filter_mode fm)
{
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_error([&] {
            *samp = new sampler(ctx, (bool)norm_coords, am, fm);
        });
}


// Kernel
error*
create_kernel(clobj_t *knl, clobj_t _prog, const char *name)
{
    auto prog = static_cast<program*>(_prog);
    return c_handle_error([&] {
            *knl = new kernel(prog, name);
        });
}

error*
kernel__set_arg_null(clobj_t knl, cl_uint arg_index)
{
    return c_handle_error([&] {
            static_cast<kernel*>(knl)->set_arg_null(arg_index);
        });
}

error*
kernel__set_arg_mem(clobj_t _knl, cl_uint arg_index, clobj_t _mem)
{
    auto knl = static_cast<kernel*>(_knl);
    auto mem = static_cast<memory_object_holder*>(_mem);
    return c_handle_error([&] {
            knl->set_arg_mem(arg_index, mem);
        });
}

error*
kernel__set_arg_sampler(clobj_t _knl, cl_uint arg_index, clobj_t _samp)
{
    auto knl = static_cast<kernel*>(_knl);
    auto samp = static_cast<sampler*>(_samp);
    return c_handle_error([&] {
            knl->set_arg_sampler(arg_index, samp);
        });
}

error*
kernel__set_arg_buf(clobj_t _knl, cl_uint arg_index,
                    const void *buffer, size_t size)
{
    auto knl = static_cast<kernel*>(_knl);
    return c_handle_error([&] {
            knl->set_arg_buf(arg_index, buffer, size);
        });
}

error*
kernel__get_work_group_info(clobj_t _knl, cl_kernel_work_group_info param,
                            clobj_t _dev, generic_info *out)
{
    auto knl = static_cast<kernel*>(_knl);
    auto dev = static_cast<device*>(_dev);
    return c_handle_error([&] {
            *out = knl->get_work_group_info(param, dev);
        });
}


// Image
error*
create_image_2d(clobj_t *img, clobj_t _ctx, cl_mem_flags flags,
                cl_image_format *fmt, size_t width, size_t height,
                size_t pitch, void *buffer)
{
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_error([&] {
            auto mem = retry_mem_error<cl_mem>([&] {
                    return pyopencl_call_guarded(
                        clCreateImage2D, ctx->data(), flags,
                        fmt, width, height, pitch, buffer);
                });
            *img = new_image(mem, (flags & CL_MEM_USE_HOST_PTR ?
                                   buffer : NULL));
        });
}

error*
create_image_3d(clobj_t *img, clobj_t _ctx, cl_mem_flags flags,
                cl_image_format *fmt, size_t width, size_t height,
                size_t depth, size_t pitch_x, size_t pitch_y, void *buffer)
{
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_error([&] {
            auto mem = retry_mem_error<cl_mem>([&] {
                    return pyopencl_call_guarded(
                        clCreateImage3D, ctx->data(), flags, fmt, width,
                        height, depth, pitch_x, pitch_y, buffer);
                });
            *img = new_image(mem, (flags & CL_MEM_USE_HOST_PTR ?
                                   buffer : NULL));
        });
}

error*
image__get_image_info(clobj_t img, cl_image_info param, generic_info *out)
{
    return c_handle_error([&] {
            *out = static_cast<image*>(img)->get_image_info(param);
        });
}


// Event
error*
event__get_profiling_info(clobj_t _evt, cl_profiling_info param,
                          generic_info *out)
{
    auto evt = static_cast<event*>(_evt);
    return c_handle_error([&] {
            *out = evt->get_profiling_info(param);
        });
}

error*
event__wait(clobj_t evt)
{
    return c_handle_error([&] {
            static_cast<event*>(evt)->wait();
        });
}

#if PYOPENCL_CL_VERSION >= 0x1010
error*
event__set_callback(clobj_t _evt, cl_int type, void (*cb)(cl_int),
                    void (*ref)(unsigned long))
{
    auto evt = static_cast<event*>(_evt);
    return c_handle_error([&] {
            unsigned long obj_id = next_obj_id();
            evt->set_callback(type, [=] (cl_int status) {
                    cb(status);
                    python_deref(obj_id);
                });
            ref(obj_id);
        });
}
#endif

error*
wait_for_events(const clobj_t *_wait_for, uint32_t num_wait_for)
{
    auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    return c_handle_error([&] {
            pyopencl_call_guarded(clWaitForEvents,
                                  num_wait_for, wait_for.get());
        });
}

// Nanny Event
unsigned long
nanny_event__get_ward(clobj_t evt)
{
    return static_cast<nanny_event*>(evt)->get_ward();
}


// enqueue_*
error*
enqueue_nd_range_kernel(clobj_t *_evt, clobj_t _queue, clobj_t _knl,
                        cl_uint work_dim, const size_t *global_work_offset,
                        const size_t *global_work_size,
                        const size_t *local_work_size,
                        const clobj_t *_wait_for, uint32_t num_wait_for)
{
    auto queue = static_cast<command_queue*>(_queue);
    auto knl = static_cast<kernel*>(_knl);
    auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    return c_handle_error([&] {
            cl_event evt;
            retry_mem_error<void>([&] {
                    pyopencl_call_guarded(
                        clEnqueueNDRangeKernel, queue->data(), knl->data(),
                        work_dim, global_work_offset, global_work_size,
                        local_work_size, num_wait_for, wait_for.get(), &evt);
                });
            *_evt = new_event(evt);
        });
}

error*
enqueue_task(clobj_t *_evt, clobj_t _queue, clobj_t _knl,
             const clobj_t *_wait_for, uint32_t num_wait_for)
{
    auto queue = static_cast<command_queue*>(_queue);
    auto knl = static_cast<kernel*>(_knl);
    auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    return c_handle_error([&] {
            cl_event evt;
            retry_mem_error<void>([&] {
                    pyopencl_call_guarded(
                        clEnqueueTask, queue->data(), knl->data(),
                        num_wait_for, wait_for.get(), &evt);
                });
            *_evt = new_event(evt);
        });
}

#if PYOPENCL_CL_VERSION >= 0x1020
error*
enqueue_marker_with_wait_list(clobj_t *_evt, clobj_t _queue,
                              const clobj_t *_wait_for, uint32_t num_wait_for)
{
    auto queue = static_cast<command_queue*>(_queue);
    auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    return c_handle_error([&] {
            cl_event evt;
            pyopencl_call_guarded(clEnqueueMarkerWithWaitList, queue->data(),
                                  num_wait_for, wait_for.get(), &evt);
            *_evt = new_event(evt);
        });
}

error*
enqueue_barrier_with_wait_list(clobj_t *_evt, clobj_t _queue,
                               const clobj_t *_wait_for, uint32_t num_wait_for)
{
    auto queue = static_cast<command_queue*>(_queue);
    auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    return c_handle_error([&] {
            cl_event evt;
            pyopencl_call_guarded(clEnqueueBarrierWithWaitList, queue->data(),
                                  num_wait_for, wait_for.get(), &evt);
            *_evt = new_event(evt);
        });
}
#endif

error*
enqueue_marker(clobj_t *_evt, clobj_t _queue)
{
    auto queue = static_cast<command_queue*>(_queue);
    return c_handle_error([&] {
            cl_event evt;
            pyopencl_call_guarded(clEnqueueMarker, queue->data(), &evt);
            *_evt = new_event(evt);
        });
}

error*
enqueue_barrier(clobj_t _queue)
{
    auto queue = static_cast<command_queue*>(_queue);
    return c_handle_error([&] {
            pyopencl_call_guarded(clEnqueueBarrier, queue->data());
        });
}

// {{{ transfer enqueues

error*
enqueue_read_buffer(clobj_t *_evt, clobj_t _queue, clobj_t _mem,
                    void *buffer, size_t size, size_t device_offset,
                    const clobj_t *_wait_for, uint32_t num_wait_for,
                    int is_blocking, void (*ref)(unsigned long))
{
    auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    auto queue = static_cast<command_queue*>(_queue);
    auto mem = static_cast<memory_object_holder*>(_mem);
    return c_handle_error([&] {
            cl_event evt;
            retry_mem_error<void>([&] {
                    pyopencl_call_guarded(
                        clEnqueueReadBuffer, queue->data(), mem->data(),
                        cast_bool(is_blocking), device_offset, size,
                        buffer, num_wait_for, wait_for.get(), &evt);
                });
            *_evt = new_nanny_event(evt, ref);
        });
}

error*
enqueue_write_buffer(clobj_t *_evt, clobj_t _queue, clobj_t _mem,
                     const void *buffer, size_t size, size_t device_offset,
                     const clobj_t *_wait_for, uint32_t num_wait_for,
                     int is_blocking, void (*ref)(unsigned long))
{
    auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    auto queue = static_cast<command_queue*>(_queue);
    auto mem = static_cast<memory_object_holder*>(_mem);
    return c_handle_error([&] {
            cl_event evt;
            retry_mem_error<void>([&] {
                    pyopencl_call_guarded(
                        clEnqueueWriteBuffer, queue->data(), mem->data(),
                        cast_bool(is_blocking), device_offset,
                        size, buffer, num_wait_for, wait_for.get(), &evt);
                });
            *_evt = new_nanny_event(evt, ref);
        });
}

error*
enqueue_copy_buffer(clobj_t *_evt, clobj_t _queue, clobj_t _src, clobj_t _dst,
                    ptrdiff_t byte_count, size_t src_offset, size_t dst_offset,
                    const clobj_t *_wait_for, uint32_t num_wait_for)
{
    auto queue = static_cast<command_queue*>(_queue);
    auto src = static_cast<memory_object_holder*>(_src);
    auto dst = static_cast<memory_object_holder*>(_dst);
    return c_handle_error([&] {
            if (byte_count < 0) {
                size_t byte_count_src = 0;
                size_t byte_count_dst = 0;
                pyopencl_call_guarded(
                    clGetMemObjectInfo, src->data(), CL_MEM_SIZE,
                    sizeof(byte_count), &byte_count_src, NULL);
                pyopencl_call_guarded(
                    clGetMemObjectInfo, src->data(), CL_MEM_SIZE,
                    sizeof(byte_count), &byte_count_dst, NULL);
                byte_count = std::min(byte_count_src, byte_count_dst);
            }
            auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
            cl_event evt;
            retry_mem_error<void>([&] {
                    pyopencl_call_guarded(
                        clEnqueueCopyBuffer, queue->data(), src->data(),
                        dst->data(), src_offset, dst_offset, byte_count,
                        num_wait_for, wait_for.get(), &evt);
                });
            *_evt = new_event(evt);
        });
}

error*
enqueue_map_buffer(clobj_t *_evt, clobj_t *map, clobj_t _queue, clobj_t _mem,
                   cl_map_flags flags, size_t offset, size_t size,
                   const clobj_t *_wait_for, uint32_t num_wait_for,
                   int block)
{
    auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    auto queue = static_cast<command_queue*>(_queue);
    auto buf = static_cast<buffer*>(_mem);
    return c_handle_error([&] {
            cl_event evt;
            void *res = retry_mem_error<void*>([&] {
                    return pyopencl_call_guarded(
                        clEnqueueMapBuffer, queue->data(), buf->data(),
                        cast_bool(block), flags, offset, size, num_wait_for,
                        wait_for.get(), &evt);
                });
            try {
                *_evt = new event(evt, false);
                evt = 0;
                *map = new memory_map(queue, buf, res, size);
            } catch (...) {
                if (evt) {
                    pyopencl_call_guarded_cleanup(clReleaseEvent, evt);
                }
                pyopencl_call_guarded_cleanup(clEnqueueUnmapMemObject,
                                              queue->data(), buf->data(),
                                              res, 0, NULL, NULL);
                throw;
            }
        });
}

error*
enqueue_read_image(clobj_t *_evt, clobj_t _queue, clobj_t _mem, size_t *origin,
                   size_t *region, void *buffer, size_t row_pitch,
                   size_t slice_pitch, const clobj_t *_wait_for,
                   uint32_t num_wait_for, int is_blocking,
                   void (*ref)(unsigned long))
{
    auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    auto queue = static_cast<command_queue*>(_queue);
    auto img = static_cast<image*>(_mem);
    return c_handle_error([&] {
            cl_event evt;
            retry_mem_error<void>([&] {
                    pyopencl_call_guarded(
                        clEnqueueReadImage, queue->data(), img->data(),
                        cast_bool(is_blocking), origin, region, row_pitch,
                        slice_pitch, buffer, num_wait_for,
                        wait_for.get(), &evt);
                });
            *_evt = new_nanny_event(evt, ref);
        });
}

// }}}

intptr_t
clobj__int_ptr(clobj_t obj)
{
    return obj->intptr();
}

error*
_from_int_ptr(clobj_t *ptr_out, intptr_t int_ptr_value, class_t class_)
{
#define FROM_INT_PTR(CLSU, CLS)                                         \
    *ptr_out = new CLS((PYOPENCL_CL_##CLSU)int_ptr_value,     \
                                 /* retain */ true);

    return c_handle_error([&] {
            SWITCHCLASS(FROM_INT_PTR);
        });
}

error*
clobj__get_info(clobj_t obj, cl_uint param, generic_info *out)
{
    return c_handle_error([&] {
            *out = obj->get_info(param);
        });
}

void
clobj__delete(clobj_t obj)
{
    delete obj;
}

// {{{ gl interop

#ifdef HAVE_GL
error*
create_from_gl_buffer(clobj_t *ptr, clobj_t _ctx,
                      cl_mem_flags flags, GLuint bufobj)
{
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_error([&] {
            cl_mem mem = pyopencl_call_guarded(clCreateFromGLBuffer,
                                               ctx->data(), flags, bufobj);
            *ptr = pyopencl_convert_obj(gl_buffer,
                                        clReleaseMemObject, mem);
        });
}

error*
create_from_gl_renderbuffer(clobj_t *ptr, clobj_t _ctx,
                            cl_mem_flags flags, GLuint bufobj)
{
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_error([&] {
            cl_mem mem = pyopencl_call_guarded(clCreateFromGLRenderbuffer,
                                               ctx->data(), flags, bufobj);
            *ptr = pyopencl_convert_obj(gl_renderbuffer,
                                        clReleaseMemObject, mem);
        });
}

error*
enqueue_acquire_gl_objects(clobj_t *_evt, clobj_t queue,
                           const clobj_t *mem_objects,
                           uint32_t num_mem_objects,
                           const clobj_t *wait_for, uint32_t num_wait_for)
{
    return c_handle_error([&] {
            *_evt = enqueue_gl_objects(
                Acquire, static_cast<command_queue*>(queue),
                mem_objects, num_mem_objects, wait_for, num_wait_for);
        });
}

error*
enqueue_release_gl_objects(clobj_t *event, clobj_t queue,
                           const clobj_t *mem_objects,
                           uint32_t num_mem_objects,
                           const clobj_t *wait_for, uint32_t num_wait_for)
{
    return c_handle_error([&] {
            *event = enqueue_gl_objects(
                Release, static_cast<command_queue*>(queue),
                mem_objects, num_mem_objects, wait_for, num_wait_for);
        });
}
#endif /* HAVE_GL */

// }}}

// }}}

// vim: foldmethod=marker
