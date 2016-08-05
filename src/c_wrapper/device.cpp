#include "device.h"
#include "platform.h"

template class clobj<cl_device_id>;
template void print_arg<cl_device_id>(std::ostream&,
                                      const cl_device_id&, bool);
template void print_clobj<device>(std::ostream&, const device*);
template void print_buf<cl_device_id>(std::ostream&, const cl_device_id*,
                                      size_t, ArgType, bool, bool);

void
device::get_version(cl_device_id dev, int *major, int *minor)
{
    cl_platform_id plat;
    pyopencl_call_guarded(clGetDeviceInfo, dev, CL_DEVICE_PLATFORM,
                          size_arg(plat), nullptr);
    platform::get_version(plat, major, minor);
}

device::~device()
{
    if (false) {
    }
#if PYOPENCL_CL_VERSION >= 0x1020
    else if (m_ref_type == REF_CL_1_2) {
        pyopencl_call_guarded_cleanup(clReleaseDevice, PYOPENCL_CL_CASTABLE_THIS);
    }
#endif
}

#ifdef CL_DEVICE_TOPOLOGY_AMD
template<typename... ArgTypes>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE generic_info
get_device_topology_amd(ArgTypes&&... args)
{
    const char * tpname = "cl_device_topology_amd*";
    cl_device_topology_amd value;
    const char * fname = "clGetDeviceInfo";
    call_guarded(clGetDeviceInfo, fname, args..., size_arg(value), nullptr);
    generic_info info;
    info.dontfree = 0;
    info.opaque_class = CLASS_NONE;
    info.type = tpname;
    info.value = cl_memdup(&value);
    return info;
}

#define pyopencl_get_device_topology_amd(...) get_device_topology_amd(__VA_ARGS__)

#endif

generic_info
device::get_info(cl_uint param_name) const
{
#define DEV_GET_INT_INF(TYPE)                                   \
    pyopencl_get_int_info(TYPE, Device, PYOPENCL_CL_CASTABLE_THIS, param_name)

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
        return pyopencl_get_array_info(size_t, Device, PYOPENCL_CL_CASTABLE_THIS, param_name);

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
    // same as CL_DEVICE_QUEUE_ON_HOST_PROPERTIES in 2.0
        return DEV_GET_INT_INF(cl_command_queue_properties);

    case CL_DEVICE_NAME:
    case CL_DEVICE_VENDOR:
    case CL_DRIVER_VERSION:
    case CL_DEVICE_PROFILE:
    case CL_DEVICE_VERSION:
    case CL_DEVICE_EXTENSIONS:
        return pyopencl_get_str_info(Device, PYOPENCL_CL_CASTABLE_THIS, param_name);

    case CL_DEVICE_PLATFORM:
        return pyopencl_get_opaque_info(platform, Device, PYOPENCL_CL_CASTABLE_THIS, param_name);
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

    case CL_DEVICE_HOST_UNIFIED_MEMORY: // deprecated in 2.0
        return DEV_GET_INT_INF(cl_bool);
    case CL_DEVICE_OPENCL_C_VERSION:
        return pyopencl_get_str_info(Device, PYOPENCL_CL_CASTABLE_THIS, param_name);
#endif
#ifdef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
    case CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV:
    case CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV:
    case CL_DEVICE_REGISTERS_PER_BLOCK_NV:
    case CL_DEVICE_WARP_SIZE_NV:
#ifdef CL_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT_NV
    case CL_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT_NV:
#endif
#ifdef CL_DEVICE_PCI_BUS_ID_NV
    case CL_DEVICE_PCI_BUS_ID_NV:
#endif
#ifdef CL_DEVICE_PCI_SLOT_ID_NV
    case CL_DEVICE_PCI_SLOT_ID_NV:
#endif
        return DEV_GET_INT_INF(cl_uint);
    case CL_DEVICE_GPU_OVERLAP_NV:
    case CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV:
    case CL_DEVICE_INTEGRATED_MEMORY_NV:
        return DEV_GET_INT_INF(cl_bool);
#endif
#if PYOPENCL_CL_VERSION >= 0x1020
    case CL_DEVICE_LINKER_AVAILABLE:
        return DEV_GET_INT_INF(cl_bool);
    case CL_DEVICE_BUILT_IN_KERNELS:
        return pyopencl_get_str_info(Device, PYOPENCL_CL_CASTABLE_THIS, param_name);
    case CL_DEVICE_IMAGE_MAX_BUFFER_SIZE:
    case CL_DEVICE_IMAGE_MAX_ARRAY_SIZE:
        return DEV_GET_INT_INF(size_t);
    case CL_DEVICE_PARENT_DEVICE:
        return pyopencl_get_opaque_info(device, Device, PYOPENCL_CL_CASTABLE_THIS, param_name);
    case CL_DEVICE_PARTITION_MAX_SUB_DEVICES:
        return DEV_GET_INT_INF(cl_uint);
    case CL_DEVICE_PARTITION_TYPE:
    case CL_DEVICE_PARTITION_PROPERTIES:
        return pyopencl_get_array_info(cl_device_partition_property,
                                       Device, PYOPENCL_CL_CASTABLE_THIS, param_name);
    case CL_DEVICE_PARTITION_AFFINITY_DOMAIN:
        return pyopencl_get_array_info(cl_device_affinity_domain,
                                       Device, PYOPENCL_CL_CASTABLE_THIS, param_name);
    case CL_DEVICE_REFERENCE_COUNT:
        return DEV_GET_INT_INF(cl_uint);
    case CL_DEVICE_PREFERRED_INTEROP_USER_SYNC:
    case CL_DEVICE_PRINTF_BUFFER_SIZE:
        return DEV_GET_INT_INF(cl_bool);
#endif
#ifdef cl_khr_image2d_from_buffer
    case CL_DEVICE_IMAGE_PITCH_ALIGNMENT:
        return DEV_GET_INT_INF(cl_uint);
    case CL_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT:
        return DEV_GET_INT_INF(cl_uint);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
    case CL_DEVICE_MAX_READ_WRITE_IMAGE_ARGS:
        return DEV_GET_INT_INF(cl_uint);
    case CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE:
        return DEV_GET_INT_INF(size_t);
    case CL_DEVICE_QUEUE_ON_DEVICE_PROPERTIES:
        return DEV_GET_INT_INF(cl_command_queue_properties);
    case CL_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE:
        return DEV_GET_INT_INF(cl_uint);
    case CL_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE:
        return DEV_GET_INT_INF(cl_uint);
    case CL_DEVICE_MAX_ON_DEVICE_QUEUES:
        return DEV_GET_INT_INF(cl_uint);
    case CL_DEVICE_MAX_ON_DEVICE_EVENTS:
        return DEV_GET_INT_INF(cl_uint);
    case CL_DEVICE_SVM_CAPABILITIES:
        return DEV_GET_INT_INF(cl_device_svm_capabilities);
    case CL_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE:
        return DEV_GET_INT_INF(size_t);
    case CL_DEVICE_MAX_PIPE_ARGS:
    case CL_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS:
    case CL_DEVICE_PIPE_MAX_PACKET_SIZE:
        return DEV_GET_INT_INF(cl_uint);
    case CL_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT:
    case CL_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT:
    case CL_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT:
        return DEV_GET_INT_INF(cl_uint);
#endif
#if PYOPENCL_CL_VERSION >= 0x2010
    case CL_DEVICE_IL_VERSION:
        return pyopencl_get_str_info(Device, PYOPENCL_CL_CASTABLE_THIS, param_name);
    case CL_DEVICE_MAX_NUM_SUB_GROUPS:
        return DEV_GET_INT_INF(cl_uint);
    case CL_DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS:
        return DEV_GET_INT_INF(cl_bool);
#endif


        // {{{ AMD dev attrs
        //
        // types of AMD dev attrs divined from
        // https://www.khronos.org/registry/cl/api/1.2/cl.hpp
#ifdef CL_DEVICE_PROFILING_TIMER_OFFSET_AMD
    case CL_DEVICE_PROFILING_TIMER_OFFSET_AMD:
        return DEV_GET_INT_INF(cl_ulong);
#endif
#ifdef CL_DEVICE_TOPOLOGY_AMD
        case CL_DEVICE_TOPOLOGY_AMD:
            return pyopencl_get_device_topology_amd(PYOPENCL_CL_CASTABLE_THIS, param_name);
#endif
#ifdef CL_DEVICE_THREAD_TRACE_SUPPORTED_AMD
    case CL_DEVICE_THREAD_TRACE_SUPPORTED_AMD:
        return DEV_GET_INT_INF(cl_bool);
#endif
#ifdef CL_DEVICE_BOARD_NAME_AMD
    case CL_DEVICE_BOARD_NAME_AMD: ;
        return pyopencl_get_str_info(Device, PYOPENCL_CL_CASTABLE_THIS, param_name);
#endif
#ifdef CL_DEVICE_GLOBAL_FREE_MEMORY_AMD
    case CL_DEVICE_GLOBAL_FREE_MEMORY_AMD:
        return pyopencl_get_array_info(size_t, Device,
                                       PYOPENCL_CL_CASTABLE_THIS, param_name);
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
#ifdef CL_DEVICE_GFXIP_MAJOR_AMD
    case CL_DEVICE_GFXIP_MAJOR_AMD:
#endif
#ifdef CL_DEVICE_GFXIP_MINOR_AMD
    case CL_DEVICE_GFXIP_MINOR_AMD:
#endif
#ifdef CL_DEVICE_AVAILABLE_ASYNC_QUEUES_AMD
    case CL_DEVICE_AVAILABLE_ASYNC_QUEUES_AMD:
#endif
        return DEV_GET_INT_INF(cl_uint);
        // }}}
#ifdef CL_DEVICE_ME_VERSION_INTEL
    case CL_DEVICE_ME_VERSION_INTEL:
#endif
#ifdef CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM
    case CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM:
#endif
#ifdef CL_DEVICE_PAGE_SIZE_QCOM
    case CL_DEVICE_PAGE_SIZE_QCOM:
#endif
#ifdef CL_DEVICE_NUM_SIMULTANEOUS_INTEROPS_INTEL
    case CL_DEVICE_NUM_SIMULTANEOUS_INTEROPS_INTEL:
#endif
        return DEV_GET_INT_INF(cl_uint);
#ifdef CL_DEVICE_SIMULTANEOUS_INTEROPS_INTEL
    case CL_DEVICE_SIMULTANEOUS_INTEROPS_INTEL:
        return pyopencl_get_array_info(cl_uint, Device, PYOPENCL_CL_CASTABLE_THIS, param_name);
#endif
#ifdef CL_DEVICE_SPIR_VERSIONS
    case CL_DEVICE_SPIR_VERSIONS:
        return pyopencl_get_str_info(Device, PYOPENCL_CL_CASTABLE_THIS, param_name);
#endif
#ifdef CL_DEVICE_CORE_TEMPERATURE_ALTERA
    case CL_DEVICE_CORE_TEMPERATURE_ALTERA:
        return DEV_GET_INT_INF(cl_int);
#endif

    default:
        throw clerror("Device.get_info", CL_INVALID_VALUE);
    }
}

#if PYOPENCL_CL_VERSION >= 0x1020
PYOPENCL_USE_RESULT pyopencl_buf<clobj_t>
device::create_sub_devices(const cl_device_partition_property *props)
{
    // TODO debug print props
    cl_uint num_devices;
    pyopencl_call_guarded(clCreateSubDevices, PYOPENCL_CL_CASTABLE_THIS, props, 0, nullptr,
                          buf_arg(num_devices));
    pyopencl_buf<cl_device_id> devices(num_devices);
    pyopencl_call_guarded(clCreateSubDevices, PYOPENCL_CL_CASTABLE_THIS, props, devices,
                          buf_arg(num_devices));
    return buf_to_base<device>(devices, true, device::REF_CL_1_2);
}
#endif

// c wrapper

error*
device__create_sub_devices(clobj_t _dev, clobj_t **_devs,
                           uint32_t *num_devices,
                           const cl_device_partition_property *props)
{
#if PYOPENCL_CL_VERSION >= 0x1020
    auto dev = static_cast<device*>(_dev);
    return c_handle_error([&] {
            auto devs = dev->create_sub_devices(props);
            *num_devices = (uint32_t)devs.len();
            *_devs = devs.release();
        });
#else
    PYOPENCL_UNSUPPORTED_BEFORE(clCreateSubDevices, "CL 1.2")
#endif
}
