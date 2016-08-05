#include "wrap_cl.h"
#include <iostream>

#ifdef CONST
#undef CONST
#endif

extern "C"
void populate_constants(void(*add)(const char*, const char*, int64_t value))
{
#define _ADD_ATTR(TYPE, PREFIX, NAME, SUFFIX, ...)      \
      add(TYPE, #NAME, CL_##PREFIX##NAME##SUFFIX)
#define ADD_ATTR(TYPE, PREFIX, NAME, ...)  \
      _ADD_ATTR(TYPE, PREFIX, NAME, __VA_ARGS__)

    // program_kind
    add("program_kind", "UNKNOWN", KND_UNKNOWN);
    add("program_kind", "SOURCE", KND_SOURCE);
    add("program_kind", "BINARY", KND_BINARY);

    // status_code
    ADD_ATTR("status_code", , SUCCESS);
    ADD_ATTR("status_code", , DEVICE_NOT_FOUND);
    ADD_ATTR("status_code", , DEVICE_NOT_AVAILABLE);
#if !(defined(CL_PLATFORM_NVIDIA) && CL_PLATFORM_NVIDIA == 0x3001)
    ADD_ATTR("status_code", , COMPILER_NOT_AVAILABLE);
#endif
    ADD_ATTR("status_code", , MEM_OBJECT_ALLOCATION_FAILURE);
    ADD_ATTR("status_code", , OUT_OF_RESOURCES);
    ADD_ATTR("status_code", , OUT_OF_HOST_MEMORY);
    ADD_ATTR("status_code", , PROFILING_INFO_NOT_AVAILABLE);
    ADD_ATTR("status_code", , MEM_COPY_OVERLAP);
    ADD_ATTR("status_code", , IMAGE_FORMAT_MISMATCH);
    ADD_ATTR("status_code", , IMAGE_FORMAT_NOT_SUPPORTED);
    ADD_ATTR("status_code", , BUILD_PROGRAM_FAILURE);
    ADD_ATTR("status_code", , MAP_FAILURE);

    ADD_ATTR("status_code", , INVALID_VALUE);
    ADD_ATTR("status_code", , INVALID_DEVICE_TYPE);
    ADD_ATTR("status_code", , INVALID_PLATFORM);
    ADD_ATTR("status_code", , INVALID_DEVICE);
    ADD_ATTR("status_code", , INVALID_CONTEXT);
    ADD_ATTR("status_code", , INVALID_QUEUE_PROPERTIES);
    ADD_ATTR("status_code", , INVALID_COMMAND_QUEUE);
    ADD_ATTR("status_code", , INVALID_HOST_PTR);
    ADD_ATTR("status_code", , INVALID_MEM_OBJECT);
    ADD_ATTR("status_code", , INVALID_IMAGE_FORMAT_DESCRIPTOR);
    ADD_ATTR("status_code", , INVALID_IMAGE_SIZE);
    ADD_ATTR("status_code", , INVALID_SAMPLER);
    ADD_ATTR("status_code", , INVALID_BINARY);
    ADD_ATTR("status_code", , INVALID_BUILD_OPTIONS);
    ADD_ATTR("status_code", , INVALID_PROGRAM);
    ADD_ATTR("status_code", , INVALID_PROGRAM_EXECUTABLE);
    ADD_ATTR("status_code", , INVALID_KERNEL_NAME);
    ADD_ATTR("status_code", , INVALID_KERNEL_DEFINITION);
    ADD_ATTR("status_code", , INVALID_KERNEL);
    ADD_ATTR("status_code", , INVALID_ARG_INDEX);
    ADD_ATTR("status_code", , INVALID_ARG_VALUE);
    ADD_ATTR("status_code", , INVALID_ARG_SIZE);
    ADD_ATTR("status_code", , INVALID_KERNEL_ARGS);
    ADD_ATTR("status_code", , INVALID_WORK_DIMENSION);
    ADD_ATTR("status_code", , INVALID_WORK_GROUP_SIZE);
    ADD_ATTR("status_code", , INVALID_WORK_ITEM_SIZE);
    ADD_ATTR("status_code", , INVALID_GLOBAL_OFFSET);
    ADD_ATTR("status_code", , INVALID_EVENT_WAIT_LIST);
    ADD_ATTR("status_code", , INVALID_EVENT);
    ADD_ATTR("status_code", , INVALID_OPERATION);
    ADD_ATTR("status_code", , INVALID_GL_OBJECT);
    ADD_ATTR("status_code", , INVALID_BUFFER_SIZE);
    ADD_ATTR("status_code", , INVALID_MIP_LEVEL);

#if defined(cl_khr_icd) && (cl_khr_icd >= 1)
    ADD_ATTR("status_code", , PLATFORM_NOT_FOUND_KHR);
#endif

#if defined(cl_khr_gl_sharing) && (cl_khr_gl_sharing >= 1)
    ADD_ATTR("status_code", , INVALID_GL_SHAREGROUP_REFERENCE_KHR);
#endif

#if PYOPENCL_CL_VERSION >= 0x1010
    ADD_ATTR("status_code", , MISALIGNED_SUB_BUFFER_OFFSET);
    ADD_ATTR("status_code", , EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
    ADD_ATTR("status_code", , INVALID_GLOBAL_WORK_SIZE);
#endif

#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR("status_code", , COMPILE_PROGRAM_FAILURE);
    ADD_ATTR("status_code", , LINKER_NOT_AVAILABLE);
    ADD_ATTR("status_code", , LINK_PROGRAM_FAILURE);
    ADD_ATTR("status_code", , DEVICE_PARTITION_FAILED);
    ADD_ATTR("status_code", , KERNEL_ARG_INFO_NOT_AVAILABLE);
    ADD_ATTR("status_code", , INVALID_IMAGE_DESCRIPTOR);
    ADD_ATTR("status_code", , INVALID_COMPILER_OPTIONS);
    ADD_ATTR("status_code", , INVALID_LINKER_OPTIONS);
    ADD_ATTR("status_code", , INVALID_DEVICE_PARTITION_COUNT);
#endif

#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR("status_code", , INVALID_PIPE_SIZE);
    ADD_ATTR("status_code", , INVALID_DEVICE_QUEUE);
#endif

    // platform_info
    ADD_ATTR("platform_info", PLATFORM_, PROFILE);
    ADD_ATTR("platform_info", PLATFORM_, VERSION);
    ADD_ATTR("platform_info", PLATFORM_, NAME);
    ADD_ATTR("platform_info", PLATFORM_, VENDOR);
#if !(defined(CL_PLATFORM_NVIDIA) && CL_PLATFORM_NVIDIA == 0x3001)
    ADD_ATTR("platform_info", PLATFORM_, EXTENSIONS);
#endif


    // device_type
    ADD_ATTR("device_type", DEVICE_TYPE_, DEFAULT);
    ADD_ATTR("device_type", DEVICE_TYPE_, CPU);
    ADD_ATTR("device_type", DEVICE_TYPE_, GPU);
    ADD_ATTR("device_type", DEVICE_TYPE_, ACCELERATOR);
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR("device_type", DEVICE_TYPE_, CUSTOM);
#endif
    ADD_ATTR("device_type", DEVICE_TYPE_, ALL);


    // device_info
    ADD_ATTR("device_info", DEVICE_, TYPE);
    ADD_ATTR("device_info", DEVICE_, VENDOR_ID);
    ADD_ATTR("device_info", DEVICE_, MAX_COMPUTE_UNITS);
    ADD_ATTR("device_info", DEVICE_, MAX_WORK_ITEM_DIMENSIONS);
    ADD_ATTR("device_info", DEVICE_, MAX_WORK_GROUP_SIZE);
    ADD_ATTR("device_info", DEVICE_, MAX_WORK_ITEM_SIZES);
    ADD_ATTR("device_info", DEVICE_, PREFERRED_VECTOR_WIDTH_CHAR);
    ADD_ATTR("device_info", DEVICE_, PREFERRED_VECTOR_WIDTH_SHORT);
    ADD_ATTR("device_info", DEVICE_, PREFERRED_VECTOR_WIDTH_INT);
    ADD_ATTR("device_info", DEVICE_, PREFERRED_VECTOR_WIDTH_LONG);
    ADD_ATTR("device_info", DEVICE_, PREFERRED_VECTOR_WIDTH_FLOAT);
    ADD_ATTR("device_info", DEVICE_, PREFERRED_VECTOR_WIDTH_DOUBLE);
    ADD_ATTR("device_info", DEVICE_, MAX_CLOCK_FREQUENCY);
    ADD_ATTR("device_info", DEVICE_, ADDRESS_BITS);
    ADD_ATTR("device_info", DEVICE_, MAX_READ_IMAGE_ARGS);
    ADD_ATTR("device_info", DEVICE_, MAX_WRITE_IMAGE_ARGS);
    ADD_ATTR("device_info", DEVICE_, MAX_MEM_ALLOC_SIZE);
    ADD_ATTR("device_info", DEVICE_, IMAGE2D_MAX_WIDTH);
    ADD_ATTR("device_info", DEVICE_, IMAGE2D_MAX_HEIGHT);
    ADD_ATTR("device_info", DEVICE_, IMAGE3D_MAX_WIDTH);
    ADD_ATTR("device_info", DEVICE_, IMAGE3D_MAX_HEIGHT);
    ADD_ATTR("device_info", DEVICE_, IMAGE3D_MAX_DEPTH);
    ADD_ATTR("device_info", DEVICE_, IMAGE_SUPPORT);
    ADD_ATTR("device_info", DEVICE_, MAX_PARAMETER_SIZE);
    ADD_ATTR("device_info", DEVICE_, MAX_SAMPLERS);
    ADD_ATTR("device_info", DEVICE_, MEM_BASE_ADDR_ALIGN);
    ADD_ATTR("device_info", DEVICE_, MIN_DATA_TYPE_ALIGN_SIZE);
    ADD_ATTR("device_info", DEVICE_, SINGLE_FP_CONFIG);
#ifdef CL_DEVICE_DOUBLE_FP_CONFIG
    ADD_ATTR("device_info", DEVICE_, DOUBLE_FP_CONFIG);
#endif
#ifdef CL_DEVICE_HALF_FP_CONFIG
    ADD_ATTR("device_info", DEVICE_, HALF_FP_CONFIG);
#endif
    ADD_ATTR("device_info", DEVICE_, GLOBAL_MEM_CACHE_TYPE);
    ADD_ATTR("device_info", DEVICE_, GLOBAL_MEM_CACHELINE_SIZE);
    ADD_ATTR("device_info", DEVICE_, GLOBAL_MEM_CACHE_SIZE);
    ADD_ATTR("device_info", DEVICE_, GLOBAL_MEM_SIZE);
    ADD_ATTR("device_info", DEVICE_, MAX_CONSTANT_BUFFER_SIZE);
    ADD_ATTR("device_info", DEVICE_, MAX_CONSTANT_ARGS);
    ADD_ATTR("device_info", DEVICE_, LOCAL_MEM_TYPE);
    ADD_ATTR("device_info", DEVICE_, LOCAL_MEM_SIZE);
    ADD_ATTR("device_info", DEVICE_, ERROR_CORRECTION_SUPPORT);
    ADD_ATTR("device_info", DEVICE_, PROFILING_TIMER_RESOLUTION);
    ADD_ATTR("device_info", DEVICE_, ENDIAN_LITTLE);
    ADD_ATTR("device_info", DEVICE_, AVAILABLE);
    ADD_ATTR("device_info", DEVICE_, COMPILER_AVAILABLE);
    ADD_ATTR("device_info", DEVICE_, EXECUTION_CAPABILITIES);
    ADD_ATTR("device_info", DEVICE_, QUEUE_PROPERTIES);
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR("device_info", DEVICE_, QUEUE_ON_HOST_PROPERTIES);
#endif
    ADD_ATTR("device_info", DEVICE_, NAME);
    ADD_ATTR("device_info", DEVICE_, VENDOR);
    ADD_ATTR("device_info", , DRIVER_VERSION);
    ADD_ATTR("device_info", DEVICE_, VERSION);
    ADD_ATTR("device_info", DEVICE_, PROFILE);
    ADD_ATTR("device_info", DEVICE_, EXTENSIONS);
    ADD_ATTR("device_info", DEVICE_, PLATFORM);
#if PYOPENCL_CL_VERSION >= 0x1010
    ADD_ATTR("device_info", DEVICE_, PREFERRED_VECTOR_WIDTH_HALF);
    ADD_ATTR("device_info", DEVICE_, HOST_UNIFIED_MEMORY); // deprecated in 2.0
    ADD_ATTR("device_info", DEVICE_, NATIVE_VECTOR_WIDTH_CHAR);
    ADD_ATTR("device_info", DEVICE_, NATIVE_VECTOR_WIDTH_SHORT);
    ADD_ATTR("device_info", DEVICE_, NATIVE_VECTOR_WIDTH_INT);
    ADD_ATTR("device_info", DEVICE_, NATIVE_VECTOR_WIDTH_LONG);
    ADD_ATTR("device_info", DEVICE_, NATIVE_VECTOR_WIDTH_FLOAT);
    ADD_ATTR("device_info", DEVICE_, NATIVE_VECTOR_WIDTH_DOUBLE);
    ADD_ATTR("device_info", DEVICE_, NATIVE_VECTOR_WIDTH_HALF);
    ADD_ATTR("device_info", DEVICE_, OPENCL_C_VERSION);
#endif
#ifdef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
    ADD_ATTR("device_info", DEVICE_, COMPUTE_CAPABILITY_MAJOR_NV);
    ADD_ATTR("device_info", DEVICE_, COMPUTE_CAPABILITY_MINOR_NV);
    ADD_ATTR("device_info", DEVICE_, REGISTERS_PER_BLOCK_NV);
    ADD_ATTR("device_info", DEVICE_, WARP_SIZE_NV);
    ADD_ATTR("device_info", DEVICE_, GPU_OVERLAP_NV);
    ADD_ATTR("device_info", DEVICE_, KERNEL_EXEC_TIMEOUT_NV);
    ADD_ATTR("device_info", DEVICE_, INTEGRATED_MEMORY_NV);
    // Nvidia specific device attributes, not defined in Khronos CL/cl_ext.h
#ifdef CL_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT_NV
    ADD_ATTR("device_info", DEVICE_, ATTRIBUTE_ASYNC_ENGINE_COUNT_NV);
#endif
#ifdef CL_DEVICE_PCI_BUS_ID_NV
    ADD_ATTR("device_info", DEVICE_, PCI_BUS_ID_NV);
#endif
#ifdef CL_DEVICE_PCI_SLOT_ID_NV
    ADD_ATTR("device_info", DEVICE_, PCI_SLOT_ID_NV);
#endif
#endif
#ifdef CL_DEVICE_PROFILING_TIMER_OFFSET_AMD
    ADD_ATTR("device_info", DEVICE_, PROFILING_TIMER_OFFSET_AMD);
#endif
#ifdef CL_DEVICE_TOPOLOGY_AMD
    ADD_ATTR("device_info", DEVICE_, TOPOLOGY_AMD);
#endif
#ifdef CL_DEVICE_BOARD_NAME_AMD
    ADD_ATTR("device_info", DEVICE_, BOARD_NAME_AMD);
#endif
#ifdef CL_DEVICE_GLOBAL_FREE_MEMORY_AMD
    ADD_ATTR("device_info", DEVICE_, GLOBAL_FREE_MEMORY_AMD);
#endif
#ifdef CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD
    ADD_ATTR("device_info", DEVICE_, SIMD_PER_COMPUTE_UNIT_AMD);
#endif
#ifdef CL_DEVICE_SIMD_WIDTH_AMD
    ADD_ATTR("device_info", DEVICE_, SIMD_WIDTH_AMD);
#endif
#ifdef CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD
    ADD_ATTR("device_info", DEVICE_, SIMD_INSTRUCTION_WIDTH_AMD);
#endif
#ifdef CL_DEVICE_WAVEFRONT_WIDTH_AMD
    ADD_ATTR("device_info", DEVICE_, WAVEFRONT_WIDTH_AMD);
#endif
#ifdef CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD
    ADD_ATTR("device_info", DEVICE_, GLOBAL_MEM_CHANNELS_AMD);
#endif
#ifdef CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD
    ADD_ATTR("device_info", DEVICE_, GLOBAL_MEM_CHANNEL_BANKS_AMD);
#endif
#ifdef CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD
    ADD_ATTR("device_info", DEVICE_, GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD);
#endif
#ifdef CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD
    ADD_ATTR("device_info", DEVICE_, LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD);
#endif
#ifdef CL_DEVICE_LOCAL_MEM_BANKS_AMD
    ADD_ATTR("device_info", DEVICE_, LOCAL_MEM_BANKS_AMD);
#endif

#ifdef CL_DEVICE_THREAD_TRACE_SUPPORTED_AMD
    ADD_ATTR("device_info", DEVICE_, THREAD_TRACE_SUPPORTED_AMD);
#endif
#ifdef CL_DEVICE_GFXIP_MAJOR_AMD
    ADD_ATTR("device_info", DEVICE_, GFXIP_MAJOR_AMD);
#endif
#ifdef CL_DEVICE_GFXIP_MINOR_AMD
    ADD_ATTR("device_info", DEVICE_, GFXIP_MINOR_AMD);
#endif
#ifdef CL_DEVICE_AVAILABLE_ASYNC_QUEUES_AMD
    ADD_ATTR("device_info", DEVICE_, AVAILABLE_ASYNC_QUEUES_AMD);
#endif

#ifdef CL_DEVICE_MAX_ATOMIC_COUNTERS_EXT
    ADD_ATTR("device_info", DEVICE_, MAX_ATOMIC_COUNTERS_EXT);
#endif
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR("device_info", DEVICE_, LINKER_AVAILABLE);
    ADD_ATTR("device_info", DEVICE_, BUILT_IN_KERNELS);
    ADD_ATTR("device_info", DEVICE_, IMAGE_MAX_BUFFER_SIZE);
    ADD_ATTR("device_info", DEVICE_, IMAGE_MAX_ARRAY_SIZE);
    ADD_ATTR("device_info", DEVICE_, PARENT_DEVICE);
    ADD_ATTR("device_info", DEVICE_, PARTITION_MAX_SUB_DEVICES);
    ADD_ATTR("device_info", DEVICE_, PARTITION_PROPERTIES);
    ADD_ATTR("device_info", DEVICE_, PARTITION_AFFINITY_DOMAIN);
    ADD_ATTR("device_info", DEVICE_, PARTITION_TYPE);
    ADD_ATTR("device_info", DEVICE_, REFERENCE_COUNT);
    ADD_ATTR("device_info", DEVICE_, PREFERRED_INTEROP_USER_SYNC);
    ADD_ATTR("device_info", DEVICE_, PRINTF_BUFFER_SIZE);
#endif
#ifdef cl_khr_image2d_from_buffer
    ADD_ATTR("device_info", DEVICE_, IMAGE_PITCH_ALIGNMENT);
    ADD_ATTR("device_info", DEVICE_, IMAGE_BASE_ADDRESS_ALIGNMENT);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR("device_info", DEVICE_, MAX_READ_WRITE_IMAGE_ARGS);
    ADD_ATTR("device_info", DEVICE_, MAX_GLOBAL_VARIABLE_SIZE);
    ADD_ATTR("device_info", DEVICE_, QUEUE_ON_DEVICE_PROPERTIES);
    ADD_ATTR("device_info", DEVICE_, QUEUE_ON_DEVICE_PREFERRED_SIZE);
    ADD_ATTR("device_info", DEVICE_, QUEUE_ON_DEVICE_MAX_SIZE);
    ADD_ATTR("device_info", DEVICE_, MAX_ON_DEVICE_QUEUES);
    ADD_ATTR("device_info", DEVICE_, MAX_ON_DEVICE_EVENTS);
    ADD_ATTR("device_info", DEVICE_, SVM_CAPABILITIES);
    ADD_ATTR("device_info", DEVICE_, GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE);
    ADD_ATTR("device_info", DEVICE_, MAX_PIPE_ARGS);
    ADD_ATTR("device_info", DEVICE_, PIPE_MAX_ACTIVE_RESERVATIONS);
    ADD_ATTR("device_info", DEVICE_, PIPE_MAX_PACKET_SIZE);
    ADD_ATTR("device_info", DEVICE_, PREFERRED_PLATFORM_ATOMIC_ALIGNMENT);
    ADD_ATTR("device_info", DEVICE_, PREFERRED_GLOBAL_ATOMIC_ALIGNMENT);
    ADD_ATTR("device_info", DEVICE_, PREFERRED_LOCAL_ATOMIC_ALIGNMENT);
#endif
#if PYOPENCL_CL_VERSION >= 0x2010
    ADD_ATTR("device_info", DEVICE_, IL_VERSION);
    ADD_ATTR("device_info", DEVICE_, MAX_NUM_SUB_GROUPS);
    ADD_ATTR("device_info", DEVICE_, SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS);
#endif
    /* cl_intel_advanced_motion_estimation */
#ifdef CL_DEVICE_ME_VERSION_INTEL
    ADD_ATTR("device_info", DEVICE_, ME_VERSION_INTEL);
#endif

    /* cl_qcom_ext_host_ptr */
#ifdef CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM
    ADD_ATTR("device_info", DEVICE_, EXT_MEM_PADDING_IN_BYTES_QCOM);
#endif
#ifdef CL_DEVICE_PAGE_SIZE_QCOM
    ADD_ATTR("device_info", DEVICE_, PAGE_SIZE_QCOM);
#endif

    /* cl_khr_spir */
#ifdef CL_DEVICE_SPIR_VERSIONS
    ADD_ATTR("device_info", DEVICE_, SPIR_VERSIONS);
#endif

    /* cl_altera_device_temperature */
#ifdef CL_DEVICE_CORE_TEMPERATURE_ALTERA
    ADD_ATTR("device_info", DEVICE_, CORE_TEMPERATURE_ALTERA);
#endif

    /* cl_intel_simultaneous_sharing */
#ifdef CL_DEVICE_SIMULTANEOUS_INTEROPS_INTEL
    ADD_ATTR("device_info", DEVICE_, SIMULTANEOUS_INTEROPS_INTEL);
#endif
#ifdef CL_DEVICE_NUM_SIMULTANEOUS_INTEROPS_INTEL
    ADD_ATTR("device_info", DEVICE_, NUM_SIMULTANEOUS_INTEROPS_INTEL);
#endif

    // device_fp_config
    ADD_ATTR("device_fp_config", FP_, DENORM);
    ADD_ATTR("device_fp_config", FP_, INF_NAN);
    ADD_ATTR("device_fp_config", FP_, ROUND_TO_NEAREST);
    ADD_ATTR("device_fp_config", FP_, ROUND_TO_ZERO);
    ADD_ATTR("device_fp_config", FP_, ROUND_TO_INF);
    ADD_ATTR("device_fp_config", FP_, FMA);
#if PYOPENCL_CL_VERSION >= 0x1010
    ADD_ATTR("device_fp_config", FP_, SOFT_FLOAT);
#endif
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR("device_fp_config", FP_, CORRECTLY_ROUNDED_DIVIDE_SQRT);
#endif


    // device_mem_cache_type
    ADD_ATTR("device_mem_cache_type",  , NONE);
    ADD_ATTR("device_mem_cache_type",  , READ_ONLY_CACHE);
    ADD_ATTR("device_mem_cache_type",  , READ_WRITE_CACHE);


    // device_local_mem_type
    ADD_ATTR("device_local_mem_type",  , LOCAL);
    ADD_ATTR("device_local_mem_type",  , GLOBAL);


    // device_exec_capabilities
    ADD_ATTR("device_exec_capabilities", EXEC_, KERNEL);
    ADD_ATTR("device_exec_capabilities", EXEC_, NATIVE_KERNEL);
#ifdef CL_EXEC_IMMEDIATE_EXECUTION_INTEL
    ADD_ATTR("device_exec_capabilities", EXEC_, IMMEDIATE_EXECUTION_INTEL);
#endif

#if PYOPENCL_CL_VERSION >= 0x2000
    // device_svm_capabilities
    ADD_ATTR("device_svm_capabilities", DEVICE_SVM_, COARSE_GRAIN_BUFFER);
    ADD_ATTR("device_svm_capabilities", DEVICE_SVM_, FINE_GRAIN_BUFFER);
    ADD_ATTR("device_svm_capabilities", DEVICE_SVM_, FINE_GRAIN_SYSTEM);
    ADD_ATTR("device_svm_capabilities", DEVICE_SVM_, ATOMICS);
#endif


    // command_queue_properties
    ADD_ATTR("command_queue_properties", QUEUE_, OUT_OF_ORDER_EXEC_MODE_ENABLE);
    ADD_ATTR("command_queue_properties", QUEUE_, PROFILING_ENABLE);
#ifdef CL_QUEUE_IMMEDIATE_EXECUTION_ENABLE_INTEL
    ADD_ATTR("command_queue_properties", QUEUE_, IMMEDIATE_EXECUTION_ENABLE_INTEL);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR("command_queue_properties", QUEUE_, ON_DEVICE);
    ADD_ATTR("command_queue_properties", QUEUE_, ON_DEVICE_DEFAULT);
#endif


    // context_info
    ADD_ATTR("context_info", CONTEXT_, REFERENCE_COUNT);
    ADD_ATTR("context_info", CONTEXT_, DEVICES);
    ADD_ATTR("context_info", CONTEXT_, PROPERTIES);
#if PYOPENCL_CL_VERSION >= 0x1010
    ADD_ATTR("context_info", CONTEXT_, NUM_DEVICES);
#endif
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR("context_info", CONTEXT_, INTEROP_USER_SYNC);
#endif


    // gl_context_info
#if defined(cl_khr_gl_sharing) && (cl_khr_gl_sharing >= 1)
    ADD_ATTR("gl_context_info", , CURRENT_DEVICE_FOR_GL_CONTEXT_KHR);
    ADD_ATTR("gl_context_info", , DEVICES_FOR_GL_CONTEXT_KHR);
#endif


    // context_properties
    ADD_ATTR("context_properties", CONTEXT_, PLATFORM);
#if defined(cl_khr_gl_sharing) && (cl_khr_gl_sharing >= 1)
    ADD_ATTR("context_properties",  ,GL_CONTEXT_KHR);
    ADD_ATTR("context_properties",  ,EGL_DISPLAY_KHR);
    ADD_ATTR("context_properties",  ,GLX_DISPLAY_KHR);
    ADD_ATTR("context_properties",  ,WGL_HDC_KHR);
    ADD_ATTR("context_properties",  ,CGL_SHAREGROUP_KHR);
#endif
#if defined(__APPLE__) && defined(HAVE_GL)
    ADD_ATTR("context_properties",  ,CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE);
#endif /* __APPLE__ */
#ifdef CL_CONTEXT_OFFLINE_DEVICES_AMD
    ADD_ATTR("context_properties", CONTEXT_, OFFLINE_DEVICES_AMD);
#endif


    // command_queue_info
    ADD_ATTR("command_queue_info", QUEUE_, CONTEXT);
    ADD_ATTR("command_queue_info", QUEUE_, DEVICE);
    ADD_ATTR("command_queue_info", QUEUE_, REFERENCE_COUNT);
    ADD_ATTR("command_queue_info", QUEUE_, PROPERTIES);


    // queue_properties
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR("queue_properties", QUEUE_, PROPERTIES);
    ADD_ATTR("queue_properties", QUEUE_, SIZE);
#endif


    // mem_flags
    ADD_ATTR("mem_flags", MEM_, READ_WRITE);
    ADD_ATTR("mem_flags", MEM_, WRITE_ONLY);
    ADD_ATTR("mem_flags", MEM_, READ_ONLY);
    ADD_ATTR("mem_flags", MEM_, USE_HOST_PTR);
    ADD_ATTR("mem_flags", MEM_, ALLOC_HOST_PTR);
    ADD_ATTR("mem_flags", MEM_, COPY_HOST_PTR);
#ifdef cl_amd_device_memory_flags
    ADD_ATTR("mem_flags", MEM_, USE_PERSISTENT_MEM_AMD);
#endif
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR("mem_flags", MEM_, HOST_WRITE_ONLY);
    ADD_ATTR("mem_flags", MEM_, HOST_READ_ONLY);
    ADD_ATTR("mem_flags", MEM_, HOST_NO_ACCESS);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR("mem_flags", MEM_, KERNEL_READ_AND_WRITE);
#endif

#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR("svm_mem_flags", MEM_, READ_WRITE);
    ADD_ATTR("svm_mem_flags", MEM_, WRITE_ONLY);
    ADD_ATTR("svm_mem_flags", MEM_, READ_ONLY);
    ADD_ATTR("svm_mem_flags", MEM_, SVM_FINE_GRAIN_BUFFER);
    ADD_ATTR("svm_mem_flags", MEM_, SVM_ATOMICS);
#endif


    // channel_order
    ADD_ATTR("channel_order",  , R);
    ADD_ATTR("channel_order",  , A);
    ADD_ATTR("channel_order",  , RG);
    ADD_ATTR("channel_order",  , RA);
    ADD_ATTR("channel_order",  , RGB);
    ADD_ATTR("channel_order",  , RGBA);
    ADD_ATTR("channel_order",  , BGRA);
    ADD_ATTR("channel_order",  , INTENSITY);
    ADD_ATTR("channel_order",  , LUMINANCE);
#if PYOPENCL_CL_VERSION >= 0x1010
    ADD_ATTR("channel_order",  , Rx);
    ADD_ATTR("channel_order",  , RGx);
    ADD_ATTR("channel_order",  , RGBx);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR("channel_order",  , sRGB);
    ADD_ATTR("channel_order",  , sRGBx);
    ADD_ATTR("channel_order",  , sRGBA);
    ADD_ATTR("channel_order",  , sBGRA);
    ADD_ATTR("channel_order",  , ABGR);
#endif


    // channel_type
    ADD_ATTR("channel_type",  , SNORM_INT8);
    ADD_ATTR("channel_type",  , SNORM_INT16);
    ADD_ATTR("channel_type",  , UNORM_INT8);
    ADD_ATTR("channel_type",  , UNORM_INT16);
    ADD_ATTR("channel_type",  , UNORM_SHORT_565);
    ADD_ATTR("channel_type",  , UNORM_SHORT_555);
    ADD_ATTR("channel_type",  , UNORM_INT_101010);
    ADD_ATTR("channel_type",  , SIGNED_INT8);
    ADD_ATTR("channel_type",  , SIGNED_INT16);
    ADD_ATTR("channel_type",  , SIGNED_INT32);
    ADD_ATTR("channel_type",  , UNSIGNED_INT8);
    ADD_ATTR("channel_type",  , UNSIGNED_INT16);
    ADD_ATTR("channel_type",  , UNSIGNED_INT32);
    ADD_ATTR("channel_type",  , HALF_FLOAT);
    ADD_ATTR("channel_type",  , FLOAT);


    // mem_object_type
    ADD_ATTR("mem_object_type", MEM_OBJECT_, BUFFER);
    ADD_ATTR("mem_object_type", MEM_OBJECT_, IMAGE2D);
    ADD_ATTR("mem_object_type", MEM_OBJECT_, IMAGE3D);
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR("mem_object_type", MEM_OBJECT_, IMAGE2D_ARRAY);
    ADD_ATTR("mem_object_type", MEM_OBJECT_, IMAGE1D);
    ADD_ATTR("mem_object_type", MEM_OBJECT_, IMAGE1D_ARRAY);
    ADD_ATTR("mem_object_type", MEM_OBJECT_, IMAGE1D_BUFFER);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR("mem_object_type", MEM_OBJECT_, PIPE);
#endif


    // mem_info
    ADD_ATTR("mem_info", MEM_, TYPE);
    ADD_ATTR("mem_info", MEM_, FLAGS);
    ADD_ATTR("mem_info", MEM_, SIZE);
    ADD_ATTR("mem_info", MEM_, HOST_PTR);
    ADD_ATTR("mem_info", MEM_, MAP_COUNT);
    ADD_ATTR("mem_info", MEM_, REFERENCE_COUNT);
    ADD_ATTR("mem_info", MEM_, CONTEXT);
#if PYOPENCL_CL_VERSION >= 0x1010
    ADD_ATTR("mem_info", MEM_, ASSOCIATED_MEMOBJECT);
    ADD_ATTR("mem_info", MEM_, OFFSET);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR("mem_info", MEM_, USES_SVM_POINTER);
#endif


    // image_info
    ADD_ATTR("image_info", IMAGE_, FORMAT);
    ADD_ATTR("image_info", IMAGE_, ELEMENT_SIZE);
    ADD_ATTR("image_info", IMAGE_, ROW_PITCH);
    ADD_ATTR("image_info", IMAGE_, SLICE_PITCH);
    ADD_ATTR("image_info", IMAGE_, WIDTH);
    ADD_ATTR("image_info", IMAGE_, HEIGHT);
    ADD_ATTR("image_info", IMAGE_, DEPTH);
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR("image_info", IMAGE_, ARRAY_SIZE);
    ADD_ATTR("image_info", IMAGE_, BUFFER);
    ADD_ATTR("image_info", IMAGE_, NUM_MIP_LEVELS);
    ADD_ATTR("image_info", IMAGE_, NUM_SAMPLES);
#endif


    // addressing_mode
    ADD_ATTR("addressing_mode", ADDRESS_, NONE);
    ADD_ATTR("addressing_mode", ADDRESS_, CLAMP_TO_EDGE);
    ADD_ATTR("addressing_mode", ADDRESS_, CLAMP);
    ADD_ATTR("addressing_mode", ADDRESS_, REPEAT);
#if PYOPENCL_CL_VERSION >= 0x1010
    ADD_ATTR("addressing_mode", ADDRESS_, MIRRORED_REPEAT);
#endif


    // filter_mode
    ADD_ATTR("filter_mode", FILTER_, NEAREST);
    ADD_ATTR("filter_mode", FILTER_, LINEAR);


    // sampler_info
    ADD_ATTR("sampler_info", SAMPLER_, REFERENCE_COUNT);
    ADD_ATTR("sampler_info", SAMPLER_, CONTEXT);
    ADD_ATTR("sampler_info", SAMPLER_, NORMALIZED_COORDS);
    ADD_ATTR("sampler_info", SAMPLER_, ADDRESSING_MODE);
    ADD_ATTR("sampler_info", SAMPLER_, FILTER_MODE);
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR("sampler_info", SAMPLER_, MIP_FILTER_MODE);
    ADD_ATTR("sampler_info", SAMPLER_, LOD_MIN);
    ADD_ATTR("sampler_info", SAMPLER_, LOD_MAX);
#endif


    // map_flags
    ADD_ATTR("map_flags", MAP_, READ);
    ADD_ATTR("map_flags", MAP_, WRITE);
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR("map_flags", MAP_, WRITE_INVALIDATE_REGION);
#endif


    // program_info
    ADD_ATTR("program_info", PROGRAM_, REFERENCE_COUNT);
    ADD_ATTR("program_info", PROGRAM_, CONTEXT);
    ADD_ATTR("program_info", PROGRAM_, NUM_DEVICES);
    ADD_ATTR("program_info", PROGRAM_, DEVICES);
    ADD_ATTR("program_info", PROGRAM_, SOURCE);
    ADD_ATTR("program_info", PROGRAM_, BINARY_SIZES);
    ADD_ATTR("program_info", PROGRAM_, BINARIES);
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR("program_info", PROGRAM_, NUM_KERNELS);
    ADD_ATTR("program_info", PROGRAM_, KERNEL_NAMES);
#endif


    // program_build_info
    ADD_ATTR("program_build_info", PROGRAM_BUILD_, STATUS);
    ADD_ATTR("program_build_info", PROGRAM_BUILD_, OPTIONS);
    ADD_ATTR("program_build_info", PROGRAM_BUILD_, LOG);
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR("program_build_info", PROGRAM_, BINARY_TYPE);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR("program_build_info", PROGRAM_BUILD_, GLOBAL_VARIABLE_TOTAL_SIZE);
#endif


    // program_binary_type
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR("program_binary_type", PROGRAM_BINARY_TYPE_, NONE);
    ADD_ATTR("program_binary_type", PROGRAM_BINARY_TYPE_, COMPILED_OBJECT);
    ADD_ATTR("program_binary_type", PROGRAM_BINARY_TYPE_, LIBRARY);
    ADD_ATTR("program_binary_type", PROGRAM_BINARY_TYPE_, EXECUTABLE);
#endif


    // kernel_info
    ADD_ATTR("kernel_info", KERNEL_, FUNCTION_NAME);
    ADD_ATTR("kernel_info", KERNEL_, NUM_ARGS);
    ADD_ATTR("kernel_info", KERNEL_, REFERENCE_COUNT);
    ADD_ATTR("kernel_info", KERNEL_, CONTEXT);
    ADD_ATTR("kernel_info", KERNEL_, PROGRAM);
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR("kernel_info", KERNEL_, ATTRIBUTES);
#endif


    // kernel_arg_info
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR("kernel_arg_info", KERNEL_ARG_, ADDRESS_QUALIFIER);
    ADD_ATTR("kernel_arg_info", KERNEL_ARG_, ACCESS_QUALIFIER);
    ADD_ATTR("kernel_arg_info", KERNEL_ARG_, TYPE_NAME);
    ADD_ATTR("kernel_arg_info", KERNEL_ARG_, TYPE_QUALIFIER);
    ADD_ATTR("kernel_arg_info", KERNEL_ARG_, NAME);
#endif


    // kernel_arg_address_qualifier
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR("kernel_arg_address_qualifier", KERNEL_ARG_ADDRESS_, GLOBAL);
    ADD_ATTR("kernel_arg_address_qualifier", KERNEL_ARG_ADDRESS_, LOCAL);
    ADD_ATTR("kernel_arg_address_qualifier", KERNEL_ARG_ADDRESS_, CONSTANT);
    ADD_ATTR("kernel_arg_address_qualifier", KERNEL_ARG_ADDRESS_, PRIVATE);
#endif


    // kernel_arg_access_qualifier
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR("kernel_arg_access_qualifier", KERNEL_ARG_ACCESS_, READ_ONLY);
    ADD_ATTR("kernel_arg_access_qualifier", KERNEL_ARG_ACCESS_, WRITE_ONLY);
    ADD_ATTR("kernel_arg_access_qualifier", KERNEL_ARG_ACCESS_, READ_WRITE);
    ADD_ATTR("kernel_arg_access_qualifier", KERNEL_ARG_ACCESS_, NONE);
#endif


    // kernel_arg_type_qualifier
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR("kernel_arg_type_qualifier", KERNEL_ARG_TYPE_, NONE);
    ADD_ATTR("kernel_arg_type_qualifier", KERNEL_ARG_TYPE_, CONST);
    ADD_ATTR("kernel_arg_type_qualifier", KERNEL_ARG_TYPE_, RESTRICT);
    ADD_ATTR("kernel_arg_type_qualifier", KERNEL_ARG_TYPE_, VOLATILE);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR("kernel_arg_type_qualifier", KERNEL_ARG_TYPE_, PIPE);
#endif


    // kernel_work_group_info
    ADD_ATTR("kernel_work_group_info", KERNEL_, WORK_GROUP_SIZE);
    ADD_ATTR("kernel_work_group_info", KERNEL_, COMPILE_WORK_GROUP_SIZE);
    ADD_ATTR("kernel_work_group_info", KERNEL_, LOCAL_MEM_SIZE);
#if PYOPENCL_CL_VERSION >= 0x1010
    ADD_ATTR("kernel_work_group_info", KERNEL_, PREFERRED_WORK_GROUP_SIZE_MULTIPLE);
    ADD_ATTR("kernel_work_group_info", KERNEL_, PRIVATE_MEM_SIZE);
#endif
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR("kernel_work_group_info", KERNEL_, GLOBAL_WORK_SIZE);
#endif


    // event_info
    ADD_ATTR("event_info", EVENT_, COMMAND_QUEUE);
    ADD_ATTR("event_info", EVENT_, COMMAND_TYPE);
    ADD_ATTR("event_info", EVENT_, REFERENCE_COUNT);
    ADD_ATTR("event_info", EVENT_, COMMAND_EXECUTION_STATUS);
#if PYOPENCL_CL_VERSION >= 0x1010
    ADD_ATTR("event_info", EVENT_, CONTEXT);
#endif


    // command_type
    ADD_ATTR("command_type", COMMAND_, NDRANGE_KERNEL);
    ADD_ATTR("command_type", COMMAND_, TASK);
    ADD_ATTR("command_type", COMMAND_, NATIVE_KERNEL);
    ADD_ATTR("command_type", COMMAND_, READ_BUFFER);
    ADD_ATTR("command_type", COMMAND_, WRITE_BUFFER);
    ADD_ATTR("command_type", COMMAND_, COPY_BUFFER);
    ADD_ATTR("command_type", COMMAND_, READ_IMAGE);
    ADD_ATTR("command_type", COMMAND_, WRITE_IMAGE);
    ADD_ATTR("command_type", COMMAND_, COPY_IMAGE);
    ADD_ATTR("command_type", COMMAND_, COPY_IMAGE_TO_BUFFER);
    ADD_ATTR("command_type", COMMAND_, COPY_BUFFER_TO_IMAGE);
    ADD_ATTR("command_type", COMMAND_, MAP_BUFFER);
    ADD_ATTR("command_type", COMMAND_, MAP_IMAGE);
    ADD_ATTR("command_type", COMMAND_, UNMAP_MEM_OBJECT);
    ADD_ATTR("command_type", COMMAND_, MARKER);
    ADD_ATTR("command_type", COMMAND_, ACQUIRE_GL_OBJECTS);
    ADD_ATTR("command_type", COMMAND_, RELEASE_GL_OBJECTS);
#if PYOPENCL_CL_VERSION >= 0x1010
    ADD_ATTR("command_type", COMMAND_, READ_BUFFER_RECT);
    ADD_ATTR("command_type", COMMAND_, WRITE_BUFFER_RECT);
    ADD_ATTR("command_type", COMMAND_, COPY_BUFFER_RECT);
    ADD_ATTR("command_type", COMMAND_, USER);
#endif
#ifdef cl_ext_migrate_memobject
    ADD_ATTR("command_type", COMMAND_, MIGRATE_MEM_OBJECT_EXT);
#endif
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR("command_type", COMMAND_, BARRIER);
    ADD_ATTR("command_type", COMMAND_, MIGRATE_MEM_OBJECTS);
    ADD_ATTR("command_type", COMMAND_, FILL_BUFFER);
    ADD_ATTR("command_type", COMMAND_, FILL_IMAGE);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR("command_type", COMMAND_, SVM_FREE);
    ADD_ATTR("command_type", COMMAND_, SVM_MEMCPY);
    ADD_ATTR("command_type", COMMAND_, SVM_MEMFILL);
    ADD_ATTR("command_type", COMMAND_, SVM_MAP);
    ADD_ATTR("command_type", COMMAND_, SVM_UNMAP);
#endif


    // command_execution_status
    ADD_ATTR("command_execution_status", , COMPLETE);
    ADD_ATTR("command_execution_status", , RUNNING);
    ADD_ATTR("command_execution_status", , SUBMITTED);
    ADD_ATTR("command_execution_status", , QUEUED);


    // profiling_info
    ADD_ATTR("profiling_info", PROFILING_COMMAND_, QUEUED);
    ADD_ATTR("profiling_info", PROFILING_COMMAND_, SUBMIT);
    ADD_ATTR("profiling_info", PROFILING_COMMAND_, START);
    ADD_ATTR("profiling_info", PROFILING_COMMAND_, END);
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR("profiling_info", PROFILING_COMMAND_, COMPLETE);
#endif


    // mem_migration_flags
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR("mem_migration_flags", MIGRATE_MEM_OBJECT_, HOST);
    ADD_ATTR("mem_migration_flags", MIGRATE_MEM_OBJECT_, CONTENT_UNDEFINED);
#endif


    // mem_migration_flags_ext
#ifdef cl_ext_migrate_memobject
    ADD_ATTR("mem_migration_flags_ext", MIGRATE_MEM_OBJECT_, HOST, _EXT);
    ADD_ATTR("mem_migration_flags_ext", MIGRATE_MEM_OBJECT_,
             CONTENT_UNDEFINED, _EXT);
#endif


    // device_partition_property
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR("device_partition_property", DEVICE_PARTITION_, EQUALLY);
    ADD_ATTR("device_partition_property", DEVICE_PARTITION_, BY_COUNTS);
    ADD_ATTR("device_partition_property", DEVICE_PARTITION_, BY_COUNTS_LIST_END);
    ADD_ATTR("device_partition_property", DEVICE_PARTITION_, BY_AFFINITY_DOMAIN);
#endif


    // device_affinity_domain
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR("device_affinity_domain", DEVICE_AFFINITY_DOMAIN_, NUMA);
    ADD_ATTR("device_affinity_domain", DEVICE_AFFINITY_DOMAIN_, L4_CACHE);
    ADD_ATTR("device_affinity_domain", DEVICE_AFFINITY_DOMAIN_, L3_CACHE);
    ADD_ATTR("device_affinity_domain", DEVICE_AFFINITY_DOMAIN_, L2_CACHE);
    ADD_ATTR("device_affinity_domain", DEVICE_AFFINITY_DOMAIN_, L1_CACHE);
    ADD_ATTR("device_affinity_domain", DEVICE_AFFINITY_DOMAIN_,
             NEXT_PARTITIONABLE);
#endif


#ifdef HAVE_GL
    // gl_object_type
    ADD_ATTR("gl_object_type", GL_OBJECT_, BUFFER);
    ADD_ATTR("gl_object_type", GL_OBJECT_, TEXTURE2D);
    ADD_ATTR("gl_object_type", GL_OBJECT_, TEXTURE3D);
    ADD_ATTR("gl_object_type", GL_OBJECT_, RENDERBUFFER);


    // gl_texture_info
    ADD_ATTR("gl_texture_info", GL_, TEXTURE_TARGET);
    ADD_ATTR("gl_texture_info", GL_, MIPMAP_LEVEL);
#endif


    // migrate_mem_object_flags_ext
#ifdef cl_ext_migrate_memobject
    ADD_ATTR("migrate_mem_object_flags_ext", MIGRATE_MEM_OBJECT_, HOST, _EXT);
#endif
}
