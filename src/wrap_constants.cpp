// Wrap CL constants and errors
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


#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pyopencl_ARRAY_API

#include "wrap_cl.hpp"


using namespace pyopencl;


namespace
{
  // {{{ 'fake' constant scopes
  class status_code { };
  class platform_info { };
  class device_type { };
  class device_info { };
  class device_topology_type_amd { };
  class device_fp_config { };
  class device_mem_cache_type { };
  class device_local_mem_type { };
  class device_exec_capabilities { };
  class device_svm_capabilities { };
  class command_queue_properties { };
  class context_info { };
  class gl_context_info { };
  class context_properties { };
  class command_queue_info { };
  class queue_properties { };
  class mem_flags { };
  class svm_mem_flags { };
  class channel_order { };
  class channel_type { };
  class mem_object_type { };
  class mem_info { };
  class image_info { };
  class pipe_info { };
  class pipe_properties { };
  class addressing_mode { };
  class filter_mode { };
  class sampler_info { };
  class sampler_properties { };
  class map_flags { };
  class program_info { };
  class program_build_info { };
  class program_binary_type { };
  class build_status { };
  class kernel_info { };
  class kernel_arg_info { };
  class kernel_arg_address_qualifier { };
  class kernel_arg_access_qualifier { };
  class kernel_arg_type_qualifier { };
  class kernel_work_group_info { };
  class kernel_sub_group_info { };
  class event_info { };
  class command_type { };
  class command_execution_status { };
  class profiling_info { };
  class buffer_create_type { };
  class mem_migration_flags { };

  class device_partition_property { };
  class device_affinity_domain { };
  class device_atomic_capabilities { };
  class device_device_enqueue_capabilities { };

  class version_bits { };
  class khronos_vendor_id { };

  class gl_object_type { };
  class gl_texture_info { };

  // }}}
}


void pyopencl_expose_constants(py::module &m)
{
  // {{{ exceptions
  {
#define DECLARE_EXC(NAME, BASE) \
  static py::exception<pyopencl::error> CL##NAME(m, #NAME, BASE);

    DECLARE_EXC(Error, NULL);
    DECLARE_EXC(MemoryError, CLError.ptr());
    DECLARE_EXC(LogicError, CLError.ptr());
    DECLARE_EXC(RuntimeError, CLError.ptr());

    py::register_exception_translator(
        [](std::exception_ptr p)
        {
          try
          {
            if (p) std::rethrow_exception(p);
          }
          catch (pyopencl::error &err)
          {
            py::object err_obj = py::cast(err);
            if (err.code() == CL_MEM_OBJECT_ALLOCATION_FAILURE)
              PyErr_SetObject(CLMemoryError.ptr(), err_obj.ptr());
            else if (err.code() <= CL_INVALID_VALUE)
              PyErr_SetObject(CLLogicError.ptr(), err_obj.ptr());
            else if (err.code() > CL_INVALID_VALUE && err.code() < CL_SUCCESS)
              PyErr_SetObject(CLRuntimeError.ptr(), err_obj.ptr());
            else
              PyErr_SetObject(CLError.ptr(), err_obj.ptr());
          }
        });
  }
  // }}}

  // {{{ error record

  {
    typedef error cls;
    py::class_<error> (m, "_ErrorRecord")
      .def(py::init<const char *, cl_int, const char *>(),
          py::arg("routine"),
          py::arg("code"),
          py::arg("msg"))
      .DEF_SIMPLE_METHOD(routine)
      .DEF_SIMPLE_METHOD(code)
      .DEF_SIMPLE_METHOD(what)
      .DEF_SIMPLE_METHOD(is_out_of_memory)
      .def("_program", &cls::get_program)
      ;
  }

  // }}}

  // {{{ constants
#define ADD_ATTR(PREFIX, NAME) \
  cls.attr(#NAME) = CL_##PREFIX##NAME
#define ADD_ATTR_SUFFIX(PREFIX, NAME, SUFFIX) \
  cls.attr(#NAME) = CL_##PREFIX##NAME##SUFFIX

  {
    py::class_<status_code> cls(m, "status_code");

    ADD_ATTR(, SUCCESS);
    ADD_ATTR(, DEVICE_NOT_FOUND);
    ADD_ATTR(, DEVICE_NOT_AVAILABLE);
#if !(defined(CL_PLATFORM_NVIDIA) && CL_PLATFORM_NVIDIA == 0x3001)
    ADD_ATTR(, COMPILER_NOT_AVAILABLE);
#endif
    ADD_ATTR(, MEM_OBJECT_ALLOCATION_FAILURE);
    ADD_ATTR(, OUT_OF_RESOURCES);
    ADD_ATTR(, OUT_OF_HOST_MEMORY);
    ADD_ATTR(, PROFILING_INFO_NOT_AVAILABLE);
    ADD_ATTR(, MEM_COPY_OVERLAP);
    ADD_ATTR(, IMAGE_FORMAT_MISMATCH);
    ADD_ATTR(, IMAGE_FORMAT_NOT_SUPPORTED);
    ADD_ATTR(, BUILD_PROGRAM_FAILURE);
    ADD_ATTR(, MAP_FAILURE);

    ADD_ATTR(, INVALID_VALUE);
    ADD_ATTR(, INVALID_DEVICE_TYPE);
    ADD_ATTR(, INVALID_PLATFORM);
    ADD_ATTR(, INVALID_DEVICE);
    ADD_ATTR(, INVALID_CONTEXT);
    ADD_ATTR(, INVALID_QUEUE_PROPERTIES);
    ADD_ATTR(, INVALID_COMMAND_QUEUE);
    ADD_ATTR(, INVALID_HOST_PTR);
    ADD_ATTR(, INVALID_MEM_OBJECT);
    ADD_ATTR(, INVALID_IMAGE_FORMAT_DESCRIPTOR);
    ADD_ATTR(, INVALID_IMAGE_SIZE);
    ADD_ATTR(, INVALID_SAMPLER);
    ADD_ATTR(, INVALID_BINARY);
    ADD_ATTR(, INVALID_BUILD_OPTIONS);
    ADD_ATTR(, INVALID_PROGRAM);
    ADD_ATTR(, INVALID_PROGRAM_EXECUTABLE);
    ADD_ATTR(, INVALID_KERNEL_NAME);
    ADD_ATTR(, INVALID_KERNEL_DEFINITION);
    ADD_ATTR(, INVALID_KERNEL);
    ADD_ATTR(, INVALID_ARG_INDEX);
    ADD_ATTR(, INVALID_ARG_VALUE);
    ADD_ATTR(, INVALID_ARG_SIZE);
    ADD_ATTR(, INVALID_KERNEL_ARGS);
    ADD_ATTR(, INVALID_WORK_DIMENSION);
    ADD_ATTR(, INVALID_WORK_GROUP_SIZE);
    ADD_ATTR(, INVALID_WORK_ITEM_SIZE);
    ADD_ATTR(, INVALID_GLOBAL_OFFSET);
    ADD_ATTR(, INVALID_EVENT_WAIT_LIST);
    ADD_ATTR(, INVALID_EVENT);
    ADD_ATTR(, INVALID_OPERATION);
    ADD_ATTR(, INVALID_GL_OBJECT);
    ADD_ATTR(, INVALID_BUFFER_SIZE);
    ADD_ATTR(, INVALID_MIP_LEVEL);

#if defined(cl_khr_icd) && (cl_khr_icd >= 1)
    ADD_ATTR(, PLATFORM_NOT_FOUND_KHR);
#endif

#if defined(cl_khr_gl_sharing) && (cl_khr_gl_sharing >= 1)
    ADD_ATTR(, INVALID_GL_SHAREGROUP_REFERENCE_KHR);
#endif

#if PYOPENCL_CL_VERSION >= 0x1010
    ADD_ATTR(, MISALIGNED_SUB_BUFFER_OFFSET);
    ADD_ATTR(, EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
    ADD_ATTR(, INVALID_GLOBAL_WORK_SIZE);
#endif

#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR(, COMPILE_PROGRAM_FAILURE);
    ADD_ATTR(, LINKER_NOT_AVAILABLE);
    ADD_ATTR(, LINK_PROGRAM_FAILURE);
    ADD_ATTR(, DEVICE_PARTITION_FAILED);
    ADD_ATTR(, KERNEL_ARG_INFO_NOT_AVAILABLE);
    ADD_ATTR(, INVALID_IMAGE_DESCRIPTOR);
    ADD_ATTR(, INVALID_COMPILER_OPTIONS);
    ADD_ATTR(, INVALID_LINKER_OPTIONS);
    ADD_ATTR(, INVALID_DEVICE_PARTITION_COUNT);
#endif

#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR(, INVALID_PIPE_SIZE);
    ADD_ATTR(, INVALID_DEVICE_QUEUE);
#endif

#if PYOPENCL_CL_VERSION >= 0x2020
    ADD_ATTR(, INVALID_SPEC_ID);
    ADD_ATTR(, MAX_SIZE_RESTRICTION_EXCEEDED);
#endif

#if defined(cl_ext_device_fission) && defined(PYOPENCL_USE_DEVICE_FISSION)
    ADD_ATTR(, DEVICE_PARTITION_FAILED_EXT);
    ADD_ATTR(, INVALID_PARTITION_COUNT_EXT);
    ADD_ATTR(, INVALID_PARTITION_NAME_EXT);
#endif
  }

  {
    py::class_<platform_info> cls(m, "platform_info");
    ADD_ATTR(PLATFORM_, PROFILE);
    ADD_ATTR(PLATFORM_, VERSION);
    ADD_ATTR(PLATFORM_, NAME);
    ADD_ATTR(PLATFORM_, VENDOR);
#if !(defined(CL_PLATFORM_NVIDIA) && CL_PLATFORM_NVIDIA == 0x3001)
    ADD_ATTR(PLATFORM_, EXTENSIONS);
#endif
#if PYOPENCL_CL_VERSION >= 0x2010
    ADD_ATTR(PLATFORM_, HOST_TIMER_RESOLUTION);
#endif
#if PYOPENCL_CL_VERSION >= 0x3000
    ADD_ATTR(PLATFORM_, NUMERIC_VERSION);
    ADD_ATTR(PLATFORM_, EXTENSIONS_WITH_VERSION);
#endif
  }

  {
    py::class_<device_type> cls(m, "device_type");
    ADD_ATTR(DEVICE_TYPE_, DEFAULT);
    ADD_ATTR(DEVICE_TYPE_, CPU);
    ADD_ATTR(DEVICE_TYPE_, GPU);
    ADD_ATTR(DEVICE_TYPE_, ACCELERATOR);
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR(DEVICE_TYPE_, CUSTOM);
#endif
    ADD_ATTR(DEVICE_TYPE_, ALL);
  }

  {
    py::class_<device_info> cls(m, "device_info");
    ADD_ATTR(DEVICE_, TYPE);
    ADD_ATTR(DEVICE_, VENDOR_ID);
    ADD_ATTR(DEVICE_, MAX_COMPUTE_UNITS);
    ADD_ATTR(DEVICE_, MAX_WORK_ITEM_DIMENSIONS);
    ADD_ATTR(DEVICE_, MAX_WORK_GROUP_SIZE);
    ADD_ATTR(DEVICE_, MAX_WORK_ITEM_SIZES);
    ADD_ATTR(DEVICE_, PREFERRED_VECTOR_WIDTH_CHAR);
    ADD_ATTR(DEVICE_, PREFERRED_VECTOR_WIDTH_SHORT);
    ADD_ATTR(DEVICE_, PREFERRED_VECTOR_WIDTH_INT);
    ADD_ATTR(DEVICE_, PREFERRED_VECTOR_WIDTH_LONG);
    ADD_ATTR(DEVICE_, PREFERRED_VECTOR_WIDTH_FLOAT);
    ADD_ATTR(DEVICE_, PREFERRED_VECTOR_WIDTH_DOUBLE);
    ADD_ATTR(DEVICE_, MAX_CLOCK_FREQUENCY);
    ADD_ATTR(DEVICE_, ADDRESS_BITS);
    ADD_ATTR(DEVICE_, MAX_READ_IMAGE_ARGS);
    ADD_ATTR(DEVICE_, MAX_WRITE_IMAGE_ARGS);
    ADD_ATTR(DEVICE_, MAX_MEM_ALLOC_SIZE);
    ADD_ATTR(DEVICE_, IMAGE2D_MAX_WIDTH);
    ADD_ATTR(DEVICE_, IMAGE2D_MAX_HEIGHT);
    ADD_ATTR(DEVICE_, IMAGE3D_MAX_WIDTH);
    ADD_ATTR(DEVICE_, IMAGE3D_MAX_HEIGHT);
    ADD_ATTR(DEVICE_, IMAGE3D_MAX_DEPTH);
    ADD_ATTR(DEVICE_, IMAGE_SUPPORT);
    ADD_ATTR(DEVICE_, MAX_PARAMETER_SIZE);
    ADD_ATTR(DEVICE_, MAX_SAMPLERS);
    ADD_ATTR(DEVICE_, MEM_BASE_ADDR_ALIGN);
    ADD_ATTR(DEVICE_, MIN_DATA_TYPE_ALIGN_SIZE);
    ADD_ATTR(DEVICE_, SINGLE_FP_CONFIG);
#ifdef CL_DEVICE_DOUBLE_FP_CONFIG
    ADD_ATTR(DEVICE_, DOUBLE_FP_CONFIG);
#endif
#ifdef CL_DEVICE_HALF_FP_CONFIG
    ADD_ATTR(DEVICE_, HALF_FP_CONFIG);
#endif
    ADD_ATTR(DEVICE_, GLOBAL_MEM_CACHE_TYPE);
    ADD_ATTR(DEVICE_, GLOBAL_MEM_CACHELINE_SIZE);
    ADD_ATTR(DEVICE_, GLOBAL_MEM_CACHE_SIZE);
    ADD_ATTR(DEVICE_, GLOBAL_MEM_SIZE);
    ADD_ATTR(DEVICE_, MAX_CONSTANT_BUFFER_SIZE);
    ADD_ATTR(DEVICE_, MAX_CONSTANT_ARGS);
    ADD_ATTR(DEVICE_, LOCAL_MEM_TYPE);
    ADD_ATTR(DEVICE_, LOCAL_MEM_SIZE);
    ADD_ATTR(DEVICE_, ERROR_CORRECTION_SUPPORT);
    ADD_ATTR(DEVICE_, PROFILING_TIMER_RESOLUTION);
    ADD_ATTR(DEVICE_, ENDIAN_LITTLE);
    ADD_ATTR(DEVICE_, AVAILABLE);
    ADD_ATTR(DEVICE_, COMPILER_AVAILABLE);
    ADD_ATTR(DEVICE_, EXECUTION_CAPABILITIES);
    ADD_ATTR(DEVICE_, QUEUE_PROPERTIES);
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR(DEVICE_, QUEUE_ON_HOST_PROPERTIES);
#endif
    ADD_ATTR(DEVICE_, NAME);
    ADD_ATTR(DEVICE_, VENDOR);
    ADD_ATTR(, DRIVER_VERSION);
    ADD_ATTR(DEVICE_, VERSION);
    ADD_ATTR(DEVICE_, PROFILE);
    ADD_ATTR(DEVICE_, VERSION);
    ADD_ATTR(DEVICE_, EXTENSIONS);
    ADD_ATTR(DEVICE_, PLATFORM);
#if PYOPENCL_CL_VERSION >= 0x1010
    ADD_ATTR(DEVICE_, PREFERRED_VECTOR_WIDTH_HALF);
    ADD_ATTR(DEVICE_, HOST_UNIFIED_MEMORY);
    ADD_ATTR(DEVICE_, NATIVE_VECTOR_WIDTH_CHAR);
    ADD_ATTR(DEVICE_, NATIVE_VECTOR_WIDTH_SHORT);
    ADD_ATTR(DEVICE_, NATIVE_VECTOR_WIDTH_INT);
    ADD_ATTR(DEVICE_, NATIVE_VECTOR_WIDTH_LONG);
    ADD_ATTR(DEVICE_, NATIVE_VECTOR_WIDTH_FLOAT);
    ADD_ATTR(DEVICE_, NATIVE_VECTOR_WIDTH_DOUBLE);
    ADD_ATTR(DEVICE_, NATIVE_VECTOR_WIDTH_HALF);
    ADD_ATTR(DEVICE_, OPENCL_C_VERSION);
#endif
// support for cl_nv_device_attribute_query
#ifdef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
    ADD_ATTR(DEVICE_, COMPUTE_CAPABILITY_MAJOR_NV);
    ADD_ATTR(DEVICE_, COMPUTE_CAPABILITY_MINOR_NV);
    ADD_ATTR(DEVICE_, REGISTERS_PER_BLOCK_NV);
    ADD_ATTR(DEVICE_, WARP_SIZE_NV);
    ADD_ATTR(DEVICE_, GPU_OVERLAP_NV);
    ADD_ATTR(DEVICE_, KERNEL_EXEC_TIMEOUT_NV);
    ADD_ATTR(DEVICE_, INTEGRATED_MEMORY_NV);
    // Nvidia specific device attributes, not defined in Khronos CL/cl_ext.h
#ifdef CL_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT_NV
    ADD_ATTR(DEVICE_, ATTRIBUTE_ASYNC_ENGINE_COUNT_NV);
#endif
#ifdef CL_DEVICE_PCI_BUS_ID_NV
    ADD_ATTR(DEVICE_, PCI_BUS_ID_NV);
#endif
#ifdef CL_DEVICE_PCI_SLOT_ID_NV
    ADD_ATTR(DEVICE_, PCI_SLOT_ID_NV);
#endif
#endif

// {{{ cl_amd_device_attribute_query
#ifdef CL_DEVICE_PROFILING_TIMER_OFFSET_AMD
    ADD_ATTR(DEVICE_, PROFILING_TIMER_OFFSET_AMD);
#endif
#ifdef CL_DEVICE_TOPOLOGY_AMD
    ADD_ATTR(DEVICE_, TOPOLOGY_AMD);
#endif
#ifdef CL_DEVICE_BOARD_NAME_AMD
    ADD_ATTR(DEVICE_, BOARD_NAME_AMD);
#endif
#ifdef CL_DEVICE_GLOBAL_FREE_MEMORY_AMD
    ADD_ATTR(DEVICE_, GLOBAL_FREE_MEMORY_AMD);
#endif
#ifdef CL_DEVICE_SIMD_PER_COMPUTE_UNIT_AMD
    ADD_ATTR(DEVICE_, SIMD_PER_COMPUTE_UNIT_AMD);
#endif
#ifdef CL_DEVICE_SIMD_WIDTH_AMD
    ADD_ATTR(DEVICE_, SIMD_WIDTH_AMD);
#endif
#ifdef CL_DEVICE_SIMD_INSTRUCTION_WIDTH_AMD
    ADD_ATTR(DEVICE_, SIMD_INSTRUCTION_WIDTH_AMD);
#endif
#ifdef CL_DEVICE_WAVEFRONT_WIDTH_AMD
    ADD_ATTR(DEVICE_, WAVEFRONT_WIDTH_AMD);
#endif
#ifdef CL_DEVICE_GLOBAL_MEM_CHANNELS_AMD
    ADD_ATTR(DEVICE_, GLOBAL_MEM_CHANNELS_AMD);
#endif
#ifdef CL_DEVICE_GLOBAL_MEM_CHANNEL_BANKS_AMD
    ADD_ATTR(DEVICE_, GLOBAL_MEM_CHANNEL_BANKS_AMD);
#endif
#ifdef CL_DEVICE_GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD
    ADD_ATTR(DEVICE_, GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD);
#endif
#ifdef CL_DEVICE_LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD
    ADD_ATTR(DEVICE_, LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD);
#endif
#ifdef CL_DEVICE_LOCAL_MEM_BANKS_AMD
    ADD_ATTR(DEVICE_, LOCAL_MEM_BANKS_AMD);
#endif
#ifdef CL_DEVICE_THREAD_TRACE_SUPPORTED_AMD
    ADD_ATTR(DEVICE_, THREAD_TRACE_SUPPORTED_AMD);
#endif
#ifdef CL_DEVICE_GFXIP_MAJOR_AMD
    ADD_ATTR(DEVICE_, GFXIP_MAJOR_AMD);
#endif
#ifdef CL_DEVICE_GFXIP_MINOR_AMD
    ADD_ATTR(DEVICE_, GFXIP_MINOR_AMD);
#endif
#ifdef CL_DEVICE_AVAILABLE_ASYNC_QUEUES_AMD
    ADD_ATTR(DEVICE_, AVAILABLE_ASYNC_QUEUES_AMD);
#endif
#ifdef CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_AMD
    ADD_ATTR(DEVICE_, PREFERRED_WORK_GROUP_SIZE_AMD);
#endif
#ifdef CL_DEVICE_MAX_WORK_GROUP_SIZE_AMD
    ADD_ATTR(DEVICE_, MAX_WORK_GROUP_SIZE_AMD);
#endif
#ifdef CL_DEVICE_PREFERRED_CONSTANT_BUFFER_SIZE_AMD
    ADD_ATTR(DEVICE_, PREFERRED_CONSTANT_BUFFER_SIZE_AMD);
#endif
#ifdef CL_DEVICE_PCIE_ID_AMD
    ADD_ATTR(DEVICE_, PCIE_ID_AMD);
#endif
// }}}

#ifdef CL_DEVICE_MAX_ATOMIC_COUNTERS_EXT
    ADD_ATTR(DEVICE_, MAX_ATOMIC_COUNTERS_EXT);
#endif
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR(DEVICE_, LINKER_AVAILABLE);
    ADD_ATTR(DEVICE_, BUILT_IN_KERNELS);
    ADD_ATTR(DEVICE_, IMAGE_MAX_BUFFER_SIZE);
    ADD_ATTR(DEVICE_, IMAGE_MAX_ARRAY_SIZE);
    ADD_ATTR(DEVICE_, PARENT_DEVICE);
    ADD_ATTR(DEVICE_, PARTITION_MAX_SUB_DEVICES);
    ADD_ATTR(DEVICE_, PARTITION_PROPERTIES);
    ADD_ATTR(DEVICE_, PARTITION_AFFINITY_DOMAIN);
    ADD_ATTR(DEVICE_, PARTITION_TYPE);
    ADD_ATTR(DEVICE_, REFERENCE_COUNT);
    ADD_ATTR(DEVICE_, PREFERRED_INTEROP_USER_SYNC);
    ADD_ATTR(DEVICE_, PRINTF_BUFFER_SIZE);
#endif
#ifdef cl_khr_image2d_from_buffer
    ADD_ATTR(DEVICE_, IMAGE_PITCH_ALIGNMENT);
    ADD_ATTR(DEVICE_, IMAGE_BASE_ADDRESS_ALIGNMENT);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR(DEVICE_, MAX_READ_WRITE_IMAGE_ARGS);
    ADD_ATTR(DEVICE_, MAX_GLOBAL_VARIABLE_SIZE);
    ADD_ATTR(DEVICE_, QUEUE_ON_DEVICE_PROPERTIES);
    ADD_ATTR(DEVICE_, QUEUE_ON_DEVICE_PREFERRED_SIZE);
    ADD_ATTR(DEVICE_, QUEUE_ON_DEVICE_MAX_SIZE);
    ADD_ATTR(DEVICE_, MAX_ON_DEVICE_QUEUES);
    ADD_ATTR(DEVICE_, MAX_ON_DEVICE_EVENTS);
    ADD_ATTR(DEVICE_, SVM_CAPABILITIES);
    ADD_ATTR(DEVICE_, GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE);
    ADD_ATTR(DEVICE_, MAX_PIPE_ARGS);
    ADD_ATTR(DEVICE_, PIPE_MAX_ACTIVE_RESERVATIONS);
    ADD_ATTR(DEVICE_, PIPE_MAX_PACKET_SIZE);
    ADD_ATTR(DEVICE_, PREFERRED_PLATFORM_ATOMIC_ALIGNMENT);
    ADD_ATTR(DEVICE_, PREFERRED_GLOBAL_ATOMIC_ALIGNMENT);
    ADD_ATTR(DEVICE_, PREFERRED_LOCAL_ATOMIC_ALIGNMENT);
#endif
#if PYOPENCL_CL_VERSION >= 0x2010
    ADD_ATTR(DEVICE_, IL_VERSION);
    ADD_ATTR(DEVICE_, MAX_NUM_SUB_GROUPS);
    ADD_ATTR(DEVICE_, SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS);
#endif
#if PYOPENCL_CL_VERSION >= 0x3000
    ADD_ATTR(DEVICE_, NUMERIC_VERSION);
    ADD_ATTR(DEVICE_, EXTENSIONS_WITH_VERSION);
    ADD_ATTR(DEVICE_, ILS_WITH_VERSION);
    ADD_ATTR(DEVICE_, BUILT_IN_KERNELS_WITH_VERSION);
    ADD_ATTR(DEVICE_, ATOMIC_MEMORY_CAPABILITIES);
    ADD_ATTR(DEVICE_, ATOMIC_FENCE_CAPABILITIES);
    ADD_ATTR(DEVICE_, NON_UNIFORM_WORK_GROUP_SUPPORT);
    ADD_ATTR(DEVICE_, OPENCL_C_ALL_VERSIONS);
    ADD_ATTR(DEVICE_, PREFERRED_WORK_GROUP_SIZE_MULTIPLE);
    ADD_ATTR(DEVICE_, WORK_GROUP_COLLECTIVE_FUNCTIONS_SUPPORT);
    ADD_ATTR(DEVICE_, GENERIC_ADDRESS_SPACE_SUPPORT);
    ADD_ATTR(DEVICE_, OPENCL_C_FEATURES);
#ifdef CL_DEVICE_DEVICE_ENQUEUE_SUPPORT
    // some busted headers shipped by Debian have this
    cls.attr("DEVICE_ENQUEUE_CAPABILITIES") = CL_DEVICE_DEVICE_ENQUEUE_SUPPORT;
#else
    ADD_ATTR(DEVICE_, DEVICE_ENQUEUE_CAPABILITIES);
#endif
    ADD_ATTR(DEVICE_, PIPE_SUPPORT);
#endif
    /* cl_intel_advanced_motion_estimation */
#ifdef CL_DEVICE_ME_VERSION_INTEL
    ADD_ATTR(DEVICE_, ME_VERSION_INTEL);
#endif

    /* cl_qcom_ext_host_ptr */
#ifdef CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM
    ADD_ATTR(DEVICE_, EXT_MEM_PADDING_IN_BYTES_QCOM);
#endif
#ifdef CL_DEVICE_PAGE_SIZE_QCOM
    ADD_ATTR(DEVICE_, PAGE_SIZE_QCOM);
#endif

    /* cl_khr_spir */
#ifdef CL_DEVICE_SPIR_VERSIONS
    ADD_ATTR(DEVICE_, SPIR_VERSIONS);
#endif

    /* cl_altera_device_temperature */
#ifdef CL_DEVICE_CORE_TEMPERATURE_ALTERA
    ADD_ATTR(DEVICE_, CORE_TEMPERATURE_ALTERA);
#endif

    /* cl_intel_simultaneous_sharing */
#ifdef CL_DEVICE_SIMULTANEOUS_INTEROPS_INTEL
    ADD_ATTR(DEVICE_, SIMULTANEOUS_INTEROPS_INTEL);
#endif
#ifdef CL_DEVICE_NUM_SIMULTANEOUS_INTEROPS_INTEL
    ADD_ATTR(DEVICE_, NUM_SIMULTANEOUS_INTEROPS_INTEL);
#endif
  }

  {
    py::class_<device_topology_type_amd> cls(m, "device_topology_type_amd");
#ifdef CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD
    cls.attr("PCIE") = CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD;
#endif
  }

  {
    py::class_<device_fp_config> cls(m, "device_fp_config");
    ADD_ATTR(FP_, DENORM);
    ADD_ATTR(FP_, INF_NAN);
    ADD_ATTR(FP_, ROUND_TO_NEAREST);
    ADD_ATTR(FP_, ROUND_TO_ZERO);
    ADD_ATTR(FP_, ROUND_TO_INF);
    ADD_ATTR(FP_, FMA);
#if PYOPENCL_CL_VERSION >= 0x1010
    ADD_ATTR(FP_, SOFT_FLOAT);
#endif
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR(FP_, CORRECTLY_ROUNDED_DIVIDE_SQRT);
#endif
  }

  {
    py::class_<device_mem_cache_type> cls(m, "device_mem_cache_type");
    ADD_ATTR( , NONE);
    ADD_ATTR( , READ_ONLY_CACHE);
    ADD_ATTR( , READ_WRITE_CACHE);
  }

  {
    py::class_<device_local_mem_type> cls(m, "device_local_mem_type");
    ADD_ATTR( , LOCAL);
    ADD_ATTR( , GLOBAL);
  }

  {
    py::class_<device_exec_capabilities> cls(m, "device_exec_capabilities");
    ADD_ATTR(EXEC_, KERNEL);
    ADD_ATTR(EXEC_, NATIVE_KERNEL);
#ifdef CL_EXEC_IMMEDIATE_EXECUTION_INTEL
    ADD_ATTR(EXEC_, IMMEDIATE_EXECUTION_INTEL);
#endif
  }

  {
    py::class_<device_svm_capabilities> cls(m, "device_svm_capabilities");
#if PYOPENCL_CL_VERSION >= 0x2000
    // device_svm_capabilities
    ADD_ATTR(DEVICE_SVM_, COARSE_GRAIN_BUFFER);
    ADD_ATTR(DEVICE_SVM_, FINE_GRAIN_BUFFER);
    ADD_ATTR(DEVICE_SVM_, FINE_GRAIN_SYSTEM);
    ADD_ATTR(DEVICE_SVM_, ATOMICS);
#endif
  }

  {
    py::class_<command_queue_properties> cls(m, "command_queue_properties");
    ADD_ATTR(QUEUE_, OUT_OF_ORDER_EXEC_MODE_ENABLE);
    ADD_ATTR(QUEUE_, PROFILING_ENABLE);
#ifdef CL_QUEUE_IMMEDIATE_EXECUTION_ENABLE_INTEL
    ADD_ATTR(QUEUE_, IMMEDIATE_EXECUTION_ENABLE_INTEL);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR(QUEUE_, ON_DEVICE);
    ADD_ATTR(QUEUE_, ON_DEVICE_DEFAULT);
#endif
  }

  {
    py::class_<context_info> cls(m, "context_info");
    ADD_ATTR(CONTEXT_, REFERENCE_COUNT);
    ADD_ATTR(CONTEXT_, DEVICES);
    ADD_ATTR(CONTEXT_, PROPERTIES);
#if PYOPENCL_CL_VERSION >= 0x1010
    ADD_ATTR(CONTEXT_, NUM_DEVICES);
#endif
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR(CONTEXT_, INTEROP_USER_SYNC);
#endif
  }

  {
    py::class_<gl_context_info> cls(m, "gl_context_info");
#if defined(cl_khr_gl_sharing) && (cl_khr_gl_sharing >= 1)
    ADD_ATTR(, CURRENT_DEVICE_FOR_GL_CONTEXT_KHR);
    ADD_ATTR(, DEVICES_FOR_GL_CONTEXT_KHR);
#endif
  }

  {
    py::class_<context_properties> cls(m, "context_properties");
    ADD_ATTR(CONTEXT_, PLATFORM);
#if defined(cl_khr_gl_sharing) && (cl_khr_gl_sharing >= 1)
    ADD_ATTR( ,GL_CONTEXT_KHR);
    ADD_ATTR( ,EGL_DISPLAY_KHR);
    ADD_ATTR( ,GLX_DISPLAY_KHR);
    ADD_ATTR( ,WGL_HDC_KHR);
    ADD_ATTR( ,CGL_SHAREGROUP_KHR);
#endif
#if defined(__APPLE__) && defined(HAVE_GL)
    ADD_ATTR( ,CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE);
#endif /* __APPLE__ */
// cl_amd_offline_devices
#ifdef CL_CONTEXT_OFFLINE_DEVICES_AMD
    ADD_ATTR(CONTEXT_, OFFLINE_DEVICES_AMD);
#endif
  }

  {
    py::class_<command_queue_info> cls(m, "command_queue_info");
    ADD_ATTR(QUEUE_, CONTEXT);
    ADD_ATTR(QUEUE_, DEVICE);
    ADD_ATTR(QUEUE_, REFERENCE_COUNT);
    ADD_ATTR(QUEUE_, PROPERTIES);
#if PYOPENCL_CL_VERSION >= 0x3000
    ADD_ATTR(QUEUE_, PROPERTIES_ARRAY);
#endif
  }

  {
    py::class_<queue_properties> cls(m, "queue_properties");
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR(QUEUE_, PROPERTIES);
    ADD_ATTR(QUEUE_, SIZE);
#endif
#if PYOPENCL_CL_VERSION >= 0x2010
    ADD_ATTR(QUEUE_, DEVICE_DEFAULT);
#endif
  }

  {
    py::class_<mem_flags> cls(m, "mem_flags");
    ADD_ATTR(MEM_, READ_WRITE);
    ADD_ATTR(MEM_, WRITE_ONLY);
    ADD_ATTR(MEM_, READ_ONLY);
    ADD_ATTR(MEM_, USE_HOST_PTR);
    ADD_ATTR(MEM_, ALLOC_HOST_PTR);
    ADD_ATTR(MEM_, COPY_HOST_PTR);
#ifdef cl_amd_device_memory_flags
    ADD_ATTR(MEM_, USE_PERSISTENT_MEM_AMD);
#endif
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR(MEM_, HOST_WRITE_ONLY);
    ADD_ATTR(MEM_, HOST_READ_ONLY);
    ADD_ATTR(MEM_, HOST_NO_ACCESS);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR(MEM_, KERNEL_READ_AND_WRITE);
#endif
  }

  {
    py::class_<svm_mem_flags> cls(m, "svm_mem_flags");
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR(MEM_, READ_WRITE);
    ADD_ATTR(MEM_, WRITE_ONLY);
    ADD_ATTR(MEM_, READ_ONLY);
    ADD_ATTR(MEM_, SVM_FINE_GRAIN_BUFFER);
    ADD_ATTR(MEM_, SVM_ATOMICS);
#endif
  }

  {
    py::class_<channel_order> cls(m, "channel_order");
    ADD_ATTR( , R);
    ADD_ATTR( , A);
    ADD_ATTR( , RG);
    ADD_ATTR( , RA);
    ADD_ATTR( , RGB);
    ADD_ATTR( , RGBA);
    ADD_ATTR( , BGRA);
    ADD_ATTR( , INTENSITY);
    ADD_ATTR( , LUMINANCE);
#if PYOPENCL_CL_VERSION >= 0x1010
    ADD_ATTR( , Rx);
    ADD_ATTR( , RGx);
    ADD_ATTR( , RGBx);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR(  , sRGB);
    ADD_ATTR(  , sRGBx);
    ADD_ATTR(  , sRGBA);
    ADD_ATTR(  , sBGRA);
    ADD_ATTR(  , ABGR);
#endif
  }

  {
    py::class_<channel_type> cls(m, "channel_type");
    ADD_ATTR( , SNORM_INT8);
    ADD_ATTR( , SNORM_INT16);
    ADD_ATTR( , UNORM_INT8);
    ADD_ATTR( , UNORM_INT16);
    ADD_ATTR( , UNORM_SHORT_565);
    ADD_ATTR( , UNORM_SHORT_555);
    ADD_ATTR( , UNORM_INT_101010);
    ADD_ATTR( , SIGNED_INT8);
    ADD_ATTR( , SIGNED_INT16);
    ADD_ATTR( , SIGNED_INT32);
    ADD_ATTR( , UNSIGNED_INT8);
    ADD_ATTR( , UNSIGNED_INT16);
    ADD_ATTR( , UNSIGNED_INT32);
    ADD_ATTR( , HALF_FLOAT);
    ADD_ATTR( , FLOAT);
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR( , UNORM_INT24);
#endif
#if PYOPENCL_CL_VERSION >= 0x2010
    ADD_ATTR( , UNORM_INT_101010_2);
#endif
  }

  {
    py::class_<mem_object_type> cls(m, "mem_object_type");
    ADD_ATTR(MEM_OBJECT_, BUFFER);
    ADD_ATTR(MEM_OBJECT_, IMAGE2D);
    ADD_ATTR(MEM_OBJECT_, IMAGE3D);
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR(MEM_OBJECT_, IMAGE2D_ARRAY);
    ADD_ATTR(MEM_OBJECT_, IMAGE1D);
    ADD_ATTR(MEM_OBJECT_, IMAGE1D_ARRAY);
    ADD_ATTR(MEM_OBJECT_, IMAGE1D_BUFFER);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR(MEM_OBJECT_, PIPE);
#endif
  }

  {
    py::class_<mem_info> cls(m, "mem_info");
    ADD_ATTR(MEM_, TYPE);
    ADD_ATTR(MEM_, FLAGS);
    ADD_ATTR(MEM_, SIZE);
    ADD_ATTR(MEM_, HOST_PTR);
    ADD_ATTR(MEM_, MAP_COUNT);
    ADD_ATTR(MEM_, REFERENCE_COUNT);
    ADD_ATTR(MEM_, CONTEXT);
#if PYOPENCL_CL_VERSION >= 0x1010
    ADD_ATTR(MEM_, ASSOCIATED_MEMOBJECT);
    ADD_ATTR(MEM_, OFFSET);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR(MEM_, USES_SVM_POINTER);
#endif
#if PYOPENCL_CL_VERSION >= 0x3000
    ADD_ATTR(MEM_, PROPERTIES);
#endif
  }

  {
    py::class_<image_info> cls(m, "image_info");
    ADD_ATTR(IMAGE_, FORMAT);
    ADD_ATTR(IMAGE_, ELEMENT_SIZE);
    ADD_ATTR(IMAGE_, ROW_PITCH);
    ADD_ATTR(IMAGE_, SLICE_PITCH);
    ADD_ATTR(IMAGE_, WIDTH);
    ADD_ATTR(IMAGE_, HEIGHT);
    ADD_ATTR(IMAGE_, DEPTH);
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR(IMAGE_, ARRAY_SIZE);
    ADD_ATTR(IMAGE_, BUFFER);
    ADD_ATTR(IMAGE_, NUM_MIP_LEVELS);
    ADD_ATTR(IMAGE_, NUM_SAMPLES);
#endif
  }

  {
    py::class_<pipe_info> cls(m, "pipe_info");
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR(PIPE_, PACKET_SIZE);
    ADD_ATTR(PIPE_, MAX_PACKETS);
#endif
#if PYOPENCL_CL_VERSION >= 0x3000
    ADD_ATTR(PIPE_, PROPERTIES);
#endif
  }

  {
    py::class_<pipe_properties> cls(m, "pipe_properties");
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR(PIPE_, PACKET_SIZE);
    ADD_ATTR(PIPE_, MAX_PACKETS);
#endif
  }
  {
    py::class_<addressing_mode> cls(m, "addressing_mode");
    ADD_ATTR(ADDRESS_, NONE);
    ADD_ATTR(ADDRESS_, CLAMP_TO_EDGE);
    ADD_ATTR(ADDRESS_, CLAMP);
    ADD_ATTR(ADDRESS_, REPEAT);
#if PYOPENCL_CL_VERSION >= 0x1010
    ADD_ATTR(ADDRESS_, MIRRORED_REPEAT);
#endif
  }

  {
    py::class_<filter_mode> cls(m, "filter_mode");
    ADD_ATTR(FILTER_, NEAREST);
    ADD_ATTR(FILTER_, LINEAR);
  }

  {
    py::class_<sampler_info> cls(m, "sampler_info");
    ADD_ATTR(SAMPLER_, REFERENCE_COUNT);
    ADD_ATTR(SAMPLER_, CONTEXT);
    ADD_ATTR(SAMPLER_, NORMALIZED_COORDS);
    ADD_ATTR(SAMPLER_, ADDRESSING_MODE);
    ADD_ATTR(SAMPLER_, FILTER_MODE);
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR(SAMPLER_, MIP_FILTER_MODE);
    ADD_ATTR(SAMPLER_, LOD_MIN);
    ADD_ATTR(SAMPLER_, LOD_MAX);
#endif
#if PYOPENCL_CL_VERSION >= 0x3000
    ADD_ATTR(SAMPLER_, PROPERTIES);
#endif
// {{{ cl_khr_mipmap_image
#ifdef CL_SAMPLER_MIP_FILTER_MODE_KHR
    ADD_ATTR(SAMPLER_, MIP_FILTER_MODE_KHR);
    ADD_ATTR(SAMPLER_, LOD_MIN_KHR);
    ADD_ATTR(SAMPLER_, LOD_MAX_KHR);
#endif
// }}}
  }

  {
    py::class_<sampler_properties> cls(m, "sampler_properties");
    ADD_ATTR(SAMPLER_, NORMALIZED_COORDS);
    ADD_ATTR(SAMPLER_, ADDRESSING_MODE);
    ADD_ATTR(SAMPLER_, FILTER_MODE);
  }

  {
    py::class_<map_flags> cls(m, "map_flags");
    ADD_ATTR(MAP_, READ);
    ADD_ATTR(MAP_, WRITE);
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR(MAP_, WRITE_INVALIDATE_REGION);
#endif
  }

  {
    py::class_<program_info> cls(m, "program_info");
    ADD_ATTR(PROGRAM_, REFERENCE_COUNT);
    ADD_ATTR(PROGRAM_, CONTEXT);
    ADD_ATTR(PROGRAM_, NUM_DEVICES);
    ADD_ATTR(PROGRAM_, DEVICES);
    ADD_ATTR(PROGRAM_, SOURCE);
    ADD_ATTR(PROGRAM_, BINARY_SIZES);
    ADD_ATTR(PROGRAM_, BINARIES);
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR(PROGRAM_, NUM_KERNELS);
    ADD_ATTR(PROGRAM_, KERNEL_NAMES);
#endif
#if PYOPENCL_CL_VERSION >= 0x2010
    ADD_ATTR(PROGRAM_, IL);
#endif
#if PYOPENCL_CL_VERSION >= 0x2020
    ADD_ATTR(PROGRAM_, SCOPE_GLOBAL_CTORS_PRESENT);
    ADD_ATTR(PROGRAM_, SCOPE_GLOBAL_DTORS_PRESENT);
#endif
  }

  {
    py::class_<program_build_info> cls(m, "program_build_info");
    ADD_ATTR(PROGRAM_BUILD_, STATUS);
    ADD_ATTR(PROGRAM_BUILD_, OPTIONS);
    ADD_ATTR(PROGRAM_BUILD_, LOG);
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR(PROGRAM_, BINARY_TYPE);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR(PROGRAM_BUILD_, GLOBAL_VARIABLE_TOTAL_SIZE);
#endif
  }

  {
    py::class_<program_binary_type> cls(m, "program_binary_type");
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR(PROGRAM_BINARY_TYPE_, NONE);
    ADD_ATTR(PROGRAM_BINARY_TYPE_, COMPILED_OBJECT);
    ADD_ATTR(PROGRAM_BINARY_TYPE_, LIBRARY);
    ADD_ATTR(PROGRAM_BINARY_TYPE_, EXECUTABLE);
#endif
  }

  {
    py::class_<kernel_info> cls(m, "kernel_info");
    ADD_ATTR(KERNEL_, FUNCTION_NAME);
    ADD_ATTR(KERNEL_, NUM_ARGS);
    ADD_ATTR(KERNEL_, REFERENCE_COUNT);
    ADD_ATTR(KERNEL_, CONTEXT);
    ADD_ATTR(KERNEL_, PROGRAM);
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR(KERNEL_, ATTRIBUTES);
#endif
  }

  {
    py::class_<kernel_arg_info> cls(m, "kernel_arg_info");
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR(KERNEL_ARG_, ADDRESS_QUALIFIER);
    ADD_ATTR(KERNEL_ARG_, ACCESS_QUALIFIER);
    ADD_ATTR(KERNEL_ARG_, TYPE_NAME);
    ADD_ATTR(KERNEL_ARG_, TYPE_QUALIFIER);
    ADD_ATTR(KERNEL_ARG_, NAME);
#endif
  }

  {
    py::class_<kernel_arg_address_qualifier> cls(
        m, "kernel_arg_address_qualifier");
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR(KERNEL_ARG_ADDRESS_, GLOBAL);
    ADD_ATTR(KERNEL_ARG_ADDRESS_, LOCAL);
    ADD_ATTR(KERNEL_ARG_ADDRESS_, CONSTANT);
    ADD_ATTR(KERNEL_ARG_ADDRESS_, PRIVATE);
#endif
  }

  {
    py::class_<kernel_arg_access_qualifier> cls(
        m, "kernel_arg_access_qualifier");
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR(KERNEL_ARG_ACCESS_, READ_ONLY);
    ADD_ATTR(KERNEL_ARG_ACCESS_, WRITE_ONLY);
    ADD_ATTR(KERNEL_ARG_ACCESS_, READ_WRITE);
    ADD_ATTR(KERNEL_ARG_ACCESS_, NONE);
#endif
  }

  {
    py::class_<kernel_arg_type_qualifier> cls(
        m, "kernel_arg_type_qualifier");
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR(KERNEL_ARG_TYPE_, NONE);
    ADD_ATTR(KERNEL_ARG_TYPE_, CONST);
    ADD_ATTR(KERNEL_ARG_TYPE_, RESTRICT);
    ADD_ATTR(KERNEL_ARG_TYPE_, VOLATILE);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR(KERNEL_ARG_TYPE_, PIPE);
#endif
  }

  {
    py::class_<kernel_work_group_info> cls(m, "kernel_work_group_info");
    ADD_ATTR(KERNEL_, WORK_GROUP_SIZE);
    ADD_ATTR(KERNEL_, COMPILE_WORK_GROUP_SIZE);
    ADD_ATTR(KERNEL_, LOCAL_MEM_SIZE);
#if PYOPENCL_CL_VERSION >= 0x1010
    ADD_ATTR(KERNEL_, PREFERRED_WORK_GROUP_SIZE_MULTIPLE);
    ADD_ATTR(KERNEL_, PRIVATE_MEM_SIZE);
#endif
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR(KERNEL_, GLOBAL_WORK_SIZE);
#endif
  }

  {
    py::class_<kernel_sub_group_info> cls(m, "kernel_sub_group_info");
#if PYOPENCL_CL_VERSION >= 0x2010
    ADD_ATTR(KERNEL_, MAX_SUB_GROUP_SIZE_FOR_NDRANGE);
    ADD_ATTR(KERNEL_, SUB_GROUP_COUNT_FOR_NDRANGE);
    ADD_ATTR(KERNEL_, LOCAL_SIZE_FOR_SUB_GROUP_COUNT);
    ADD_ATTR(KERNEL_, MAX_NUM_SUB_GROUPS);
    ADD_ATTR(KERNEL_, COMPILE_NUM_SUB_GROUPS);
#endif
  }

  {
    py::class_<event_info> cls(m, "event_info");
    ADD_ATTR(EVENT_, COMMAND_QUEUE);
    ADD_ATTR(EVENT_, COMMAND_TYPE);
    ADD_ATTR(EVENT_, REFERENCE_COUNT);
    ADD_ATTR(EVENT_, COMMAND_EXECUTION_STATUS);
#if PYOPENCL_CL_VERSION >= 0x1010
    ADD_ATTR(EVENT_, CONTEXT);
#endif
  }

  {
    py::class_<command_type> cls(m, "command_type");
    ADD_ATTR(COMMAND_, NDRANGE_KERNEL);
    ADD_ATTR(COMMAND_, TASK);
    ADD_ATTR(COMMAND_, NATIVE_KERNEL);
    ADD_ATTR(COMMAND_, READ_BUFFER);
    ADD_ATTR(COMMAND_, WRITE_BUFFER);
    ADD_ATTR(COMMAND_, COPY_BUFFER);
    ADD_ATTR(COMMAND_, READ_IMAGE);
    ADD_ATTR(COMMAND_, WRITE_IMAGE);
    ADD_ATTR(COMMAND_, COPY_IMAGE);
    ADD_ATTR(COMMAND_, COPY_IMAGE_TO_BUFFER);
    ADD_ATTR(COMMAND_, COPY_BUFFER_TO_IMAGE);
    ADD_ATTR(COMMAND_, MAP_BUFFER);
    ADD_ATTR(COMMAND_, MAP_IMAGE);
    ADD_ATTR(COMMAND_, UNMAP_MEM_OBJECT);
    ADD_ATTR(COMMAND_, MARKER);
    ADD_ATTR(COMMAND_, ACQUIRE_GL_OBJECTS);
    ADD_ATTR(COMMAND_, RELEASE_GL_OBJECTS);
#if PYOPENCL_CL_VERSION >= 0x1010
    ADD_ATTR(COMMAND_, READ_BUFFER_RECT);
    ADD_ATTR(COMMAND_, WRITE_BUFFER_RECT);
    ADD_ATTR(COMMAND_, COPY_BUFFER_RECT);
    ADD_ATTR(COMMAND_, USER);
#endif
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR(COMMAND_, BARRIER);
    ADD_ATTR(COMMAND_, MIGRATE_MEM_OBJECTS);
    ADD_ATTR(COMMAND_, FILL_BUFFER);
    ADD_ATTR(COMMAND_, FILL_IMAGE);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR(COMMAND_, SVM_FREE);
    ADD_ATTR(COMMAND_, SVM_MEMCPY);
    ADD_ATTR(COMMAND_, SVM_MEMFILL);
    ADD_ATTR(COMMAND_, SVM_MAP);
    ADD_ATTR(COMMAND_, SVM_UNMAP);
#endif
#if PYOPENCL_CL_VERSION >= 0x3000
    ADD_ATTR(COMMAND_, SVM_MIGRATE_MEM);
#endif
  }

  {
    py::class_<command_execution_status> cls(m, "command_execution_status");
    ADD_ATTR(, COMPLETE);
    ADD_ATTR(, RUNNING);
    ADD_ATTR(, SUBMITTED);
    ADD_ATTR(, QUEUED);
  }

  {
    py::class_<profiling_info> cls(m, "profiling_info");
    ADD_ATTR(PROFILING_COMMAND_, QUEUED);
    ADD_ATTR(PROFILING_COMMAND_, SUBMIT);
    ADD_ATTR(PROFILING_COMMAND_, START);
    ADD_ATTR(PROFILING_COMMAND_, END);
#if PYOPENCL_CL_VERSION >= 0x2000
    ADD_ATTR(PROFILING_COMMAND_, COMPLETE);
#endif
  }

/* not needed--filled in automatically by implementation.
#if PYOPENCL_CL_VERSION >= 0x1010
  {
    py::class_<buffer_create_type> cls(m, "buffer_create_type");
    ADD_ATTR(BUFFER_CREATE_TYPE_, REGION);
  }
#endif
*/

  {
    py::class_<mem_migration_flags> cls(
        m, "mem_migration_flags");
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR(MIGRATE_MEM_OBJECT_, HOST);
    ADD_ATTR(MIGRATE_MEM_OBJECT_, CONTENT_UNDEFINED);
#endif
  }

  {
    py::class_<device_partition_property> cls(
        m, "device_partition_property");
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR(DEVICE_PARTITION_, EQUALLY);
    ADD_ATTR(DEVICE_PARTITION_, BY_COUNTS);
    ADD_ATTR(DEVICE_PARTITION_, BY_COUNTS_LIST_END);
    ADD_ATTR(DEVICE_PARTITION_, BY_AFFINITY_DOMAIN);
#endif
  }

  {
    py::class_<device_affinity_domain> cls(m, "device_affinity_domain");
#if PYOPENCL_CL_VERSION >= 0x1020
    ADD_ATTR(DEVICE_AFFINITY_DOMAIN_, NUMA);
    ADD_ATTR(DEVICE_AFFINITY_DOMAIN_, L4_CACHE);
    ADD_ATTR(DEVICE_AFFINITY_DOMAIN_, L3_CACHE);
    ADD_ATTR(DEVICE_AFFINITY_DOMAIN_, L2_CACHE);
    ADD_ATTR(DEVICE_AFFINITY_DOMAIN_, L1_CACHE);
    ADD_ATTR(DEVICE_AFFINITY_DOMAIN_, NEXT_PARTITIONABLE);
#endif
  }

  {
    py::class_<device_atomic_capabilities> cls(m, "device_atomic_capabilities");
#if PYOPENCL_CL_VERSION >= 0x3000
    ADD_ATTR(DEVICE_ATOMIC_, ORDER_RELAXED);
    ADD_ATTR(DEVICE_ATOMIC_, ORDER_ACQ_REL);
    ADD_ATTR(DEVICE_ATOMIC_, ORDER_SEQ_CST);
    ADD_ATTR(DEVICE_ATOMIC_, SCOPE_WORK_ITEM);
    ADD_ATTR(DEVICE_ATOMIC_, SCOPE_WORK_GROUP);
    ADD_ATTR(DEVICE_ATOMIC_, SCOPE_DEVICE);
    ADD_ATTR(DEVICE_ATOMIC_, SCOPE_ALL_DEVICES);
#endif
  }
  {
    py::class_<device_device_enqueue_capabilities> cls(m, "device_device_enqueue_capabilities");
#if (PYOPENCL_CL_VERSION >= 0x3000) && defined(CL_DEVICE_DEVICE_ENQUEUE_CAPABILITIES)
    ADD_ATTR(DEVICE_QUEUE_, SUPPORTED);
    ADD_ATTR(DEVICE_QUEUE_, REPLACEABLE_DEFAULT);
#endif
  }

  {
    py::class_<version_bits> cls(m, "version_bits");
#if PYOPENCL_CL_VERSION >= 0x3000
    ADD_ATTR(VERSION_, MAJOR_BITS);
    ADD_ATTR(VERSION_, MINOR_BITS);
    ADD_ATTR(VERSION_, PATCH_BITS);

    ADD_ATTR(VERSION_, MAJOR_MASK);
    ADD_ATTR(VERSION_, MINOR_MASK);
    ADD_ATTR(VERSION_, PATCH_MASK);
#endif
  }

  {
    py::class_<khronos_vendor_id> cls(m, "khronos_vendor_id");
#if PYOPENCL_CL_VERSION >= 0x3000
    ADD_ATTR(KHRONOS_VENDOR_ID_, CODEPLAY);
#endif
  }


#ifdef HAVE_GL
  {
    py::class_<gl_object_type> cls(m, "gl_object_type");
    ADD_ATTR(GL_OBJECT_, BUFFER);
    ADD_ATTR(GL_OBJECT_, TEXTURE2D);
    ADD_ATTR(GL_OBJECT_, TEXTURE3D);
    ADD_ATTR(GL_OBJECT_, RENDERBUFFER);
  }

  {
    py::class_<gl_texture_info> cls(m, "gl_texture_info");
    ADD_ATTR(GL_, TEXTURE_TARGET);
    ADD_ATTR(GL_, MIPMAP_LEVEL);
  }
#endif

  // }}}

  // {{{ cl_name_version
#if PYOPENCL_CL_VERSION >= 0x3000
  {
    typedef cl_name_version cls;
    py::class_<cls>(m, "NameVersion")
      .def(py::init(
            [](cl_version version, const char* name)
            {
              cl_name_version result;
              result.version = version;
              result.name[0] = '\0';
              // https://stackoverflow.com/a/1258577
              strncat(result.name, name, CL_NAME_VERSION_MAX_NAME_SIZE-1);
              return result;
            }),
          py::arg("version")=0,
          py::arg("name")=0)

      .def_property("version",
          [](cls &t) { return t.version; },
          [](cls &t, cl_version val) { t.version = val; })
      .def_property("name",
          [](cls &t) { return t.name; },
          [](cls &t, const char *name)
          {
              t.name[0] = '\0';
              // https://stackoverflow.com/a/1258577
              strncat(t.name, name, CL_NAME_VERSION_MAX_NAME_SIZE-1);
          })
      ;
  }
#endif
  // }}}

  // {{{ CL_DEVICE_TOPOLOGY_AMD

#ifdef CL_DEVICE_TOPOLOGY_AMD
  {
    typedef cl_device_topology_amd cls;
    py::class_<cls>(m, "DeviceTopologyAmd")
      .def(py::init(
            [](cl_char bus, cl_char device, cl_char function)
            {
              cl_device_topology_amd result;
              result.pcie.type = CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD;
              result.pcie.bus = bus;
              result.pcie.device = device;
              result.pcie.function = function;
              return result;
            }),
          py::arg("bus")=0,
          py::arg("device")=0,
          py::arg("function")=0)

      .def_property("type",
          [](cls &t) { return t.pcie.type; },
          [](cls &t, cl_uint val) { t.pcie.type = val; })

      .def_property("bus",
          [](cls &t) { return t.pcie.bus; },
          [](cls &t, cl_char val) { t.pcie.bus = val; })
      .def_property("device",
          [](cls &t) { return t.pcie.device; },
          [](cls &t, cl_char val) { t.pcie.device = val; })
      .def_property("function",
          [](cls &t) { return t.pcie.function; },
          [](cls &t, cl_char val) { t.pcie.function = val; })
      ;
  }
#endif

  // }}}

}




// vim: foldmethod=marker
