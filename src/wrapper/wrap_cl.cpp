#include <cl.hpp>
#include "wrap_helpers.hpp"




namespace py = boost::python;




namespace
{
  py::handle<> 
    CLError, 
    CLMemoryError, 
    CLLogicError, 
    CLRuntimeError;




  void translate_cl_error(const cl::error &err)
  {
    if (err.code() == CL_MEM_OBJECT_ALLOCATION_FAILURE)
      PyErr_SetString(CLMemoryError.get(), err.what());
    else if (err.code() <= CL_INVALID_VALUE)
      PyErr_SetString(CLLogicError.get(), err.what());
    else if (err.code() > CL_INVALID_VALUE && err.code() < CL_SUCCESS)
      PyErr_SetString(CLRuntimeError.get(), err.what());
    else 
      PyErr_SetString(CLError.get(), err.what());
  }

  // 'fake' constant scopes
  class device_type { };
  class device_info { };
  class device_address_info { };
  class device_fp_config { };
  class device_mem_cache_type { };
  class device_local_mem_type { };
  class device_exec_capabilities { };
  class command_queue_properties { };
  class context_info { };
  class command_queue_info { };
  class mem_flags { };
  class channel_order { };
  class channel_type { };
  class mem_object_type { };
  class mem_info { };
  class image_info { };
  class addressing_mode { };
  class filter_mode { };
  class sampler_info { };
  class map_flags { };
  class program_info { };
  class program_build_info { };
  class build_status { };
  class kernel_info { };
  class kernel_work_group_info { };
  class event_info { };
  class command_type { };
  class command_execution_status { };
  class profiling_info { };
}




BOOST_PYTHON_MODULE(_cl)
{
#define DECLARE_EXC(NAME, BASE) \
  CL##NAME = py::handle<>(PyErr_NewException("pyopencl._cl." #NAME, BASE, NULL)); \
  py::scope().attr(#NAME) = CL##NAME;

  {
    DECLARE_EXC(Error, NULL);
    py::tuple memerr_bases = py::make_tuple(
        CLError, 
        py::handle<>(py::borrowed(PyExc_MemoryError)));
    DECLARE_EXC(MemoryError, memerr_bases.ptr());
    DECLARE_EXC(LogicError, CLLogicError.get());
    DECLARE_EXC(RuntimeError, CLError.get());

    py::register_exception_translator<cl::error>(translate_cl_error);
  }

  {
    py::class_<device_type> cls("device_type", py::no_init);
    cls.attr("DEFAULT") = CL_DEVICE_TYPE_DEFAULT;
    cls.attr("CPU") = CL_DEVICE_TYPE_CPU;
    cls.attr("GPU") = CL_DEVICE_TYPE_GPU;
    cls.attr("ACCELERATOR") = CL_DEVICE_TYPE_ACCELERATOR;
    cls.attr("ALL") = CL_DEVICE_TYPE_ALL;
  }

  {
    py::class_<device_info> cls("device_info", py::no_init);
    cls.attr("TYPE") = CL_DEVICE_TYPE;
    cls.attr("VENDOR_ID") = CL_DEVICE_VENDOR_ID;
    cls.attr("MAX_COMPUTE_UNITS") = CL_DEVICE_MAX_COMPUTE_UNITS;
    cls.attr("MAX_WORK_ITEM_DIMENSIONS") = CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS;
    cls.attr("MAX_WORK_GROUP_SIZE") = CL_DEVICE_MAX_WORK_GROUP_SIZE;
    cls.attr("MAX_WORK_ITEM_SIZES") = CL_DEVICE_MAX_WORK_ITEM_SIZES;
    cls.attr("PREFERRED_VECTOR_WIDTH_CHAR") = CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR;
    cls.attr("PREFERRED_VECTOR_WIDTH_SHORT") = CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT;
    cls.attr("PREFERRED_VECTOR_WIDTH_INT") = CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT;
    cls.attr("PREFERRED_VECTOR_WIDTH_LONG") = CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG;
    cls.attr("PREFERRED_VECTOR_WIDTH_FLOAT") = CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT;
    cls.attr("PREFERRED_VECTOR_WIDTH_DOUBLE") = CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE;
    cls.attr("MAX_CLOCK_FREQUENCY") = CL_DEVICE_MAX_CLOCK_FREQUENCY;
    cls.attr("ADDRESS_BITS") = CL_DEVICE_ADDRESS_BITS;
    cls.attr("MAX_READ_IMAGE_ARGS") = CL_DEVICE_MAX_READ_IMAGE_ARGS;
    cls.attr("MAX_WRITE_IMAGE_ARGS") = CL_DEVICE_MAX_WRITE_IMAGE_ARGS;
    cls.attr("MAX_MEM_ALLOC_SIZE") = CL_DEVICE_MAX_MEM_ALLOC_SIZE;
    cls.attr("IMAGE2D_MAX_WIDTH") = CL_DEVICE_IMAGE2D_MAX_WIDTH;
    cls.attr("IMAGE2D_MAX_HEIGHT") = CL_DEVICE_IMAGE2D_MAX_HEIGHT;
    cls.attr("IMAGE3D_MAX_WIDTH") = CL_DEVICE_IMAGE3D_MAX_WIDTH;
    cls.attr("IMAGE3D_MAX_HEIGHT") = CL_DEVICE_IMAGE3D_MAX_HEIGHT;
    cls.attr("IMAGE3D_MAX_DEPTH") = CL_DEVICE_IMAGE3D_MAX_DEPTH;
    cls.attr("IMAGE_SUPPORT") = CL_DEVICE_IMAGE_SUPPORT;
    cls.attr("MAX_PARAMETER_SIZE") = CL_DEVICE_MAX_PARAMETER_SIZE;
    cls.attr("MAX_SAMPLERS") = CL_DEVICE_MAX_SAMPLERS;
    cls.attr("MEM_BASE_ADDR_ALIGN") = CL_DEVICE_MEM_BASE_ADDR_ALIGN;
    cls.attr("MIN_DATA_TYPE_ALIGN_SIZE") = CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE;
    cls.attr("SINGLE_FP_CONFIG") = CL_DEVICE_SINGLE_FP_CONFIG;
    cls.attr("GLOBAL_MEM_CACHE_TYPE") = CL_DEVICE_GLOBAL_MEM_CACHE_TYPE;
    cls.attr("GLOBAL_MEM_CACHELINE_SIZE") = CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE;
    cls.attr("GLOBAL_MEM_CACHE_SIZE") = CL_DEVICE_GLOBAL_MEM_CACHE_SIZE;
    cls.attr("GLOBAL_MEM_SIZE") = CL_DEVICE_GLOBAL_MEM_SIZE;
    cls.attr("MAX_CONSTANT_BUFFER_SIZE") = CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE;
    cls.attr("MAX_CONSTANT_ARGS") = CL_DEVICE_MAX_CONSTANT_ARGS;
    cls.attr("LOCAL_MEM_TYPE") = CL_DEVICE_LOCAL_MEM_TYPE;
    cls.attr("LOCAL_MEM_SIZE") = CL_DEVICE_LOCAL_MEM_SIZE;
    cls.attr("ERROR_CORRECTION_SUPPORT") = CL_DEVICE_ERROR_CORRECTION_SUPPORT;
    cls.attr("PROFILING_TIMER_RESOLUTION") = CL_DEVICE_PROFILING_TIMER_RESOLUTION;
    cls.attr("ENDIAN_LITTLE") = CL_DEVICE_ENDIAN_LITTLE;
    cls.attr("AVAILABLE") = CL_DEVICE_AVAILABLE;
    cls.attr("COMPILER_AVAILABLE") = CL_DEVICE_COMPILER_AVAILABLE;
    cls.attr("EXECUTION_CAPABILITIES") = CL_DEVICE_EXECUTION_CAPABILITIES;
    cls.attr("QUEUE_PROPERTIES") = CL_DEVICE_QUEUE_PROPERTIES;
    cls.attr("NAME") = CL_DEVICE_NAME;
    cls.attr("VENDOR") = CL_DEVICE_VENDOR;
    cls.attr("VERSION") = CL_DRIVER_VERSION;
    cls.attr("PROFILE") = CL_DEVICE_PROFILE;
    cls.attr("VERSION") = CL_DEVICE_VERSION;
    cls.attr("EXTENSIONS") = CL_DEVICE_EXTENSIONS;
  }

  {
    py::class_<device_address_info> cls("device_address_info", py::no_init);
    cls.attr("32_BITS") = CL_DEVICE_ADDRESS_32_BITS;
    cls.attr("64_BITS") = CL_DEVICE_ADDRESS_64_BITS;
  }

  {
    py::class_<device_fp_config> cls("device_fp_config", py::no_init);
    cls.attr("DENORM") = CL_FP_DENORM;
    cls.attr("INF_NAN") = CL_FP_INF_NAN;
    cls.attr("ROUND_TO_NEAREST") = CL_FP_ROUND_TO_NEAREST;
    cls.attr("ROUND_TO_ZERO") = CL_FP_ROUND_TO_ZERO;
    cls.attr("ROUND_TO_INF") = CL_FP_ROUND_TO_INF;
    cls.attr("FMA") = CL_FP_FMA;
  }

  {
    py::class_<device_mem_cache_type> cls("device_mem_cache_type", py::no_init);
    cls.attr("NONE") = CL_NONE;
    cls.attr("READ_ONLY_CACHE") = CL_READ_ONLY_CACHE;
    cls.attr("READ_WRITE_CACHE") = CL_READ_WRITE_CACHE;
  }

  {
    py::class_<device_local_mem_type> cls("device_local_mem_type", py::no_init);
    cls.attr("LOCAL") = CL_LOCAL;
    cls.attr("GLOBAL") = CL_GLOBAL;
  }

  {
    py::class_<device_exec_capabilities> cls("device_exec_capabilities", py::no_init);
    cls.attr("KERNEL") = CL_EXEC_KERNEL;
    cls.attr("NATIVE_KERNEL") = CL_EXEC_NATIVE_KERNEL;
  }

  {
    py::class_<command_queue_properties> cls("command_queue_properties", py::no_init);
    cls.attr("OUT_OF_ORDER_EXEC_MODE_ENABLE") = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    cls.attr("PROFILING_ENABLE") = CL_QUEUE_PROFILING_ENABLE;
  }

  {
    py::class_<context_info> cls("context_info", py::no_init);
    cls.attr("REFERENCE_COUNT") = CL_CONTEXT_REFERENCE_COUNT;
    cls.attr("NUM_DEVICES") = CL_CONTEXT_NUM_DEVICES;
    cls.attr("DEVICES") = CL_CONTEXT_DEVICES;
    cls.attr("PROPERTIES") = CL_CONTEXT_PROPERTIES;
  }

  {
    py::class_<command_queue_info> cls("command_queue_info", py::no_init);
    cls.attr("CONTEXT") = CL_QUEUE_CONTEXT;
    cls.attr("DEVICE") = CL_QUEUE_DEVICE;
    cls.attr("REFERENCE_COUNT") = CL_QUEUE_REFERENCE_COUNT;
    cls.attr("PROPERTIES") = CL_QUEUE_PROPERTIES;
  }

  {
    py::class_<mem_flags> cls("mem_flags", py::no_init);
    cls.attr("CL_MEM_READ_WRITE") = CL_MEM_READ_WRITE;
    cls.attr("CL_MEM_WRITE_ONLY") = CL_MEM_WRITE_ONLY;
    cls.attr("CL_MEM_READ_ONLY") = CL_MEM_READ_ONLY;
    cls.attr("CL_MEM_USE_HOST_PTR") = CL_MEM_USE_HOST_PTR;
    cls.attr("CL_MEM_ALLOC_HOST_PTR") = CL_MEM_ALLOC_HOST_PTR;
    cls.attr("CL_MEM_COPY_HOST_PTR") = CL_MEM_COPY_HOST_PTR;
  }

  {
    py::class_<channel_order> cls("channel_order", py::no_init);
    cls.attr("R") = CL_R;
    cls.attr("A") = CL_A;
    cls.attr("RG") = CL_RG;
    cls.attr("RA") = CL_RA;
    cls.attr("RGB") = CL_RGB;
    cls.attr("RGBA") = CL_RGBA;
    cls.attr("BGRA") = CL_BGRA;
    cls.attr("ARGB") = CL_ARGB;
  }
  
  {
    py::class_<channel_type> cls("channel_type", py::no_init);
    cls.attr("SNORM_INT8") = CL_SNORM_INT8;
    cls.attr("SNORM_INT16") = CL_SNORM_INT16;
    cls.attr("UNORM_INT8") = CL_UNORM_INT8;
    cls.attr("UNORM_INT16") = CL_UNORM_INT16;
    cls.attr("UNORM_SHORT_565") = CL_UNORM_SHORT_565;
    cls.attr("UNORM_SHORT_555") = CL_UNORM_SHORT_555;
    cls.attr("UNORM_INT_101010") = CL_UNORM_INT_101010;
    cls.attr("SIGNED_INT8") = CL_SIGNED_INT8;
    cls.attr("SIGNED_INT16") = CL_SIGNED_INT16;
    cls.attr("SIGNED_INT32") = CL_SIGNED_INT32;
    cls.attr("UNSIGNED_INT8") = CL_UNSIGNED_INT8;
    cls.attr("UNSIGNED_INT16") = CL_UNSIGNED_INT16;
    cls.attr("UNSIGNED_INT32") = CL_UNSIGNED_INT32;
    cls.attr("HALF_FLOAT") = CL_HALF_FLOAT;
    cls.attr("FLOAT") = CL_FLOAT;
  }

  {
    py::class_<mem_object_type> cls("mem_object_type", py::no_init);
    cls.attr("BUFFER") = CL_MEM_OBJECT_BUFFER;
    cls.attr("IMAGE2D") = CL_MEM_OBJECT_IMAGE2D;
    cls.attr("IMAGE3D") = CL_MEM_OBJECT_IMAGE3D;
  }

  {
    py::class_<mem_info> cls("mem_info", py::no_init);
    cls.attr("TYPE") = CL_MEM_TYPE;
    cls.attr("FLAGS") = CL_MEM_FLAGS;
    cls.attr("SIZE") = CL_MEM_SIZE;
    cls.attr("HOST_PTR") = CL_MEM_HOST_PTR;
    cls.attr("MAP_COUNT") = CL_MEM_MAP_COUNT;
    cls.attr("REFERENCE_COUNT") = CL_MEM_REFERENCE_COUNT;
    cls.attr("CONTEXT") = CL_MEM_CONTEXT;
  }

  {
    py::class_<image_info> cls("image_info", py::no_init);
    cls.attr("FORMAT") = CL_IMAGE_FORMAT;
    cls.attr("ELEMENT_SIZE") = CL_IMAGE_ELEMENT_SIZE;
    cls.attr("ROW_PITCH") = CL_IMAGE_ROW_PITCH;
    cls.attr("SLICE_PITCH") = CL_IMAGE_SLICE_PITCH;
    cls.attr("WIDTH") = CL_IMAGE_WIDTH;
    cls.attr("HEIGHT") = CL_IMAGE_HEIGHT;
    cls.attr("DEPTH") = CL_IMAGE_DEPTH;
  }

  {
    py::class_<addressing_mode> cls("addressing_mode", py::no_init);
    cls.attr("NONE") = CL_ADDRESS_NONE;
    cls.attr("CLAMP_TO_EDGE") = CL_ADDRESS_CLAMP_TO_EDGE;
    cls.attr("CLAMP") = CL_ADDRESS_CLAMP;
    cls.attr("REPEAT") = CL_ADDRESS_REPEAT;
  }

  {
    py::class_<filter_mode> cls("filter_mode", py::no_init);
    cls.attr("NEAREST") = CL_FILTER_NEAREST;
    cls.attr("LINEAR") = CL_FILTER_LINEAR;
  }

  {
    py::class_<sampler_info> cls("sampler_info", py::no_init);
    cls.attr("REFERENCE_COUNT") = CL_SAMPLER_REFERENCE_COUNT;
    cls.attr("CONTEXT") = CL_SAMPLER_CONTEXT;
    cls.attr("NORMALIZED_COORDS") = CL_SAMPLER_NORMALIZED_COORDS;
    cls.attr("ADDRESSING_MODE") = CL_SAMPLER_ADDRESSING_MODE;
    cls.attr("FILTER_MODE") = CL_SAMPLER_FILTER_MODE;
  }

  {
    py::class_<map_flags> cls("map_flags", py::no_init);
    cls.attr("READ") = CL_MAP_READ;
    cls.attr("WRITE") = CL_MAP_WRITE;
  }

  {
    py::class_<program_info> cls("program_info", py::no_init);
    cls.attr("REFERENCE_COUNT") = CL_PROGRAM_REFERENCE_COUNT;
    cls.attr("CONTEXT") = CL_PROGRAM_CONTEXT;
    cls.attr("NUM_DEVICES") = CL_PROGRAM_NUM_DEVICES;
    cls.attr("DEVICES") = CL_PROGRAM_DEVICES;
    cls.attr("SOURCE") = CL_PROGRAM_SOURCE;
    cls.attr("BINARY_SIZES") = CL_PROGRAM_BINARY_SIZES;
    cls.attr("BINARIES") = CL_PROGRAM_BINARIES;
  }

  {
    py::class_<program_build_info> cls("program_build_info", py::no_init);
    cls.attr("STATUS") = CL_PROGRAM_BUILD_STATUS;
    cls.attr("OPTIONS") = CL_PROGRAM_BUILD_OPTIONS;
    cls.attr("LOG") = CL_PROGRAM_BUILD_LOG;
  }

  {
    py::class_<kernel_info> cls("kernel_info", py::no_init);
    cls.attr("FUNCTION_NAME") = CL_KERNEL_FUNCTION_NAME;
    cls.attr("NUM_ARGS") = CL_KERNEL_NUM_ARGS;
    cls.attr("REFERENCE_COUNT") = CL_KERNEL_REFERENCE_COUNT;
    cls.attr("CONTEXT") = CL_KERNEL_CONTEXT;
    cls.attr("PROGRAM") = CL_KERNEL_PROGRAM;
  }

  {
    py::class_<kernel_work_group_info> cls("kernel_work_group_info", py::no_init);
    cls.attr("WORK_GROUP_SIZE") = CL_KERNEL_WORK_GROUP_SIZE;
    cls.attr("COMPILE_WORK_GROUP_SIZE") = CL_KERNEL_COMPILE_WORK_GROUP_SIZE;
  }

  {
    py::class_<event_info> cls("event_info", py::no_init);
    cls.attr("COMMAND_QUEUE") = CL_EVENT_COMMAND_QUEUE;
    cls.attr("COMMAND_TYPE") = CL_EVENT_COMMAND_TYPE;
    cls.attr("REFERENCE_COUNT") = CL_EVENT_REFERENCE_COUNT;
    cls.attr("COMMAND_EXECUTION_STATUS") = CL_EVENT_COMMAND_EXECUTION_STATUS;
  }

  {
    py::class_<command_type> cls("command_type", py::no_init);
    cls.attr("NDRANGE_KERNEL") = CL_COMMAND_NDRANGE_KERNEL;
    cls.attr("TASK") = CL_COMMAND_TASK;
    cls.attr("NATIVE_KERNEL") = CL_COMMAND_NATIVE_KERNEL;
    cls.attr("READ_BUFFER") = CL_COMMAND_READ_BUFFER;
    cls.attr("WRITE_BUFFER") = CL_COMMAND_WRITE_BUFFER;
    cls.attr("COPY_BUFFER") = CL_COMMAND_COPY_BUFFER;
    cls.attr("READ_IMAGE") = CL_COMMAND_READ_IMAGE;
    cls.attr("WRITE_IMAGE") = CL_COMMAND_WRITE_IMAGE;
    cls.attr("COPY_IMAGE") = CL_COMMAND_COPY_IMAGE;
    cls.attr("COPY_IMAGE_TO_BUFFER") = CL_COMMAND_COPY_IMAGE_TO_BUFFER;
    cls.attr("COPY_BUFFER_TO_IMAGE") = CL_COMMAND_COPY_BUFFER_TO_IMAGE;
    cls.attr("MAP_BUFFER") = CL_COMMAND_MAP_BUFFER;
    cls.attr("MAP_IMAGE") = CL_COMMAND_MAP_IMAGE;
    cls.attr("UNMAP_MEM_OBJECT") = CL_COMMAND_UNMAP_MEM_OBJECT;
    cls.attr("MARKER") = CL_COMMAND_MARKER;
    cls.attr("WAIT_FOR_EVENTS") = CL_COMMAND_WAIT_FOR_EVENTS;
    cls.attr("BARRIER") = CL_COMMAND_BARRIER;
    cls.attr("ACQUIRE_GL_OBJECTS") = CL_COMMAND_ACQUIRE_GL_OBJECTS;
    cls.attr("RELEASE_GL_OBJECTS") = CL_COMMAND_RELEASE_GL_OBJECTS;
  }

  {
    py::class_<command_execution_status> cls("command_execution_status", py::no_init);
    cls.attr("COMPLETE") = CL_COMPLETE;
    cls.attr("RUNNING") = CL_RUNNING;
    cls.attr("SUBMITTED") = CL_SUBMITTED;
    cls.attr("QUEUED") = CL_QUEUED;
  }

  {
    py::class_<profiling_info> cls("profiling_info", py::no_init);
    cls.attr("QUEUED") = CL_PROFILING_COMMAND_QUEUED;
    cls.attr("SUBMIT") = CL_PROFILING_COMMAND_SUBMIT;
    cls.attr("START") = CL_PROFILING_COMMAND_START;
    cls.attr("END") = CL_PROFILING_COMMAND_END;
  }

}
