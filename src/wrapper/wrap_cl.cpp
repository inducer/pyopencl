#include "wrap_cl.hpp"




using namespace pyopencl;




namespace
{
  py::handle<>
    CLError,
    CLMemoryError,
    CLLogicError,
    CLRuntimeError;




  void translate_cl_error(const error &err)
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
  class platform_info { };
  class device_type { };
  class device_info { };
  class device_fp_config { };
  class device_mem_cache_type { };
  class device_local_mem_type { };
  class device_exec_capabilities { };
  class command_queue_properties { };
  class context_info { };
  class context_properties { };
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

  class gl_object_type { };
  class gl_texture_info { };
}




BOOST_PYTHON_MODULE(_cl)
{
#define DECLARE_EXC(NAME, BASE) \
  CL##NAME = py::handle<>(PyErr_NewException("pyopencl." #NAME, BASE, NULL)); \
  py::scope().attr(#NAME) = CL##NAME;

  {
    DECLARE_EXC(Error, NULL);
    py::tuple memerr_bases = py::make_tuple(
        CLError,
        py::handle<>(py::borrowed(PyExc_MemoryError)));
    DECLARE_EXC(MemoryError, memerr_bases.ptr());
    DECLARE_EXC(LogicError, CLLogicError.get());
    DECLARE_EXC(RuntimeError, CLError.get());

    py::register_exception_translator<error>(translate_cl_error);
  }

#define ADD_ATTR(PREFIX, NAME) \
  cls.attr(#NAME) = CL_##PREFIX##NAME

  {
    py::class_<platform_info> cls("platform_info", py::no_init);
    ADD_ATTR(PLATFORM_, PROFILE);
    ADD_ATTR(PLATFORM_, VERSION);
    ADD_ATTR(PLATFORM_, NAME);
    ADD_ATTR(PLATFORM_, VENDOR);
#if !(defined(CL_PLATFORM_NVIDIA) && CL_PLATFORM_NVIDIA == 0x3001)
    ADD_ATTR(PLATFORM_, EXTENSIONS);
#endif
  }

  {
    py::class_<device_type> cls("device_type", py::no_init);
    ADD_ATTR(DEVICE_TYPE_, DEFAULT);
    ADD_ATTR(DEVICE_TYPE_, CPU);
    ADD_ATTR(DEVICE_TYPE_, GPU);
    ADD_ATTR(DEVICE_TYPE_, ACCELERATOR);
    ADD_ATTR(DEVICE_TYPE_, ALL);
  }

  {
    py::class_<device_info> cls("device_info", py::no_init);
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
    ADD_ATTR(DEVICE_, NAME);
    ADD_ATTR(DEVICE_, VENDOR);
    ADD_ATTR(DEVICE_, VERSION);
    ADD_ATTR(DEVICE_, PROFILE);
    ADD_ATTR(DEVICE_, VERSION);
    ADD_ATTR(DEVICE_, EXTENSIONS);
    ADD_ATTR(DEVICE_, PLATFORM);
  }

  {
    py::class_<device_fp_config> cls("device_fp_config", py::no_init);
    ADD_ATTR(FP_, DENORM);
    ADD_ATTR(FP_, INF_NAN);
    ADD_ATTR(FP_, ROUND_TO_NEAREST);
    ADD_ATTR(FP_, ROUND_TO_ZERO);
    ADD_ATTR(FP_, ROUND_TO_INF);
    ADD_ATTR(FP_, FMA);
  }

  {
    py::class_<device_mem_cache_type> cls("device_mem_cache_type", py::no_init);
    ADD_ATTR( , NONE);
    ADD_ATTR( , READ_ONLY_CACHE);
    ADD_ATTR( , READ_WRITE_CACHE);
  }

  {
    py::class_<device_local_mem_type> cls("device_local_mem_type", py::no_init);
    ADD_ATTR( , LOCAL);
    ADD_ATTR( , GLOBAL);
  }

  {
    py::class_<device_exec_capabilities> cls("device_exec_capabilities", py::no_init);
    ADD_ATTR(EXEC_, KERNEL);
    ADD_ATTR(EXEC_, NATIVE_KERNEL);
  }

  {
    py::class_<command_queue_properties> cls("command_queue_properties", py::no_init);
    ADD_ATTR(QUEUE_, OUT_OF_ORDER_EXEC_MODE_ENABLE);
    ADD_ATTR(QUEUE_, PROFILING_ENABLE);
  }

  {
    py::class_<context_info> cls("context_info", py::no_init);
    ADD_ATTR(CONTEXT_, REFERENCE_COUNT);
    ADD_ATTR(CONTEXT_, DEVICES);
    ADD_ATTR(CONTEXT_, PROPERTIES);
  }

  {
    py::class_<context_properties> cls("context_properties", py::no_init);
    ADD_ATTR(CONTEXT_, PLATFORM);
  }

  {
    py::class_<command_queue_info> cls("command_queue_info", py::no_init);
    ADD_ATTR(QUEUE_, CONTEXT);
    ADD_ATTR(QUEUE_, DEVICE);
    ADD_ATTR(QUEUE_, REFERENCE_COUNT);
    ADD_ATTR(QUEUE_, PROPERTIES);
  }

  {
    py::class_<mem_flags> cls("mem_flags", py::no_init);
    ADD_ATTR(MEM_, READ_WRITE);
    ADD_ATTR(MEM_, WRITE_ONLY);
    ADD_ATTR(MEM_, READ_ONLY);
    ADD_ATTR(MEM_, USE_HOST_PTR);
    ADD_ATTR(MEM_, ALLOC_HOST_PTR);
    ADD_ATTR(MEM_, COPY_HOST_PTR);
  }

  {
    py::class_<channel_order> cls("channel_order", py::no_init);
    ADD_ATTR( , R);
    ADD_ATTR( , A);
    ADD_ATTR( , RG);
    ADD_ATTR( , RA);
    ADD_ATTR( , RGB);
    ADD_ATTR( , RGBA);
    ADD_ATTR( , BGRA);
    ADD_ATTR( , INTENSITY);
    ADD_ATTR( , LUMINANCE);
  }

  {
    py::class_<channel_type> cls("channel_type", py::no_init);
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
  }

  {
    py::class_<mem_object_type> cls("mem_object_type", py::no_init);
    ADD_ATTR(MEM_OBJECT_, BUFFER);
    ADD_ATTR(MEM_OBJECT_, IMAGE2D);
    ADD_ATTR(MEM_OBJECT_, IMAGE3D);
  }

  {
    py::class_<mem_info> cls("mem_info", py::no_init);
    ADD_ATTR(MEM_, TYPE);
    ADD_ATTR(MEM_, FLAGS);
    ADD_ATTR(MEM_, SIZE);
    ADD_ATTR(MEM_, HOST_PTR);
    ADD_ATTR(MEM_, MAP_COUNT);
    ADD_ATTR(MEM_, REFERENCE_COUNT);
    ADD_ATTR(MEM_, CONTEXT);
  }

  {
    py::class_<image_info> cls("image_info", py::no_init);
    ADD_ATTR(IMAGE_, FORMAT);
    ADD_ATTR(IMAGE_, ELEMENT_SIZE);
    ADD_ATTR(IMAGE_, ROW_PITCH);
    ADD_ATTR(IMAGE_, SLICE_PITCH);
    ADD_ATTR(IMAGE_, WIDTH);
    ADD_ATTR(IMAGE_, HEIGHT);
    ADD_ATTR(IMAGE_, DEPTH);
  }

  {
    py::class_<addressing_mode> cls("addressing_mode", py::no_init);
    ADD_ATTR(ADDRESS_, NONE);
    ADD_ATTR(ADDRESS_, CLAMP_TO_EDGE);
    ADD_ATTR(ADDRESS_, CLAMP);
    ADD_ATTR(ADDRESS_, REPEAT);
  }

  {
    py::class_<filter_mode> cls("filter_mode", py::no_init);
    ADD_ATTR(FILTER_, NEAREST);
    ADD_ATTR(FILTER_, LINEAR);
  }

  {
    py::class_<sampler_info> cls("sampler_info", py::no_init);
    ADD_ATTR(SAMPLER_, REFERENCE_COUNT);
    ADD_ATTR(SAMPLER_, CONTEXT);
    ADD_ATTR(SAMPLER_, NORMALIZED_COORDS);
    ADD_ATTR(SAMPLER_, ADDRESSING_MODE);
    ADD_ATTR(SAMPLER_, FILTER_MODE);
  }

  {
    py::class_<map_flags> cls("map_flags", py::no_init);
    ADD_ATTR(MAP_, READ);
    ADD_ATTR(MAP_, WRITE);
  }

  {
    py::class_<program_info> cls("program_info", py::no_init);
    ADD_ATTR(PROGRAM_, REFERENCE_COUNT);
    ADD_ATTR(PROGRAM_, CONTEXT);
    ADD_ATTR(PROGRAM_, NUM_DEVICES);
    ADD_ATTR(PROGRAM_, DEVICES);
    ADD_ATTR(PROGRAM_, SOURCE);
    ADD_ATTR(PROGRAM_, BINARY_SIZES);
    ADD_ATTR(PROGRAM_, BINARIES);
  }

  {
    py::class_<program_build_info> cls("program_build_info", py::no_init);
    ADD_ATTR(PROGRAM_BUILD_, STATUS);
    ADD_ATTR(PROGRAM_BUILD_, OPTIONS);
    ADD_ATTR(PROGRAM_BUILD_, LOG);
  }

  {
    py::class_<kernel_info> cls("kernel_info", py::no_init);
    ADD_ATTR(KERNEL_, FUNCTION_NAME);
    ADD_ATTR(KERNEL_, NUM_ARGS);
    ADD_ATTR(KERNEL_, REFERENCE_COUNT);
    ADD_ATTR(KERNEL_, CONTEXT);
    ADD_ATTR(KERNEL_, PROGRAM);
  }

  {
    py::class_<kernel_work_group_info> cls("kernel_work_group_info", py::no_init);
    ADD_ATTR(KERNEL_, WORK_GROUP_SIZE);
    ADD_ATTR(KERNEL_, COMPILE_WORK_GROUP_SIZE);
  }

  {
    py::class_<event_info> cls("event_info", py::no_init);
    ADD_ATTR(EVENT_, COMMAND_QUEUE);
    ADD_ATTR(EVENT_, COMMAND_TYPE);
    ADD_ATTR(EVENT_, REFERENCE_COUNT);
    ADD_ATTR(EVENT_, COMMAND_EXECUTION_STATUS);
  }

  {
    py::class_<command_type> cls("command_type", py::no_init);
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
  }

  {
    py::class_<command_execution_status> cls("command_execution_status", py::no_init);
    ADD_ATTR(, COMPLETE);
    ADD_ATTR(, RUNNING);
    ADD_ATTR(, SUBMITTED);
    ADD_ATTR(, QUEUED);
  }

  {
    py::class_<profiling_info> cls("profiling_info", py::no_init);
    ADD_ATTR(PROFILING_COMMAND_, QUEUED);
    ADD_ATTR(PROFILING_COMMAND_, SUBMIT);
    ADD_ATTR(PROFILING_COMMAND_, START);
    ADD_ATTR(PROFILING_COMMAND_, END);
  }

  DEF_SIMPLE_FUNCTION(get_platforms);

  {
    typedef platform cls;
    py::class_<cls, boost::noncopyable>("Platform", py::no_init)
      .DEF_SIMPLE_METHOD(get_info)
      .def("get_devices", &cls::get_devices,
          py::arg("device_type")=CL_DEVICE_TYPE_ALL)
      .add_property("obj_ptr", &cls::obj_ptr)
      .def(py::self == py::self)
      .def(py::self != py::self)
      ;
  }

  {
    typedef device cls;
    py::class_<cls, boost::noncopyable>("Device", py::no_init)
      .DEF_SIMPLE_METHOD(get_info)
      .add_property("obj_ptr", &cls::obj_ptr)
      .def(py::self == py::self)
      .def(py::self != py::self)
      ;
  }

  {
    typedef context cls;
    py::class_<cls, boost::noncopyable>("Context", py::no_init)
      .def("__init__", make_constructor(create_context,
            py::default_call_policies(),
            (py::arg("devices")=py::object(),
             py::arg("properties")=py::object(),
             py::arg("dev_type")=py::object()
            )))
      .DEF_SIMPLE_METHOD(get_info)
      .add_property("obj_ptr", &cls::obj_ptr)
      .def(py::self == py::self)
      .def(py::self != py::self)
      ;
  }

  {
    typedef command_queue cls;
    py::class_<cls, boost::noncopyable>("CommandQueue", 
        py::init<const context &, 
          const device *, cl_command_queue_properties>
        ((py::arg("context"), py::arg("device")=py::object(), py::arg("properties")=0)))
      .DEF_SIMPLE_METHOD(get_info)
      .DEF_SIMPLE_METHOD(set_property)
      .DEF_SIMPLE_METHOD(flush)
      .DEF_SIMPLE_METHOD(finish)
      .add_property("obj_ptr", &cls::obj_ptr)
      .def(py::self == py::self)
      .def(py::self != py::self)
      ;
  }
  {
    typedef event cls;
    py::class_<cls, boost::noncopyable>("Event", py::no_init)
      .DEF_SIMPLE_METHOD(get_info)
      .DEF_SIMPLE_METHOD(get_profiling_info)
      .add_property("obj_ptr", &cls::obj_ptr)
      .def(py::self == py::self)
      .def(py::self != py::self)
      ;
  }

  DEF_SIMPLE_FUNCTION(wait_for_events);
  py::def("enqueue_marker", enqueue_marker,
      py::return_value_policy<py::manage_new_object>());
  DEF_SIMPLE_FUNCTION(enqueue_wait_for_events);

  // memory_object ------------------------------------------------------------
  {
    typedef memory_object cls;
    py::class_<cls, boost::noncopyable>("MemoryObject", py::no_init)
      .DEF_SIMPLE_METHOD(get_info)
      .DEF_SIMPLE_METHOD(release)
      .add_property("obj_ptr", &cls::obj_ptr)
      .add_property("hostbuf", &cls::hostbuf)
      .def(py::self == py::self)
      .def(py::self != py::self)
      ;
  }

  {
    typedef buffer cls;
    py::class_<cls, py::bases<memory_object>, boost::noncopyable>(
        "Buffer", py::no_init)
      .def("__init__", make_constructor(create_buffer,
            py::default_call_policies(),
            (py::args("context", "flags"),
             py::arg("size")=0,
             py::arg("hostbuf")=py::object()
            )))
      ;
  }

  py::def("enqueue_read_buffer", enqueue_read_buffer,
      (py::args("queue", "mem", "hostbuf"), 
       py::arg("device_offset")=0,
       py::arg("wait_for")=py::object(),
       py::arg("is_blocking")=false,
       py::arg("host_buffer")=py::object()
       ),
      py::return_value_policy<py::manage_new_object>());
  py::def("enqueue_write_buffer", enqueue_write_buffer,
      (py::args("queue", "mem", "hostbuf"), 
       py::arg("device_offset")=0,
       py::arg("wait_for")=py::object(),
       py::arg("is_blocking")=false,
       py::arg("host_buffer")=py::object()
       ),
      py::return_value_policy<py::manage_new_object>());

  // image --------------------------------------------------------------------
  {
    typedef image cls;
    py::class_<cls, py::bases<memory_object>, boost::noncopyable>(
        "Image", py::no_init)
      .def("__init__", make_constructor(create_image,
            py::default_call_policies(),
            (py::args("context", "flags", "format"), 
             py::arg("shape")=py::object(),
             py::arg("pitches")=py::object(),
             py::arg("hostbuf")=py::object(),
             py::arg("host_buffer")=py::object()
            )))
      .DEF_SIMPLE_METHOD(get_image_info)
      ;
  }

  {
    typedef cl_image_format cls;
    py::class_<cls>("ImageFormat")
      .def("__init__", py::make_constructor(make_image_format))
      .def_readwrite("channel_order", &cls::image_channel_order)
      .def_readwrite("channel_data_type", &cls::image_channel_data_type)
      ;
  }

  DEF_SIMPLE_FUNCTION(get_supported_image_formats);

  py::def("enqueue_read_image", enqueue_read_image,
      (py::args("queue", "mem", "origin", "region", "hostbuf"), 
       py::arg("row_pitch")=0,
       py::arg("slice_pitch")=0,
       py::arg("wait_for")=py::object(),
       py::arg("is_blocking")=false,
       py::arg("host_buffer")=py::object()
       ),
      py::return_value_policy<py::manage_new_object>());
  py::def("enqueue_write_image", enqueue_write_image,
      (py::args("queue", "mem", "origin", "region", "hostbuf"), 
       py::arg("row_pitch")=0,
       py::arg("slice_pitch")=0,
       py::arg("wait_for")=py::object(),
       py::arg("is_blocking")=false,
       py::arg("host_buffer")=py::object()
       ),
      py::return_value_policy<py::manage_new_object>());

  py::def("enqueue_copy_image", enqueue_copy_image,
      (py::args("queue", "src", "dest", "src_origin", "dest_origin", "region"), 
       py::arg("wait_for")=py::object()),
      py::return_value_policy<py::manage_new_object>());
  py::def("enqueue_copy_image_to_buffer", enqueue_copy_image_to_buffer,
      (py::args("queue", "src", "dest", "origin", "region", "offset"), 
       py::arg("wait_for")=py::object()),
      py::return_value_policy<py::manage_new_object>());
  py::def("enqueue_copy_buffer_to_image", enqueue_copy_image_to_buffer,
      (py::args("queue", "src", "dest", "offset", "origin", "region"), 
       py::arg("wait_for")=py::object()),
      py::return_value_policy<py::manage_new_object>());

  // memory_map ---------------------------------------------------------------
  {
    typedef memory_map cls;
    py::class_<cls, boost::noncopyable>("MemoryMap", py::no_init)
      .def("release", &cls::release,
          (py::arg("queue")=0, py::arg("wait_for")=py::object()),
          py::return_value_policy<py::manage_new_object>())
      ;
  }

  py::def("enqueue_map_buffer", enqueue_map_buffer,
      (py::args("queue", "buf", "flags", 
                "offset", 
                "shape", "dtype", "order"), 
       py::arg("wait_for")=py::object(),
       py::arg("is_blocking")=false));
  py::def("enqueue_map_image", enqueue_map_image,
      (py::args("queue", "img", "flags", 
                "origin", "region",
                "shape", "dtype", "order"), 
       py::arg("wait_for")=py::object(),
       py::arg("is_blocking")=false));


  // sampler ------------------------------------------------------------------
  {
    typedef sampler cls;
    py::class_<cls, boost::noncopyable>("Sampler", 
        py::init<context const &, bool, cl_addressing_mode, cl_filter_mode>())
      .DEF_SIMPLE_METHOD(get_info)
      .add_property("obj_ptr", &cls::obj_ptr)
      .def(py::self == py::self)
      .def(py::self != py::self)
      ;
  }

  // program ------------------------------------------------------------------
  {
    typedef program cls;
    py::class_<cls, boost::noncopyable>("Program", py::no_init)
      .def("__init__", make_constructor(
            create_program_with_source,
            py::default_call_policies(),
            py::args("context", "src")))
      .def("__init__", make_constructor(
            create_program_with_binary,
            py::default_call_policies(),
            py::args("context", "devices", "binaries")))
      .DEF_SIMPLE_METHOD(get_info)
      .DEF_SIMPLE_METHOD(get_build_info)
      .def("_build", &cls::build,
          (py::arg("options")="", py::arg("devices")=py::object()))
      .add_property("obj_ptr", &cls::obj_ptr)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("all_kernels", create_kernels_in_program)
      ;
  }

  py::def("unload_compiler", unload_compiler);

  {
    typedef kernel cls;
    py::class_<cls, boost::noncopyable>("Kernel", 
        py::init<const program &, std::string const &>())
      .DEF_SIMPLE_METHOD(get_info)
      .DEF_SIMPLE_METHOD(get_work_group_info)
      .DEF_SIMPLE_METHOD(set_arg)
      .add_property("obj_ptr", &cls::obj_ptr)
      .def(py::self == py::self)
      .def(py::self != py::self)
      ;
  }

  {
    typedef local_memory cls;
    py::class_<cls, boost::noncopyable>("LocalMemory", 
        py::init<size_t>(py::arg("size")))
      .add_property("size", &cls::size)
      ;
  }


  py::def("enqueue_nd_range_kernel", enqueue_nd_range_kernel,
      (py::args("queue", "kernel"),
      py::arg("global_work_size"),
      py::arg("local_work_size"),
      py::arg("global_work_offset")=py::object(),
      py::arg("wait_for")=py::object()
      ),
      py::return_value_policy<py::manage_new_object>());
  py::def("enqueue_task", enqueue_task,
      (py::args("queue", "kernel"),
      py::arg("wait_for")=py::object()
      ),
      py::return_value_policy<py::manage_new_object>());

  // TODO: clEnqueueNativeKernel

  // GL interop ---------------------------------------------------------------
  DEF_SIMPLE_FUNCTION(have_gl);

#ifdef HAVE_GL
  {
    py::class_<gl_object_type> cls("gl_object_type", py::no_init);
    ADD_ATTR(GL_OBJECT_, BUFFER);
    ADD_ATTR(GL_OBJECT_, TEXTURE2D);
    ADD_ATTR(GL_OBJECT_, TEXTURE3D);
    ADD_ATTR(GL_OBJECT_, RENDERBUFFER);
  }

  {
    py::class_<gl_texture_info> cls("gl_texture_info", py::no_init);
    ADD_ATTR(GL_, TEXTURE_TARGET);
    ADD_ATTR(GL_, MIPMAP_LEVEL);
  }

  {
    typedef gl_buffer cls;
    py::class_<cls, py::bases<memory_object>, boost::noncopyable>(
        "GLBuffer", py::no_init)
      .def("__init__", make_constructor(create_from_gl_buffer,
            py::default_call_policies(),
            (py::args("context", "flags" "bufobj"))))
      .def("get_gl_object_info", get_gl_object_info)
      ;
  }

  {
    typedef gl_renderbuffer cls;
    py::class_<cls, py::bases<memory_object>, boost::noncopyable>(
        "GLRenderBuffer", py::no_init)
      .def("__init__", make_constructor(create_from_gl_renderbuffer,
            py::default_call_policies(),
            (py::args("context", "flags" "bufobj"))))
      .def("get_gl_object_info", get_gl_object_info)
      ;
  }

  {
    typedef gl_texture cls;
    py::class_<cls, py::bases<image>, boost::noncopyable>(
        "GLTexture", py::no_init)
      .def("__init__", make_constructor(create_from_gl_texture,
            py::default_call_policies(),
            (py::args("context", "flags", 
                      "texture_target", "miplevel",
                      "texture", "dims"))))
      .def("get_gl_object_info", get_gl_object_info)
      .DEF_SIMPLE_METHOD(get_gl_texture_info)
      ;
  }

  py::def("enqueue_acquire_gl_objects", enqueue_acquire_gl_objects,
      (py::args("queue", "mem_objects"),
      py::arg("wait_for")=py::object()
      ),
      py::return_value_policy<py::manage_new_object>());
  py::def("enqueue_release_gl_objects", enqueue_release_gl_objects,
      (py::args("queue", "mem_objects"),
      py::arg("wait_for")=py::object()
      ),
      py::return_value_policy<py::manage_new_object>());

#endif
}
