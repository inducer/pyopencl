#include "wrap_cl.hpp"




using namespace pyopencl;




void pyopencl_expose_part_2()
{
  // {{{ image
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
      .add_property("channel_count", &get_image_format_channel_count)
      .add_property("dtype_size", &get_image_format_channel_dtype_size)
      .add_property("itemsize", &get_image_format_item_size)
      ;
  }

  DEF_SIMPLE_FUNCTION(get_supported_image_formats);

  py::def("_enqueue_read_image", enqueue_read_image,
      (py::args("queue", "mem", "origin", "region", "hostbuf"),
       py::arg("row_pitch")=0,
       py::arg("slice_pitch")=0,
       py::arg("wait_for")=py::object(),
       py::arg("is_blocking")=true,
       py::arg("host_buffer")=py::object()
       ),
      py::return_value_policy<py::manage_new_object>());
  py::def("_enqueue_write_image", enqueue_write_image,
      (py::args("queue", "mem", "origin", "region", "hostbuf"),
       py::arg("row_pitch")=0,
       py::arg("slice_pitch")=0,
       py::arg("wait_for")=py::object(),
       py::arg("is_blocking")=true,
       py::arg("host_buffer")=py::object()
       ),
      py::return_value_policy<py::manage_new_object>());

  py::def("_enqueue_copy_image", enqueue_copy_image,
      (py::args("queue", "src", "dest", "src_origin", "dest_origin", "region"),
       py::arg("wait_for")=py::object()),
      py::return_value_policy<py::manage_new_object>());
  py::def("_enqueue_copy_image_to_buffer", enqueue_copy_image_to_buffer,
      (py::args("queue", "src", "dest", "origin", "region", "offset"),
       py::arg("wait_for")=py::object()),
      py::return_value_policy<py::manage_new_object>());
  py::def("_enqueue_copy_buffer_to_image", enqueue_copy_image_to_buffer,
      (py::args("queue", "src", "dest", "offset", "origin", "region"),
       py::arg("wait_for")=py::object()),
      py::return_value_policy<py::manage_new_object>());

  // }}}

  // {{{ memory_map
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
       py::arg("is_blocking")=true));
  py::def("enqueue_map_image", enqueue_map_image,
      (py::args("queue", "img", "flags",
                "origin", "region",
                "shape", "dtype", "order"),
       py::arg("wait_for")=py::object(),
       py::arg("is_blocking")=true));

  // }}}

  // {{{ sampler
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

  // }}}

  // {{{ program
  {
    typedef program cls;
    py::class_<cls, boost::noncopyable>("_Program", py::no_init)
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
      py::arg("wait_for")=py::object(),
      py::arg("g_times_l")=false
      ),
      py::return_value_policy<py::manage_new_object>());
  py::def("enqueue_task", enqueue_task,
      (py::args("queue", "kernel"),
      py::arg("wait_for")=py::object()
      ),
      py::return_value_policy<py::manage_new_object>());

  // TODO: clEnqueueNativeKernel
  // }}}

  // {{{ GL interop
  DEF_SIMPLE_FUNCTION(have_gl);

#ifdef HAVE_GL

#ifdef __APPLE__
  DEF_SIMPLE_FUNCTION(get_apple_cgl_share_group);
#endif /* __APPLE__ */

  {
    typedef gl_buffer cls;
    py::class_<cls, py::bases<memory_object>, boost::noncopyable>(
        "GLBuffer", py::no_init)
      .def("__init__", make_constructor(create_from_gl_buffer,
            py::default_call_policies(),
            (py::args("context", "flags", "bufobj"))))
      .def("get_gl_object_info", get_gl_object_info)
      ;
  }

  {
    typedef gl_renderbuffer cls;
    py::class_<cls, py::bases<memory_object>, boost::noncopyable>(
        "GLRenderBuffer", py::no_init)
      .def("__init__", make_constructor(create_from_gl_renderbuffer,
            py::default_call_policies(),
            (py::args("context", "flags", "bufobj"))))
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

#if defined(cl_khr_gl_sharing) && (cl_khr_gl_sharing >= 1)
  py::def("get_gl_context_info_khr", get_gl_context_info_khr,
      py::args("properties", "param_name"));
#endif

#endif
  // }}}
}




// vim: foldmethod=marker
