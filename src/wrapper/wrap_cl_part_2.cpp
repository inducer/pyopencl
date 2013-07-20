#include "wrap_cl.hpp"




namespace pyopencl {
#if PYOPENCL_CL_VERSION >= 0x1020
  py::object image_desc_dummy_getter(cl_image_desc &desc)
  {
    return py::object();
  }

  void image_desc_set_shape(cl_image_desc &desc, py::object py_shape)
  {
    COPY_PY_REGION_TRIPLE(shape);
    desc.image_width = shape[0];
    desc.image_height = shape[1];
    desc.image_depth = shape[2];
    desc.image_array_size = shape[2];
  }

  void image_desc_set_pitches(cl_image_desc &desc, py::object py_pitches)
  {
    COPY_PY_PITCH_TUPLE(pitches);
    desc.image_row_pitch = pitches[0];
    desc.image_slice_pitch = pitches[1];
  }

  void image_desc_set_buffer(cl_image_desc &desc, memory_object *mobj)
  {
    if (mobj)
      desc.buffer = mobj->data();
    else
      desc.buffer = 0;
  }

#endif
}




using namespace pyopencl;




void pyopencl_expose_part_2()
{
  py::docstring_options doc_op;
  doc_op.disable_cpp_signatures();

  // {{{ image

#if PYOPENCL_CL_VERSION >= 0x1020
  {
    typedef cl_image_desc cls;
    py::class_<cls>("ImageDescriptor")
      .def_readwrite("image_type", &cls::image_type)
      .add_property("shape", &image_desc_dummy_getter, image_desc_set_shape)
      .def_readwrite("array_size", &cls::image_array_size)
      .add_property("pitches", &image_desc_dummy_getter, image_desc_set_pitches)
      .def_readwrite("num_mip_levels", &cls::num_mip_levels)
      .def_readwrite("num_samples", &cls::num_samples)
      .add_property("buffer", &image_desc_dummy_getter, image_desc_set_buffer)
      ;
  }
#endif

  {
    typedef image cls;
    py::class_<cls, py::bases<memory_object>, boost::noncopyable>(
        "Image", py::no_init)
      .def("__init__", make_constructor(create_image,
            py::default_call_policies(),
            (py::args("context", "flags", "format"),
             py::arg("shape")=py::object(),
             py::arg("pitches")=py::object(),
             py::arg("hostbuf")=py::object()
            )))
#if PYOPENCL_CL_VERSION >= 0x1020
      .def("__init__", make_constructor(create_image_from_desc,
            py::default_call_policies(),
            (py::args("context", "flags", "format", "desc"),
             py::arg("hostbuf")=py::object())))
#endif
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
       py::arg("is_blocking")=true
       ),
      py::return_value_policy<py::manage_new_object>());
  py::def("_enqueue_write_image", enqueue_write_image,
      (py::args("queue", "mem", "origin", "region", "hostbuf"),
       py::arg("row_pitch")=0,
       py::arg("slice_pitch")=0,
       py::arg("wait_for")=py::object(),
       py::arg("is_blocking")=true
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
  py::def("_enqueue_copy_buffer_to_image", enqueue_copy_buffer_to_image,
      (py::args("queue", "src", "dest", "offset", "origin", "region"),
       py::arg("wait_for")=py::object()),
      py::return_value_policy<py::manage_new_object>());

#if PYOPENCL_CL_VERSION >= 0x1020
  py::def("enqueue_fill_image", enqueue_write_image,
      (py::args("queue", "mem", "color", "origin", "region"),
       py::arg("wait_for")=py::object()),
      py::return_value_policy<py::manage_new_object>());
#endif

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
                "shape", "dtype"),
       py::arg("order")="C",
       py::arg("strides")=py::object(),
       py::arg("wait_for")=py::object(),
       py::arg("is_blocking")=true));
  py::def("enqueue_map_image", enqueue_map_image,
      (py::args("queue", "img", "flags",
                "origin", "region",
                "shape", "dtype"),
       py::arg("order")="C",
       py::arg("strides")=py::object(),
       py::arg("wait_for")=py::object(),
       py::arg("is_blocking")=true));

  // }}}

  // {{{ sampler
  {
    typedef sampler cls;
    py::class_<cls, boost::noncopyable>("Sampler",
        py::init<context const &, bool, cl_addressing_mode, cl_filter_mode>())
      .DEF_SIMPLE_METHOD(get_info)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__hash__", &cls::hash)
      PYOPENCL_EXPOSE_TO_FROM_INT_PTR(cl_sampler)
      ;
  }

  // }}}

  // {{{ program
  {
    typedef program cls;
    py::enum_<cls::program_kind_type>("program_kind")
      .value("UNKNOWN", cls::KND_UNKNOWN)
      .value("SOURCE", cls::KND_SOURCE)
      .value("BINARY", cls::KND_BINARY)
      ;

    py::class_<cls, boost::noncopyable>("_Program", py::no_init)
      .def("__init__", make_constructor(
            create_program_with_source,
            py::default_call_policies(),
            py::args("context", "src")))
      .def("__init__", make_constructor(
            create_program_with_binary,
            py::default_call_policies(),
            py::args("context", "devices", "binaries")))
#if (PYOPENCL_CL_VERSION >= 0x1020) && \
      ((PYOPENCL_CL_VERSION >= 0x1030) && defined(__APPLE__))
      .def("create_with_built_in_kernels",
          create_program_with_built_in_kernels,
          py::args("context", "devices", "kernel_names"),
          py::return_value_policy<py::manage_new_object>())
      .staticmethod("create_with_built_in_kernels")
#endif
      .DEF_SIMPLE_METHOD(kind)
      .DEF_SIMPLE_METHOD(get_info)
      .DEF_SIMPLE_METHOD(get_build_info)
      .def("_build", &cls::build,
          (py::arg("options")="", py::arg("devices")=py::object()))
#if PYOPENCL_CL_VERSION >= 0x1020
      .def("compile", &cls::compile,
          (py::arg("options")="", py::arg("devices")=py::object(),
           py::arg("headers")=py::list()))
      .def("link", &link_program,
          (py::arg("context"),
           py::arg("programs"),
           py::arg("options")="",
           py::arg("devices")=py::object()),
          py::return_value_policy<py::manage_new_object>())
      .staticmethod("link")
#endif
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__hash__", &cls::hash)
      .def("all_kernels", create_kernels_in_program)
      PYOPENCL_EXPOSE_TO_FROM_INT_PTR(cl_program)
      ;
  }

#if PYOPENCL_CL_VERSION >= 0x1020
  py::def("unload_platform_compiler", unload_platform_compiler);
#endif

  // }}}

  // {{{ kernel

  {
    typedef kernel cls;
    py::class_<cls, boost::noncopyable>("Kernel",
        py::init<const program &, std::string const &>())
      .DEF_SIMPLE_METHOD(get_info)
      .DEF_SIMPLE_METHOD(get_work_group_info)
      .DEF_SIMPLE_METHOD(set_arg)
#if PYOPENCL_CL_VERSION >= 0x1020
      .DEF_SIMPLE_METHOD(get_arg_info)
#endif
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__hash__", &cls::hash)
      PYOPENCL_EXPOSE_TO_FROM_INT_PTR(cl_kernel)
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
      (py::args("properties", "param_name"), py::arg("platform")=py::object()));
#endif

#endif
  // }}}
}




// vim: foldmethod=marker
