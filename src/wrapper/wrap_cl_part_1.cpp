#include "wrap_cl.hpp"




using namespace pyopencl;




void pyopencl_expose_part_1()
{
  py::def("get_cl_header_version", get_cl_header_version);

  // {{{ platform
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

  // }}}

  // {{{ device
  {
    typedef device cls;
    py::class_<cls, boost::noncopyable>("Device", py::no_init)
      .DEF_SIMPLE_METHOD(get_info)
      .add_property("obj_ptr", &cls::obj_ptr)
      .def(py::self == py::self)
      .def(py::self != py::self)
#if defined(cl_ext_device_fission) && defined(PYOPENCL_USE_DEVICE_FISSION)
      .DEF_SIMPLE_METHOD(create_sub_devices)
#endif
      ;
  }

  // }}}

  // {{{ context

  {
    typedef context cls;
    py::class_<cls, boost::noncopyable, 
      boost::shared_ptr<cls> >("Context", py::no_init)
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

  // }}}

  // {{{ command queue
  {
    typedef command_queue cls;
    py::class_<cls, boost::noncopyable>("CommandQueue",
        py::init<const context &,
          const device *, cl_command_queue_properties>
        ((py::arg("context"), py::arg("device")=py::object(), py::arg("properties")=0)))
      .DEF_SIMPLE_METHOD(get_info)
#ifndef CL_VERSION_1_1
      .DEF_SIMPLE_METHOD(set_property)
#endif
      .DEF_SIMPLE_METHOD(flush)
      .DEF_SIMPLE_METHOD(finish)
      .add_property("obj_ptr", &cls::obj_ptr)
      .def(py::self == py::self)
      .def(py::self != py::self)
      ;
  }

  // }}}

  // {{{ events/synchronization
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
  DEF_SIMPLE_FUNCTION(enqueue_barrier);

#ifdef CL_VERSION_1_1
  {
    typedef user_event cls;
    py::class_<cls, py::bases<event>, boost::noncopyable>("UserEvent", py::no_init)
      .def("__init__", make_constructor(
            create_user_event, py::default_call_policies(), py::args("context")))
      .DEF_SIMPLE_METHOD(set_status)
      ;
  }
#endif

  // }}}

  // {{{ memory_object

  {
    typedef memory_object_holder cls;
    py::class_<cls, boost::noncopyable>(
        "MemoryObjectHolder", py::no_init)
      .DEF_SIMPLE_METHOD(get_info)
      .def("get_host_array", get_mem_obj_host_array,
          (py::arg("shape"), py::arg("dtype"), py::arg("order")="C"))
      .def(py::self == py::self)
      .def(py::self != py::self)
      ;
  }
  {
    typedef memory_object cls;
    py::class_<cls, boost::noncopyable, py::bases<memory_object_holder> >(
        "MemoryObject", py::no_init)
      .DEF_SIMPLE_METHOD(release)
      .add_property("obj_ptr", &cls::obj_ptr)
      .add_property("hostbuf", &cls::hostbuf)
      ;
  }

  // }}}

  // {{{ buffer
  {
    typedef buffer cls;
    py::class_<cls, py::bases<memory_object>, boost::noncopyable>(
        "Buffer", py::no_init)
      .def("__init__", make_constructor(create_buffer_py,
            py::default_call_policies(),
            (py::args("context", "flags"),
             py::arg("size")=0,
             py::arg("hostbuf")=py::object()
            )))
#ifdef CL_VERSION_1_1
      .def("get_sub_region", &cls::get_sub_region,
          (py::args("origin", "size"), py::arg("flags")=0),
          py::return_value_policy<py::manage_new_object>())
      .def("__getitem__", &cls::getitem,
          py::return_value_policy<py::manage_new_object>())
#endif
      ;
  }

  // {{{ transfers

  // {{{ byte-for-byte
  py::def("_enqueue_read_buffer", enqueue_read_buffer,
      (py::args("queue", "mem", "hostbuf"),
       py::arg("device_offset")=0,
       py::arg("wait_for")=py::object(),
       py::arg("is_blocking")=true,
       py::arg("host_buffer")=py::object()
       ),
      py::return_value_policy<py::manage_new_object>());
  py::def("_enqueue_write_buffer", enqueue_write_buffer,
      (py::args("queue", "mem", "hostbuf"),
       py::arg("device_offset")=0,
       py::arg("wait_for")=py::object(),
       py::arg("is_blocking")=true,
       py::arg("host_buffer")=py::object()
       ),
      py::return_value_policy<py::manage_new_object>());
  py::def("_enqueue_copy_buffer", enqueue_copy_buffer,
      (py::args("queue", "src", "dst"),
       py::arg("byte_count")=0,
       py::arg("src_offset")=0,
       py::arg("dst_offset")=0,
       py::arg("wait_for")=py::object()
       ),
      py::return_value_policy<py::manage_new_object>());

  // }}}

  // {{{ rectangular

#ifdef CL_VERSION_1_1
  py::def("_enqueue_read_buffer_rect", enqueue_read_buffer_rect,
      (py::args("queue", "mem", "hostbuf",
                "buffer_origin", "host_origin", "region"),
       py::arg("buffer_pitches")=py::object(),
       py::arg("host_pitches")=py::object(),
       py::arg("wait_for")=py::object(),
       py::arg("is_blocking")=true
       ),
      py::return_value_policy<py::manage_new_object>());
  py::def("_enqueue_write_buffer_rect", enqueue_write_buffer_rect,
      (py::args("queue", "mem", "hostbuf",
                "buffer_origin", "host_origin", "region"),
       py::arg("buffer_pitches")=py::object(),
       py::arg("host_pitches")=py::object(),
       py::arg("wait_for")=py::object(),
       py::arg("is_blocking")=true
       ),
      py::return_value_policy<py::manage_new_object>());
  py::def("_enqueue_copy_buffer_rect", enqueue_copy_buffer_rect,
      (py::args("queue", "src", "dst",
                "src_origin", "dst_origin", "region"),
       py::arg("src_pitches")=py::object(),
       py::arg("dst_pitches")=py::object(),
       py::arg("wait_for")=py::object()
       ),
      py::return_value_policy<py::manage_new_object>());
#endif

  // }}}

  // }}}

  // }}}
}

// vim: foldmethod=marker
