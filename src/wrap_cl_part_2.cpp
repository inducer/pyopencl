// Wrap CL
//
// Copyright (C) 2009-18 Andreas Kloeckner
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




namespace pyopencl {
#if PYOPENCL_CL_VERSION >= 0x1020
  py::object image_desc_dummy_getter(cl_image_desc &desc)
  {
    return py::none();
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




void pyopencl_expose_part_2(py::module &m)
{
  // {{{ image

#if PYOPENCL_CL_VERSION >= 0x1020
  {
    typedef cl_image_desc cls;
    py::class_<cls>(m, "ImageDescriptor")
      .def(py::init<>())
      .def_readwrite("image_type", &cls::image_type)
      .def_property("shape", &image_desc_dummy_getter, image_desc_set_shape)
      .def_readwrite("array_size", &cls::image_array_size)
      .def_property("pitches", &image_desc_dummy_getter, image_desc_set_pitches)
      .def_readwrite("num_mip_levels", &cls::num_mip_levels)
      .def_readwrite("num_samples", &cls::num_samples)
      .def_property("buffer", &image_desc_dummy_getter, image_desc_set_buffer)
      ;
  }
#endif

  {
    typedef image cls;
    py::class_<cls, memory_object>(m, "Image", py::dynamic_attr())
      .def(
          py::init(
            [](
              context const &ctx,
              cl_mem_flags flags,
              cl_image_format const &fmt,
              py::sequence shape,
              py::sequence pitches,
              py::object buffer)
            {
              return create_image(ctx, flags, fmt, shape, pitches, buffer);
            }),
          py::arg("context"),
          py::arg("flags"),
          py::arg("format"),
          py::arg("shape")=py::none(),
          py::arg("pitches")=py::none(),
          py::arg("hostbuf")=py::none()
          )
#if PYOPENCL_CL_VERSION >= 0x1020
      .def(
          py::init(
            [](
              context const &ctx,
              cl_mem_flags flags,
              cl_image_format const &fmt,
              cl_image_desc &desc,
              py::object buffer)
            {
              return create_image_from_desc(ctx, flags, fmt, desc, buffer);
            }),
          py::arg("context"),
          py::arg("flags"),
          py::arg("format"),
          py::arg("desc"),
          py::arg("hostbuf")=py::none()
          )
#endif
      .DEF_SIMPLE_METHOD(get_image_info)
      ;
  }

  {
    typedef cl_image_format cls;
    py::class_<cls>(m, "ImageFormat")
      .def(
          py::init(
            [](cl_channel_order ord, cl_channel_type tp)
            {
              return make_image_format(ord, tp);
            }))
      .def_readwrite("channel_order", &cls::image_channel_order)
      .def_readwrite("channel_data_type", &cls::image_channel_data_type)
      .def_property_readonly("channel_count", &get_image_format_channel_count)
      .def_property_readonly("dtype_size", &get_image_format_channel_dtype_size)
      .def_property_readonly("itemsize", &get_image_format_item_size)
      ;
  }

  DEF_SIMPLE_FUNCTION(get_supported_image_formats);

  m.def("_enqueue_read_image", enqueue_read_image,
      py::arg("queue"),
      py::arg("mem"),
      py::arg("origin"),
      py::arg("region"),
      py::arg("hostbuf"),
      py::arg("row_pitch")=0,
      py::arg("slice_pitch")=0,
      py::arg("wait_for")=py::none(),
      py::arg("is_blocking")=true
      );
  m.def("_enqueue_write_image", enqueue_write_image,
      py::arg("queue"),
      py::arg("mem"),
      py::arg("origin"),
      py::arg("region"),
      py::arg("hostbuf"),
      py::arg("row_pitch")=0,
      py::arg("slice_pitch")=0,
      py::arg("wait_for")=py::none(),
      py::arg("is_blocking")=true
      );

  m.def("_enqueue_copy_image", enqueue_copy_image,
      py::arg("queue"),
      py::arg("src"),
      py::arg("dest"),
      py::arg("src_origin"),
      py::arg("dest_origin"),
      py::arg("region"),
      py::arg("wait_for")=py::none()
      );
  m.def("_enqueue_copy_image_to_buffer", enqueue_copy_image_to_buffer,
      py::arg("queue"),
      py::arg("src"),
      py::arg("dest"),
      py::arg("origin"),
      py::arg("region"),
      py::arg("offset"),
      py::arg("wait_for")=py::none()
      );
  m.def("_enqueue_copy_buffer_to_image", enqueue_copy_buffer_to_image,
      py::arg("queue"),
      py::arg("src"),
      py::arg("dest"),
      py::arg("offset"),
      py::arg("origin"),
      py::arg("region"),
      py::arg("wait_for")=py::none()
      );

#if PYOPENCL_CL_VERSION >= 0x1020
  m.def("enqueue_fill_image", enqueue_fill_image,
      py::arg("queue"),
      py::arg("mem"),
      py::arg("color"),
      py::arg("origin"),
      py::arg("region"),
      py::arg("wait_for")=py::none()
      );
#endif

  // }}}

  // {{{ pipe

  {
    typedef pyopencl::pipe cls;
    py::class_<cls, memory_object>(m, "Pipe", py::dynamic_attr())
#if PYOPENCL_CL_VERSION >= 0x2000
      .def(
          py::init(
            [](
              context const &ctx,
              cl_mem_flags flags,
              cl_uint pipe_packet_size,
              cl_uint pipe_max_packets,
              py::sequence py_props)
            {
              return create_pipe(ctx, flags, pipe_packet_size, pipe_max_packets, py_props);
            }),
          py::arg("context"),
          py::arg("flags"),
          py::arg("packet_size"),
          py::arg("max_packets"),
          py::arg("properties")=py::make_tuple()
          )
#endif
      .DEF_SIMPLE_METHOD(get_pipe_info)
      ;
  }

  // }}}

  // {{{ memory_map
  {
    typedef memory_map cls;
    py::class_<cls>(m, "MemoryMap", py::dynamic_attr())
      .def("release", &cls::release,
          py::arg("queue").none(true)=nullptr,
          py::arg("wait_for")=py::none()
          )
      ;
  }

  // FIXME: Reenable in pypy
#ifndef PYPY_VERSION
  m.def("enqueue_map_buffer", enqueue_map_buffer,
      py::arg("queue"),
      py::arg("buf"),
      py::arg("flags"),
      py::arg("offset"),
      py::arg("shape"),
      py::arg("dtype"),
      py::arg("order")="C",
      py::arg("strides")=py::none(),
      py::arg("wait_for")=py::none(),
      py::arg("is_blocking")=true);
  m.def("enqueue_map_image", enqueue_map_image,
      py::arg("queue"),
      py::arg("img"),
      py::arg("flags"),
      py::arg("origin"),
      py::arg("region"),
      py::arg("shape"),
      py::arg("dtype"),
      py::arg("order")="C",
      py::arg("strides")=py::none(),
      py::arg("wait_for")=py::none(),
      py::arg("is_blocking")=true);
#endif

  // }}}

  // {{{ svm

#if PYOPENCL_CL_VERSION >= 0x2000
  {
    typedef svm_arg_wrapper cls;
    py::class_<cls>(m, "SVM", py::dynamic_attr())
      .def(py::init<py::object>())
      ;
  }

  {
    typedef svm_allocation cls;
    py::class_<cls>(m, "SVMAllocation", py::dynamic_attr())
      .def(py::init<std::shared_ptr<context>, size_t, cl_uint, cl_svm_mem_flags>())
      .DEF_SIMPLE_METHOD(release)
      .def("enqueue_release", &cls::enqueue_release,
          ":returns: a :class:`pyopencl.Event`\n\n"
          "|std-enqueue-blurb|")
      .def("_ptr_as_int", &cls::ptr_as_int)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__hash__", &cls::ptr_as_int)
      ;
  }

  m.def("_enqueue_svm_memcpy", enqueue_svm_memcpy,
      py::arg("queue"),
      py::arg("is_blocking"),
      py::arg("dst"),
      py::arg("src"),
      py::arg("wait_for")=py::none()
      );

  m.def("_enqueue_svm_memfill", enqueue_svm_memfill,
      py::arg("queue"),
      py::arg("dst"),
      py::arg("pattern"),
      py::arg("byte_count")=py::none(),
      py::arg("wait_for")=py::none()
      );

  m.def("_enqueue_svm_map", enqueue_svm_map,
      py::arg("queue"),
      py::arg("is_blocking"),
      py::arg("flags"),
      py::arg("svm"),
      py::arg("wait_for")=py::none()
      );

  m.def("_enqueue_svm_unmap", enqueue_svm_unmap,
      py::arg("queue"),
      py::arg("svm"),
      py::arg("wait_for")=py::none()
      );
#endif

#if PYOPENCL_CL_VERSION >= 0x2010
  m.def("_enqueue_svm_migrate_mem", enqueue_svm_migratemem,
      py::arg("queue"),
      py::arg("svms"),
      py::arg("flags")=py::none(),
      py::arg("wait_for")=py::none()
      );
#endif

  // }}}

  // {{{ sampler
  {
    typedef sampler cls;
    py::class_<cls>(m, "Sampler", py::dynamic_attr())
#if PYOPENCL_CL_VERSION >= 0x2000
      .def(py::init<context const &, py::sequence>())
#endif
      .def(py::init<context const &, bool, cl_addressing_mode, cl_filter_mode>())
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
    py::enum_<cls::program_kind_type>(m, "program_kind")
      .value("UNKNOWN", cls::KND_UNKNOWN)
      .value("SOURCE", cls::KND_SOURCE)
      .value("BINARY", cls::KND_BINARY)
      .value("IL", cls::KND_IL)
      ;

    py::class_<cls>(m, "_Program", py::dynamic_attr())
      .def(
          py::init(
            [](context &ctx, std::string const &src)
            {
              return create_program_with_source(ctx, src);
            }),
          py::arg("context"),
          py::arg("src"))
      .def(
          py::init(
            [](context &ctx, py::sequence devices, py::sequence binaries)
            {
              return create_program_with_binary(ctx, devices, binaries);
            }),
          py::arg("context"),
          py::arg("devices"),
          py::arg("binaries"))
#if (PYOPENCL_CL_VERSION >= 0x1020) || \
      ((PYOPENCL_CL_VERSION >= 0x1030) && defined(__APPLE__))
      .def_static("create_with_built_in_kernels",
          create_program_with_built_in_kernels,
          py::arg("context"),
          py::arg("devices"),
          py::arg("kernel_names"))
#endif
      .DEF_SIMPLE_METHOD(kind)
      .DEF_SIMPLE_METHOD(get_info)
      .DEF_SIMPLE_METHOD(get_build_info)
      .def("_build", &cls::build,
          py::arg("options")="",
          py::arg("devices")=py::none())
#if PYOPENCL_CL_VERSION >= 0x1020
      .def("compile", &cls::compile,
          py::arg("options")="",
          py::arg("devices")=py::none(),
          py::arg("headers")=py::list())
      .def_static("link", &link_program,
          py::arg("context"),
          py::arg("programs"),
          py::arg("options")="",
          py::arg("devices")=py::none()
          )
#endif
#if PYOPENCL_CL_VERSION >= 0x2020
      .def("set_specialization_constant", &cls::set_specialization_constant,
          py::arg("spec_id"),
          py::arg("buffer"))
#endif
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__hash__", &cls::hash)
      .def("all_kernels", create_kernels_in_program)
      PYOPENCL_EXPOSE_TO_FROM_INT_PTR(cl_program)
      ;
  }

#if (PYOPENCL_CL_VERSION >= 0x2010)
  m.def("_create_program_with_il", create_program_with_il);
#endif

#if PYOPENCL_CL_VERSION >= 0x1020
  m.def("unload_platform_compiler", unload_platform_compiler);
#endif

  // }}}

  // {{{ kernel

  {
    typedef kernel cls;
    py::class_<cls>(m, "Kernel", py::dynamic_attr())
      .def(py::init<const program &, std::string const &>())
      .DEF_SIMPLE_METHOD(get_info)
      .DEF_SIMPLE_METHOD(get_work_group_info)
#if PYOPENCL_CL_VERSION >= 0x2010
      .DEF_SIMPLE_METHOD(clone)
#endif
      .def("_set_arg_null", &cls::set_arg_null)
      .def("_set_arg_buf", &cls::set_arg_buf)
#if PYOPENCL_CL_VERSION >= 0x2000
      .def("_set_arg_svm", &cls::set_arg_svm)
#endif
      .def("_set_arg_multi",
          [](cls &knl, py::tuple indices_and_args)
          {
            set_arg_multi(
                [&](cl_uint i, py::handle arg) { knl.set_arg(i, arg); },
                indices_and_args);
          })
      .def("_set_arg_buf_multi",
          [](cls &knl, py::tuple indices_and_args)
          {
            set_arg_multi(
                [&](cl_uint i, py::handle arg) { knl.set_arg_buf(i, arg); },
                indices_and_args);
          })
      .def("_set_arg_buf_pack_multi",
          [](cls &knl, py::tuple indices_chars_and_args)
          {
            set_arg_multi(
                [&](cl_uint i, py::handle typechar, py::handle arg)
                { knl.set_arg_buf_pack(i, typechar, arg); },
                indices_chars_and_args);
          })
      .DEF_SIMPLE_METHOD(set_arg)
#if PYOPENCL_CL_VERSION >= 0x1020
      .DEF_SIMPLE_METHOD(get_arg_info)
#endif
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__hash__", &cls::hash)
      PYOPENCL_EXPOSE_TO_FROM_INT_PTR(cl_kernel)
#if PYOPENCL_CL_VERSION >= 0x2010
      .def("get_sub_group_info", &cls::get_sub_group_info,
          py::arg("device"),
          py::arg("param"),
          py::arg("input_value")=py::none()
          )
#endif
      ;
  }

  {
    typedef local_memory cls;
    py::class_<cls>(m, "LocalMemory", py::dynamic_attr())
      .def(
          py::init<size_t>(),
          py::arg("size"))
      .def_property_readonly("size", &cls::size)
      ;
  }


  m.def("enqueue_nd_range_kernel", enqueue_nd_range_kernel,
      py::arg("queue"),
      py::arg("kernel"),
      py::arg("global_work_size"),
      py::arg("local_work_size"),
      py::arg("global_work_offset")=py::none(),
      py::arg("wait_for")=py::none(),
      py::arg("g_times_l")=false,
      py::arg("allow_empty_ndrange")=false
      );

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
    py::class_<cls, memory_object>(m, "GLBuffer", py::dynamic_attr())
      .def(
          py::init(
            [](context &ctx, cl_mem_flags flags, GLuint bufobj)
            {
              return create_from_gl_buffer(ctx, flags, bufobj);
            }),
          py::arg("context"),
          py::arg("flags"),
          py::arg("bufobj"))
      .def("get_gl_object_info", get_gl_object_info)
      ;
  }

  {
    typedef gl_renderbuffer cls;
    py::class_<cls, memory_object>(m, "GLRenderBuffer", py::dynamic_attr())
      .def(
          py::init(
            [](context &ctx, cl_mem_flags flags, GLuint bufobj)
            {
              return create_from_gl_renderbuffer(ctx, flags, bufobj);
            }),
          py::arg("context"),
          py::arg("flags"),
          py::arg("bufobj"))
      .def("get_gl_object_info", get_gl_object_info)
      ;
  }

  {
    typedef gl_texture cls;
    py::class_<cls, image>(m, "GLTexture", py::dynamic_attr())
      .def(
          py::init(
            [](context &ctx, cl_mem_flags flags, GLenum texture_target,
              GLint miplevel, GLuint texture, unsigned dims)
            {
              return create_from_gl_texture(ctx, flags, texture_target, miplevel, texture, dims);
            }),
          py::arg("context"),
          py::arg("flags"),
          py::arg("texture_target"),
          py::arg("miplevel"),
          py::arg("texture"),
          py::arg("dims"))
      .def("get_gl_object_info", get_gl_object_info)
      .DEF_SIMPLE_METHOD(get_gl_texture_info)
      ;
  }

  m.def("enqueue_acquire_gl_objects", enqueue_acquire_gl_objects,
      py::arg("queue"),
      py::arg("mem_objects"),
      py::arg("wait_for")=py::none()
      );
  m.def("enqueue_release_gl_objects", enqueue_release_gl_objects,
      py::arg("queue"),
      py::arg("mem_objects"),
      py::arg("wait_for")=py::none()
      );

#if defined(cl_khr_gl_sharing) && (cl_khr_gl_sharing >= 1)
  m.def("get_gl_context_info_khr", get_gl_context_info_khr,
      py::arg("properties"),
      py::arg("param_name"),
      py::arg("platform")=py::none()
      );
#endif

#endif
  // }}}

}


// vim: foldmethod=marker
