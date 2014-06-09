#include "image.h"
#include "context.h"
#include "command_queue.h"
#include "event.h"

namespace pyopencl {

generic_info
image::get_image_info(cl_image_info param) const
{
    switch (param) {
    case CL_IMAGE_FORMAT:
        return pyopencl_get_int_info(cl_image_format, Image, this, param);
    case CL_IMAGE_ELEMENT_SIZE:
    case CL_IMAGE_ROW_PITCH:
    case CL_IMAGE_SLICE_PITCH:
    case CL_IMAGE_WIDTH:
    case CL_IMAGE_HEIGHT:
    case CL_IMAGE_DEPTH:
#if PYOPENCL_CL_VERSION >= 0x1020
    case CL_IMAGE_ARRAY_SIZE:
#endif
        return pyopencl_get_int_info(size_t, Image, this, param);

#if PYOPENCL_CL_VERSION >= 0x1020
        // TODO:
        //    case CL_IMAGE_BUFFER:
        //      {
        //        cl_mem param_value;
        //        PYOPENCL_CALL_GUARDED(clGetImageInfo, (this, param, sizeof(param_value), &param_value, 0));
        //        if (param_value == 0)
        //               {
        //                 // no associated memory object? no problem.
        //                 return py::object();
        //               }
        //        return create_mem_object_wrapper(param_value);
        //      }
    case CL_IMAGE_NUM_MIP_LEVELS:
    case CL_IMAGE_NUM_SAMPLES:
        return pyopencl_get_int_info(cl_uint, Image, this, param);
#endif
    default:
        throw clerror("Image.get_image_info", CL_INVALID_VALUE);
    }
}

// {{{ image creation

// #if PYOPENCL_CL_VERSION >= 0x1020

//   PYOPENCL_INLINE
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

//   PYOPENCL_INLINE
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

//   PYOPENCL_INLINE
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

// }}}

}

// c wrapper
// Import all the names in pyopencl namespace for c wrappers.
using namespace pyopencl;

// Image
error*
create_image_2d(clobj_t *img, clobj_t _ctx, cl_mem_flags flags,
                cl_image_format *fmt, size_t width, size_t height,
                size_t pitch, void *buffer)
{
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_error([&] {
            auto mem = retry_mem_error([&] {
                    return pyopencl_call_guarded(
                        clCreateImage2D, ctx, flags,
                        fmt, width, height, pitch, buffer);
                });
            *img = new_image(mem, (flags & CL_MEM_USE_HOST_PTR ?
                                   buffer : nullptr), fmt);
        });
}

error*
create_image_3d(clobj_t *img, clobj_t _ctx, cl_mem_flags flags,
                cl_image_format *fmt, size_t width, size_t height,
                size_t depth, size_t pitch_x, size_t pitch_y, void *buffer)
{
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_error([&] {
            auto mem = retry_mem_error([&] {
                    return pyopencl_call_guarded(
                        clCreateImage3D, ctx, flags, fmt, width,
                        height, depth, pitch_x, pitch_y, buffer);
                });
            *img = new_image(mem, (flags & CL_MEM_USE_HOST_PTR ?
                                   buffer : nullptr), fmt);
        });
}

error*
image__get_image_info(clobj_t _img, cl_image_info param, generic_info *out)
{
    auto img = static_cast<image*>(_img);
    return c_handle_error([&] {
            *out = img->get_image_info(param);
        });
}

type_t
image__get_fill_type(clobj_t img)
{
    return static_cast<image*>(img)->get_fill_type();
}

error*
enqueue_read_image(clobj_t *_evt, clobj_t _queue, clobj_t _mem,
                   const size_t *_origin, size_t origin_l,
                   const size_t *_region, size_t region_l, void *buffer,
                   size_t row_pitch, size_t slice_pitch,
                   const clobj_t *_wait_for, uint32_t num_wait_for,
                   int is_blocking, void *pyobj)
{
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    auto queue = static_cast<command_queue*>(_queue);
    auto img = static_cast<image*>(_mem);
    ConstBuffer<size_t, 3> origin(_origin, origin_l);
    ConstBuffer<size_t, 3> region(_region, region_l);
    return c_handle_error([&] {
            cl_event evt;
            retry_mem_error([&] {
                    pyopencl_call_guarded(
                        clEnqueueReadImage, queue, img,
                        cast_bool(is_blocking), origin, region, row_pitch,
                        slice_pitch, buffer, wait_for, &evt);
                });
            *_evt = new_nanny_event(evt, pyobj);
        });
}

error*
enqueue_copy_image(clobj_t *_evt, clobj_t _queue, clobj_t _src, clobj_t _dst,
                   const size_t *_src_origin, size_t src_origin_l,
                   const size_t *_dst_origin, size_t dst_origin_l,
                   const size_t *_region, size_t region_l,
                   const clobj_t *_wait_for, uint32_t num_wait_for)
{
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    auto queue = static_cast<command_queue*>(_queue);
    auto src = static_cast<image*>(_src);
    auto dst = static_cast<image*>(_dst);
    ConstBuffer<size_t, 3> src_origin(_src_origin, src_origin_l);
    ConstBuffer<size_t, 3> dst_origin(_dst_origin, dst_origin_l);
    ConstBuffer<size_t, 3> region(_region, region_l);
    return c_handle_error([&] {
            cl_event evt;
            retry_mem_error([&] {
                    pyopencl_call_guarded(
                        clEnqueueCopyImage, queue, src, dst, src_origin,
                        dst_origin, region, wait_for, &evt);
                });
            *_evt = new_event(evt);
        });
}

error*
enqueue_write_image(clobj_t *_evt, clobj_t _queue, clobj_t _mem,
                    const size_t *_origin, size_t origin_l,
                    const size_t *_region, size_t region_l,
                    const void *buffer, size_t row_pitch, size_t slice_pitch,
                    const clobj_t *_wait_for, uint32_t num_wait_for,
                    int is_blocking, void *pyobj)
{
    auto queue = static_cast<command_queue*>(_queue);
    auto img = static_cast<image*>(_mem);
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    ConstBuffer<size_t, 3> origin(_origin, origin_l);
    ConstBuffer<size_t, 3> region(_region, region_l);
    return c_handle_error([&] {
            cl_event evt;
            retry_mem_error([&] {
                    pyopencl_call_guarded(
                        clEnqueueWriteImage, queue, img,
                        cast_bool(is_blocking), origin, region, row_pitch,
                        slice_pitch, buffer, wait_for, &evt);
                });
            *_evt = new_nanny_event(evt, pyobj);
        });
}

#if PYOPENCL_CL_VERSION >= 0x1020
//   PYOPENCL_INLINE
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
#endif
