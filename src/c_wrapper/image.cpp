#include "image.h"
#include "context.h"
#include "command_queue.h"
#include "event.h"
#include "buffer.h"

template void print_clobj<image>(std::ostream&, const image*);

PYOPENCL_USE_RESULT static PYOPENCL_INLINE image*
new_image(cl_mem mem, const cl_image_format *fmt)
{
    return pyopencl_convert_obj(image, clReleaseMemObject, mem, fmt);
}

generic_info
image::get_image_info(cl_image_info param) const
{
    switch (param) {
    case CL_IMAGE_FORMAT:
        return pyopencl_get_int_info(cl_image_format, Image, PYOPENCL_CL_CASTABLE_THIS, param);
    case CL_IMAGE_ELEMENT_SIZE:
    case CL_IMAGE_ROW_PITCH:
    case CL_IMAGE_SLICE_PITCH:
    case CL_IMAGE_WIDTH:
    case CL_IMAGE_HEIGHT:
    case CL_IMAGE_DEPTH:
#if PYOPENCL_CL_VERSION >= 0x1020
    case CL_IMAGE_ARRAY_SIZE:
#endif
        return pyopencl_get_int_info(size_t, Image, PYOPENCL_CL_CASTABLE_THIS, param);

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
        return pyopencl_get_int_info(cl_uint, Image, PYOPENCL_CL_CASTABLE_THIS, param);
#endif
    default:
        throw clerror("Image.get_image_info", CL_INVALID_VALUE);
    }
}

// c wrapper

// Image
error*
create_image_2d(clobj_t *img, clobj_t _ctx, cl_mem_flags flags,
                cl_image_format *fmt, size_t width, size_t height,
                size_t pitch, void *buf)
{
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_retry_mem_error([&] {
            auto mem = pyopencl_call_guarded(clCreateImage2D, ctx, flags, fmt,
                                             width, height, pitch, buf);
            *img = new_image(mem, fmt);
        });
}

error*
create_image_3d(clobj_t *img, clobj_t _ctx, cl_mem_flags flags,
                cl_image_format *fmt, size_t width, size_t height,
                size_t depth, size_t pitch_x, size_t pitch_y, void *buf)
{
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_retry_mem_error([&] {
            auto mem = pyopencl_call_guarded(clCreateImage3D, ctx, flags, fmt,
                                             width, height, depth, pitch_x,
                                             pitch_y, buf);
            *img = new_image(mem, fmt);
        });
}


error*
create_image_from_desc(clobj_t *img, clobj_t _ctx, cl_mem_flags flags,
                       cl_image_format *fmt, cl_image_desc *desc, void *buf)
{
#if PYOPENCL_CL_VERSION >= 0x1020
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_error([&] {
            auto mem = pyopencl_call_guarded(clCreateImage, ctx, flags, fmt,
                                             desc, buf);
            *img = new_image(mem, fmt);
        });
#else
    PYOPENCL_UNSUPPORTED(clCreateImage, "CL 1.1 and below")
#endif
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
enqueue_read_image(clobj_t *evt, clobj_t _queue, clobj_t _mem,
                   const size_t *_orig, size_t orig_l,
                   const size_t *_reg, size_t reg_l, void *buf,
                   size_t row_pitch, size_t slice_pitch,
                   const clobj_t *_wait_for, uint32_t num_wait_for,
                   int block, void *pyobj)
{
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    auto queue = static_cast<command_queue*>(_queue);
    auto img = static_cast<image*>(_mem);
    ConstBuffer<size_t, 3> orig(_orig, orig_l);
    ConstBuffer<size_t, 3> reg(_reg, reg_l, 1);
    return c_handle_retry_mem_error([&] {
            pyopencl_call_guarded(clEnqueueReadImage, queue, img, bool(block),
                                  orig, reg, row_pitch, slice_pitch, buf,
                                  wait_for, nanny_event_out(evt, pyobj));
        });
}

error*
enqueue_copy_image(clobj_t *evt, clobj_t _queue, clobj_t _src, clobj_t _dst,
                   const size_t *_src_orig, size_t src_orig_l,
                   const size_t *_dst_orig, size_t dst_orig_l,
                   const size_t *_reg, size_t reg_l,
                   const clobj_t *_wait_for, uint32_t num_wait_for)
{
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    auto queue = static_cast<command_queue*>(_queue);
    auto src = static_cast<image*>(_src);
    auto dst = static_cast<image*>(_dst);
    ConstBuffer<size_t, 3> src_orig(_src_orig, src_orig_l);
    ConstBuffer<size_t, 3> dst_orig(_dst_orig, dst_orig_l);
    ConstBuffer<size_t, 3> reg(_reg, reg_l, 1);
    return c_handle_retry_mem_error([&] {
            pyopencl_call_guarded(clEnqueueCopyImage, queue, src, dst, src_orig,
                                  dst_orig, reg, wait_for, event_out(evt));
        });
}

error*
enqueue_write_image(clobj_t *evt, clobj_t _queue, clobj_t _mem,
                    const size_t *_orig, size_t orig_l,
                    const size_t *_reg, size_t reg_l,
                    const void *buf, size_t row_pitch, size_t slice_pitch,
                    const clobj_t *_wait_for, uint32_t num_wait_for,
                    int block, void *pyobj)
{
    auto queue = static_cast<command_queue*>(_queue);
    auto img = static_cast<image*>(_mem);
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    ConstBuffer<size_t, 3> orig(_orig, orig_l);
    ConstBuffer<size_t, 3> reg(_reg, reg_l, 1);
    return c_handle_retry_mem_error([&] {
            pyopencl_call_guarded(clEnqueueWriteImage, queue, img, bool(block),
                                  orig, reg, row_pitch, slice_pitch, buf,
                                  wait_for, nanny_event_out(evt, pyobj));
        });
}

error*
enqueue_fill_image(clobj_t *evt, clobj_t _queue, clobj_t mem,
                   const void *color, const size_t *_orig, size_t orig_l,
                   const size_t *_reg, size_t reg_l,
                   const clobj_t *_wait_for, uint32_t num_wait_for)
{
#if PYOPENCL_CL_VERSION >= 0x1020
    // TODO debug color
    auto queue = static_cast<command_queue*>(_queue);
    auto img = static_cast<image*>(mem);
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    ConstBuffer<size_t, 3> orig(_orig, orig_l);
    ConstBuffer<size_t, 3> reg(_reg, reg_l, 1);
    return c_handle_retry_mem_error([&] {
            pyopencl_call_guarded(clEnqueueFillImage, queue, img, color, orig,
                                  reg, wait_for, event_out(evt));
        });
#else
    PYOPENCL_UNSUPPORTED(clEnqueueFillImage, "CL 1.1 and below")
#endif
}

// {{{ image transfers

error*
enqueue_copy_image_to_buffer(clobj_t *evt, clobj_t _queue, clobj_t _src,
                             clobj_t _dst, const size_t *_orig, size_t orig_l,
                             const size_t *_reg, size_t reg_l, size_t offset,
                             const clobj_t *_wait_for, uint32_t num_wait_for)
{
    auto queue = static_cast<command_queue*>(_queue);
    auto src = static_cast<image*>(_src);
    auto dst = static_cast<buffer*>(_dst);
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    ConstBuffer<size_t, 3> orig(_orig, orig_l);
    ConstBuffer<size_t, 3> reg(_reg, reg_l, 1);
    return c_handle_retry_mem_error([&] {
            pyopencl_call_guarded(clEnqueueCopyImageToBuffer, queue, src, dst,
                                  orig, reg, offset, wait_for, event_out(evt));
        });
}

error*
enqueue_copy_buffer_to_image(clobj_t *evt, clobj_t _queue, clobj_t _src,
                             clobj_t _dst, size_t offset, const size_t *_orig,
                             size_t orig_l, const size_t *_reg, size_t reg_l,
                             const clobj_t *_wait_for, uint32_t num_wait_for)
{
    auto queue = static_cast<command_queue*>(_queue);
    auto src = static_cast<buffer*>(_src);
    auto dst = static_cast<image*>(_dst);
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    ConstBuffer<size_t, 3> orig(_orig, orig_l);
    ConstBuffer<size_t, 3> reg(_reg, reg_l, 1);
    return c_handle_retry_mem_error([&] {
            pyopencl_call_guarded(clEnqueueCopyBufferToImage, queue, src, dst,
                                  offset, orig, reg, wait_for, event_out(evt));
        });
}

// }}}
