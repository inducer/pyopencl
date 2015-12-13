#include <algorithm>
#include "buffer.h"
#include "context.h"
#include "command_queue.h"
#include "event.h"

template void print_clobj<buffer>(std::ostream&, const buffer*);

PYOPENCL_USE_RESULT static PYOPENCL_INLINE buffer*
new_buffer(cl_mem mem)
{
    return pyopencl_convert_obj(buffer, clReleaseMemObject, mem);
}

#if PYOPENCL_CL_VERSION >= 0x1010
PYOPENCL_USE_RESULT buffer*
buffer::get_sub_region(size_t orig, size_t size, cl_mem_flags flags) const
{
    cl_buffer_region reg = {orig, size};

    auto mem = retry_mem_error([&] {
            return pyopencl_call_guarded(clCreateSubBuffer, PYOPENCL_CL_CASTABLE_THIS, flags,
                                         CL_BUFFER_CREATE_TYPE_REGION, &reg);
        });
    return new_buffer(mem);
}

#endif

// c wrapper

// Buffer
error*
create_buffer(clobj_t *buffer, clobj_t _ctx, cl_mem_flags flags,
              size_t size, void *hostbuf)
{
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_retry_mem_error([&] {
            auto mem = pyopencl_call_guarded(clCreateBuffer, ctx,
                                             flags, size, hostbuf);
            *buffer = new_buffer(mem);
        });
}

error*
enqueue_read_buffer(clobj_t *evt, clobj_t _queue, clobj_t _mem,
                    void *buffer, size_t size, size_t device_offset,
                    const clobj_t *_wait_for, uint32_t num_wait_for,
                    int block, void *pyobj)
{
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    auto queue = static_cast<command_queue*>(_queue);
    auto mem = static_cast<memory_object*>(_mem);
    return c_handle_retry_mem_error([&] {
            pyopencl_call_guarded(
                clEnqueueReadBuffer, queue, mem, bool(block), device_offset,
                size, buffer, wait_for, nanny_event_out(evt, pyobj));
        });
}

error*
enqueue_write_buffer(clobj_t *evt, clobj_t _queue, clobj_t _mem,
                     const void *buffer, size_t size, size_t device_offset,
                     const clobj_t *_wait_for, uint32_t num_wait_for,
                     int block, void *pyobj)
{
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    auto queue = static_cast<command_queue*>(_queue);
    auto mem = static_cast<memory_object*>(_mem);
    return c_handle_retry_mem_error([&] {
            pyopencl_call_guarded(
                clEnqueueWriteBuffer, queue, mem, bool(block), device_offset,
                size, buffer, wait_for, nanny_event_out(evt, pyobj));
        });
}

error*
enqueue_copy_buffer(clobj_t *evt, clobj_t _queue, clobj_t _src, clobj_t _dst,
                    ptrdiff_t byte_count, size_t src_offset, size_t dst_offset,
                    const clobj_t *_wait_for, uint32_t num_wait_for)
{
    auto queue = static_cast<command_queue*>(_queue);
    auto src = static_cast<memory_object*>(_src);
    auto dst = static_cast<memory_object*>(_dst);
    return c_handle_error([&] {
            if (byte_count < 0) {
                size_t byte_count_src = 0;
                size_t byte_count_dst = 0;
                pyopencl_call_guarded(
                    clGetMemObjectInfo, src, CL_MEM_SIZE,
                    sizeof(byte_count), &byte_count_src, nullptr);
                pyopencl_call_guarded(
                    clGetMemObjectInfo, src, CL_MEM_SIZE,
                    sizeof(byte_count), &byte_count_dst, nullptr);
                byte_count = std::min(byte_count_src, byte_count_dst);
            }
            const auto wait_for = buf_from_class<event>(_wait_for,
                                                        num_wait_for);
            retry_mem_error([&] {
                    pyopencl_call_guarded(
                        clEnqueueCopyBuffer, queue, src, dst, src_offset,
                        dst_offset, byte_count, wait_for, event_out(evt));
                });
        });
}


error*
enqueue_fill_buffer(clobj_t *evt, clobj_t _queue, clobj_t _mem, void *pattern,
                    size_t psize, size_t offset, size_t size,
                    const clobj_t *_wait_for, uint32_t num_wait_for)
{
#if PYOPENCL_CL_VERSION >= 0x1020
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    auto queue = static_cast<command_queue*>(_queue);
    auto mem = static_cast<memory_object*>(_mem);
    // TODO debug print pattern
    return c_handle_retry_mem_error([&] {
            pyopencl_call_guarded(clEnqueueFillBuffer, queue, mem, pattern,
                                  psize, offset, size, wait_for,
                                  event_out(evt));
        });
#else
    PYOPENCL_UNSUPPORTED(clEnqueueFillBuffer, "CL 1.1 and below")
#endif
}


// {{{ rectangular transfers

error*
enqueue_read_buffer_rect(clobj_t *evt, clobj_t _queue, clobj_t _mem, void *buf,
                         const size_t *_buf_orig, size_t buf_orig_l,
                         const size_t *_host_orig, size_t host_orig_l,
                         const size_t *_reg, size_t reg_l,
                         const size_t *_buf_pitches, size_t buf_pitches_l,
                         const size_t *_host_pitches, size_t host_pitches_l,
                         const clobj_t *_wait_for, uint32_t num_wait_for,
                         int block, void *pyobj)
{
#if PYOPENCL_CL_VERSION >= 0x1010
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    auto queue = static_cast<command_queue*>(_queue);
    auto mem = static_cast<memory_object*>(_mem);
    ConstBuffer<size_t, 3> buf_orig(_buf_orig, buf_orig_l);
    ConstBuffer<size_t, 3> host_orig(_host_orig, host_orig_l);
    ConstBuffer<size_t, 3> reg(_reg, reg_l, 1);
    ConstBuffer<size_t, 2> buf_pitches(_buf_pitches, buf_pitches_l);
    ConstBuffer<size_t, 2> host_pitches(_host_pitches, host_pitches_l);
    return c_handle_retry_mem_error([&] {
            pyopencl_call_guarded(
                clEnqueueReadBufferRect, queue, mem, bool(block), buf_orig,
                host_orig, reg, buf_pitches[0], buf_pitches[1], host_pitches[0],
                host_pitches[1], buf, wait_for, nanny_event_out(evt, pyobj));
        });
#else
    PYOPENCL_UNSUPPORTED(clEnqueueReadBufferRect, "CL 1.0")
#endif
}

error*
enqueue_write_buffer_rect(clobj_t *evt, clobj_t _queue, clobj_t _mem, void *buf,
                          const size_t *_buf_orig, size_t buf_orig_l,
                          const size_t *_host_orig, size_t host_orig_l,
                          const size_t *_reg, size_t reg_l,
                          const size_t *_buf_pitches, size_t buf_pitches_l,
                          const size_t *_host_pitches, size_t host_pitches_l,
                          const clobj_t *_wait_for, uint32_t num_wait_for,
                          int block, void *pyobj)
{
#if PYOPENCL_CL_VERSION >= 0x1010
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    auto queue = static_cast<command_queue*>(_queue);
    auto mem = static_cast<memory_object*>(_mem);
    ConstBuffer<size_t, 3> buf_orig(_buf_orig, buf_orig_l);
    ConstBuffer<size_t, 3> host_orig(_host_orig, host_orig_l);
    ConstBuffer<size_t, 3> reg(_reg, reg_l, 1);
    ConstBuffer<size_t, 2> buf_pitches(_buf_pitches, buf_pitches_l);
    ConstBuffer<size_t, 2> host_pitches(_host_pitches, host_pitches_l);
    return c_handle_retry_mem_error([&] {
            pyopencl_call_guarded(
                clEnqueueWriteBufferRect, queue, mem, bool(block), buf_orig,
                host_orig, reg, buf_pitches[0], buf_pitches[1], host_pitches[0],
                host_pitches[1], buf, wait_for, nanny_event_out(evt, pyobj));
        });
#else
    PYOPENCL_UNSUPPORTED(clEnqueueWriteBufferRect, "CL 1.0")
#endif
}

error*
enqueue_copy_buffer_rect(clobj_t *evt, clobj_t _queue, clobj_t _src,
                         clobj_t _dst, const size_t *_src_orig,
                         size_t src_orig_l, const size_t *_dst_orig,
                         size_t dst_orig_l, const size_t *_reg, size_t reg_l,
                         const size_t *_src_pitches, size_t src_pitches_l,
                         const size_t *_dst_pitches, size_t dst_pitches_l,
                         const clobj_t *_wait_for, uint32_t num_wait_for)
{
#if PYOPENCL_CL_VERSION >= 0x1010
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    auto queue = static_cast<command_queue*>(_queue);
    auto src = static_cast<memory_object*>(_src);
    auto dst = static_cast<memory_object*>(_dst);
    ConstBuffer<size_t, 3> src_orig(_src_orig, src_orig_l);
    ConstBuffer<size_t, 3> dst_orig(_dst_orig, dst_orig_l);
    ConstBuffer<size_t, 3> reg(_reg, reg_l, 1);
    ConstBuffer<size_t, 2> src_pitches(_src_pitches, src_pitches_l);
    ConstBuffer<size_t, 2> dst_pitches(_dst_pitches, dst_pitches_l);
    return c_handle_retry_mem_error([&] {
            pyopencl_call_guarded(
                clEnqueueCopyBufferRect, queue, src, dst, src_orig, dst_orig,
                reg, src_pitches[0], src_pitches[1], dst_pitches[0],
                dst_pitches[1], wait_for, event_out(evt));
        });
#else
    PYOPENCL_UNSUPPORTED(clEnqueueCopyBufferRect, "CL 1.0")
#endif
}

// }}}

error*
buffer__get_sub_region(clobj_t *_sub_buf, clobj_t _buf, size_t orig,
                       size_t size, cl_mem_flags flags)
{
#if PYOPENCL_CL_VERSION >= 0x1010
    auto buf = static_cast<buffer*>(_buf);
    return c_handle_error([&] {
            *_sub_buf = buf->get_sub_region(orig, size, flags);
        });
#else
    PYOPENCL_UNSUPPORTED(clCreateSubBuffer, "CL 1.0")
#endif
}
