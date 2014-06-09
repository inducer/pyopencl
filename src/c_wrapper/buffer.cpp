#include "buffer.h"
#include "context.h"
#include "command_queue.h"
#include "event.h"

namespace pyopencl {

#if PYOPENCL_CL_VERSION >= 0x1010
PYOPENCL_USE_RESULT buffer*
buffer::get_sub_region(size_t origin, size_t size, cl_mem_flags flags) const
{
    cl_buffer_region region = {origin, size};

    auto mem = retry_mem_error([&] {
            return pyopencl_call_guarded(clCreateSubBuffer, this, flags,
                                         CL_BUFFER_CREATE_TYPE_REGION,
                                         &region);
        });
    return new_buffer(mem);
}

//       buffer *getitem(py::slice slc) const
//       {
//         PYOPENCL_BUFFER_SIZE_T start, end, stride, length;

//         size_t my_length;
//         PYOPENCL_CALL_GUARDED(clGetMemObjectInfo,
//             (this, CL_MEM_SIZE, sizeof(my_length), &my_length, 0));

// #if PY_VERSION_HEX >= 0x03020000
//         if (PySlice_GetIndicesEx(slc.ptr(),
// #else
//         if (PySlice_GetIndicesEx(reinterpret_cast<PySliceObject *>(slc.ptr()),
// #endif
//               my_length, &start, &end, &stride, &length) != 0)
//           throw py::error_already_set();

//         if (stride != 1)
//           throw clerror("Buffer.__getitem__", CL_INVALID_VALUE,
//               "Buffer slice must have stride 1");

//         cl_mem_flags my_flags;
//         PYOPENCL_CALL_GUARDED(clGetMemObjectInfo,
//             (this, CL_MEM_FLAGS, sizeof(my_flags), &my_flags, 0));

//         return get_sub_region(start, end, my_flags);
//       }
#endif

}

// c wrapper
// Import all the names in pyopencl namespace for c wrappers.
using namespace pyopencl;

// Buffer
error*
create_buffer(clobj_t *buffer, clobj_t _ctx, cl_mem_flags flags,
              size_t size, void *hostbuf)
{
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_error([&] {
            auto mem = retry_mem_error([&] {
                    return pyopencl_call_guarded(clCreateBuffer, ctx,
                                                 flags, size, hostbuf);
                });
            *buffer = new_buffer(mem, (flags & CL_MEM_USE_HOST_PTR ?
                                       hostbuf : nullptr));
        });
}

error*
enqueue_read_buffer(clobj_t *_evt, clobj_t _queue, clobj_t _mem,
                    void *buffer, size_t size, size_t device_offset,
                    const clobj_t *_wait_for, uint32_t num_wait_for,
                    int is_blocking, void *pyobj)
{
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    auto queue = static_cast<command_queue*>(_queue);
    auto mem = static_cast<memory_object*>(_mem);
    return c_handle_error([&] {
            cl_event evt;
            retry_mem_error([&] {
                    pyopencl_call_guarded(
                        clEnqueueReadBuffer, queue, mem,
                        cast_bool(is_blocking), device_offset, size,
                        buffer, wait_for, &evt);
                });
            *_evt = new_nanny_event(evt, pyobj);
        });
}

error*
enqueue_write_buffer(clobj_t *_evt, clobj_t _queue, clobj_t _mem,
                     const void *buffer, size_t size, size_t device_offset,
                     const clobj_t *_wait_for, uint32_t num_wait_for,
                     int is_blocking, void *pyobj)
{
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    auto queue = static_cast<command_queue*>(_queue);
    auto mem = static_cast<memory_object*>(_mem);
    return c_handle_error([&] {
            cl_event evt;
            retry_mem_error([&] {
                    pyopencl_call_guarded(
                        clEnqueueWriteBuffer, queue, mem,
                        cast_bool(is_blocking), device_offset, size, buffer,
                        wait_for, &evt);
                });
            *_evt = new_nanny_event(evt, pyobj);
        });
}

error*
enqueue_copy_buffer(clobj_t *_evt, clobj_t _queue, clobj_t _src, clobj_t _dst,
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
            cl_event evt;
            retry_mem_error([&] {
                    pyopencl_call_guarded(
                        clEnqueueCopyBuffer, queue, src, dst, src_offset,
                        dst_offset, byte_count, wait_for, &evt);
                });
            *_evt = new_event(evt);
        });
}

error*
enqueue_fill_buffer(clobj_t *_evt, clobj_t _queue, clobj_t _mem, void *pattern,
                    size_t psize, size_t offset, size_t size,
                    const clobj_t *_wait_for, uint32_t num_wait_for)
{
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    auto queue = static_cast<command_queue*>(_queue);
    auto mem = static_cast<memory_object*>(_mem);
    return c_handle_error([&] {
            cl_event evt;
            retry_mem_error([&] {
                    pyopencl_call_guarded(
                        clEnqueueFillBuffer, queue, mem, pattern, psize,
                        offset, size, wait_for, &evt);
                });
            *_evt = new_event(evt);
        });
}
