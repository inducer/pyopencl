#include "context.h"
#include "command_queue.h"
#include "event.h"

error*
svm_alloc(
    clobj_t _ctx, cl_mem_flags flags, size_t size, cl_uint alignment,
    void **result)
{
#if PYOPENCL_CL_VERSION >= 0x2000
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_retry_mem_error([&] {
            *result = clSVMAlloc(ctx->data(), flags, size, alignment);
            if (!*result)
                throw clerror("clSVMalloc", CL_INVALID_VALUE,
                    "(allocation failure, unspecified reason)");
        });
#else
    PYOPENCL_UNSUPPORTED_BEFORE(clSVMAlloc, "CL 2.0")
#endif
}


error*
svm_free(clobj_t _ctx, void *svm_pointer)
{
#if PYOPENCL_CL_VERSION >= 0x2000
    auto ctx = static_cast<context*>(_ctx);
    // no error returns (?!)
    clSVMFree(ctx->data(), svm_pointer);
    return nullptr;
#else
    PYOPENCL_UNSUPPORTED_BEFORE(clSVMFree, "CL 2.0")
#endif
}


error*
enqueue_svm_free(
    clobj_t *evt, clobj_t _queue,
    cl_uint num_svm_pointers,
    void *svm_pointers[],
    const clobj_t *_wait_for, uint32_t num_wait_for)
{
#if PYOPENCL_CL_VERSION >= 0x2000
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    auto queue = static_cast<command_queue*>(_queue);
    return c_handle_retry_mem_error([&] {
        pyopencl_call_guarded(
            clEnqueueSVMFree, queue,
            num_svm_pointers, svm_pointers,
            /* pfn_free_func*/ nullptr,
            /* user_data */ nullptr,
            wait_for, event_out(evt));
        });
#else
    PYOPENCL_UNSUPPORTED_BEFORE(clEnqueueSVMFree, "CL 2.0")
#endif
}


error*
enqueue_svm_memcpy(
    clobj_t *evt, clobj_t _queue,
    cl_bool is_blocking,
    void *dst_ptr, const void *src_ptr, size_t size,
    const clobj_t *_wait_for, uint32_t num_wait_for, void *pyobj)
{
#if PYOPENCL_CL_VERSION >= 0x2000
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    auto queue = static_cast<command_queue*>(_queue);
    return c_handle_retry_mem_error([&] {
        pyopencl_call_guarded(
            clEnqueueSVMMemcpy, queue,
            is_blocking,
            dst_ptr, src_ptr, size,
            wait_for, nanny_event_out(evt, pyobj));
        });
#else
    PYOPENCL_UNSUPPORTED_BEFORE(clEnqueueSVMMemcpy, "CL 2.0")
#endif
}


error*
enqueue_svm_memfill(
    clobj_t *evt, clobj_t _queue,
    void *svm_ptr,
    const void *pattern, size_t pattern_size, size_t size,
    const clobj_t *_wait_for, uint32_t num_wait_for)
{
#if PYOPENCL_CL_VERSION >= 0x2000
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    auto queue = static_cast<command_queue*>(_queue);
    return c_handle_retry_mem_error([&] {
        pyopencl_call_guarded(
            clEnqueueSVMMemFill, queue,
            svm_ptr,
            pattern, pattern_size, size,
            wait_for, event_out(evt));
        });
#else
    PYOPENCL_UNSUPPORTED_BEFORE(clEnqueueSVMMemFill, "CL 2.0")
#endif
}


error*
enqueue_svm_map(
    clobj_t *evt, clobj_t _queue,
    cl_bool blocking_map, cl_map_flags map_flags,
    void *svm_ptr, size_t size,
    const clobj_t *_wait_for, uint32_t num_wait_for)
{
#if PYOPENCL_CL_VERSION >= 0x2000
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    auto queue = static_cast<command_queue*>(_queue);
    return c_handle_retry_mem_error([&] {
        pyopencl_call_guarded(
            clEnqueueSVMMap, queue,
            blocking_map, map_flags,
            svm_ptr, size,
            wait_for, event_out(evt));
        });
#else
    PYOPENCL_UNSUPPORTED_BEFORE(clEnqueueSVMMap, "CL 2.0")
#endif
}


error*
enqueue_svm_unmap(
    clobj_t *evt, clobj_t _queue,
    void *svm_ptr,
    const clobj_t *_wait_for, uint32_t num_wait_for)
{
#if PYOPENCL_CL_VERSION >= 0x2000
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    auto queue = static_cast<command_queue*>(_queue);
    return c_handle_retry_mem_error([&] {
        pyopencl_call_guarded(
            clEnqueueSVMUnmap, queue,
            svm_ptr,
            wait_for, event_out(evt));
        });
#else
    PYOPENCL_UNSUPPORTED_BEFORE(clEnqueueSVMUnmap, "CL 2.0")
#endif
}


error*
enqueue_svm_migrate_mem(
    clobj_t *evt, clobj_t _queue,
    cl_uint num_svm_pointers,
    const void **svm_pointers,
    const size_t *sizes,
    cl_mem_migration_flags flags,
    const clobj_t *_wait_for, uint32_t num_wait_for)
{
#if PYOPENCL_CL_VERSION >= 0x2010
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    auto queue = static_cast<command_queue*>(_queue);
    return c_handle_retry_mem_error([&] {
        pyopencl_call_guarded(
            clEnqueueSVMMigrateMem, queue,
            num_svm_pointers, svm_pointers, sizes, flags,
            wait_for, event_out(evt));
        });
#else
    PYOPENCL_UNSUPPORTED_BEFORE(clEnqueueSVMMigrateMem, "CL 2.1")
#endif
}
