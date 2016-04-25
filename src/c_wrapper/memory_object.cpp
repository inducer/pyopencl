#include "memory_object.h"
#include "context.h"
#include "event.h"
#include "command_queue.h"
#include "clhelper.h"

template class clobj<cl_mem>;
template void print_arg<cl_mem>(std::ostream&, const cl_mem&, bool);
template void print_buf<cl_mem>(std::ostream&, const cl_mem*,
                                size_t, ArgType, bool, bool);

generic_info
memory_object::get_info(cl_uint param_name) const
{
    switch ((cl_mem_info)param_name) {
    case CL_MEM_TYPE:
        return pyopencl_get_int_info(cl_mem_object_type, MemObject,
                                     PYOPENCL_CL_CASTABLE_THIS, param_name);
    case CL_MEM_FLAGS:
        return pyopencl_get_int_info(cl_mem_flags, MemObject,
                                     PYOPENCL_CL_CASTABLE_THIS, param_name);
    case CL_MEM_SIZE:
        return pyopencl_get_int_info(size_t, MemObject, PYOPENCL_CL_CASTABLE_THIS, param_name);
    case CL_MEM_HOST_PTR:
        throw clerror("MemoryObject.get_info", CL_INVALID_VALUE,
                      "Use MemoryObject.get_host_array to get "
                      "host pointer.");
    case CL_MEM_MAP_COUNT:
    case CL_MEM_REFERENCE_COUNT:
        return pyopencl_get_int_info(cl_uint, MemObject,
                                     PYOPENCL_CL_CASTABLE_THIS, param_name);
    case CL_MEM_CONTEXT:
        return pyopencl_get_opaque_info(context, MemObject, PYOPENCL_CL_CASTABLE_THIS, param_name);

#if PYOPENCL_CL_VERSION >= 0x1010
        // TODO
        //       case CL_MEM_ASSOCIATED_MEMOBJECT:
        //      {
        //        cl_mem param_value;
        //        PYOPENCL_CALL_GUARDED(clGetMemObjectInfo, (this, param_name, sizeof(param_value), &param_value, 0));
        //        if (param_value == 0)
        //          {
        //            // no associated memory object? no problem.
        //            return py::object();
        //          }

        //        return create_mem_object_wrapper(param_value);
        //      }
    case CL_MEM_OFFSET:
        return pyopencl_get_int_info(size_t, MemObject, PYOPENCL_CL_CASTABLE_THIS, param_name);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
    case CL_MEM_USES_SVM_POINTER:
        return pyopencl_get_int_info(cl_bool, MemObject, PYOPENCL_CL_CASTABLE_THIS, param_name);
#endif

    default:
        throw clerror("MemoryObject.get_info", CL_INVALID_VALUE);
    }
}

memory_object::~memory_object()
{
    if (!m_valid.exchange(false))
        return;
    pyopencl_call_guarded_cleanup(clReleaseMemObject, PYOPENCL_CL_CASTABLE_THIS);
}

// c wrapper

// Memory Object
error*
memory_object__release(clobj_t obj)
{
    return c_handle_error([&] {
            static_cast<memory_object*>(obj)->release();
        });
}

error*
memory_object__get_host_array(clobj_t _obj, void **hostptr, size_t *size)
{
    auto obj = static_cast<memory_object*>(_obj);
    return c_handle_error([&] {
            cl_mem_flags flags;
            pyopencl_call_guarded(clGetMemObjectInfo, obj, CL_MEM_FLAGS,
                                  size_arg(flags), nullptr);
            if (!(flags & CL_MEM_USE_HOST_PTR))
                throw clerror("MemoryObject.get_host_array", CL_INVALID_VALUE,
                              "Only MemoryObject with USE_HOST_PTR "
                              "is supported.");
            pyopencl_call_guarded(clGetMemObjectInfo, obj, CL_MEM_HOST_PTR,
                                  size_arg(*hostptr), nullptr);
            pyopencl_call_guarded(clGetMemObjectInfo, obj, CL_MEM_SIZE,
                                  size_arg(*size), nullptr);
        });
}

error*
enqueue_migrate_mem_objects(clobj_t *evt, clobj_t _queue,
                            const clobj_t *_mem_obj, uint32_t num_mem_obj,
                            cl_mem_migration_flags flags,
                            const clobj_t *_wait_for, uint32_t num_wait_for)
{
#if PYOPENCL_CL_VERSION >= 0x1020
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    const auto mem_obj = buf_from_class<memory_object>(_mem_obj, num_mem_obj);
    auto queue = static_cast<command_queue*>(_queue);
    return c_handle_retry_mem_error([&] {
            pyopencl_call_guarded(clEnqueueMigrateMemObjects, queue,
                                  mem_obj, flags, wait_for, event_out(evt));
        });
#else
    PYOPENCL_UNSUPPORTED_BEFORE(clEnqueueMigrateMemObjects, "CL 1.2")
#endif
}
