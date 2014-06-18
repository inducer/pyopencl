#include "memory_object.h"
#include "context.h"
#include "clhelper.h"

namespace pyopencl {

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
                                     this, param_name);
    case CL_MEM_FLAGS:
        return pyopencl_get_int_info(cl_mem_flags, MemObject,
                                     this, param_name);
    case CL_MEM_SIZE:
        return pyopencl_get_int_info(size_t, MemObject, this, param_name);
    case CL_MEM_HOST_PTR:
        throw clerror("MemoryObject.get_info", CL_INVALID_VALUE,
                      "Use MemoryObject.get_host_array to get "
                      "host pointer.");
    case CL_MEM_MAP_COUNT:
    case CL_MEM_REFERENCE_COUNT:
        return pyopencl_get_int_info(cl_uint, MemObject,
                                     this, param_name);
    case CL_MEM_CONTEXT:
        return pyopencl_get_opaque_info(context, MemObject, this, param_name);

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
        return pyopencl_get_int_info(size_t, MemObject, this, param_name);
#endif

    default:
        throw clerror("MemoryObject.get_info", CL_INVALID_VALUE);
    }
}

memory_object::~memory_object()
{
    if (!m_valid.exchange(false))
        return;
    pyopencl_call_guarded_cleanup(clReleaseMemObject, this);
}

}

// c wrapper
// Import all the names in pyopencl namespace for c wrappers.
using namespace pyopencl;

// Memory Object
error*
memory_object__release(clobj_t obj)
{
    return c_handle_error([&] {
            static_cast<memory_object*>(obj)->release();
        });
}
