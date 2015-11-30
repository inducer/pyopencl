#include "error.h"
#include <atomic>

#ifndef __PYOPENCL_MEMORY_OBJECT_H
#define __PYOPENCL_MEMORY_OBJECT_H

// {{{ memory_object

extern template class clobj<cl_mem>;
extern template void print_arg<cl_mem>(std::ostream&, const cl_mem&, bool);
extern template void print_buf<cl_mem>(std::ostream&, const cl_mem*,
                                       size_t, ArgType, bool, bool);

class memory_object : public clobj<cl_mem> {
private:
    mutable volatile std::atomic_bool m_valid;
public:
    constexpr static const char *class_name = "MEMORY_OBJECT";
    PYOPENCL_INLINE
    memory_object(cl_mem mem, bool retain)
        : clobj(mem), m_valid(true)
    {
        if (retain) {
            pyopencl_call_guarded(clRetainMemObject, PYOPENCL_CL_CASTABLE_THIS);
        }
    }
    PYOPENCL_INLINE
    memory_object(const memory_object &mem)
        : memory_object(mem.data(), true)
    {}
    ~memory_object();
    generic_info get_info(cl_uint param_name) const;
    void
    release() const
    {
        if (PYOPENCL_UNLIKELY(!m_valid.exchange(false))) {
            throw clerror("MemoryObject.release", CL_INVALID_VALUE,
                          "trying to double-unref mem object");
        }
        pyopencl_call_guarded(clReleaseMemObject, PYOPENCL_CL_CASTABLE_THIS);
    }
#if 0
    PYOPENCL_USE_RESULT size_t
    size() const
    {
        size_t param_value;
        pyopencl_call_guarded(clGetMemObjectInfo, this, CL_MEM_SIZE,
                              size_arg(param_value), nullptr);
        return param_value;
    }
#endif
};

// }}}

#endif
