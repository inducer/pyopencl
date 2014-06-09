#include "memory_object.h"
#include "clhelper.h"

#ifndef __PYOPENCL_BUFFER_H
#define __PYOPENCL_BUFFER_H

namespace pyopencl {

// {{{ buffer

class buffer : public memory_object {
public:
    PYOPENCL_DEF_CL_CLASS(BUFFER);
    PYOPENCL_INLINE
    buffer(cl_mem mem, bool retain, void *hostbuf=0)
        : memory_object(mem, retain, hostbuf)
    {}

#if PYOPENCL_CL_VERSION >= 0x1010
    PYOPENCL_USE_RESULT buffer *get_sub_region(size_t origin, size_t size,
                                               cl_mem_flags flags) const;
#endif
};
PYOPENCL_USE_RESULT static PYOPENCL_INLINE buffer*
new_buffer(cl_mem mem, void *buff=0)
{
    return pyopencl_convert_obj(buffer, clReleaseMemObject, mem, buff);
}

// }}}

}

#endif
