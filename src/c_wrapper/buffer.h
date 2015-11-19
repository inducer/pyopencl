#include "memory_object.h"
#include "clhelper.h"

#ifndef __PYOPENCL_BUFFER_H
#define __PYOPENCL_BUFFER_H

// {{{ buffer

class buffer : public memory_object {
public:
    PYOPENCL_DEF_CL_CLASS(BUFFER);
    PYOPENCL_INLINE
    buffer(cl_mem mem, bool retain)
        : memory_object(mem, retain)
    {}

#if PYOPENCL_CL_VERSION >= 0x1010
    PYOPENCL_USE_RESULT buffer *get_sub_region(size_t orig, size_t size,
                                               cl_mem_flags flags) const;
#endif
};

extern template void print_clobj<buffer>(std::ostream&, const buffer*);

// }}}

#endif
