#include "error.h"

#ifndef __PYOPENCL_CONTEXT_H
#define __PYOPENCL_CONTEXT_H

namespace pyopencl {

// {{{ context

class context : public clobj<cl_context> {
public:
    PYOPENCL_DEF_CL_CLASS(CONTEXT);
    PYOPENCL_INLINE
    context(cl_context ctx, bool retain)
        : clobj(ctx)
    {
        if (retain) {
            pyopencl_call_guarded(clRetainContext, this);
        }
    }
    ~context();
    generic_info get_info(cl_uint param_name) const;
};

// }}}

}

#endif
