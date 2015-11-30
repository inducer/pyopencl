#include "error.h"

#ifndef __PYOPENCL_CONTEXT_H
#define __PYOPENCL_CONTEXT_H

// {{{ context

extern template class clobj<cl_context>;
extern template void print_arg<cl_context>(std::ostream&,
                                           const cl_context&, bool);
extern template void print_buf<cl_context>(std::ostream&, const cl_context*,
                                           size_t, ArgType, bool, bool);

class context : public clobj<cl_context> {
public:
    PYOPENCL_DEF_CL_CLASS(CONTEXT);
    static void get_version(cl_context ctx, int *major, int *minor);
    PYOPENCL_INLINE
    context(cl_context ctx, bool retain)
        : clobj(ctx)
    {
        if (retain) {
            pyopencl_call_guarded(clRetainContext, PYOPENCL_CL_CASTABLE_THIS);
        }
    }
    ~context();
    generic_info get_info(cl_uint param_name) const;
};

extern template void print_clobj<context>(std::ostream&, const context*);

// }}}

#endif
