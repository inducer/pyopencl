#include "error.h"

#ifndef __PYOPENCL_KERNEL_H
#define __PYOPENCL_KERNEL_H

class device;

// {{{ kernel

extern template class clobj<cl_kernel>;
extern template void print_arg<cl_kernel>(std::ostream&,
                                          const cl_kernel&, bool);
extern template void print_buf<cl_kernel>(std::ostream&, const cl_kernel*,
                                          size_t, ArgType, bool, bool);

class kernel : public clobj<cl_kernel> {
public:
    PYOPENCL_DEF_CL_CLASS(KERNEL);
    PYOPENCL_INLINE
    kernel(cl_kernel knl, bool retain)
        : clobj(knl)
    {
        if (retain) {
            pyopencl_call_guarded(clRetainKernel, PYOPENCL_CL_CASTABLE_THIS);
        }
    }
    ~kernel();
    generic_info get_info(cl_uint param) const;

    PYOPENCL_USE_RESULT generic_info
    get_work_group_info(cl_kernel_work_group_info param,
                        const device *dev) const;

#if PYOPENCL_CL_VERSION >= 0x1020
    PYOPENCL_USE_RESULT generic_info
    get_arg_info(cl_uint idx, cl_kernel_arg_info param) const;
#endif
};

extern template void print_clobj<kernel>(std::ostream&, const kernel*);

// }}}

#endif
