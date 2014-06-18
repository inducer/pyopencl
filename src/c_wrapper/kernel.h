#include "error.h"

#ifndef __PYOPENCL_KERNEL_H
#define __PYOPENCL_KERNEL_H

namespace pyopencl {

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
            pyopencl_call_guarded(clRetainKernel, this);
        }
    }
    ~kernel();
    generic_info get_info(cl_uint param) const;

    PYOPENCL_USE_RESULT generic_info
    get_work_group_info(cl_kernel_work_group_info param,
                        const device *dev) const;

    // #if PYOPENCL_CL_VERSION >= 0x1020
    //       py::object get_arg_info(
    //           cl_uint arg_index,
    //           cl_kernel_arg_info param_name
    //           ) const
    //       {
    //         switch (param_name)
    //         {
    // #define PYOPENCL_FIRST_ARG data(), arg_index // hackety hack
    //           case CL_KERNEL_ARG_ADDRESS_QUALIFIER:
    //             PYOPENCL_GET_INTEGRAL_INFO(KernelArg,
    //                 PYOPENCL_FIRST_ARG, param_name,
    //                 cl_kernel_arg_address_qualifier);

    //           case CL_KERNEL_ARG_ACCESS_QUALIFIER:
    //             PYOPENCL_GET_INTEGRAL_INFO(KernelArg,
    //                 PYOPENCL_FIRST_ARG, param_name,
    //                 cl_kernel_arg_access_qualifier);

    //           case CL_KERNEL_ARG_TYPE_NAME:
    //           case CL_KERNEL_ARG_NAME:
    //             PYOPENCL_GET_STR_INFO(KernelArg, PYOPENCL_FIRST_ARG, param_name);
    // #undef PYOPENCL_FIRST_ARG
    //           default:
    //             throw error("Kernel.get_arg_info", CL_INVALID_VALUE);
    //         }
    //       }
    // #endif
};

extern template void print_clobj<kernel>(std::ostream&, const kernel*);

// }}}

}

#endif
