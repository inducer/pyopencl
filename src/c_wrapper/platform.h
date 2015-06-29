#include "error.h"

#ifndef __PYOPENCL_PLATFORM_H
#define __PYOPENCL_PLATFORM_H

// {{{ platform

extern template class clobj<cl_platform_id>;
extern template void print_arg<cl_platform_id>(std::ostream&,
                                               const cl_platform_id&, bool);
extern template void print_buf<cl_platform_id>(
    std::ostream&, const cl_platform_id*, size_t, ArgType, bool, bool);

class platform : public clobj<cl_platform_id> {
public:
    static void get_version(cl_platform_id plat, int *major, int *minor);
    using clobj::clobj;
    PYOPENCL_DEF_CL_CLASS(PLATFORM);

    generic_info get_info(cl_uint param_name) const;
};

extern template void print_clobj<platform>(std::ostream&, const platform*);

// }}}

#endif
