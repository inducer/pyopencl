#include "error.h"

#ifndef __PYOPENCL_SAMPLER_H
#define __PYOPENCL_SAMPLER_H

// {{{ sampler

extern template class clobj<cl_sampler>;
extern template void print_arg<cl_sampler>(std::ostream&,
                                           const cl_sampler&, bool);
extern template void print_buf<cl_sampler>(std::ostream&, const cl_sampler*,
                                           size_t, ArgType, bool, bool);

class sampler : public clobj<cl_sampler> {
public:
    PYOPENCL_DEF_CL_CLASS(SAMPLER);
    PYOPENCL_INLINE
    sampler(cl_sampler samp, bool retain)
        : clobj(samp)
    {
        if (retain) {
            pyopencl_call_guarded(clRetainSampler, PYOPENCL_CL_CASTABLE_THIS);
        }
    }
    ~sampler();
    generic_info get_info(cl_uint param_name) const;
};

extern template void print_clobj<sampler>(std::ostream&, const sampler*);

// }}}

#endif
