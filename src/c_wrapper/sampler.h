#include "error.h"

#ifndef __PYOPENCL_SAMPLER_H
#define __PYOPENCL_SAMPLER_H

namespace pyopencl {

// {{{ sampler

class sampler : public clobj<cl_sampler> {
public:
    PYOPENCL_DEF_CL_CLASS(SAMPLER);
    PYOPENCL_INLINE
    sampler(cl_sampler samp, bool retain)
        : clobj(samp)
    {
        if (retain) {
            pyopencl_call_guarded(clRetainSampler, this);
        }
    }
    ~sampler();
    generic_info get_info(cl_uint param_name) const;
};

// }}}

}

#endif
