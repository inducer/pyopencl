#include "clhelper.h"

#ifndef __PYOPENCL_EVENT_H
#define __PYOPENCL_EVENT_H

namespace pyopencl {

// {{{ event

extern template class clobj<cl_event>;
extern template void print_arg<cl_event>(std::ostream&, const cl_event&, bool);
extern template void print_buf<cl_event>(std::ostream&, const cl_event*,
                                         size_t, ArgType, bool, bool);

class event_private;

class event : public clobj<cl_event> {
    event_private *m_p;
    void release_private() noexcept;
protected:
    PYOPENCL_INLINE event_private*
    get_p() const
    {
        return m_p;
    }
public:
    PYOPENCL_DEF_CL_CLASS(EVENT);
    event(cl_event event, bool retain, event_private *p=nullptr);
    ~event();
    generic_info get_info(cl_uint param) const;
    PYOPENCL_USE_RESULT generic_info
    get_profiling_info(cl_profiling_info param) const;
    void wait() const;
#if PYOPENCL_CL_VERSION >= 0x1010
    bool support_cb;
    void set_callback(cl_int type, const std::function<void(cl_int)> &func);
#endif
};
static PYOPENCL_INLINE auto
event_out(clobj_t *ret) -> decltype(pyopencl_outarg(event, ret, clReleaseEvent))
{
    return pyopencl_outarg(event, ret, clReleaseEvent);
}

extern template void print_clobj<event>(std::ostream&, const event*);

class nanny_event : public event {
public:
    nanny_event(cl_event evt, bool retain, void *ward=nullptr);
    PYOPENCL_USE_RESULT void *get_ward() const noexcept;
};
static PYOPENCL_INLINE auto
nanny_event_out(clobj_t *ret, void *ward)
    -> decltype(pyopencl_outarg(nanny_event, ret, clReleaseEvent, ward))
{
    return pyopencl_outarg(nanny_event, ret, clReleaseEvent, ward);
}

// }}}

}

#endif
