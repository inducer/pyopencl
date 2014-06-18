#include "clhelper.h"

#ifndef __PYOPENCL_EVENT_H
#define __PYOPENCL_EVENT_H

namespace pyopencl {

// {{{ event

extern template class clobj<cl_event>;
extern template void print_arg<cl_event>(std::ostream&, const cl_event&, bool);
extern template void print_buf<cl_event>(std::ostream&, const cl_event*,
                                         size_t, ArgType, bool, bool);

class event : public clobj<cl_event> {
public:
    PYOPENCL_DEF_CL_CLASS(EVENT);
    PYOPENCL_INLINE
    event(cl_event event, bool retain)
        : clobj(event)
    {
        if (retain) {
            pyopencl_call_guarded(clRetainEvent, this);
        }
    }
    ~event();
    generic_info get_info(cl_uint param) const;
    PYOPENCL_USE_RESULT generic_info
    get_profiling_info(cl_profiling_info param) const;
    virtual void
    finished()
    {}
    void wait();
#if PYOPENCL_CL_VERSION >= 0x1010
    void set_callback(cl_int type, const std::function<void(cl_int)> &func);
#endif
};
static PYOPENCL_INLINE auto
event_out(clobj_t *ret)
    -> decltype(pyopencl_outarg(event, ret, clReleaseEvent))
{
    return pyopencl_outarg(event, ret, clReleaseEvent);
}

extern template void print_clobj<event>(std::ostream&, const event*);

class nanny_event : public event {
private:
    void *m_ward;
public:
    nanny_event(cl_event evt, bool retain, void *ward=nullptr)
        : event(evt, retain), m_ward(nullptr)
    {
        if (ward) {
            m_ward = py::ref(ward);
        }
    }
    ~nanny_event();
    PYOPENCL_USE_RESULT PYOPENCL_INLINE void*
    get_ward() const
    {
        return m_ward;
    }
    void finished();
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
