#include "clhelper.h"

#ifndef __PYOPENCL_EVENT_H
#define __PYOPENCL_EVENT_H

namespace pyopencl {

// {{{ event

extern template class clobj<cl_event>;

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
PYOPENCL_USE_RESULT static PYOPENCL_INLINE event*
new_event(cl_event evt)
{
    return pyopencl_convert_obj(event, clReleaseEvent, evt);
}

class nanny_event : public event {
private:
    void *m_ward;
public:
    nanny_event(cl_event evt, bool retain, void *ward=nullptr)
        : event(evt, retain), m_ward(ward)
    {
        if (ward) {
            py::ref(ward);
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
PYOPENCL_USE_RESULT static PYOPENCL_INLINE event*
new_nanny_event(cl_event evt, void *ward)
{
    return pyopencl_convert_obj(nanny_event, clReleaseEvent, evt, ward);
}

// }}}

}

#endif
