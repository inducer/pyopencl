#include "clhelper.h"
#include <thread>

#ifndef __PYOPENCL_EVENT_H
#define __PYOPENCL_EVENT_H

// {{{ event

extern template class clobj<cl_event>;
extern template void print_arg<cl_event>(std::ostream&, const cl_event&, bool);
extern template void print_buf<cl_event>(std::ostream&, const cl_event*,
                                         size_t, ArgType, bool, bool);

class event_private;

class event : public clobj<cl_event> {
    event_private *m_p;
    // return whether the event need to be released.
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
#if PYOPENCL_CL_VERSION >= 0x1010 && defined(PYOPENCL_HAVE_EVENT_SET_CALLBACK)
    template<typename Func>
    PYOPENCL_INLINE void
    set_callback(cl_int type, Func &&_func)
    {
        auto func = new rm_ref_t<Func>(std::forward<Func>(_func));
        try {
            pyopencl_call_guarded(
                clSetEventCallback, PYOPENCL_CL_CASTABLE_THIS, type,
                static_cast<void (CL_CALLBACK * /* pfn_notify */)(cl_event, cl_int, void *)>(
                    [] (cl_event, cl_int status, void *data) {
                        rm_ref_t<Func> *func = static_cast<rm_ref_t<Func>*>(data);

                        // We won't necessarily be able to acquire the GIL inside this
                        // handler without deadlocking. Create a thread that *can*
                        // wait.

                        std::thread t([func, status] () {
                                (*func)(status);
                                delete func;
                            });
                        t.detach();

                    }), (void*)func);
        } catch (...) {
            delete func;
            throw;
        }
    }
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

#endif
