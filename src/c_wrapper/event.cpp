#include "event.h"
#include "command_queue.h"
#include "context.h"
#include "async.h"
#include "pyhelper.h"

namespace pyopencl {

template class clobj<cl_event>;

#if PYOPENCL_CL_VERSION >= 0x1010
class event_callback {
    std::function<void(cl_int)> m_func;
    event_callback(const std::function<void(cl_int)> &func)
        : m_func(func)
    {}
    static void
    cl_call_and_free(cl_event, cl_int status, void *data)
    {
        auto cb = static_cast<event_callback*>(data);
        auto func = cb->m_func;
        try {
            call_async([func, status] {func(status);});
        } catch (...) {
        }
        delete cb;
    }

    friend class event;
};
#endif

event::~event()
{
    pyopencl_call_guarded_cleanup(clReleaseEvent, this);
}

generic_info
event::get_info(cl_uint param_name) const
{
    switch ((cl_event_info)param_name) {
    case CL_EVENT_COMMAND_QUEUE:
        return pyopencl_get_opaque_info(command_queue, Event, this, param_name);
    case CL_EVENT_COMMAND_TYPE:
        return pyopencl_get_int_info(cl_command_type, Event,
                                     this, param_name);
    case CL_EVENT_COMMAND_EXECUTION_STATUS:
        return pyopencl_get_int_info(cl_int, Event, this, param_name);
    case CL_EVENT_REFERENCE_COUNT:
        return pyopencl_get_int_info(cl_uint, Event, this, param_name);
#if PYOPENCL_CL_VERSION >= 0x1010
    case CL_EVENT_CONTEXT:
        return pyopencl_get_opaque_info(context, Event, this, param_name);
#endif

    default:
        throw clerror("Event.get_info", CL_INVALID_VALUE);
    }
}

generic_info
event::get_profiling_info(cl_profiling_info param) const
{
    switch (param) {
    case CL_PROFILING_COMMAND_QUEUED:
    case CL_PROFILING_COMMAND_SUBMIT:
    case CL_PROFILING_COMMAND_START:
    case CL_PROFILING_COMMAND_END:
        return pyopencl_get_int_info(cl_ulong, EventProfiling, this, param);
    default:
        throw clerror("Event.get_profiling_info", CL_INVALID_VALUE);
    }
}

void
event::wait()
{
    pyopencl_call_guarded(clWaitForEvents,
                          make_argbuf<ArgType::Length>(data()));
    finished();
}

#if PYOPENCL_CL_VERSION >= 0x1010
void
event::set_callback(cl_int type, const std::function<void(cl_int)> &func)
{
    auto cb = new event_callback(func);
    try {
        pyopencl_call_guarded(clSetEventCallback, this, type,
                              &event_callback::cl_call_and_free, cb);
    } catch (...) {
        delete cb;
        throw;
    }
    init_async();
}
#endif

nanny_event::~nanny_event()
{
    if (m_ward) {
        wait();
    }
}

void
nanny_event::finished()
{
    // No lock needed because multiple release is safe here.
    void *ward = m_ward;
    m_ward = nullptr;
    py::deref(ward);
}

}

// c wrapper
// Import all the names in pyopencl namespace for c wrappers.
using namespace pyopencl;

// Event
error*
event__get_profiling_info(clobj_t _evt, cl_profiling_info param,
                          generic_info *out)
{
    auto evt = static_cast<event*>(_evt);
    return c_handle_error([&] {
            *out = evt->get_profiling_info(param);
        });
}

error*
event__wait(clobj_t evt)
{
    return c_handle_error([&] {
            static_cast<event*>(evt)->wait();
        });
}

#if PYOPENCL_CL_VERSION >= 0x1010
error*
event__set_callback(clobj_t _evt, cl_int type, void *pyobj)
{
    auto evt = static_cast<event*>(_evt);
    return c_handle_error([&] {
            evt->set_callback(type, [=] (cl_int status) {
                    py::call(pyobj, status);
                    py::deref(pyobj);
                });
            py::ref(pyobj);
        });
}
#endif

// Nanny Event
void*
nanny_event__get_ward(clobj_t evt)
{
    return static_cast<nanny_event*>(evt)->get_ward();
}

error*
wait_for_events(const clobj_t *_wait_for, uint32_t num_wait_for)
{
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    return c_handle_error([&] {
            pyopencl_call_guarded(clWaitForEvents, wait_for);
        });
}

error*
enqueue_wait_for_events(clobj_t _queue, const clobj_t *_wait_for,
                        uint32_t num_wait_for)
{
    auto queue = static_cast<command_queue*>(_queue);
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    return c_handle_error([&] {
            pyopencl_call_guarded(clEnqueueWaitForEvents, queue, wait_for);
        });
}
