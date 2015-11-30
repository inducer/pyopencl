#include "error.h"

#ifndef __PYOPENCL_COMMAND_QUEUE_H
#define __PYOPENCL_COMMAND_QUEUE_H

// {{{ command_queue

extern template class clobj<cl_command_queue>;
extern template void print_arg<cl_command_queue>(
    std::ostream&, const cl_command_queue&, bool);
extern template void print_buf<cl_command_queue>(
    std::ostream&, const cl_command_queue*, size_t, ArgType, bool, bool);

class command_queue : public clobj<cl_command_queue> {
public:
    PYOPENCL_DEF_CL_CLASS(COMMAND_QUEUE);
    PYOPENCL_INLINE
    command_queue(cl_command_queue q, bool retain)
        : clobj(q)
    {
        if (retain) {
            pyopencl_call_guarded(clRetainCommandQueue, PYOPENCL_CL_CASTABLE_THIS);
        }
    }
    PYOPENCL_INLINE
    command_queue(const command_queue &queue)
        : command_queue(queue.data(), true)
    {}
    ~command_queue();

    generic_info get_info(cl_uint param_name) const;

#if 0

    PYOPENCL_USE_RESULT std::unique_ptr<context>
    get_context() const
    {
        cl_context param_value;
        pyopencl_call_guarded(clGetCommandQueueInfo, this, CL_QUEUE_CONTEXT,
                              size_arg(param_value), nullptr);
        return std::unique_ptr<context>(
            new context(param_value, /*retain*/ true));
    }

#if PYOPENCL_CL_VERSION < 0x1010
    cl_command_queue_properties
    set_property(cl_command_queue_properties prop, bool enable) const
    {
        cl_command_queue_properties old_prop;
        pyopencl_call_guarded(clSetCommandQueueProperty, this, prop,
                              enable, buf_arg(old_prop));
        return old_prop;
    }
#endif

#endif
};

extern template void print_clobj<command_queue>(std::ostream&,
                                                const command_queue*);

// }}}

#endif
