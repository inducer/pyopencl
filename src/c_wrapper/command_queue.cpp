#include "command_queue.h"
#include "device.h"
#include "context.h"
#include "event.h"
#include "clhelper.h"

template class clobj<cl_command_queue>;
template void print_arg<cl_command_queue>(std::ostream&,
                                          const cl_command_queue&, bool);
template void print_clobj<command_queue>(std::ostream&, const command_queue*);
template void print_buf<cl_command_queue>(
    std::ostream&, const cl_command_queue*, size_t, ArgType, bool, bool);

command_queue::~command_queue()
{
    pyopencl_call_guarded_cleanup(clReleaseCommandQueue, PYOPENCL_CL_CASTABLE_THIS);
}

generic_info
command_queue::get_info(cl_uint param_name) const
{
    switch ((cl_command_queue_info)param_name) {
    case CL_QUEUE_CONTEXT:
        return pyopencl_get_opaque_info(context, CommandQueue,
                                        PYOPENCL_CL_CASTABLE_THIS, param_name);
    case CL_QUEUE_DEVICE:
        return pyopencl_get_opaque_info(device, CommandQueue, PYOPENCL_CL_CASTABLE_THIS, param_name);
    case CL_QUEUE_REFERENCE_COUNT:
        return pyopencl_get_int_info(cl_uint, CommandQueue,
                                     PYOPENCL_CL_CASTABLE_THIS, param_name);
    case CL_QUEUE_PROPERTIES:
        return pyopencl_get_int_info(cl_command_queue_properties,
                                     CommandQueue, PYOPENCL_CL_CASTABLE_THIS, param_name);
    default:
        throw clerror("CommandQueue.get_info", CL_INVALID_VALUE);
    }
}

// c wrapper

// Command Queue
error*
create_command_queue(clobj_t *queue, clobj_t _ctx,
                     clobj_t _dev, cl_command_queue_properties props)
{
    auto ctx = static_cast<context*>(_ctx);
    auto py_dev = static_cast<device*>(_dev);
    return c_handle_error([&] {
            cl_device_id dev;
            if (py_dev) {
                dev = py_dev->data();
            } else {
                auto devs = pyopencl_get_vec_info(cl_device_id, Context,
                                                  ctx, CL_CONTEXT_DEVICES);
                if (devs.len() == 0) {
                    throw clerror("CommandQueue", CL_INVALID_VALUE,
                                  "context doesn't have any devices? -- "
                                  "don't know which one to default to");
                }
                dev = devs[0];
            }
            cl_command_queue cl_queue =
                pyopencl_call_guarded(clCreateCommandQueue, ctx, dev, props);
            *queue = new command_queue(cl_queue, false);
        });
}

error*
command_queue__finish(clobj_t queue)
{
    return c_handle_error([&] {
            pyopencl_call_guarded(clFinish, static_cast<command_queue*>(queue));
        });
}

error*
command_queue__flush(clobj_t queue)
{
    return c_handle_error([&] {
            pyopencl_call_guarded(clFlush, static_cast<command_queue*>(queue));
        });
}

error*
enqueue_marker_with_wait_list(clobj_t *evt, clobj_t _queue,
                              const clobj_t *_wait_for, uint32_t num_wait_for)
{
#if PYOPENCL_CL_VERSION >= 0x1020
    auto queue = static_cast<command_queue*>(_queue);
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    return c_handle_error([&] {
            pyopencl_call_guarded(clEnqueueMarkerWithWaitList, queue,
                                  wait_for, event_out(evt));
        });
#else
    PYOPENCL_UNSUPPORTED_BEFORE(clEnqueueMarkerWithWaitList, "CL 1.2")
#endif
}

error*
enqueue_barrier_with_wait_list(clobj_t *evt, clobj_t _queue,
                               const clobj_t *_wait_for, uint32_t num_wait_for)
{
#if PYOPENCL_CL_VERSION >= 0x1020
    auto queue = static_cast<command_queue*>(_queue);
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    return c_handle_error([&] {
            pyopencl_call_guarded(clEnqueueBarrierWithWaitList, queue,
                                  wait_for, event_out(evt));
        });
#else
    PYOPENCL_UNSUPPORTED_BEFORE(clEnqueueBarrierWithWaitList, "CL 1.2")
#endif
}

error*
enqueue_marker(clobj_t *evt, clobj_t _queue)
{
    auto queue = static_cast<command_queue*>(_queue);
    return c_handle_error([&] {
            pyopencl_call_guarded(clEnqueueMarker, queue, event_out(evt));
        });
}

error*
enqueue_barrier(clobj_t _queue)
{
    auto queue = static_cast<command_queue*>(_queue);
    return c_handle_error([&] {
            pyopencl_call_guarded(clEnqueueBarrier, queue);
        });
}
