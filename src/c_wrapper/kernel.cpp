#include "kernel.h"
#include "context.h"
#include "device.h"
#include "program.h"
#include "memory_object.h"
#include "sampler.h"
#include "command_queue.h"
#include "event.h"
#include "clhelper.h"

template class clobj<cl_kernel>;
template void print_arg<cl_kernel>(std::ostream&, const cl_kernel&, bool);
template void print_clobj<kernel>(std::ostream&, const kernel*);
template void print_buf<cl_kernel>(std::ostream&, const cl_kernel*,
                                   size_t, ArgType, bool, bool);

kernel::~kernel()
{
    pyopencl_call_guarded_cleanup(clReleaseKernel, PYOPENCL_CL_CASTABLE_THIS);
}

generic_info
kernel::get_info(cl_uint param) const
{
    switch ((cl_kernel_info)param) {
    case CL_KERNEL_FUNCTION_NAME:
        return pyopencl_get_str_info(Kernel, PYOPENCL_CL_CASTABLE_THIS, param);
    case CL_KERNEL_NUM_ARGS:
    case CL_KERNEL_REFERENCE_COUNT:
        return pyopencl_get_int_info(cl_uint, Kernel, PYOPENCL_CL_CASTABLE_THIS, param);
    case CL_KERNEL_CONTEXT:
        return pyopencl_get_opaque_info(context, Kernel, PYOPENCL_CL_CASTABLE_THIS, param);
    case CL_KERNEL_PROGRAM:
        return pyopencl_get_opaque_info(program, Kernel, PYOPENCL_CL_CASTABLE_THIS, param);
#if PYOPENCL_CL_VERSION >= 0x1020
    case CL_KERNEL_ATTRIBUTES:
        return pyopencl_get_str_info(Kernel, PYOPENCL_CL_CASTABLE_THIS, param);
#endif
    default:
        throw clerror("Kernel.get_info", CL_INVALID_VALUE);
    }
}

generic_info
kernel::get_work_group_info(cl_kernel_work_group_info param,
                            const device *dev) const
{
    switch (param) {
#if PYOPENCL_CL_VERSION >= 0x1010
    case CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE:
#endif
    case CL_KERNEL_WORK_GROUP_SIZE:
        return pyopencl_get_int_info(size_t, KernelWorkGroup, PYOPENCL_CL_CASTABLE_THIS, dev, param);
    case CL_KERNEL_COMPILE_WORK_GROUP_SIZE:
        return pyopencl_get_array_info(size_t, KernelWorkGroup,
                                       PYOPENCL_CL_CASTABLE_THIS, dev, param);
    case CL_KERNEL_LOCAL_MEM_SIZE:
#if PYOPENCL_CL_VERSION >= 0x1010
    case CL_KERNEL_PRIVATE_MEM_SIZE:
#endif
        return pyopencl_get_int_info(cl_ulong, KernelWorkGroup,
                                     PYOPENCL_CL_CASTABLE_THIS, dev, param);
    default:
        throw clerror("Kernel.get_work_group_info", CL_INVALID_VALUE);
    }
}

#if PYOPENCL_CL_VERSION >= 0x1020
PYOPENCL_USE_RESULT generic_info
kernel::get_arg_info(cl_uint idx, cl_kernel_arg_info param) const
{
    switch (param) {
    case CL_KERNEL_ARG_ADDRESS_QUALIFIER:
        return pyopencl_get_int_info(cl_kernel_arg_address_qualifier,
                                     KernelArg, PYOPENCL_CL_CASTABLE_THIS, idx, param);
    case CL_KERNEL_ARG_ACCESS_QUALIFIER:
        return pyopencl_get_int_info(cl_kernel_arg_access_qualifier,
                                     KernelArg, PYOPENCL_CL_CASTABLE_THIS, idx, param);
    case CL_KERNEL_ARG_TYPE_QUALIFIER:
        return pyopencl_get_int_info(cl_kernel_arg_type_qualifier,
                                     KernelArg, PYOPENCL_CL_CASTABLE_THIS, idx, param);
    case CL_KERNEL_ARG_TYPE_NAME:
    case CL_KERNEL_ARG_NAME:
        return pyopencl_get_str_info(KernelArg, PYOPENCL_CL_CASTABLE_THIS, idx, param);
    default:
        throw clerror("Kernel.get_arg_info", CL_INVALID_VALUE);
    }
}
#endif

// c wrapper

// Kernel
error*
create_kernel(clobj_t *knl, clobj_t _prog, const char *name)
{
    auto prog = static_cast<const program*>(_prog);
    return c_handle_error([&] {
            *knl = new kernel(pyopencl_call_guarded(clCreateKernel, prog,
                                                    name), false);
        });
}

error*
kernel__set_arg_null(clobj_t _knl, cl_uint arg_index)
{
    auto knl = static_cast<kernel*>(_knl);
    return c_handle_error([&] {
            const cl_mem m = 0;
            pyopencl_call_guarded(clSetKernelArg, knl, arg_index, size_arg(m));
        });
}

error*
kernel__set_arg_mem(clobj_t _knl, cl_uint arg_index, clobj_t _mem)
{
    auto knl = static_cast<kernel*>(_knl);
    auto mem = static_cast<memory_object*>(_mem);
    return c_handle_error([&] {
            pyopencl_call_guarded(clSetKernelArg, knl, arg_index,
                                  size_arg(mem->data()));
        });
}

error*
kernel__set_arg_sampler(clobj_t _knl, cl_uint arg_index, clobj_t _samp)
{
    auto knl = static_cast<kernel*>(_knl);
    auto samp = static_cast<sampler*>(_samp);
    return c_handle_error([&] {
            pyopencl_call_guarded(clSetKernelArg, knl, arg_index,
                                  size_arg(samp->data()));
        });
}

error*
kernel__set_arg_buf(clobj_t _knl, cl_uint arg_index,
                    const void *buffer, size_t size)
{
    auto knl = static_cast<kernel*>(_knl);
    return c_handle_error([&] {
            pyopencl_call_guarded(clSetKernelArg, knl, arg_index,
                                  size_arg(buffer, size));
        });
}

error*
kernel__set_arg_svm_pointer(clobj_t _knl, cl_uint arg_index, void *value)
{
#if PYOPENCL_CL_VERSION >= 0x2000
    auto knl = static_cast<kernel*>(_knl);
    return c_handle_error([&] {
            pyopencl_call_guarded(clSetKernelArgSVMPointer, knl, arg_index, value);
        });
#else
    PYOPENCL_UNSUPPORTED_BEFORE(clSetKernelArgSVMPointer, "CL 2.0")
#endif
}

error*
kernel__get_work_group_info(clobj_t _knl, cl_kernel_work_group_info param,
                            clobj_t _dev, generic_info *out)
{
    auto knl = static_cast<kernel*>(_knl);
    auto dev = static_cast<device*>(_dev);
    return c_handle_error([&] {
            *out = knl->get_work_group_info(param, dev);
        });
}

error*
kernel__get_arg_info(clobj_t _knl, cl_uint idx, cl_kernel_arg_info param,
                     generic_info *out)
{
#if PYOPENCL_CL_VERSION >= 0x1020
    auto knl = static_cast<kernel*>(_knl);
    return c_handle_error([&] {
            *out = knl->get_arg_info(idx, param);
        });
#else
    PYOPENCL_UNSUPPORTED(clKernelGetArgInfo, "CL 1.1 and below")
#endif
}

error*
enqueue_nd_range_kernel(clobj_t *evt, clobj_t _queue, clobj_t _knl,
                        cl_uint work_dim, const size_t *global_work_offset,
                        const size_t *global_work_size,
                        const size_t *local_work_size,
                        const clobj_t *_wait_for, uint32_t num_wait_for)
{
    auto queue = static_cast<command_queue*>(_queue);
    auto knl = static_cast<kernel*>(_knl);
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    return c_handle_retry_mem_error([&] {
            pyopencl_call_guarded(clEnqueueNDRangeKernel, queue, knl, work_dim,
                                  global_work_offset, global_work_size,
                                  local_work_size, wait_for, event_out(evt));
        });
}

error*
enqueue_task(clobj_t *evt, clobj_t _queue, clobj_t _knl,
             const clobj_t *_wait_for, uint32_t num_wait_for)
{
    auto queue = static_cast<command_queue*>(_queue);
    auto knl = static_cast<kernel*>(_knl);
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    return c_handle_retry_mem_error([&] {
            pyopencl_call_guarded(clEnqueueTask, queue, knl, wait_for,
                                  event_out(evt));
        });
}
