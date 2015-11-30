#include "context.h"
#include "device.h"
#include "platform.h"
#include "clhelper.h"

template class clobj<cl_context>;
template void print_arg<cl_context>(std::ostream&, const cl_context&, bool);
template void print_clobj<context>(std::ostream&, const context*);
template void print_buf<cl_context>(std::ostream&, const cl_context*,
                                    size_t, ArgType, bool, bool);

void
context::get_version(cl_context ctx, int *major, int *minor)
{
    cl_device_id s_buff[16];
    size_t size;
    pyopencl_buf<cl_device_id> d_buff(0);
    cl_device_id *devs = s_buff;
    pyopencl_call_guarded(clGetContextInfo, ctx, CL_CONTEXT_DEVICES,
                          0, nullptr, buf_arg(size));
    if (PYOPENCL_UNLIKELY(!size)) {
        throw clerror("Context.get_version", CL_INVALID_VALUE,
                      "Cannot get devices from context.");
    }
    if (PYOPENCL_UNLIKELY(size > sizeof(s_buff))) {
        d_buff.resize(size / sizeof(cl_device_id));
        devs = d_buff.get();
    }
    pyopencl_call_guarded(clGetContextInfo, ctx, CL_CONTEXT_DEVICES,
                          size_arg(devs, size), buf_arg(size));
    device::get_version(devs[0], major, minor);
}

context::~context()
{
    pyopencl_call_guarded_cleanup(clReleaseContext, PYOPENCL_CL_CASTABLE_THIS);
}

generic_info
context::get_info(cl_uint param_name) const
{
    switch ((cl_context_info)param_name) {
    case CL_CONTEXT_REFERENCE_COUNT:
        return pyopencl_get_int_info(cl_uint, Context,
                                     PYOPENCL_CL_CASTABLE_THIS, param_name);
    case CL_CONTEXT_DEVICES:
        return pyopencl_get_opaque_array_info(device, Context,
                                              PYOPENCL_CL_CASTABLE_THIS, param_name);
    case CL_CONTEXT_PROPERTIES: {
        auto result = pyopencl_get_vec_info(
            cl_context_properties, Context, PYOPENCL_CL_CASTABLE_THIS, param_name);
        pyopencl_buf<generic_info> py_result(result.len() / 2);
        size_t i = 0;
        for (;i < py_result.len();i++) {
            cl_context_properties key = result[i * 2];
            if (key == 0)
                break;
            cl_context_properties value = result[i * 2 + 1];
            generic_info &info = py_result[i];
            info.dontfree = 0;
            info.opaque_class = CLASS_NONE;
            switch (key) {
            case CL_CONTEXT_PLATFORM:
                info.opaque_class = CLASS_PLATFORM;
                info.type = "void *";
                info.value = new platform(
                    reinterpret_cast<cl_platform_id>(value));
                break;

#if defined(PYOPENCL_GL_SHARING_VERSION) && (PYOPENCL_GL_SHARING_VERSION >= 1)
#if defined(__APPLE__) && defined(HAVE_GL)
            case CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE:
#else
            case CL_GL_CONTEXT_KHR:
            case CL_EGL_DISPLAY_KHR:
            case CL_GLX_DISPLAY_KHR:
            case CL_WGL_HDC_KHR:
            case CL_CGL_SHAREGROUP_KHR:
#endif
                info.type = "intptr_t *";
                info.value = (void*)value;
                // we do not own this object
                info.dontfree = 1;
                break;

#endif
            default:
                throw clerror("Context.get_info", CL_INVALID_VALUE,
                              "unknown context_property key encountered");
            }
        }
        py_result.resize(i);
        return pyopencl_convert_array_info(generic_info, py_result);
    }

#if PYOPENCL_CL_VERSION >= 0x1010
    case CL_CONTEXT_NUM_DEVICES:
        return pyopencl_get_int_info(cl_uint, Context,
                                     PYOPENCL_CL_CASTABLE_THIS, param_name);
#endif

    default:
        throw clerror("Context.get_info", CL_INVALID_VALUE);
    }
}

// c wrapper

// Context
error*
create_context(clobj_t *_ctx, const cl_context_properties *props,
               cl_uint num_devices, const clobj_t *_devices)
{
    // TODO debug print properties
    return c_handle_error([&] {
            const auto devices = buf_from_class<device>(_devices, num_devices);
            *_ctx = new context(
                pyopencl_call_guarded(
                    clCreateContext,
                    const_cast<cl_context_properties*>(props),
                    devices, nullptr, nullptr), false);
        });
}

// Context
error*
create_context_from_type(clobj_t *_ctx, const cl_context_properties *props,
                         cl_device_type dev_type)
{
    // TODO debug print properties
    return c_handle_error([&] {
            *_ctx = new context(
                pyopencl_call_guarded(
                    clCreateContextFromType,
                    const_cast<cl_context_properties*>(props),
                    dev_type, nullptr, nullptr), false);
        });
}

error*
context__get_supported_image_formats(clobj_t _ctx, cl_mem_flags flags,
                                     cl_mem_object_type image_type,
                                     generic_info *out)
{
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_error([&] {
            cl_uint num;
            pyopencl_call_guarded(clGetSupportedImageFormats, ctx, flags,
                                  image_type, 0, nullptr, buf_arg(num));
            pyopencl_buf<cl_image_format> formats(num);
            pyopencl_call_guarded(clGetSupportedImageFormats, ctx, flags,
                                  image_type, formats, buf_arg(num));
            *out = pyopencl_convert_array_info(cl_image_format, formats);
        });
}
