#include "platform.h"
#include "device.h"
#include "clhelper.h"

#include <stdlib.h>

template class clobj<cl_platform_id>;
template void print_arg<cl_platform_id>(std::ostream&,
                                        const cl_platform_id&, bool);
template void print_clobj<platform>(std::ostream&, const platform*);
template void print_buf<cl_platform_id>(std::ostream&, const cl_platform_id*,
                                        size_t, ArgType, bool, bool);

generic_info
platform::get_info(cl_uint param_name) const
{
    switch ((cl_platform_info)param_name) {
    case CL_PLATFORM_PROFILE:
    case CL_PLATFORM_VERSION:
    case CL_PLATFORM_NAME:
    case CL_PLATFORM_VENDOR:
#if !(defined(CL_PLATFORM_NVIDIA) && CL_PLATFORM_NVIDIA == 0x3001)
    case CL_PLATFORM_EXTENSIONS:
#endif
        return pyopencl_get_str_info(Platform, PYOPENCL_CL_CASTABLE_THIS, param_name);
    default:
        throw clerror("Platform.get_info", CL_INVALID_VALUE);
    }
}

void
platform::get_version(cl_platform_id plat, int *major, int *minor)
{
    char s_buff[128];
    size_t size;
    pyopencl_buf<char> d_buff(0);
    char *name = s_buff;
    pyopencl_call_guarded(clGetPlatformInfo, plat, CL_PLATFORM_VERSION,
                          0, nullptr, buf_arg(size));
    if (PYOPENCL_UNLIKELY(size > sizeof(s_buff))) {
        d_buff.resize(size);
        name = d_buff.get();
    }
    pyopencl_call_guarded(clGetPlatformInfo, plat, CL_PLATFORM_VERSION,
                          size_arg(name, size), buf_arg(size));
    *major = *minor = -1;
    sscanf(name, "OpenCL %d.%d", major, minor);
    // Well, hopefully there won't be a negative OpenCL version =)
    if (*major < 0 || *minor < 0) {
        throw clerror("Platform.get_version", CL_INVALID_VALUE,
                      "platform returned non-conformant "
                      "platform version string");
    }
}

// c wrapper

error*
get_platforms(clobj_t **_platforms, uint32_t *num_platforms)
{
    return c_handle_error([&] {
            *num_platforms = 0;
            pyopencl_call_guarded(clGetPlatformIDs, 0, nullptr,
                                  buf_arg(*num_platforms));
            pyopencl_buf<cl_platform_id> platforms(*num_platforms);
            pyopencl_call_guarded(clGetPlatformIDs, platforms,
                                  buf_arg(*num_platforms));
            *_platforms = buf_to_base<platform>(platforms).release();
        });
}

error*
platform__get_devices(clobj_t _plat, clobj_t **_devices,
                      uint32_t *num_devices, cl_device_type devtype)
{
    auto plat = static_cast<platform*>(_plat);
    return c_handle_error([&] {
            *num_devices = 0;
            try {
                pyopencl_call_guarded(clGetDeviceIDs, plat, devtype, 0, nullptr,
                                      buf_arg(*num_devices));
            } catch (const clerror &e) {
                if (e.code() != CL_DEVICE_NOT_FOUND)
                    throw e;
                *num_devices = 0;
            }
            if (*num_devices == 0) {
                *_devices = nullptr;
                return;
            }
            pyopencl_buf<cl_device_id> devices(*num_devices);
            pyopencl_call_guarded(clGetDeviceIDs, plat, devtype, devices,
                                  buf_arg(*num_devices));
            *_devices = buf_to_base<device>(devices).release();
        });
}

error*
platform__unload_compiler(clobj_t plat)
{
#if PYOPENCL_CL_VERSION >= 0x1020
    return c_handle_error([&] {
            pyopencl_call_guarded(clUnloadPlatformCompiler,
                                  static_cast<platform*>(plat));
        });
#else
    PYOPENCL_UNSUPPORTED(clUnloadPlatformCompiler, "CL 1.1 and below")
#endif
}
