#include "platform.h"
#include "device.h"
#include "clhelper.h"

namespace pyopencl {

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
        return pyopencl_get_str_info(Platform, this, param_name);
    default:
        throw clerror("Platform.get_info", CL_INVALID_VALUE);
    }
}

}

// c wrapper
// Import all the names in pyopencl namespace for c wrappers.
using namespace pyopencl;

error*
get_platforms(clobj_t **_platforms, uint32_t *num_platforms)
{
    return c_handle_error([&] {
            *num_platforms = 0;
            pyopencl_call_guarded(clGetPlatformIDs, 0, nullptr,
                                  make_argbuf(*num_platforms));
            pyopencl_buf<cl_platform_id> platforms(*num_platforms);
            pyopencl_call_guarded(clGetPlatformIDs, platforms,
                                  make_argbuf(*num_platforms));
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
                                      make_argbuf(*num_devices));
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
                                  make_argbuf(*num_devices));
            *_devices = buf_to_base<device>(devices).release();
        });
}
