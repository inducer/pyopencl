#include "clhelper.h"

#ifndef __PYOPENCL_DEVICE_H
#define __PYOPENCL_DEVICE_H

// {{{ device

extern template class clobj<cl_device_id>;
extern template void print_arg<cl_device_id>(std::ostream&,
                                             const cl_device_id&, bool);
extern template void print_buf<cl_device_id>(std::ostream&, const cl_device_id*,
                                             size_t, ArgType, bool, bool);

class device : public clobj<cl_device_id> {
public:
    PYOPENCL_DEF_CL_CLASS(DEVICE);
    enum reference_type_t {
        REF_NOT_OWNABLE,
        REF_FISSION_EXT,
        REF_CL_1_2,
    };

private:
    reference_type_t m_ref_type;

public:
    static void get_version(cl_device_id dev, int *major, int *minor);
    device(cl_device_id did, bool retain=false,
           reference_type_t ref_type=REF_NOT_OWNABLE)
        : clobj(did), m_ref_type(ref_type)
    {
        if (retain && ref_type != REF_NOT_OWNABLE) {
            if (false) {
            }
#if (defined(cl_ext_device_fission) && defined(PYOPENCL_USE_DEVICE_FISSION))
            else if (ref_type == REF_FISSION_EXT) {
#if PYOPENCL_CL_VERSION >= 0x1020
                cl_platform_id plat;
                pyopencl_call_guarded(clGetDeviceInfo, data(),
                                      CL_DEVICE_PLATFORM, size_arg(plat),
                                      nullptr);
#endif
                pyopencl_call_guarded(
                    pyopencl_get_ext_fun(plat, clRetainDeviceEXT), data());
            }
#endif
#if PYOPENCL_CL_VERSION >= 0x1020
            else if (ref_type == REF_CL_1_2) {
                pyopencl_call_guarded(clRetainDevice, data());
            }
#endif

            else {
                throw clerror("Device", CL_INVALID_VALUE,
                              "cannot own references to devices when device "
                              "fission or CL 1.2 is not available");
            }
        }
    }

    ~device();

    generic_info get_info(cl_uint param_name) const;
#if PYOPENCL_CL_VERSION >= 0x1020
    PYOPENCL_USE_RESULT pyopencl_buf<clobj_t>
    create_sub_devices(const cl_device_partition_property *props);
#endif

#if defined(cl_ext_device_fission) && defined(PYOPENCL_USE_DEVICE_FISSION)
    PYOPENCL_USE_RESULT pyopencl_buf<clobj_t>
    create_sub_devices_ext(const cl_device_partition_property_ext *props);
#endif
};

extern template void print_clobj<device>(std::ostream&, const device*);

// }}}

#endif
