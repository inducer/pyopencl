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
#if PYOPENCL_CL_VERSION >= 0x1020
            else if (ref_type == REF_CL_1_2) {
                pyopencl_call_guarded(clRetainDevice, PYOPENCL_CL_CASTABLE_THIS);
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
};

extern template void print_clobj<device>(std::ostream&, const device*);

// }}}

#endif
