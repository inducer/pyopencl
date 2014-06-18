#include "clhelper.h"

#ifndef __PYOPENCL_DEVICE_H
#define __PYOPENCL_DEVICE_H

namespace pyopencl {

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
#if PYOPENCL_CL_VERSION >= 0x1020
        REF_CL_1_2,
#endif
    };

private:
    reference_type_t m_ref_type;

public:
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
                pyopencl_call_guarded(clGetDeviceInfo, this,
                                      CL_DEVICE_PLATFORM, make_sizearg(plat),
                                      nullptr);
#endif
                pyopencl_call_guarded(
                    pyopencl_get_ext_fun(plat, clRetainDeviceEXT), this);
            }
#endif
#if PYOPENCL_CL_VERSION >= 0x1020
            else if (ref_type == REF_CL_1_2) {
                pyopencl_call_guarded(clRetainDevice, this);
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
    // TODO: sub-devices
    // #if PYOPENCL_CL_VERSION >= 0x1020
    //       py::list create_sub_devices(py::object py_properties)
    //       {
    //         std::vector<cl_device_partition_property> properties;

    //         COPY_PY_LIST(cl_device_partition_property, properties);
    //         properties.push_back(0);

    //         cl_device_partition_property *props_ptr
    //           = properties.empty( ) ? nullptr : &properties.front();

    //         cl_uint num_entries;
    //         PYOPENCL_CALL_GUARDED(clCreateSubDevices,
    //             (m_device, props_ptr, 0, nullptr, &num_entries));

    //         std::vector<cl_device_id> result;
    //         result.resize(num_entries);

    //         PYOPENCL_CALL_GUARDED(clCreateSubDevices,
    //             (m_device, props_ptr, num_entries, &result.front(), nullptr));

    //         py::list py_result;
    //         BOOST_FOREACH(cl_device_id did, result)
    //           py_result.append(handle_from_new_ptr(
    //                 new pyopencl::device(did, /*retain*/true,
    //                   device::REF_CL_1_2)));
    //         return py_result;
    //       }
    // #endif

    // #if defined(cl_ext_device_fission) && defined(PYOPENCL_USE_DEVICE_FISSION)
    //       py::list create_sub_devices_ext(py::object py_properties)
    //       {
    //         std::vector<cl_device_partition_property_ext> properties;

    // #if PYOPENCL_CL_VERSION >= 0x1020
    //         cl_platform_id plat;
    //         PYOPENCL_CALL_GUARDED(clGetDeviceInfo, (m_device, CL_DEVICE_PLATFORM,
    //               sizeof(plat), &plat, nullptr));
    // #endif

    //         PYOPENCL_GET_EXT_FUN(plat, clCreateSubDevicesEXT, create_sub_dev);

    //         COPY_PY_LIST(cl_device_partition_property_ext, properties);
    //         properties.push_back(CL_PROPERTIES_LIST_END_EXT);

    //         cl_device_partition_property_ext *props_ptr
    //           = properties.empty( ) ? nullptr : &properties.front();

    //         cl_uint num_entries;
    //         PYOPENCL_CALL_GUARDED(create_sub_dev,
    //             (m_device, props_ptr, 0, nullptr, &num_entries));

    //         std::vector<cl_device_id> result;
    //         result.resize(num_entries);

    //         PYOPENCL_CALL_GUARDED(create_sub_dev,
    //             (m_device, props_ptr, num_entries, &result.front(), nullptr));

    //         py::list py_result;
    //         BOOST_FOREACH(cl_device_id did, result)
    //           py_result.append(handle_from_new_ptr(
    //                 new pyopencl::device(did, /*retain*/true,
    //                   device::REF_FISSION_EXT)));
    //         return py_result;
    //       }
    // #endif
};

extern template void print_clobj<device>(std::ostream&, const device*);

// }}}

}

#endif
