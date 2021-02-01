#ifndef _PYOPENCL_EXT_H
#define _PYOPENCL_EXT_H

#ifdef PYOPENCL_USE_SHIPPED_EXT

#include "clinfo_ext.h"

#else

#if (defined(__APPLE__) && !defined(PYOPENCL_APPLE_USE_CL_H))

#include <OpenCL/opencl.h>

#else

#include <CL/cl.h>
#include <CL/cl_ext.h>

#endif

#ifndef CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD
#define CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD        1

typedef union
{
    struct { cl_uint type; cl_uint data[5]; } raw;
    struct { cl_uint type; cl_char unused[17]; cl_char bus; cl_char device; cl_char function; } pcie;
} cl_device_topology_amd;
#endif

/* {{{ these NV defines are often missing from the system headers */

#ifndef CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV
#define CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV                0x4005
#endif
#ifndef CL_DEVICE_INTEGRATED_MEMORY_NV
#define CL_DEVICE_INTEGRATED_MEMORY_NV                  0x4006
#endif

#ifndef CL_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT_NV
#define CL_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT_NV       0x4007
#endif

#ifndef CL_DEVICE_PCI_BUS_ID_NV
#define CL_DEVICE_PCI_BUS_ID_NV                         0x4008
#endif

#ifndef CL_DEVICE_PCI_SLOT_ID_NV
#define CL_DEVICE_PCI_SLOT_ID_NV                        0x4009
#endif

#ifndef CL_DEVICE_PCI_DOMAIN_ID_NV
#define CL_DEVICE_PCI_DOMAIN_ID_NV                      0x400A
#endif

/* }}} */

#endif

#endif

/* vim: foldmethod=marker */
