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

#ifndef CL_DEVICE_P2P_DEVICES_AMD
#define CL_DEVICE_P2P_DEVICES_AMD               0x4089

typedef CL_API_ENTRY cl_int
(CL_API_CALL * clEnqueueCopyBufferP2PAMD_fn)(cl_command_queue /*command_queue*/,
                                             cl_mem /*src_buffer*/,
                                             cl_mem /*dst_buffer*/,
                                             size_t /*src_offset*/,
                                             size_t /*dst_offset*/,
                                             size_t /*cb*/,
                                             cl_uint /*num_events_in_wait_list*/,
                                             const cl_event* /*event_wait_list*/,
                                             cl_event* /*event*/);
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
