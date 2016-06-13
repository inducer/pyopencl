#ifndef _PYOPENCL_EXT_H
#define _PYOPENCL_EXT_H

#ifdef PYOPENCL_USE_SHIPPED_EXT

#include "clinfo_ext.h"

#else

#ifdef __APPLE__

#include <OpenCL/opencl.h>

#else

#include <CL/cl.h>
#include <CL/cl_ext.h>

#ifndef CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD
#define CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD        1

typedef union
{
    struct { cl_uint type; cl_uint data[5]; } raw;
    struct { cl_uint type; cl_char unused[17]; cl_char bus; cl_char device; cl_char function; } pcie;
} cl_device_topology_amd;
#endif

#endif

#endif

#endif

