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

#endif

#endif

#endif

