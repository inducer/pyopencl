#ifndef _WRAP_CL_H
#define _WRAP_CL_H


// CL 1.2 undecided:
// clSetPrintfCallback

// {{{ includes

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#ifdef __APPLE__

// Mac ------------------------------------------------------------------------
#include <OpenCL/opencl.h>
#ifdef HAVE_GL

#define PYOPENCL_GL_SHARING_VERSION 1

#include <OpenGL/OpenGL.h>
#include <OpenCL/cl_gl.h>
#include <OpenCL/cl_gl_ext.h>
#endif

#else

// elsewhere ------------------------------------------------------------------
#include <CL/cl.h>
#include <CL/cl_ext.h>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

#ifdef HAVE_GL
#include <GL/gl.h>
#include <CL/cl_gl.h>
#endif

#if defined(cl_khr_gl_sharing) && (cl_khr_gl_sharing >= 1)
#define PYOPENCL_GL_SHARING_VERSION cl_khr_gl_sharing
#endif

#endif


#ifdef PYOPENCL_PRETEND_CL_VERSION
#define PYOPENCL_CL_VERSION PYOPENCL_PRETEND_CL_VERSION
#else

#if defined(CL_VERSION_1_2)
#define PYOPENCL_CL_VERSION 0x1020
#elif defined(CL_VERSION_1_1)
#define PYOPENCL_CL_VERSION 0x1010
#else
#define PYOPENCL_CL_VERSION 0x1000
#endif

#endif

namespace pyopencl {
struct clbase;
}
typedef pyopencl::clbase *clobj_t;

#ifdef __cplusplus
extern "C" {
#endif

#include "wrap_cl_core.h"

#ifdef HAVE_GL
#include "wrap_cl_gl_core.h"
#endif

#ifdef __cplusplus
}
#endif

#endif
