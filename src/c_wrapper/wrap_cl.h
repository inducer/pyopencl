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
#else
#include <unistd.h>
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

#if defined(CL_VERSION_2_0)
#define PYOPENCL_CL_VERSION 0x2000
#elif defined(CL_VERSION_1_2)
#define PYOPENCL_CL_VERSION 0x1020
#elif defined(CL_VERSION_1_1)
#define PYOPENCL_CL_VERSION 0x1010
#else
#define PYOPENCL_CL_VERSION 0x1000
#endif

#endif

#ifndef CL_VERSION_2_0
typedef void* CLeglImageKHR;
typedef void* CLeglDisplayKHR;
typedef void* CLeglSyncKHR;
typedef intptr_t cl_egl_image_properties_khr;
typedef cl_bitfield         cl_device_svm_capabilities;
typedef cl_bitfield         cl_svm_mem_flags;
typedef intptr_t            cl_pipe_properties;
typedef cl_uint             cl_pipe_info;
typedef cl_bitfield         cl_sampler_properties;
typedef cl_uint             cl_kernel_exec_info;
#endif

#ifndef CL_VERSION_1_2
typedef intptr_t cl_device_partition_property;
typedef cl_uint cl_kernel_arg_info;
typedef struct _cl_image_desc cl_image_desc;
typedef cl_bitfield cl_mem_migration_flags;
#endif

#ifndef cl_ext_migrate_memobject
typedef cl_bitfield cl_mem_migration_flags_ext;
#endif

#ifndef cl_ext_device_fission
typedef cl_ulong cl_device_partition_property_ext;
#endif

struct clbase;
typedef clbase *clobj_t;

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

#if defined __GUNC__ || defined __GNUG__
#define PYOPENCL_USE_RESULT __attribute__((warn_unused_result))
#else
#define PYOPENCL_USE_RESULT
#endif

#endif
