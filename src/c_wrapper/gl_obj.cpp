#include "gl_obj.h"
#include "context.h"
#include "command_queue.h"
#include "event.h"
#include "clhelper.h"

#ifdef HAVE_GL

namespace pyopencl {

generic_info
gl_texture::get_gl_texture_info(cl_gl_texture_info param_name) const
{
    switch (param_name) {
    case CL_GL_TEXTURE_TARGET:
        return pyopencl_get_int_info(GLenum, GLTexture, this, param_name);
    case CL_GL_MIPMAP_LEVEL:
        return pyopencl_get_int_info(GLint, GLTexture, this, param_name);
    default:
        throw clerror("MemoryObject.get_gl_texture_info", CL_INVALID_VALUE);
    }
}

#if 0
PYOPENCL_USE_RESULT static gl_texture*
create_from_gl_texture(const context *ctx, cl_mem_flags flags,
                       GLenum texture_target, GLint miplevel,
                       GLuint texture, unsigned dims)
{
    if (dims == 2) {
        cl_mem mem = pyopencl_call_guarded(clCreateFromGLTexture2D,
                                           ctx, flags, texture_target,
                                           miplevel, texture);
        return pyopencl_convert_obj(gl_texture, clReleaseMemObject, mem);
    } else if (dims == 3) {
        cl_mem mem = pyopencl_call_guarded(clCreateFromGLTexture3D,
                                           ctx, flags, texture_target,
                                           miplevel, texture);
        return pyopencl_convert_obj(gl_texture, clReleaseMemObject, mem);
    } else {
        throw clerror("Image", CL_INVALID_VALUE, "invalid dimension");
    }
}
#endif

// TODO:
// PYOPENCL_INLINE
// py::tuple get_gl_object_info(memory_object_holder const &mem)
// {
//   cl_gl_object_type otype;
//   GLuint gl_name;
//   PYOPENCL_CALL_GUARDED(clGetGLObjectInfo, (mem, &otype, &gl_name));
//   return py::make_tuple(otype, gl_name);
// }

typedef cl_int (*clEnqueueGLObjectFunc)(cl_command_queue, cl_uint,
                                        const cl_mem*, cl_uint,
                                        const cl_event*, cl_event*);

static PYOPENCL_INLINE void
enqueue_gl_objects(clEnqueueGLObjectFunc func, const char *name,
                   clobj_t *evt, command_queue *cq, const clobj_t *mem_objects,
                   uint32_t num_mem_objects, const clobj_t *wait_for,
                   uint32_t num_wait_for)
{
    const auto _wait_for = buf_from_class<event>(wait_for, num_wait_for);
    const auto _mem_objs = buf_from_class<memory_object>(
        mem_objects, num_mem_objects);
    call_guarded(func, name, cq, _mem_objs, _wait_for, event_out(evt));
}
#define enqueue_gl_objects(what, args...)                       \
    enqueue_gl_objects(clEnqueue##what##GLObjects,              \
                       "clEnqueue" #what "GLObjects", args)

// #if defined(cl_khr_gl_sharing) && (cl_khr_gl_sharing >= 1)
//   PYOPENCL_INLINE
//   py::object get_gl_context_info_khr(
//       py::object py_properties,
//       cl_gl_context_info param_name,
//       py::object py_platform
//       )
//   {
//     std::vector<cl_context_properties> props
//       = parse_context_properties(py_properties);

//     typedef CL_API_ENTRY cl_int (CL_API_CALL
//       *func_ptr_type)(const cl_context_properties * /* properties */,
//           cl_gl_context_info            /* param_name */,
//           size_t                        /* param_value_size */,
//           void *                        /* param_value */,
//           size_t *                      /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;

//     func_ptr_type func_ptr;

// #if PYOPENCL_CL_VERSION >= 0x1020
//     if (py_platform.ptr() != Py_None)
//     {
//       platform &plat = py::extract<platform &>(py_platform);

//       func_ptr = (func_ptr_type) clGetExtensionFunctionAddressForPlatform(
//             plat.data(), "clGetGLContextInfoKHR");
//     }
//     else
//     {
//       PYOPENCL_DEPRECATED("get_gl_context_info_khr with platform=None", "2013.1", );

//       func_ptr = (func_ptr_type) clGetExtensionFunctionAddress(
//             "clGetGLContextInfoKHR");
//     }
// #else
//     func_ptr = (func_ptr_type) clGetExtensionFunctionAddress(
//           "clGetGLContextInfoKHR");
// #endif


//     if (!func_ptr)
//       throw error("Context.get_info", CL_INVALID_PLATFORM,
//           "clGetGLContextInfoKHR extension function not present");

//     cl_context_properties *props_ptr
//       = props.empty( ) ? nullptr : &props.front();

//     switch (param_name)
//     {
//       case CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR:
//         {
//           cl_device_id param_value;
//           PYOPENCL_CALL_GUARDED(func_ptr,
//               (props_ptr, param_name, sizeof(param_value), &param_value, 0));
//           return py::object(handle_from_new_ptr( new device(param_value, /*retain*/ true)));
//         }

//       case CL_DEVICES_FOR_GL_CONTEXT_KHR:
//         {
//           size_t size;
//           PYOPENCL_CALL_GUARDED(func_ptr,
//               (props_ptr, param_name, 0, 0, &size));

//           std::vector<cl_device_id> devices;

//           devices.resize(size / sizeof(devices.front()));

//           PYOPENCL_CALL_GUARDED(func_ptr,
//               (props_ptr, param_name, size,
//                devices.empty( ) ? nullptr : &devices.front(), &size));

//           py::list result;
//           BOOST_FOREACH(cl_device_id did, devices)
//             result.append(handle_from_new_ptr(
//                   new device(did)));

//           return result;
//         }

//       default:
//         throw error("get_gl_context_info_khr", CL_INVALID_VALUE);
//     }
//   }

// #endif

}

// c wrapper
// Import all the names in pyopencl namespace for c wrappers.
using namespace pyopencl;

error*
create_from_gl_buffer(clobj_t *ptr, clobj_t _ctx,
                      cl_mem_flags flags, GLuint bufobj)
{
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_error([&] {
            cl_mem mem = pyopencl_call_guarded(clCreateFromGLBuffer,
                                               ctx, flags, bufobj);
            *ptr = pyopencl_convert_obj(gl_buffer, clReleaseMemObject, mem);
        });
}

error*
create_from_gl_renderbuffer(clobj_t *ptr, clobj_t _ctx,
                            cl_mem_flags flags, GLuint bufobj)
{
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_error([&] {
            cl_mem mem = pyopencl_call_guarded(clCreateFromGLRenderbuffer,
                                               ctx, flags, bufobj);
            *ptr = pyopencl_convert_obj(gl_renderbuffer,
                                        clReleaseMemObject, mem);
        });
}

error*
enqueue_acquire_gl_objects(clobj_t *evt, clobj_t queue,
                           const clobj_t *mem_objects,
                           uint32_t num_mem_objects,
                           const clobj_t *wait_for, uint32_t num_wait_for)
{
    return c_handle_error([&] {
            enqueue_gl_objects(
                Acquire, evt, static_cast<command_queue*>(queue),
                mem_objects, num_mem_objects, wait_for, num_wait_for);
        });
}

error*
enqueue_release_gl_objects(clobj_t *evt, clobj_t queue,
                           const clobj_t *mem_objects,
                           uint32_t num_mem_objects,
                           const clobj_t *wait_for, uint32_t num_wait_for)
{
    return c_handle_error([&] {
            enqueue_gl_objects(
                Release, evt, static_cast<command_queue*>(queue),
                mem_objects, num_mem_objects, wait_for, num_wait_for);
        });
}

#endif

int
have_gl()
{
#ifdef HAVE_GL
    return 1;
#else
    return 0;
#endif
}
