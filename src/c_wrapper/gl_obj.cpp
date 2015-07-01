#include "gl_obj.h"
#include "context.h"
#include "command_queue.h"
#include "event.h"
#include "clhelper.h"

#ifdef HAVE_GL

template void print_clobj<gl_buffer>(std::ostream&, const gl_buffer*);
template void print_clobj<gl_renderbuffer>(std::ostream&,
                                           const gl_renderbuffer*);

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

// c wrapper

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

error*
get_gl_object_info(clobj_t mem, cl_gl_object_type *otype, GLuint *gl_name)
{
    auto globj = static_cast<memory_object*>(mem);
    return c_handle_error([&] {
            pyopencl_call_guarded(clGetGLObjectInfo, globj, buf_arg(*otype),
                                  buf_arg(*gl_name));
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

cl_context_properties
get_apple_cgl_share_group()
{
#ifdef __APPLE__
    #ifdef HAVE_GL
        CGLContextObj kCGLContext = CGLGetCurrentContext();
        CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);

        return (cl_context_properties)kCGLShareGroup;
    #else
        throw clerror("get_apple_cgl_share_group unavailable: "
            "GL interop not compiled",
            CL_INVALID_VALUE);
    #endif
#else
    throw clerror("get_apple_cgl_share_group unavailable: non-Apple platform",
        CL_INVALID_VALUE);
#endif /* __APPLE__ */
}
