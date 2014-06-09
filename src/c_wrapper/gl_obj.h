#include "image.h"

#ifndef __PYOPENCL_GL_OBJ_H
#define __PYOPENCL_GL_OBJ_H

#ifdef HAVE_GL

namespace pyopencl {

// {{{ gl interop

#ifdef __APPLE__
static PYOPENCL_INLINE cl_context_properties
get_apple_cgl_share_group()
{
    CGLContextObj kCGLContext = CGLGetCurrentContext();
    CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);

    return (cl_context_properties)kCGLShareGroup;
}
#endif /* __APPLE__ */

class gl_buffer : public memory_object {
public:
    PYOPENCL_DEF_CL_CLASS(GL_BUFFER);
    PYOPENCL_INLINE
    gl_buffer(cl_mem mem, bool retain, void *hostbuf=0)
        : memory_object(mem, retain, hostbuf)
    {}
};

class gl_renderbuffer : public memory_object {
public:
    PYOPENCL_DEF_CL_CLASS(GL_RENDERBUFFER);
    PYOPENCL_INLINE
    gl_renderbuffer(cl_mem mem, bool retain, void *hostbuf=0)
        : memory_object(mem, retain, hostbuf)
    {}
};

class gl_texture : public image {
  public:
    PYOPENCL_INLINE
    gl_texture(cl_mem mem, bool retain, void *hostbuf=0)
      : image(mem, retain, hostbuf)
    {}
    PYOPENCL_USE_RESULT generic_info
    get_gl_texture_info(cl_gl_texture_info param_name) const;
};

// }}}

#endif

}

#endif
