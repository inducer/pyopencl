#include "image.h"

#ifndef __PYOPENCL_GL_OBJ_H
#define __PYOPENCL_GL_OBJ_H

#ifdef HAVE_GL

// {{{ gl interop

class gl_buffer : public memory_object {
public:
    PYOPENCL_DEF_CL_CLASS(GL_BUFFER);
    PYOPENCL_INLINE
    gl_buffer(cl_mem mem, bool retain)
        : memory_object(mem, retain)
    {}
};

class gl_renderbuffer : public memory_object {
public:
    PYOPENCL_DEF_CL_CLASS(GL_RENDERBUFFER);
    PYOPENCL_INLINE
    gl_renderbuffer(cl_mem mem, bool retain)
        : memory_object(mem, retain)
    {}
};

extern template void print_clobj<gl_buffer>(std::ostream&, const gl_buffer*);
extern template void print_clobj<gl_renderbuffer>(std::ostream&,
                                                  const gl_renderbuffer*);

class gl_texture : public image {
  public:
    PYOPENCL_INLINE
    gl_texture(cl_mem mem, bool retain)
      : image(mem, retain)
    {}
    PYOPENCL_USE_RESULT generic_info
    get_gl_texture_info(cl_gl_texture_info param_name) const;
};

// }}}

#endif

#endif
