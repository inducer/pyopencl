#include "pyhelper.h"
#include "clhelper.h"
#include "platform.h"
#include "device.h"
#include "context.h"
#include "command_queue.h"
#include "event.h"
#include "memory_object.h"
#include "image.h"
#include "gl_obj.h"
#include "memory_map.h"
#include "buffer.h"
#include "sampler.h"
#include "program.h"
#include "kernel.h"

template void print_buf<char>(std::ostream&, const char*, size_t,
                              ArgType, bool, bool);
template void print_buf<cl_int>(std::ostream&, const cl_int*, size_t,
                                ArgType, bool, bool);
template void print_buf<cl_uint>(std::ostream&, const cl_uint*, size_t,
                                 ArgType, bool, bool);
template void print_buf<cl_long>(std::ostream&, const cl_long*, size_t,
                                 ArgType, bool, bool);
template void print_buf<cl_ulong>(std::ostream&, const cl_ulong*, size_t,
                                  ArgType, bool, bool);
template void print_buf<cl_image_format>(std::ostream&,
                                         const cl_image_format*, size_t,
                                         ArgType, bool, bool);

// {{{ c wrapper

// Generic functions
int
get_cl_version()
{
    return PYOPENCL_CL_VERSION;
}

void
free_pointer(void *p)
{
    free(p);
}

void
free_pointer_array(void **p, uint32_t size)
{
    for (uint32_t i = 0;i < size;i++) {
        free(p[i]);
    }
}


intptr_t
clobj__int_ptr(clobj_t obj)
{
    return PYOPENCL_LIKELY(obj) ? obj->intptr() : 0l;
}

static PYOPENCL_INLINE clobj_t
_from_int_ptr(intptr_t ptr, class_t class_, bool retain)
{
    switch(class_) {
    case CLASS_PLATFORM:
        return clobj_from_int_ptr<platform>(ptr, retain);
    case CLASS_DEVICE:
        return clobj_from_int_ptr<device>(ptr, retain);
    case CLASS_KERNEL:
        return clobj_from_int_ptr<kernel>(ptr, retain);
    case CLASS_CONTEXT:
        return clobj_from_int_ptr<context>(ptr, retain);
    case CLASS_COMMAND_QUEUE:
        return clobj_from_int_ptr<command_queue>(ptr, retain);
    case CLASS_BUFFER:
        return clobj_from_int_ptr<buffer>(ptr, retain);
    case CLASS_PROGRAM:
        return clobj_from_int_ptr<program>(ptr, retain);
    case CLASS_EVENT:
        return clobj_from_int_ptr<event>(ptr, retain);
    case CLASS_IMAGE:
        return clobj_from_int_ptr<image>(ptr, retain);
    case CLASS_SAMPLER:
        return clobj_from_int_ptr<sampler>(ptr, retain);
#ifdef HAVE_GL
    case CLASS_GL_BUFFER:
        return clobj_from_int_ptr<gl_buffer>(ptr, retain);
    case CLASS_GL_RENDERBUFFER:
        return clobj_from_int_ptr<gl_renderbuffer>(ptr, retain);
#endif
    default:
        throw clerror("unknown class", CL_INVALID_VALUE);
  }
}

error*
clobj__from_int_ptr(clobj_t *out, intptr_t ptr, class_t class_, int retain)
{
    return c_handle_error([&] {
            *out = _from_int_ptr(ptr, class_, retain);
        });
}

error*
clobj__get_info(clobj_t obj, cl_uint param, generic_info *out)
{
    return c_handle_error([&] {
            if (PYOPENCL_UNLIKELY(!obj)) {
                throw clerror("NULL input", CL_INVALID_VALUE);
            }
            *out = obj->get_info(param);
        });
}

void
clobj__delete(clobj_t obj)
{
    delete obj;
}

// }}}

// vim: foldmethod=marker
