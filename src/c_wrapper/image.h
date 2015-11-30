#include "memory_object.h"
#include "clhelper.h"

#ifndef __PYOPENCL_IMAGE_H
#define __PYOPENCL_IMAGE_H

// {{{ image

class image : public memory_object {
private:
    cl_image_format m_format;
public:
    PYOPENCL_DEF_CL_CLASS(IMAGE);
    PYOPENCL_INLINE
    image(cl_mem mem, bool retain, const cl_image_format *fmt=0)
        : memory_object(mem, retain), m_format(fmt ? *fmt : cl_image_format())
    {}
    PYOPENCL_INLINE const cl_image_format&
    format()
    {
        if (!m_format.image_channel_data_type) {
            pyopencl_call_guarded(clGetImageInfo, PYOPENCL_CL_CASTABLE_THIS, CL_IMAGE_FORMAT,
                                  size_arg(m_format), nullptr);
        }
        return m_format;
    }
    PYOPENCL_USE_RESULT generic_info get_image_info(cl_image_info param) const;
    PYOPENCL_INLINE type_t
    get_fill_type()
    {
        switch (format().image_channel_data_type) {
        case CL_SIGNED_INT8:
        case CL_SIGNED_INT16:
        case CL_SIGNED_INT32:
            return TYPE_INT;
        case CL_UNSIGNED_INT8:
        case CL_UNSIGNED_INT16:
        case CL_UNSIGNED_INT32:
            return TYPE_UINT;
        default:
            return TYPE_FLOAT;
        }
    }
};

extern template void print_clobj<image>(std::ostream&, const image*);

// }}}

#endif
