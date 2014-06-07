#include "error.h"
#include "clobj.h"

#ifndef __PYOPENCL_CLHELPER_H
#define __PYOPENCL_CLHELPER_H

namespace pyopencl {

// {{{ GetInfo helpers

template<typename T, typename... ArgTypes, typename... ArgTypes2>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE pyopencl_buf<T>
get_vec_info(cl_int (*func)(ArgTypes...), const char *name,
             ArgTypes2&&... args)
{
    size_t size = 0;
    call_guarded(func, name, args..., 0, nullptr, &size);
    pyopencl_buf<T> buf(size / sizeof(T));
    call_guarded(func, name, args..., size, buf.get(), &size);
    return buf;
}
#define pyopencl_get_vec_info(type, what, args...)                      \
    pyopencl::get_vec_info<type>(clGet##what##Info, "clGet" #what "Info", args)

template<typename T>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE generic_info
convert_array_info(const char *tname, pyopencl_buf<T> &buf)
{
    generic_info info;
    info.dontfree = 0;
    info.opaque_class = CLASS_NONE;
    info.type = _copy_str(std::string(tname) + "[" +
                          tostring(buf.len()) + "]");
    info.value = buf.release();
    return info;
}

template<typename T>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE generic_info
convert_array_info(const char *tname, pyopencl_buf<T> &&_buf)
{
    pyopencl_buf<T> &buf = _buf;
    return convert_array_info<T>(tname, buf);
}

#define pyopencl_convert_array_info(type, buf)          \
    pyopencl::convert_array_info<type>(#type, buf)
#define pyopencl_get_array_info(type, what, args...)                    \
    pyopencl_convert_array_info(type, pyopencl_get_vec_info(type, what, args))

template<typename T, typename Cls>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE generic_info
convert_opaque_array_info(pyopencl_buf<T> &buf)
{
    generic_info info;
    info.dontfree = 0;
    info.opaque_class = Cls::get_class_t();
    info.type = _copy_str(std::string("void*[") + tostring(buf.len()) + "]");
    info.value = buf_to_base<Cls>(buf).release();
    return info;
}

template<typename T, typename Cls>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE generic_info
convert_opaque_array_info(pyopencl_buf<T> &&_buf)
{
    pyopencl_buf<T> &buf = _buf;
    return convert_opaque_array_info<T, Cls>(buf);
}
#define pyopencl_get_opaque_array_info(type, cls, what, args...)  \
    pyopencl::convert_opaque_array_info<type, cls>(              \
        pyopencl_get_vec_info(type, what, args))

template<typename CLType, typename Cls,
         typename... ArgTypes, typename... ArgTypes2>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE generic_info
get_opaque_info(cl_int (*func)(ArgTypes...), const char *name,
                ArgTypes2&&... args)
{
    CLType param_value;
    call_guarded(func, name, args..., sizeof(param_value),
                 &param_value, nullptr);
    generic_info info;
    info.dontfree = 0;
    info.opaque_class = Cls::get_class_t();
    info.type = "void *";
    if (param_value) {
        info.value = (void*)(new Cls(param_value, /*retain*/ true));
    } else {
        info.value = nullptr;
    }
    return info;
}
#define pyopencl_get_opaque_info(type, cls, what, args...)              \
    pyopencl::get_opaque_info<type, cls>(clGet##what##Info,             \
                                         "clGet" #what "Info", args)

template<typename... ArgTypes, typename... ArgTypes2>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE generic_info
get_str_info(cl_int (*func)(ArgTypes...), const char *name,
             ArgTypes2&&... args)
{
    size_t param_value_size;
    call_guarded(func, name, args..., 0, nullptr, &param_value_size);
    pyopencl_buf<char> param_value(param_value_size);
    call_guarded(func, name, args..., param_value_size,
                 param_value.get(), &param_value_size);
    generic_info info;
    info.dontfree = 0;
    info.opaque_class = CLASS_NONE;
    info.type = "char*";
    info.value = (void*)param_value.release();
    return info;
}
#define pyopencl_get_str_info(what, args...)                            \
    pyopencl::get_str_info(clGet##what##Info, "clGet" #what "Info", args)

template<typename T, typename... ArgTypes, typename... ArgTypes2>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE generic_info
get_int_info(cl_int (*func)(ArgTypes...), const char *name,
             const char *tpname, ArgTypes2&&... args)
{
    pyopencl_buf<T> param_value;
    call_guarded(func, name, args..., sizeof(T), param_value.get(), nullptr);
    generic_info info;
    info.dontfree = 0;
    info.opaque_class = CLASS_NONE;
    info.type = tpname;
    info.value = (void*)param_value.release();
    return info;
}
#define pyopencl_get_int_info(type, what, args...)                      \
    pyopencl::get_int_info<type>(clGet##what##Info, "clGet" #what "Info", \
                                 #type "*", args)

// }}}

template<typename T, typename CLType, typename... ArgTypes>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE T*
convert_obj(cl_int (*clRelease)(CLType), const char *name, CLType cl_obj,
            ArgTypes&&... args)
{
    try {
        return new T(cl_obj, false, std::forward<ArgTypes>(args)...);
    } catch (...) {
        call_guarded_cleanup(clRelease, name, cl_obj);
        throw;
    }
}
#define pyopencl_convert_obj(type, func, args...)       \
    pyopencl::convert_obj<type>(func, #func, args)

}

#endif
