#include "wrap_cl.h"
#include "error.h"
#include <string>
#include <sstream>
#include <string.h>
#include <memory>

#ifndef __PYOPENCL_UTILS_H
#define __PYOPENCL_UTILS_H

#define PYOPENCL_DEF_GET_CLASS_T(name)          \
    static inline class_t                       \
    get_class_t()                               \
    {                                           \
        return CLASS_##name;                    \
    }


template<class T>
PYOPENCL_USE_RESULT static inline std::string
tostring(const T& v)
{
    std::ostringstream ostr;
    ostr << v;
    return ostr.str();
}


template<typename T>
struct _D {
    void operator()(T *p) {
        free((void*)p);
    }
};

template<typename T>
class pyopencl_buf : public std::unique_ptr<T, _D<T> > {
    size_t m_len;
public:
    pyopencl_buf(size_t len=1) :
        std::unique_ptr<T, _D<T> >((T*)(len ? malloc(sizeof(T) * len) : NULL)),
        m_len(len)
    {
    }
    inline size_t
    len() const
    {
        return m_len;
    }
    inline T&
    operator[](int i)
    {
        return this->get()[i];
    }
    inline const T&
    operator[](int i) const
    {
        return this->get()[i];
    }
    inline void
    resize(size_t len)
    {
        if (len == m_len)
            return;
        m_len = len;
        this->reset((T*)realloc((void*)this->release(), len * sizeof(T)));
    }
};

namespace pyopencl {

template<typename T>
static inline cl_bool
cast_bool(const T &v)
{
    return v ? CL_TRUE : CL_FALSE;
}

struct clbase {
private:
    // non-copyable
    clbase(const clbase&) = delete;
    clbase &operator=(const clbase&) = delete;
    bool operator==(clbase const &other) const = delete;
    bool operator!=(clbase const &other) const = delete;
public:
    clbase() = default;
    virtual ~clbase() = default;
    virtual intptr_t intptr() const = 0;
    virtual generic_info get_info(cl_uint) const = 0;
};

template<typename CLType>
class clobj : public clbase {
private:
    CLType m_obj;
public:
    typedef CLType cl_type;
    clobj(CLType obj, bool=false) : m_obj(obj)
    {}
    inline const CLType&
    data() const
    {
        return m_obj;
    }
    intptr_t
    intptr() const
    {
        return (intptr_t)m_obj;
    }
};

template<typename T, typename T2>
PYOPENCL_USE_RESULT static inline pyopencl_buf<typename T::cl_type>
buf_from_class(const T2 *buf2, size_t len)
{
    pyopencl_buf<typename T::cl_type> buf(len);
    for (size_t i = 0;i < len;i++) {
        buf[i] = static_cast<const T*>(buf2[i])->data();
    }
    return buf;
}

template<typename T, typename T2>
PYOPENCL_USE_RESULT static inline pyopencl_buf<typename T::cl_type>
buf_from_class(const pyopencl_buf<T2> &&buf)
{
    return buf_from_class(buf.get(), buf.len());
}

template<typename T, typename T2>
PYOPENCL_USE_RESULT static inline pyopencl_buf<typename T::cl_type>
buf_from_class(const pyopencl_buf<T2> &buf)
{
    return buf_from_class(buf.get(), buf.len());
}

template<typename T, typename T2, typename... ArgTypes>
PYOPENCL_USE_RESULT static inline pyopencl_buf<clbase*>
buf_to_base(const T2 *buf2, size_t len, ArgTypes&&... args)
{
    pyopencl_buf<clbase*> buf(len);
    size_t i = 0;
    try {
        for (;i < len;i++) {
            buf[i] = static_cast<clbase*>(
                new T((typename T::cl_type)buf2[i],
                      std::forward<ArgTypes>(args)...));
        }
    } catch (...) {
        for (size_t j = 0;j < i;j++) {
            delete buf[i];
        }
        throw;
    }
    return buf;
}

template<typename T, typename T2, typename... ArgTypes>
PYOPENCL_USE_RESULT static inline pyopencl_buf<clbase*>
buf_to_base(const pyopencl_buf<T2> &&buf2, ArgTypes&&... args)
{
    return buf_to_base<T>(buf2.get(), buf2.len(),
                           std::forward<ArgTypes>(args)...);
}

template<typename T, typename T2, typename... ArgTypes>
PYOPENCL_USE_RESULT static inline pyopencl_buf<clbase*>
buf_to_base(const pyopencl_buf<T2> &buf2, ArgTypes&&... args)
{
    return buf_to_base<T>(buf2.get(), buf2.len(),
                          std::forward<ArgTypes>(args)...);
}

// FIXME
PYOPENCL_USE_RESULT static inline char*
_copy_str(const std::string& str)
{
    return strdup(str.c_str());
}

// {{{ GetInfo helpers

template<typename T, typename... ArgTypes, typename... ArgTypes2>
PYOPENCL_USE_RESULT static inline pyopencl_buf<T>
get_vec_info(cl_int (*func)(ArgTypes...), const char *name,
             ArgTypes2&&... args)
{
    size_t size = 0;
    call_guarded(func, name, args..., 0, NULL, &size);
    pyopencl_buf<T> buf(size / sizeof(T));
    call_guarded(func, name, args..., size, buf.get(), &size);
    return buf;
}
#define pyopencl_get_vec_info(type, what, args...)                      \
    pyopencl::get_vec_info<type>(clGet##what##Info, "clGet" #what "Info", args)

template<typename T>
PYOPENCL_USE_RESULT static inline generic_info
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
PYOPENCL_USE_RESULT static inline generic_info
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
PYOPENCL_USE_RESULT static inline generic_info
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
PYOPENCL_USE_RESULT static inline generic_info
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
PYOPENCL_USE_RESULT static inline generic_info
get_opaque_info(cl_int (*func)(ArgTypes...), const char *name,
                ArgTypes2&&... args)
{
    CLType param_value;
    call_guarded(func, name, args..., sizeof(param_value), &param_value, NULL);
    generic_info info;
    info.dontfree = 0;
    info.opaque_class = Cls::get_class_t();
    info.type = "void *";
    if (param_value) {
        info.value = (void*)(new Cls(param_value, /*retain*/ true));
    } else {
        info.value = NULL;
    }
    return info;
}
#define pyopencl_get_opaque_info(type, cls, what, args...)              \
    pyopencl::get_opaque_info<type, cls>(clGet##what##Info,             \
                                         "clGet" #what "Info", args)

template<typename... ArgTypes, typename... ArgTypes2>
PYOPENCL_USE_RESULT static inline generic_info
get_str_info(cl_int (*func)(ArgTypes...), const char *name,
             ArgTypes2&&... args)
{
    size_t param_value_size;
    call_guarded(func, name, args..., 0, NULL, &param_value_size);
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
PYOPENCL_USE_RESULT static inline generic_info
get_int_info(cl_int (*func)(ArgTypes...), const char *name,
             const char *tpname, ArgTypes2&&... args)
{
    pyopencl_buf<T> param_value;
    call_guarded(func, name, args..., sizeof(T), param_value.get(), NULL);
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

unsigned long next_obj_id();
extern void (*python_deref)(unsigned long);

}

#endif
