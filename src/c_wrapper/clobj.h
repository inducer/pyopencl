#include "utils.h"

#ifndef __PYOPENCL_CLOBJ_H
#define __PYOPENCL_CLOBJ_H

#define PYOPENCL_DEF_CL_CLASS(name)                     \
    constexpr static class_t class_id = CLASS_##name;   \
    constexpr static const char *class_name = #name;

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
    PYOPENCL_INLINE
    clobj(CLType obj, bool=false) : m_obj(obj)
    {}
    PYOPENCL_INLINE const CLType&
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

template<typename CLObj>
void
print_clobj(std::ostream &stm, const CLObj *obj)
{
    stm << CLObj::class_name << "(" << (const void*)obj << ")<"
        << (const void*)obj->data() << ">";
}

template<typename CLObj>
class CLArg<CLObj, enable_if_t<std::is_base_of<clobj<typename CLObj::cl_type>,
                                               CLObj>::value> > {
private:
    CLObj &m_obj;
public:
    CLArg(CLObj &obj) : m_obj(obj)
    {
    }
    PYOPENCL_INLINE const typename CLObj::cl_type&
    convert() const
    {
        return m_obj.data();
    }
    PYOPENCL_INLINE void
    print(std::ostream &stm)
    {
        print_clobj(stm, &m_obj);
    }
};

template<typename CLObj>
class CLArg<CLObj*, enable_if_t<std::is_base_of<clobj<typename CLObj::cl_type>,
                                                CLObj>::value> > {
private:
    CLObj *m_obj;
public:
    CLArg(CLObj *obj) : m_obj(obj)
    {
    }
    PYOPENCL_INLINE const typename CLObj::cl_type&
    convert() const
    {
        return m_obj->data();
    }
    PYOPENCL_INLINE void
    print(std::ostream &stm)
    {
        print_clobj(stm, m_obj);
    }
};

template<typename CLObj>
static PYOPENCL_INLINE CLObj*
clobj_from_int_ptr(intptr_t ptr, bool retain)
{
    return new CLObj(reinterpret_cast<typename CLObj::cl_type>(ptr), retain);
}

template<typename T, typename T2>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE pyopencl_buf<typename T::cl_type>
buf_from_class(T2 *buf2, size_t len)
{
    pyopencl_buf<typename T::cl_type> buf(len);
    for (size_t i = 0;i < len;i++) {
        buf[i] = static_cast<const T*>(buf2[i])->data();
    }
    return buf;
}

template<typename T, typename T2>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE pyopencl_buf<typename T::cl_type>
buf_from_class(T2 &&buf)
{
    return buf_from_class(buf.get(), buf.len());
}

template<typename T, typename T2, typename... ArgTypes>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE pyopencl_buf<clbase*>
buf_to_base(T2 *buf2, size_t len, ArgTypes&&... args)
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
PYOPENCL_USE_RESULT static PYOPENCL_INLINE pyopencl_buf<clbase*>
buf_to_base(T2 &&buf2, ArgTypes&&... args)
{
    return buf_to_base<T>(buf2.get(), buf2.len(),
                           std::forward<ArgTypes>(args)...);
}

#endif
