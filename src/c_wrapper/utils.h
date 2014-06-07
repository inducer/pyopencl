#include "wrap_cl.h"
#include "function.h"

#include <string>
#include <sstream>
#include <string.h>
#include <memory>

#ifndef __PYOPENCL_UTILS_H
#define __PYOPENCL_UTILS_H

template<class T>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE std::string
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
        std::unique_ptr<T, _D<T> >((T*)(len ? malloc(sizeof(T) * len) :
                                        nullptr)),
        m_len(len)
    {
    }
    PYOPENCL_INLINE size_t
    len() const
    {
        return m_len;
    }
    PYOPENCL_INLINE T&
    operator[](int i)
    {
        return this->get()[i];
    }
    PYOPENCL_INLINE const T&
    operator[](int i) const
    {
        return this->get()[i];
    }
    PYOPENCL_INLINE void
    resize(size_t len)
    {
        if (len == m_len)
            return;
        m_len = len;
        this->reset((T*)realloc((void*)this->release(), len * sizeof(T)));
    }
};

namespace pyopencl {

template<typename T, class = void>
class CLArg {
private:
    T &m_arg;
public:
    CLArg(T &arg)
        : m_arg(arg)
    {
    }
    PYOPENCL_INLINE T&
    convert()
    {
        return m_arg;
    }
};

template<typename T, size_t n>
class ConstBuffer {
private:
    T m_intern_buf[n];
    const T *m_buf;
public:
    ConstBuffer(const T *buf, size_t l)
        : m_buf(buf)
    {
        if (l < n) {
            memcpy(m_intern_buf, buf, sizeof(T) * std::min(l, n));
            m_buf = m_intern_buf;
        }
    }
    operator const T*()
    {
        return m_buf;
    }
};

template<typename T>
static PYOPENCL_INLINE cl_bool
cast_bool(const T &v)
{
    return v ? CL_TRUE : CL_FALSE;
}

// FIXME
PYOPENCL_USE_RESULT static PYOPENCL_INLINE char*
_copy_str(const std::string& str)
{
    return strdup(str.c_str());
}

}

#endif
