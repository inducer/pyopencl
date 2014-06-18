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

namespace pyopencl {

template<typename T, class = void>
class CLArg {
private:
    T &m_arg;
public:
    CLArg(T &arg) noexcept
        : m_arg(arg)
    {}
    CLArg(CLArg &&other) noexcept
        : m_arg(other.m_arg)
    {}
    PYOPENCL_INLINE T&
    convert() noexcept
    {
        return m_arg;
    }
};

enum class ArgType {
    None,
    SizeOf,
    Length,
};

template<typename T, ArgType AT=ArgType::None>
class ArgBuffer {
private:
    T *m_buf;
    size_t m_len;
protected:
    PYOPENCL_INLINE void
    set(T *buf) noexcept
    {
        m_buf = buf;
    }
public:
    typedef T type;
    constexpr static ArgType arg_type = AT;
    ArgBuffer(T *buf, size_t l) noexcept
        : m_buf(buf), m_len(l)
    {}
    ArgBuffer(ArgBuffer<T, AT> &&other) noexcept
        : ArgBuffer(other.m_buf, other.m_len)
    {}
    PYOPENCL_INLINE T*
    get() const noexcept
    {
        return m_buf;
    }
    PYOPENCL_INLINE size_t
    len() const noexcept
    {
        return m_len;
    }
};

template<ArgType AT=ArgType::None, typename T>
static PYOPENCL_INLINE ArgBuffer<T, AT>
make_argbuf(T &buf)
{
    return ArgBuffer<T, AT>(&buf, 1);
}

template<ArgType AT=ArgType::None, typename T>
static PYOPENCL_INLINE ArgBuffer<T, AT>
make_argbuf(T *buf, size_t l)
{
    return ArgBuffer<T, AT>(buf, l);
}

template<typename T>
static PYOPENCL_INLINE ArgBuffer<T, ArgType::SizeOf>
make_sizearg(T &buf)
{
    return ArgBuffer<T, ArgType::SizeOf>(&buf, 1);
}

template<typename Buff, class = void>
struct _ArgBufferConverter;

template<typename Buff>
struct _ArgBufferConverter<Buff, typename std::enable_if<
                                     Buff::arg_type == ArgType::None>::type> {
    static PYOPENCL_INLINE typename Buff::type*
    convert(Buff &buff)
    {
        return buff.get();
    }
};

template<typename Buff>
struct _ArgBufferConverter<Buff, typename std::enable_if<
                                     Buff::arg_type == ArgType::SizeOf>::type> {
    static PYOPENCL_INLINE auto
    convert(Buff &buff)
        -> decltype(std::make_tuple(sizeof(typename Buff::type) * buff.len(),
                                    buff.get()))
    {
        return std::make_tuple(sizeof(typename Buff::type) * buff.len(),
                               buff.get());
    }
};

template<typename Buff>
struct _ArgBufferConverter<Buff, typename std::enable_if<
                                     Buff::arg_type == ArgType::Length>::type> {
    static PYOPENCL_INLINE auto
    convert(Buff &buff)
        -> decltype(std::make_tuple(buff.len(), buff.get()))
    {
        return std::make_tuple(buff.len(), buff.get());
    }
};

template<typename Buff>
class CLArg<Buff, typename std::enable_if<std::is_base_of<
                                              ArgBuffer<typename Buff::type,
                                                        Buff::arg_type>,
                                              Buff>::value>::type> {
private:
    Buff &m_buff;
public:
    CLArg(Buff &buff) noexcept
        : m_buff(buff)
    {}
    CLArg(CLArg<Buff> &&other) noexcept
        : m_buff(other.m_buff)
    {}
    PYOPENCL_INLINE auto
    convert() const noexcept
        -> decltype(_ArgBufferConverter<Buff>::convert(m_buff))
    {
        return _ArgBufferConverter<Buff>::convert(m_buff);
    }
};

template<typename T, size_t n, ArgType AT=ArgType::None>
class ConstBuffer : public ArgBuffer<const T, AT> {
private:
    T m_intern_buf[n];
    ConstBuffer(ConstBuffer<T, n, AT>&&) = delete;
public:
    ConstBuffer(const T *buf, size_t l)
        : ArgBuffer<const T, AT>(buf, n)
    {
        if (l < n) {
            memcpy(m_intern_buf, buf, sizeof(T) * std::min(l, n));
            this->set(m_intern_buf);
        }
    }
};

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

template<typename Buff>
class CLArg<Buff, typename std::enable_if<
                      std::is_base_of<
                          pyopencl_buf<typename Buff::element_type>,
                          Buff>::value>::type> {
private:
    Buff &m_buff;
public:
    CLArg(Buff &buff) noexcept
        : m_buff(buff)
    {}
    CLArg(CLArg<Buff> &&other) noexcept
        : m_buff(other.m_buff)
    {}
    PYOPENCL_INLINE auto
    convert() const noexcept
        -> decltype(std::make_tuple(m_buff.len(), m_buff.get()))
    {
        return std::make_tuple(m_buff.len(), m_buff.get());
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
