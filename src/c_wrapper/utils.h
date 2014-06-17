#include "wrap_cl.h"
#include "function.h"

#include <string>
#include <sstream>
#include <string.h>
#include <memory>

#ifndef __PYOPENCL_UTILS_H
#define __PYOPENCL_UTILS_H

#if (defined(__GNUC__) && (__GNUC__ > 2))
#  define PYOPENCL_EXPECT(exp, var) __builtin_expect(exp, var)
#else
#  define PYOPENCL_EXPECT(exp, var) (exp)
#endif

#define PYOPENCL_LIKELY(x) PYOPENCL_EXPECT(bool(x), true)
#define PYOPENCL_UNLIKELY(x) PYOPENCL_EXPECT(bool(x), false)

template<class T>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE std::string
tostring(const T& v)
{
    std::ostringstream ostr;
    ostr << v;
    return ostr.str();
}

namespace pyopencl {

// TODO
template<typename T, bool, class = void>
struct CLGenericArgPrinter {
    static PYOPENCL_INLINE void
    print(std::ostream &stm, T &arg)
    {
        stm << arg;
    }
};

template<bool out>
struct CLGenericArgPrinter<std::nullptr_t, out, void> {
    static PYOPENCL_INLINE void
    print(std::ostream &stm, std::nullptr_t&)
    {
        stm << (void*)nullptr;
    }
};

template<typename T, class = void>
struct CLGenericArgOut {
    constexpr static bool value = false;
};

template<typename T, class = void>
class CLArg {
private:
    T &m_arg;
public:
    constexpr static bool is_out = CLGenericArgOut<T>::value;
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
    template<bool out>
    PYOPENCL_INLINE void
    print(std::ostream &stm)
    {
        CLGenericArgPrinter<T, out>::print(stm, m_arg);
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
    constexpr static size_t ele_size = sizeof(T);
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
static PYOPENCL_INLINE void
_print_buf(std::ostream &stm, Buff &&buff, ArgType arg_type, bool content)
{
    typedef decltype(buff.len()) len_t;
    len_t len = buff.len();
    typedef typename std::remove_reference<Buff>::type _Buff;
    size_t ele_size = _Buff::ele_size;
    if (content) {
        stm << "[";
        for (len_t i = 0;i < len;i++) {
            stm << buff.get()[i];
            if (i != len - 1) {
                stm << ", ";
            }
        }
        stm << "] <";
    }
    switch (arg_type) {
    case ArgType::SizeOf:
        stm << ele_size * len << ", ";
    case ArgType::Length:
        stm << len << ", ";
    default:
        break;
    }
    stm << buff.get();
    if (content) {
        stm << ">";
    }
}

template<typename Buff>
class CLArg<Buff, typename std::enable_if<std::is_base_of<
                                              ArgBuffer<typename Buff::type,
                                                        Buff::arg_type>,
                                              Buff>::value>::type> {
private:
    Buff &m_buff;
public:
    constexpr static bool is_out = !(std::is_const<Buff>::value ||
                                     std::is_const<typename Buff::type>::value);
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
    template<bool out>
    PYOPENCL_INLINE void
    print(std::ostream &stm)
    {
        _print_buf(stm, m_buff, Buff::arg_type, out || !is_out);
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

struct OutArg {
    PYOPENCL_INLINE void
    convert()
    {}
    PYOPENCL_INLINE void
    cleanup(bool)
    {}
};

template<typename T>
class _SimpleOutArg : public OutArg {
    T *m_t;
public:
    _SimpleOutArg(T *t)
        : m_t(t)
    {}
    PYOPENCL_INLINE T*
    get()
    {
        return m_t;
    }
    template<bool out>
    PYOPENCL_INLINE void
    print(std::ostream &stm)
    {
        if (!out) {
            stm << m_t;
        } else {
            stm << "*(" << m_t << "): " << *m_t;
        }
    }
};

template<typename T>
static PYOPENCL_INLINE _SimpleOutArg<T>
out_arg(T *t)
{
    return _SimpleOutArg<T>(t);
}

template<typename T>
class CLArg<T, typename std::enable_if<
                      std::is_base_of<OutArg, T>::value>::type> {
private:
    bool m_converted;
    bool m_need_cleanup;
    T &m_arg;
public:
    constexpr static bool is_out = true;
    CLArg(T &arg)
        : m_converted(false), m_need_cleanup(false), m_arg(arg)
    {
    }
    CLArg(CLArg<T> &&other) noexcept
        : m_converted(other.m_converted), m_need_cleanup(other.m_need_cleanup),
        m_arg(other.m_arg)
    {
        other.m_need_cleanup = false;
    }
    PYOPENCL_INLINE auto
    convert()
        -> decltype(m_arg.get())
    {
        return m_arg.get();
    }
    PYOPENCL_INLINE void
    finish(bool converted) noexcept
    {
        m_need_cleanup = !converted;
    }
    PYOPENCL_INLINE void
    post()
    {
        m_arg.convert();
        m_converted = true;
    }
    ~CLArg()
    {
        if (m_need_cleanup) {
            m_arg.cleanup(m_converted);
        }
    }
    template<bool out>
    PYOPENCL_INLINE void
    print(std::ostream &stm)
    {
        m_arg.template print<out>(stm);
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
    constexpr static size_t ele_size = sizeof(T);
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
    constexpr static bool is_out =
        !(std::is_const<Buff>::value ||
          std::is_const<typename Buff::element_type>::value);
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
    template<bool out>
    PYOPENCL_INLINE void
    print(std::ostream &stm)
    {
        _print_buf(stm, m_buff, ArgType::Length, out || !is_out);
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
