#include "wrap_cl.h"
#include "function.h"
#include "debug.h"

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

template<typename T, class = void>
struct CLGenericArgPrinter {
    static PYOPENCL_INLINE void
    print(std::ostream &stm, T &arg)
    {
        stm << arg;
    }
};

PYOPENCL_USE_RESULT static PYOPENCL_INLINE void*
cl_memdup(const void *p, size_t size)
{
    void *res = malloc(size);
    memcpy(res, p, size);
    return res;
}

template<typename T>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE T*
cl_memdup(const T *p)
{
    // Not supported by libstdc++ yet...
    // static_assert(std::is_trivially_copy_constructible<T>::value);
    return static_cast<T*>(cl_memdup(static_cast<const void*>(p), sizeof(T)));
}

enum class ArgType {
    None,
    SizeOf,
    Length,
};

template<typename T, class = void>
struct type_size : std::integral_constant<size_t, sizeof(T)> {};
template<typename T>
struct type_size<T, enable_if_t<std::is_same<rm_const_t<T>, void>::value> > :
        std::integral_constant<size_t, 1> {};

template<typename T>
static PYOPENCL_INLINE void
_print_buf_content(std::ostream &stm, const T *p, size_t len)
{
    if (len > 1) {
        stm << "[";
    }
    for (size_t i = 0;i < len;i++) {
        CLGenericArgPrinter<const T>::print(stm, p[i]);
        if (i != len - 1) {
            stm << ", ";
        }
    }
    if (len > 1) {
        stm << "]";
    }
}

template<>
PYOPENCL_INLINE void
_print_buf_content<char>(std::ostream &stm, const char *p, size_t len)
{
    dbg_print_str(stm, p, len);
}

template<>
PYOPENCL_INLINE void
_print_buf_content<unsigned char>(std::ostream &stm,
                                  const unsigned char *p, size_t len)
{
    dbg_print_bytes(stm, p, len);
}

template<>
PYOPENCL_INLINE void
_print_buf_content<void>(std::ostream &stm, const void *p, size_t len)
{
    dbg_print_bytes(stm, static_cast<const unsigned char*>(p), len);
}

template<typename T>
void
print_buf(std::ostream &stm, const T *p, size_t len,
          ArgType arg_type, bool content, bool out)
{
    const size_t ele_size = type_size<T>::value;
    if (out) {
        stm << "*(" << (const void*)p << "): ";
        if (p) {
            _print_buf_content(stm, p, len);
        } else {
            stm << "NULL";
        }
    } else {
        bool need_quote = content || arg_type != ArgType::None;
        if (content) {
            if (p) {
                _print_buf_content(stm, p, len);
                stm << " ";
            } else {
                stm << "NULL ";
            }
        }
        if (need_quote) {
            stm << "<";
        }
        switch (arg_type) {
        case ArgType::SizeOf:
            stm << ele_size * len << ", ";
            break;
        case ArgType::Length:
            stm << len << ", ";
            break;
        default:
            break;
        }
        stm << (const void*)p;
        if (need_quote) {
            stm << ">";
        }
    }
}

template<typename T>
void
print_arg(std::ostream &stm, const T &v, bool out)
{
    if (!out) {
        stm << (const void*)&v;
    } else {
        stm << "*(" << (const void*)&v << "): " << v;
    }
}
extern template void print_buf<char>(std::ostream&, const char*, size_t,
                                     ArgType, bool, bool);
extern template void print_buf<cl_int>(std::ostream&, const cl_int*, size_t,
                                       ArgType, bool, bool);
extern template void print_buf<cl_uint>(std::ostream&, const cl_uint*, size_t,
                                        ArgType, bool, bool);
extern template void print_buf<cl_long>(std::ostream&, const cl_long*, size_t,
                                        ArgType, bool, bool);
extern template void print_buf<cl_ulong>(std::ostream&, const cl_ulong*, size_t,
                                         ArgType, bool, bool);
extern template void print_buf<cl_image_format>(std::ostream&,
                                                const cl_image_format*, size_t,
                                                ArgType, bool, bool);

template<>
struct CLGenericArgPrinter<std::nullptr_t, void> {
    static PYOPENCL_INLINE void
    print(std::ostream &stm, std::nullptr_t&)
    {
        stm << (void*)nullptr;
    }
};

template<typename T>
struct CLGenericArgPrinter<
    T, enable_if_t<std::is_same<const char*, rm_const_t<T> >::value ||
                   std::is_same<char*, rm_const_t<T> >::value> > {
    static PYOPENCL_INLINE void
    print(std::ostream &stm, const char *str)
    {
        dbg_print_str(stm, str);
    }
};

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
    PYOPENCL_INLINE void
    print(std::ostream &stm)
    {
        CLGenericArgPrinter<T>::print(stm, m_arg);
    }
};

template<>
class CLArg<bool> : public CLArg<cl_bool> {
    cl_bool m_arg;
public:
    CLArg(bool arg) noexcept
        : CLArg<cl_bool>(m_arg), m_arg(arg ? CL_TRUE : CL_FALSE)
    {}
    CLArg(CLArg<bool> &&other) noexcept
        : CLArg<bool>(bool(other.m_arg))
    {}
    PYOPENCL_INLINE void
    print(std::ostream &stm)
    {
        stm << (m_arg ? "true" : "false");
    }
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
    PYOPENCL_INLINE rm_const_t<T>*
    get() const noexcept
    {
        return const_cast<rm_const_t<T>*>(m_buf);
    }
    template<typename T2 = T>
    PYOPENCL_INLINE T2&
    operator[](int i) const
    {
        return m_buf[i];
    }
    PYOPENCL_INLINE size_t
    len() const noexcept
    {
        return m_len;
    }
};

template<ArgType AT, typename T, class = void>
struct _ToArgBuffer {
    static PYOPENCL_INLINE ArgBuffer<rm_ref_t<T>, AT>
    convert(T &buf)
    {
        return ArgBuffer<rm_ref_t<T>, AT>(&buf, 1);
    }
};

template<ArgType AT=ArgType::None, typename T>
static PYOPENCL_INLINE auto
buf_arg(T &&buf) -> decltype(_ToArgBuffer<AT, T>::convert(std::forward<T>(buf)))
{
    return _ToArgBuffer<AT, T>::convert(std::forward<T>(buf));
}

template<ArgType AT=ArgType::None, typename T>
static PYOPENCL_INLINE ArgBuffer<T, AT>
buf_arg(T *buf, size_t l)
{
    return ArgBuffer<T, AT>(buf, l);
}

template<typename... T>
static PYOPENCL_INLINE auto
size_arg(T&&... buf)
    -> decltype(buf_arg<ArgType::SizeOf>(std::forward<T>(buf)...))
{
    return buf_arg<ArgType::SizeOf>(std::forward<T>(buf)...);
}

template<typename... T>
static PYOPENCL_INLINE auto
len_arg(T&&... buf)
    -> decltype(buf_arg<ArgType::Length>(std::forward<T>(buf)...))
{
    return buf_arg<ArgType::Length>(std::forward<T>(buf)...);
}

template<typename Buff, class = void>
struct _ArgBufferConverter;

template<typename Buff>
struct _ArgBufferConverter<Buff,
                           enable_if_t<Buff::arg_type == ArgType::None> > {
    static PYOPENCL_INLINE auto
    convert(Buff &buff) -> decltype(buff.get())
    {
        return buff.get();
    }
};

template<typename Buff>
struct _ArgBufferConverter<Buff,
                           enable_if_t<Buff::arg_type == ArgType::SizeOf> > {
    static PYOPENCL_INLINE auto
    convert(Buff &buff)
        -> decltype(std::make_tuple(type_size<typename Buff::type>::value *
                                    buff.len(), buff.get()))
    {
        return std::make_tuple(type_size<typename Buff::type>::value *
                               buff.len(), buff.get());
    }
};

template<typename Buff>
struct _ArgBufferConverter<Buff,
                           enable_if_t<Buff::arg_type == ArgType::Length> > {
    static PYOPENCL_INLINE auto
    convert(Buff &buff) -> decltype(std::make_tuple(buff.len(), buff.get()))
    {
        return std::make_tuple(buff.len(), buff.get());
    }
};

template<typename Buff>
class CLArg<Buff, enable_if_t<std::is_base_of<ArgBuffer<typename Buff::type,
                                                        Buff::arg_type>,
                                              Buff>::value> > {
private:
    Buff &m_buff;
public:
    constexpr static bool is_out = !std::is_const<typename Buff::type>::value;
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
    PYOPENCL_INLINE void
    print(std::ostream &stm, bool out=false)
    {
        print_buf(stm, m_buff.get(), m_buff.len(),
                  Buff::arg_type, out || !is_out, out);
    }
};

template<typename T, size_t n, ArgType AT=ArgType::None>
class ConstBuffer : public ArgBuffer<const T, AT> {
private:
    T m_intern_buf[n];
    ConstBuffer(ConstBuffer<T, n, AT>&&) = delete;
    ConstBuffer() = delete;
public:
    ConstBuffer(const T *buf, size_t l, T content=0)
        : ArgBuffer<const T, AT>(buf, n)
    {
        if (l < n) {
            memcpy(m_intern_buf, buf, type_size<T>::value * l);
            for (size_t i = l;i < n;i++) {
                m_intern_buf[i] = content;
            }
            this->set(m_intern_buf);
        }
    }
};

struct OutArg {
};

template<typename T>
class CLArg<T, enable_if_t<std::is_base_of<OutArg, T>::value> > {
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
    convert() -> decltype(m_arg.get())
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
    PYOPENCL_INLINE void
    print(std::ostream &stm, bool out=false)
    {
        m_arg.print(stm, out);
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
    PYOPENCL_INLINE
    pyopencl_buf(size_t len=1)
        : std::unique_ptr<T, _D<T> >((T*)(len ? malloc(sizeof(T) * (len + 1)) :
                                          nullptr)), m_len(len)
    {
        if (len) {
            memset((void*)this->get(), 0, (len + 1) * sizeof(T));
        }
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
        this->reset((T*)realloc((void*)this->release(),
                                (len + 1) * sizeof(T)));
    }
};

template<typename T>
using pyopencl_buf_ele_t = typename rm_ref_t<T>::element_type;

template<typename T, class = void>
struct is_pyopencl_buf : std::false_type {};

template<typename T>
struct is_pyopencl_buf<
    T, enable_if_t<std::is_base_of<pyopencl_buf<pyopencl_buf_ele_t<T> >,
                                   rm_ref_t<T> >::value> > : std::true_type {};

template<ArgType AT, typename T>
struct _ToArgBuffer<AT, T, enable_if_t<is_pyopencl_buf<T>::value &&
                                       std::is_const<rm_ref_t<T> >::value> > {
    static PYOPENCL_INLINE ArgBuffer<const pyopencl_buf_ele_t<T>, AT>
    convert(T &&buf)
    {
        return ArgBuffer<const pyopencl_buf_ele_t<T>, AT>(buf.get(), buf.len());
    }
};

template<ArgType AT, typename T>
struct _ToArgBuffer<AT, T, enable_if_t<is_pyopencl_buf<T>::value &&
                                       !std::is_const<rm_ref_t<T> >::value> > {
    static PYOPENCL_INLINE ArgBuffer<pyopencl_buf_ele_t<T>, AT>
    convert(T &&buf)
    {
        return ArgBuffer<pyopencl_buf_ele_t<T>, AT>(buf.get(), buf.len());
    }
};

template<typename Buff>
using __pyopencl_buf_arg_type =
    rm_ref_t<decltype(len_arg(std::declval<Buff&>()))>;

template<typename Buff>
class CLArg<Buff, enable_if_t<is_pyopencl_buf<Buff>::value> >
    : public CLArg<__pyopencl_buf_arg_type<Buff> > {
    typedef __pyopencl_buf_arg_type<Buff> BufType;
    BufType m_buff;
public:
    PYOPENCL_INLINE
    CLArg(Buff &buff) noexcept
        : CLArg<BufType>(m_buff), m_buff(len_arg(buff))
    {}
    PYOPENCL_INLINE
    CLArg(CLArg<Buff> &&other) noexcept
        : CLArg<BufType>(m_buff), m_buff(std::move(other.m_buff))
    {}
};

// FIXME
PYOPENCL_USE_RESULT static PYOPENCL_INLINE char*
_copy_str(const std::string& str)
{
    return strdup(str.c_str());
}

#endif
