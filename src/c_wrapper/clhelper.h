#include "error.h"
#include "clobj.h"

#ifndef __PYOPENCL_CLHELPER_H
#define __PYOPENCL_CLHELPER_H

template<typename CLObj, typename... T>
class _CLObjOutArg : public OutArg {
    typedef typename CLObj::cl_type CLType;
    clobj_t *const m_ret;
    CLType m_clobj;
    cl_int (CL_API_CALL *m_release)(CLType);
    const char *m_name;
    std::tuple<T...> m_t1;
    template<int... S>
    PYOPENCL_INLINE CLObj*
    __new_obj(seq<S...>)
    {
        return new CLObj(m_clobj, false, std::get<S>(m_t1)...);
    }
public:
    PYOPENCL_INLINE
    _CLObjOutArg(clobj_t *ret, cl_int (CL_API_CALL *release)(CLType),
                 const char *name, T... t1) noexcept
        : m_ret(ret), m_clobj(nullptr), m_release(release),
          m_name(name), m_t1(t1...)
    {
    }
    PYOPENCL_INLINE
    _CLObjOutArg(_CLObjOutArg<CLObj, T...> &&other) noexcept
        : m_ret(other.m_ret), m_clobj(other.m_clobj),
          m_release(other.m_release), m_name(other.m_name)
    {
        std::swap(m_t1, other.m_t1);
    }
    PYOPENCL_INLINE typename CLObj::cl_type*
    get()
    {
        return &m_clobj;
    }
    PYOPENCL_INLINE void
    convert()
    {
        *m_ret = __new_obj(typename gens<sizeof...(T)>::type());
    }
    PYOPENCL_INLINE void
    cleanup(bool converted)
    {
        if (converted) {
            delete *m_ret;
            *m_ret = nullptr;
        } else {
            call_guarded_cleanup(m_release, m_name, m_clobj);
        }
    }
    PYOPENCL_INLINE void
    print(std::ostream &stm, bool out=false) const
    {
        print_arg(stm, m_clobj, out);
    }
};

template<typename CLObj, typename... T>
static PYOPENCL_INLINE _CLObjOutArg<CLObj, T...>
make_cloutarg(clobj_t *ret, cl_int (CL_API_CALL *release)(typename CLObj::cl_type),
              const char *name, T... t1)
{
    return _CLObjOutArg<CLObj, T...>(ret, release, name, t1...);
}
#define pyopencl_outarg(type, ret, func, ...)               \
    make_cloutarg<type>(ret, func, #func, ##__VA_ARGS__)

// {{{ GetInfo helpers

template<typename T, typename... ArgTypes, typename... ArgTypes2>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE pyopencl_buf<T>
get_vec_info(cl_int (CL_API_CALL *func)(ArgTypes...), const char *name,
             ArgTypes2&&... args)
{
    size_t size = 0;
    call_guarded(func, name, args..., 0, nullptr, buf_arg(size));
    pyopencl_buf<T> buf(size / sizeof(T));
    call_guarded(func, name, args..., size_arg(buf), buf_arg(size));
    return buf;
}
#define pyopencl_get_vec_info(type, what, ...)                      \
    get_vec_info<type>(clGet##what##Info, "clGet" #what "Info", __VA_ARGS__)

inline generic_info make_generic_info(class_t opaque_class, const char *type, bool free_type, void *value, bool free_value)
{
  generic_info result;
  result.opaque_class = opaque_class;
  result.type = type;
  result.free_type = free_type;
  result.value = value;
  result.free_value = free_value;
  return result;
}

template<typename T>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE generic_info
convert_array_info(const char *tname, pyopencl_buf<T> &buf)
{
    return make_generic_info(
        CLASS_NONE,
        _copy_str(std::string(tname) + "[" + tostring(buf.len()) + "]"),
        true,
        buf.release(),
        true);
}

template<typename T>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE generic_info
convert_array_info(const char *tname, pyopencl_buf<T> &&_buf)
{
    pyopencl_buf<T> &buf = _buf;
    return convert_array_info<T>(tname, buf);
}

#define pyopencl_convert_array_info(type, buf)          \
    convert_array_info<type>(#type, buf)
#define pyopencl_get_array_info(type, what, ...)                    \
    pyopencl_convert_array_info(type, pyopencl_get_vec_info(type, what, __VA_ARGS__))

template<typename CLObj, typename T>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE generic_info
convert_opaque_array_info(T &&buf)
{
    return make_generic_info(
        CLObj::class_id,
        _copy_str(std::string("void*[") + tostring(buf.len()) + "]"),
        true,
        buf_to_base<CLObj>(std::forward<T>(buf)).release(),
        true);
}
#define pyopencl_get_opaque_array_info(cls, what, ...)  \
    convert_opaque_array_info<cls>(               \
        pyopencl_get_vec_info(cls::cl_type, what, __VA_ARGS__))

template<typename CLObj, typename... ArgTypes, typename... ArgTypes2>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE generic_info
get_opaque_info(cl_int (CL_API_CALL *func)(ArgTypes...), const char *name,
                ArgTypes2&&... args)
{
    typename CLObj::cl_type param_value;
    call_guarded(func, name, args..., size_arg(param_value), nullptr);
    void *value;
    if (param_value) {
        value = (void*)(new CLObj(param_value, /*retain*/ true));
    } else {
        value = nullptr;
    }
    return make_generic_info(CLObj::class_id, "void *", false, value, true);
}
#define pyopencl_get_opaque_info(clobj, what, ...)              \
    get_opaque_info<clobj>(clGet##what##Info,             \
                                     "clGet" #what "Info", __VA_ARGS__)

template<typename... ArgTypes, typename... ArgTypes2>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE generic_info
get_str_info(cl_int (CL_API_CALL *func)(ArgTypes...), const char *name,
             ArgTypes2&&... args)
{
    size_t size;
    call_guarded(func, name, args..., 0, nullptr, buf_arg(size));
    pyopencl_buf<char> param_value(size);
    call_guarded(func, name, args..., param_value, buf_arg(size));
    return make_generic_info(CLASS_NONE, "char*", false, (void*)param_value.release(), true);
}
#define pyopencl_get_str_info(what, ...)                            \
    get_str_info(clGet##what##Info, "clGet" #what "Info", __VA_ARGS__)

template<typename T, typename... ArgTypes, typename... ArgTypes2>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE generic_info
get_int_info(cl_int (CL_API_CALL *func)(ArgTypes...), const char *name,
             const char *tpname, ArgTypes2&&... args)
{
    T value;
    call_guarded(func, name, args..., size_arg(value), nullptr);
    return make_generic_info(CLASS_NONE, tpname, false, cl_memdup(&value), true);
}
#define pyopencl_get_int_info(type, what, ...)                      \
    get_int_info<type>(clGet##what##Info, "clGet" #what "Info", \
                                 #type "*", __VA_ARGS__)

// }}}

template<typename T, typename CLType, typename... ArgTypes>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE T*
convert_obj(cl_int (CL_API_CALL *clRelease)(CLType), const char *name, CLType cl_obj,
            ArgTypes&&... args)
{
    try {
        return new T(cl_obj, false, std::forward<ArgTypes>(args)...);
    } catch (...) {
        call_guarded_cleanup(clRelease, name, cl_obj);
        throw;
    }
}
#define pyopencl_convert_obj(type, func, ...)       \
    convert_obj<type>(func, #func, __VA_ARGS__)

// {{{ extension function pointers

#if PYOPENCL_CL_VERSION >= 0x1020
template<typename T>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE T
get_ext_fun(cl_platform_id plat, const char *name, const char *err)
{
    T func = (T)clGetExtensionFunctionAddressForPlatform(plat, name);
    if (!func) {
        throw clerror(name, CL_INVALID_VALUE, err);
    }
    return func;
}
#define pyopencl_get_ext_fun(plat, name)                                \
    get_ext_fun<name##_fn>(plat, #name, #name " not available")
#else
template<typename T>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE T
get_ext_fun(const char *name, const char *err)
{
    T func = (T)clGetExtensionFunctionAddress(name);
    if (!func) {
        throw clerror(name, CL_INVALID_VALUE, err);
    }
    return func;
}
#define pyopencl_get_ext_fun(plat, name)                                \
    get_ext_fun<name##_fn>(#name, #name " not available")
#endif

// }}}

static PYOPENCL_INLINE std::ostream&
operator<<(std::ostream &stm, const cl_image_format &fmt)
{
    stm << "channel_order: " << fmt.image_channel_order
        << ",\nchannel_data_type: " << fmt.image_channel_data_type;
    return stm;
}

#ifdef CL_DEVICE_TOPOLOGY_AMD
static PYOPENCL_INLINE std::ostream&
operator<<(std::ostream &stm, const cl_device_topology_amd &topol)
{
    stm << "pcie.bus: " << topol.pcie.bus
        << ",\npcie.device: " << topol.pcie.device
        << ",\npcie.function: " << topol.pcie.function
        << ",\npcie.type: " << topol.pcie.type;
    return stm;
}
#endif
#endif
