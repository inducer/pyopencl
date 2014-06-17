#include "wrap_cl.h"
#include "pyhelper.h"
#include "clobj.h"
#include "debug.h"

#include <string.h>
#include <stdexcept>
#include <iostream>
#include <utility>
#include <functional>
#include <atomic>

#ifndef __PYOPENCL_ERROR_H
#define __PYOPENCL_ERROR_H

namespace pyopencl {

template<typename FirstType, typename... ArgTypes>
static PYOPENCL_INLINE void
_print_args(std::ostream &stm, FirstType &&arg1, ArgTypes&&... args)
{
    stm << arg1 << "; ";
    _print_args(stm, std::forward<ArgTypes>(args)...);
}

template<typename FirstType>
static PYOPENCL_INLINE void
_print_args(std::ostream &stm, FirstType &&arg1)
{
    stm << arg1 << "; ";
}

static PYOPENCL_INLINE void
print_call_trace(const char *name)
{
    if (!DEBUG_ON)
        return;
    std::cerr << name << std::endl;
}

template<typename... ArgTypes>
static PYOPENCL_INLINE void
print_call_trace(const char *name, ArgTypes&&... args)
{
    if (!DEBUG_ON)
        return;
    std::cerr << name << "(";
    _print_args(std::cerr, args...);
    std::cerr << ")" << std::endl;
}

// {{{ error

class clerror : public std::runtime_error {
private:
    const char *m_routine;
    cl_int m_code;

public:
    clerror(const char *rout, cl_int c, const char *msg="")
        : std::runtime_error(msg), m_routine(rout), m_code(c)
    {
        std::cout << rout <<";" << msg<< ";" << c << std::endl;
    }
    PYOPENCL_INLINE const char*
    routine() const
    {
        return m_routine;
    }

    PYOPENCL_INLINE cl_int
    code() const
    {
        return m_code;
    }

    PYOPENCL_INLINE bool
    is_out_of_memory() const
    {
        return (code() == CL_MEM_OBJECT_ALLOCATION_FAILURE ||
                code() == CL_OUT_OF_RESOURCES ||
                code() == CL_OUT_OF_HOST_MEMORY);
    }
};

// }}}

// {{{ tracing and error reporting

template<typename>
struct __CLArgGetter {
    template<typename T>
    static PYOPENCL_INLINE auto
    get(T&& clarg) -> decltype(clarg.convert())
    {
        return clarg.convert();
    }
};

template<typename T, class = void>
struct __CLFinish {
    static PYOPENCL_INLINE void
    call(T)
    {
    }
};

template<typename T>
struct __CLFinish<T, decltype((void)(std::declval<T>().finish()))> {
    static PYOPENCL_INLINE void
    call(T v)
    {
        v.finish();
    }
};

template<typename T, class = void>
struct __CLCleanup {
    static PYOPENCL_INLINE void
    call(T)
    {
    }
};

template<typename T>
struct __CLCleanup<T, decltype((void)(std::declval<T>().cleanup()))> {
    static PYOPENCL_INLINE void
    call(T v)
    {
        v.cleanup();
    }
};

template<template<typename...> class Caller, size_t n, typename T>
struct __CLCall {
    static PYOPENCL_INLINE void
    call(T &&t)
    {
        __CLCall<Caller, n - 1, T>::call(std::forward<T>(t));
        Caller<decltype(std::get<n>(t))>::call(std::get<n>(t));
    }
};

template<template<typename...> class Caller, typename T>
struct __CLCall<Caller, 0, T> {
    static PYOPENCL_INLINE void
    call(T &&t)
    {
        Caller<decltype(std::get<0>(t))>::call(std::get<0>(t));
    }
};

template<typename... Types>
class CLArgPack : public ArgPack<CLArg, Types...> {
public:
    using ArgPack<CLArg, Types...>::ArgPack;
    template<typename Func>
    PYOPENCL_INLINE auto
    clcall(Func func)
        -> decltype(this->template call<__CLArgGetter>(func))
    {
        auto res = this->template call<__CLArgGetter>(func);
        typename CLArgPack::tuple_base *that = this;
        __CLCall<__CLFinish, sizeof...(Types) - 1,
                 decltype(*that)>::call(*that);
        __CLCall<__CLCleanup, sizeof...(Types) - 1,
                 decltype(*that)>::call(*that);
        return res;
    }
};

template<typename... Types>
static PYOPENCL_INLINE CLArgPack<typename std::remove_reference<Types>::type...>
make_clargpack(Types&&... args)
{
    return CLArgPack<typename std::remove_reference<Types>::type...>(
        std::forward<Types>(args)...);
}

template<typename... ArgTypes2, typename... ArgTypes>
static PYOPENCL_INLINE void
call_guarded(cl_int (*func)(ArgTypes...), const char *name, ArgTypes2&&... args)
{
    print_call_trace(name);
    auto argpack = make_clargpack(std::forward<ArgTypes2>(args)...);
    cl_int status_code = argpack.clcall(func);
    if (status_code != CL_SUCCESS) {
        throw clerror(name, status_code);
    }
}

template<typename T, typename... ArgTypes, typename... ArgTypes2>
PYOPENCL_USE_RESULT static PYOPENCL_INLINE T
call_guarded(T (*func)(ArgTypes...), const char *name, ArgTypes2&&... args)
{
    print_call_trace(name);
    cl_int status_code = CL_SUCCESS;
    auto argpack = make_clargpack(std::forward<ArgTypes2>(args)...,
                                  &status_code);
    T res = argpack.clcall(func);
    if (status_code != CL_SUCCESS) {
        throw clerror(name, status_code);
    }
    return res;
}
#define pyopencl_call_guarded(func, args...)    \
    pyopencl::call_guarded(func, #func, args)

template<typename... ArgTypes, typename... ArgTypes2>
static PYOPENCL_INLINE void
call_guarded_cleanup(cl_int (*func)(ArgTypes...), const char *name,
                     ArgTypes2&&... args)
{
    print_call_trace(name);
    auto argpack = make_clargpack(std::forward<ArgTypes2>(args)...);
    cl_int status_code = argpack.clcall(func);
    if (status_code != CL_SUCCESS) {
        std::cerr
            << ("PyOpenCL WARNING: a clean-up operation failed "
                "(dead context maybe?)") << std::endl
            << name << " failed with code " << status_code << std::endl;
    }
}
#define pyopencl_call_guarded_cleanup(func, args...)    \
    pyopencl::call_guarded_cleanup(func, #func, args)

PYOPENCL_USE_RESULT static PYOPENCL_INLINE ::error*
c_handle_error(std::function<void()> func) noexcept
{
    try {
        func();
        return nullptr;
    } catch(const clerror &e) {
        auto err = (::error*)malloc(sizeof(::error));
        err->routine = strdup(e.routine());
        err->msg = strdup(e.what());
        err->code = e.code();
        err->other = 0;
        return err;
    } catch (const std::exception &e) {
        /* non-pyopencl exceptions need to be converted as well */
        auto err = (::error*)malloc(sizeof(::error));
        err->other = 1;
        err->msg = strdup(e.what());
        return err;
    }
}

template<typename Func>
static PYOPENCL_INLINE auto
retry_mem_error(Func func) -> decltype(func())
{
    try {
        return func();
    } catch (clerror &e) {
        if (PYOPENCL_LIKELY(!e.is_out_of_memory()) || !py::gc()) {
            throw;
        }
    }
    return func();
}

// }}}

}

#endif
