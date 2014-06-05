#include "wrap_cl.h"
#include "pyhelper.h"
#include "function.h"

#include <string.h>
#include <stdexcept>
#include <iostream>
#include <utility>
#include <functional>
#include <atomic>

#ifndef __PYOPENCL_ERROR_H
#define __PYOPENCL_ERROR_H

namespace pyopencl {

#ifdef PYOPENCL_TRACE

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
    std::cerr << name << std::endl;
}

template<typename... ArgTypes>
static PYOPENCL_INLINE void
print_call_trace(const char *name, ArgTypes&&... args)
{
    std::cerr << name << "(";
    _print_args(std::cerr, args...);
    std::cerr << ")" << std::endl;
}

#else

template<typename... ArgTypes>
static PYOPENCL_INLINE void
print_call_trace(ArgTypes&&...)
{
}

#endif

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

template<typename... ArgTypes2, typename... ArgTypes>
static PYOPENCL_INLINE void
call_guarded(cl_int (*func)(ArgTypes...), const char *name, ArgTypes2&&... args)
{
    print_call_trace(name);
    cl_int status_code = func(ArgTypes(args)...);
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
    T res = func(args..., &status_code);
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
    cl_int status_code = func(ArgTypes(args)...);
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
c_handle_error(std::function<void()> func)
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

template<typename T>
static PYOPENCL_INLINE T
retry_mem_error(std::function<T()> func)
{
    try {
        return func();
    } catch (clerror &e) {
        if (!e.is_out_of_memory() || !py::gc()) {
            throw;
        }
    }
    return func();
}

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
