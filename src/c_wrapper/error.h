#include "wrap_cl.h"
#include <stdexcept>
#include <iostream>
#include <utility>

namespace pyopencl {

#ifdef PYOPENCL_TRACE

template<typename FirstType, typename... ArgTypes>
static inline void
_print_args(std::ostream &stm, FirstType &&arg1, ArgTypes&&... args)
{
    stm << arg1 << "; ";
    _print_args(stm, std::forward<ArgTypes>(args)...);
}

template<typename FirstType>
static inline void
_print_args(std::ostream &stm, FirstType &&arg1)
{
    stm << arg1 << "; ";
}

static inline void
print_call_trace(const char *name)
{
    std::cerr << name << std::endl;
}

template<typename... ArgTypes>
static inline void
print_call_trace(const char *name, ArgTypes&&... args)
{
    std::cerr << name << "(";
    _print_args(std::cerr, args...);
    std::cerr << ")" << std::endl;
}

#else

template<typename... ArgTypes>
static inline void
print_call_trace(ArgTypes&&...)
{
}

#endif

// {{{ error

class error : public std::runtime_error {
private:
    const char *m_routine;
    cl_int m_code;

public:
    error(const char *rout, cl_int c, const char *msg="")
        : std::runtime_error(msg), m_routine(rout), m_code(c)
    {
        std::cout << rout <<";" << msg<< ";" << c << std::endl;
    }
    inline const char*
    routine() const
    {
        return m_routine;
    }

    inline cl_int
    code() const
    {
        return m_code;
    }

    inline bool
    is_out_of_memory() const
    {
        return (code() == CL_MEM_OBJECT_ALLOCATION_FAILURE ||
                code() == CL_OUT_OF_RESOURCES ||
                code() == CL_OUT_OF_HOST_MEMORY);
    }
};

// }}}

template<typename... ArgTypes, typename... ArgTypes2>
static inline void
call_guarded(cl_int (*func)(ArgTypes...), const char *name,
             ArgTypes2&&... args)
{
    print_call_trace(name);
    cl_int status_code = func(ArgTypes(args)...);
    if (status_code != CL_SUCCESS) {
        throw pyopencl::error(name, status_code);
    }
}
#define pyopencl_call_guarded(func, args...)    \
    pyopencl::call_guarded(func, #func, args)

template<typename... ArgTypes, typename... ArgTypes2>
static inline void
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

}
