#ifndef __PYOPENCL_PYHELPER_H
#define __PYOPENCL_PYHELPER_H

#include "wrap_cl.h"
#include "function.h"

template<typename _Signature>
class WrapFunc;

template<typename Ret, typename... Args>
class WrapFunc<Ret(Args...)> {
    typedef Ret (*_FuncType)(Args...);
    _FuncType m_func;
    static PYOPENCL_INLINE _FuncType
    check_func(_FuncType f)
    {
        return f ? f : ([] (Args...) {return Ret();});
    }
public:
    WrapFunc(_FuncType func=nullptr)
        : m_func(check_func(func))
    {}
    Ret
    operator()(Args... args)
    {
        return m_func(std::forward<Args>(args)...);
    }
    WrapFunc&
    operator=(_FuncType func)
    {
        m_func = check_func(func);
        return *this;
    }
};

namespace py {
extern WrapFunc<int()> gc;
extern WrapFunc<void*(void*)> ref;
extern WrapFunc<void(void*)> deref;
extern WrapFunc<void(void*, cl_int)> call;
}

#endif
