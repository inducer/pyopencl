#include <functional>
#include <utility>

#ifndef __PYOPENCL_FUNCTION_H
#define __PYOPENCL_FUNCTION_H

#if defined __GNUC__ &&  __GNUC__ > 3
#define PYOPENCL_INLINE inline __attribute__((__always_inline__))
#else
#define PYOPENCL_INLINE inline
#endif

namespace pyopencl {

template<int...>
struct seq {
};

template<int N, int... S>
struct gens : gens<N - 1, N - 1, S...> {
};

template<int ...S>
struct gens<0, S...> {
    typedef seq<S...> type;
};

template<typename Function, int... S, typename... Arg2>
static inline auto
_call_func(Function func, seq<S...>, std::tuple<Arg2...> &args)
    -> decltype(func(std::forward<Arg2>(std::get<S>(args))...))
{
    return func(static_cast<Arg2&&>(std::get<S>(args))...);
}

template<typename Function, typename T>
static inline auto
call_tuple(Function &&func, T args)
    -> decltype(_call_func(std::forward<Function>(func),
                           typename gens<std::tuple_size<T>::value>::type(),
                           args))
{
    return _call_func(std::forward<Function>(func),
                      typename gens<std::tuple_size<T>::value>::type(), args);
}

template<typename T>
using _ArgType = typename std::remove_reference<T>::type;

template<template<typename...> class Convert, typename... Types>
using _ArgPackBase = std::tuple<Convert<_ArgType<Types> >...>;

template<template<typename...> class Convert, typename... Types>
class ArgPack : public _ArgPackBase<Convert, Types...> {
    typedef _ArgPackBase<Convert, Types...> _base;
    template<typename T>
    static inline std::tuple<T&&>
    ensure_tuple(T &&v)
    {
        return std::tuple<T&&>(std::forward<T>(v));
    }
    template<typename... T>
    static inline std::tuple<T...>&&
    ensure_tuple(std::tuple<T...> &&t)
    {
        return std::move(t);
    }

    template<typename T>
    using ArgConvert = Convert<_ArgType<T> >;
    template<template<typename...> class Getter, int... S>
    inline auto
    __get(seq<S...>)
    -> decltype(std::tuple_cat(ensure_tuple(Getter<ArgConvert<Types> >::get(
                                                std::get<S>(*(_base*)this)))...))
    {
        return std::tuple_cat(ensure_tuple(Getter<ArgConvert<Types> >::get(
                                               std::get<S>(*(_base*)this)))...);
    }
public:
    template<typename... Types2>
    ArgPack(Types2&&... arg_orig)
        : _base(ArgConvert<Types2>(arg_orig)...)
    {
    }
    ArgPack(ArgPack &&other)
        : _base(static_cast<_base&&>(other))
    {
    }
    // GCC Bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=57543
    template<template<typename...> class Getter>
    inline auto
    get() -> decltype(this->__get<Getter>(
                          typename gens<sizeof...(Types)>::type()))
    {
        return __get<Getter>(typename gens<sizeof...(Types)>::type());
    }
    template<template<typename...> class Getter, typename Func>
    inline auto
    call(Func func)
        -> decltype(call_tuple(func, this->get<Getter>()))
    {
        return call_tuple(func, this->get<Getter>());
    }
};

template<template<typename...> class Convert, typename... Types>
static inline ArgPack<Convert, _ArgType<Types>...>
make_argpack(Types&&... args)
{
    return ArgPack<Convert, _ArgType<Types>...>(
        std::forward<Types>(args)...);
}

}

#endif
