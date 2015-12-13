#include <functional>
#include <utility>

#ifndef __PYOPENCL_FUNCTION_H
#define __PYOPENCL_FUNCTION_H

#if defined __GNUC__ &&  __GNUC__ > 3
#define PYOPENCL_INLINE inline __attribute__((__always_inline__))
#else
#define PYOPENCL_INLINE inline
#endif

template<typename T>
using rm_ref_t = typename std::remove_reference<T>::type;
template<typename T>
using rm_const_t = typename std::remove_const<T>::type;
template<bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

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
static PYOPENCL_INLINE auto
_call_func(Function func, seq<S...>, std::tuple<Arg2...> &args)
    -> decltype(func(std::forward<Arg2>(std::get<S>(args))...))
{
    return func(static_cast<Arg2&&>(std::get<S>(args))...);
}

template<typename Function, typename T>
static PYOPENCL_INLINE auto
call_tuple(Function &&func, T &&args)
    -> decltype(_call_func(std::forward<Function>(func),
                           typename gens<std::tuple_size<T>::value>::type(),
                           args))
{
    return _call_func(std::forward<Function>(func),
                      typename gens<std::tuple_size<T>::value>::type(), args);
}

template<template<typename...> class Convert, typename... Types>
using _ArgPackBase = std::tuple<Convert<typename std::remove_reference<Types>::type>...>;

template<template<typename...> class Convert, typename... Types>
class ArgPack : public _ArgPackBase<Convert, Types...> {
public:
    typedef _ArgPackBase<Convert, Types...> tuple_base;
private:
    template<typename T>
    static PYOPENCL_INLINE std::tuple<T>
    ensure_tuple(T &&v)
    {
        return std::tuple<T>(std::forward<T>(v));
    }
    template<typename... T>
    static PYOPENCL_INLINE std::tuple<T...>
    ensure_tuple(std::tuple<T...> &&t)
    {
        return t;
    }

    template<typename T>
    using ArgConvert = Convert<rm_ref_t<T> >;
    template<template<typename...> class Getter, int... S>
    PYOPENCL_INLINE auto
    __get(seq<S...>)
    -> decltype(std::tuple_cat(
                    ensure_tuple(Getter<ArgConvert<Types> >::get(
                                     std::get<S>(*(tuple_base*)this)))...))
    {
        return std::tuple_cat(
            ensure_tuple(Getter<ArgConvert<Types> >::get(
                             std::get<S>(*(tuple_base*)this)))...);
    }
public:
    template<typename... Types2>
    ArgPack(Types2&&... arg_orig)
        : tuple_base(ArgConvert<rm_ref_t<Types> >(arg_orig)...)
    {
    }
    ArgPack(ArgPack<Convert, Types...> &&other)
        : tuple_base(static_cast<tuple_base&&>(other))
    {
    }
    // GCC Bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=57543
    template<template<typename...> class Getter>
    PYOPENCL_INLINE auto
    get() -> decltype(this->__get<Getter>(
                          typename gens<sizeof...(Types)>::type()))
    {
        return __get<Getter>(typename gens<sizeof...(Types)>::type());
    }
    template<template<typename...> class Getter, typename Func>
    PYOPENCL_INLINE auto
    call(Func func) -> decltype(call_tuple(func, this->get<Getter>()))
    {
        return call_tuple(func, this->get<Getter>());
    }
};

template<template<typename...> class Convert, typename... Types>
static PYOPENCL_INLINE ArgPack<Convert, rm_ref_t<Types>...>
make_argpack(Types&&... args)
{
    return ArgPack<Convert, rm_ref_t<Types>...>(std::forward<Types>(args)...);
}

#endif
