//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_INVOKE_HPP
#define HPX_UTIL_INVOKE_HPP

#include <hpx/config.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/move.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/void_guard.hpp>
#include <hpx/util/detail/pack.hpp>

#include <boost/ref.hpp>
#include <boost/type_traits/is_function.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/type_traits/is_member_function_pointer.hpp>
#include <boost/type_traits/is_member_pointer.hpp>
#include <boost/type_traits/is_pointer.hpp>
#include <boost/type_traits/remove_pointer.hpp>
#include <boost/utility/enable_if.hpp>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    struct invoke_result_of
      : hpx::util::result_of<F>
    {};

    ///////////////////////////////////////////////////////////////////////////
    // (t1.*f)(t2, ..., tN)
    template <typename R, typename FR, typename C, typename ...Ps
      , typename T, typename ...Ts>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     && !boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(Ps...), T&& t, Ts&&... vs)
    {
        return util::void_guard<R>(),
            (std::forward<T>(t).*f)(std::forward<Ts>(vs)...);
    }

    template <typename R, typename FR, typename C, typename ...Ps
      , typename T, typename ...Ts>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     && !boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(Ps...) const, T&& t, Ts&&... vs)
    {
        return util::void_guard<R>(),
            (std::forward<T>(t).*f)(std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    // (t1.get().*f)(t2, ..., tN)
    template <typename R, typename FR, typename C, typename ...Ps
      , typename T, typename ...Ts>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(Ps...), T&& t, Ts&&... vs)
    {
        return
            util::void_guard<R>(), ((t.get()).*f)
                (std::forward<Ts>(vs)...);
    }

    template <typename R, typename FR, typename C, typename ...Ps
      , typename T, typename ...Ts>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(Ps...) const, T&& t, Ts&&... vs)
    {
        return util::void_guard<R>(),
            ((t.get()).*f)(std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    // ((*t1).*f)(t2, ..., tN)
    template <typename R, typename FR, typename C, typename ...Ps
      , typename T, typename ...Ts>
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     || boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(Ps...), T&& t, Ts&&... vs)
    {
        return util::void_guard<R>(),
            ((*std::forward<T>(t)).*f)(std::forward<Ts>(vs)...);
    }

    template <typename R, typename FR, typename C, typename ...Ps
      , typename T, typename ...Ts>
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     || boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR (C::*f)(Ps...) const, T&& t, Ts&&... vs)
    {
        return util::void_guard<R>(),
            ((*std::forward<T>(t)).*f)(std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    // t1.*f
    template <typename R, typename FR, typename C, typename T>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     && !boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR C::*f, T && t)
    {
        return util::void_guard<R>(), (std::forward<T>(t).*f);
    }

    ///////////////////////////////////////////////////////////////////////////
    // t1.get().*f
    template <typename R, typename FR, typename C, typename T>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR C::*f, T && t)
    {
        return util::void_guard<R>(), ((t.get()).*f);
    }

    ///////////////////////////////////////////////////////////////////////////
    // (*t1).*f
    template <typename R, typename FR, typename C, typename T>
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_base_of<C, typename util::decay<T>::type>::value
     || boost::is_reference_wrapper<typename util::decay<T>::type>::value
      , R
    >::type
    invoke_r(FR C::*f, T && t)
    {
        return util::void_guard<R>(), ((*std::forward<T>(t)).*f);
    }

    ///////////////////////////////////////////////////////////////////////////
    // f(t1, t2, ..., tN)
    template <typename R, typename FR, typename ...Ps, typename ...Ts>
    HPX_MAYBE_FORCEINLINE
    R
    invoke_r(FR (*f)(Ps...), Ts&&... vs)
    {
        return util::void_guard<R>(),
            f(std::forward<Ts>(vs)...);
    }

    template <typename R, typename F, typename ...Ts>
    HPX_MAYBE_FORCEINLINE
    typename boost::enable_if_c<
        boost::is_reference_wrapper<typename util::decay<F>::type>::value
      , R
    >::type
    invoke_r(F&& f, Ts&&... vs)
    {
        return util::void_guard<R>(),
            (f.get())(std::forward<Ts>(vs)...);
    }

    template <typename R, typename F, typename ...Ts>
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_function<typename boost::remove_pointer<typename
                           util::decay<F>::type>::type>::value
     || boost::is_member_pointer<typename util::decay<F>::type>::value
     || boost::is_reference_wrapper<typename util::decay<F>::type>::value
      , R
    >::type
    invoke_r(F&& f, Ts&&... vs)
    {
        return util::void_guard<R>(),
            std::forward<F>(f)(std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename R, typename ...Ps, typename ...Ts>
    HPX_MAYBE_FORCEINLINE
    R
    invoke(R (*f)(Ps...), Ts&&... vs)
    {
        return util::invoke_r<R>(
            f, std::forward<Ts>(vs)...);
    }

    template <typename R, typename C, typename ...Ps
      , typename T, typename ...Ts>
    HPX_MAYBE_FORCEINLINE
    R
    invoke(R (C::*f)(Ps...), T&& t, Ts&&... vs)
    {
        return util::invoke_r<R>(
            f, std::forward<T>(t), std::forward<Ts>(vs)...);
    }

    template <typename R, typename C, typename ...Ps
      , typename T, typename ...Ts>
    HPX_MAYBE_FORCEINLINE
    R
    invoke(R (C::*f)(Ps...) const, T&& t, Ts&&... vs)
    {
        return util::invoke_r<R>(
            f, std::forward<T>(t), std::forward<Ts>(vs)...);
    }

    template <typename F, typename ...Ts>
    HPX_MAYBE_FORCEINLINE
    typename boost::disable_if_c<
        boost::is_function<typename boost::remove_pointer<typename
                           util::decay<F>::type>::type>::value
     || boost::is_member_function_pointer<typename util::decay<F>::type>::value
      , typename invoke_result_of<F(Ts...)>::type
    >::type
    invoke(F&& f, Ts&&... vs)
    {
        typedef typename invoke_result_of<F(Ts...)>::type result_type;

        return util::invoke_r<result_type>(
            std::forward<F>(f), std::forward<Ts>(vs)...);
    }
}}

#endif
