//  Copyright (c) 2013-2015 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_INVOKE_HPP
#define HPX_UTIL_INVOKE_HPP

#include <hpx/config.hpp>
#include <hpx/util/result_of.hpp>

#include <boost/ref.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename FD>
        struct invoke_impl
        {
            // f(t0, t1, ..., tN)
            template <typename F, typename ...Ts>
            inline typename util::result_of<F&&(Ts&&...)>::type
            operator()(F&& f, Ts&&... vs)
            {
                return std::forward<F>(f)(std::forward<Ts>(vs)...);
            }
        };

        template <typename M, typename C>
        struct invoke_impl<M C::*>
        {
            // t0.*f
            template <typename F, typename T0>
            inline typename std::enable_if<
                std::is_base_of<C, typename std::decay<T0>::type>::value,
                typename util::result_of<F&&(T0&&)>::type
            >::type
            operator()(F f, T0&& v0)
            {
                return (std::forward<T0>(v0).*f);
            }

            // (*t0).*f
            template <typename F, typename T0>
            inline typename std::enable_if<
                !std::is_base_of<C, typename std::decay<T0>::type>::value,
                typename util::result_of<F&&(T0&&)>::type
            >::type
            operator()(F f, T0&& v0)
            {
                return ((*std::forward<T0>(v0)).*f);
            }
        };

        template <typename R, typename C, typename ...Ps>
        struct invoke_impl<R (C::*)(Ps...)>
        {
            // (t0.*f)(t1, ..., tN)
            template <typename F, typename T0, typename ...Ts>
            inline typename std::enable_if<
                std::is_base_of<C, typename std::decay<T0>::type>::value,
                typename util::result_of<F&&(T0&&, Ts&&...)>::type
            >::type
            operator()(F f, T0&& v0, Ts&&... vs)
            {
                return (std::forward<T0>(v0).*f)(std::forward<Ts>(vs)...);
            }

            // ((*t0).*f)(t1, ..., tN)
            template <typename F, typename T0, typename ...Ts>
            inline typename std::enable_if<
                !std::is_base_of<C, typename std::decay<T0>::type>::value,
                typename util::result_of<F&&(T0&&, Ts&&...)>::type
            >::type
            operator()(F f, T0&& v0, Ts&&... vs)
            {
                return ((*std::forward<T0>(v0)).*f)(std::forward<Ts>(vs)...);
            }
        };

        template <typename R, typename C, typename ...Ps>
        struct invoke_impl<R (C::*)(Ps...) const>
          : invoke_impl<R (C::*)(Ps...)>
        {};

        template <typename X>
        struct invoke_impl< ::boost::reference_wrapper<X> >
          : invoke_impl<X&>
        {
            // support boost::[c]ref, which is not callable as std::[c]ref
            template <typename F, typename ...Ts>
            inline typename util::result_of<F&&(Ts&&...)>::type
            operator()(F f, Ts&&... vs)
            {
                return f.get()(std::forward<Ts>(vs)...);
            }
        };
    }

    template <typename F, typename ...Ts>
    inline typename util::result_of<F&&(Ts&&...)>::type
    invoke(F&& f, Ts&&... vs)
    {
        return detail::invoke_impl<typename std::decay<F>::type>()(
            std::forward<F>(f), std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename R>
        struct invoke_guard
        {
            template <typename F, typename ...Ts>
            inline R operator()(F&& f, Ts&&... vs)
            {
                return detail::invoke_impl<typename std::decay<F>::type>()(
                    std::forward<F>(f), std::forward<Ts>(vs)...);
            }
        };

        template <>
        struct invoke_guard<void>
        {
            template <typename F, typename ...Ts>
            inline void operator()(F&& f, Ts&&... vs)
            {
                detail::invoke_impl<typename std::decay<F>::type>()(
                    std::forward<F>(f), std::forward<Ts>(vs)...);
            }
        };
    }

    template <typename R, typename F, typename ...Ts>
    inline R invoke(F&& f, Ts&&... vs)
    {
        return detail::invoke_guard<R>()(
            std::forward<F>(f), std::forward<Ts>(vs)...);
    }
}}

#endif
