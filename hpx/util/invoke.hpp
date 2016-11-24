//  Copyright (c) 2013-2015 Agustin Berge
//  Copyright (c) 2016 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedinclude:boost/ref.hpp
// hpxinspect:nodeprecatedname:boost::reference_wrapper

#ifndef HPX_UTIL_INVOKE_HPP
#define HPX_UTIL_INVOKE_HPP

#include <hpx/config.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/void_guard.hpp>

#include <boost/ref.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename R,typename FD>
        struct invoke_impl
        {
            // f(t0, t1, ..., tN)
            template <typename F, typename ...Ts>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            R operator()(F&& f, Ts&&... vs)
            {
                return hpx::util::void_guard<R>(),
                    std::forward<F>(f)(std::forward<Ts>(vs)...);
            }
        };


        template <typename R, typename M, typename C>
        struct invoke_impl<R, M C::*>
        {
            // t0.*f
            template <typename F, typename T0>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            typename std::enable_if<
                std::is_base_of<C, typename std::decay<T0>::type>::value,
                R
            >::type
            operator()(F f, T0& v0)
            {
                return hpx::util::void_guard<R>(), (v0.*f);
            }

            template <typename F, typename T0>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            typename std::enable_if<
                std::is_base_of<C, typename std::decay<T0>::type>::value
             && !std::is_lvalue_reference<T0>::value,
                R
            >::type
            operator()(F f, T0&& v0)
            {
                return hpx::util::void_guard<R>(), std::move(v0.*f);
            }

            // (*t0).*f
            template <typename F, typename T0>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            typename std::enable_if<
                !std::is_base_of<C, typename std::decay<T0>::type>::value,
                R
            >::type
            operator()(F f, T0&& v0)
            {
                return hpx::util::void_guard<R>(), (*this)(f, *std::forward<T0>(v0));
            }
        };

        template <typename R, typename RR, typename C, typename ...Ps>
        struct invoke_impl<R, RR (C::*)(Ps...)>
        {
            // (t0.*f)(t1, ..., tN)
            template <typename F, typename T0, typename ...Ts>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            typename std::enable_if<
                std::is_base_of<C, typename std::decay<T0>::type>::value,
                R
            >::type
            operator()(F f, T0&& v0, Ts&&... vs)
            {
                return hpx::util::void_guard<R>(),
                    (std::forward<T0>(v0).*f)(std::forward<Ts>(vs)...);
            }

            // ((*t0).*f)(t1, ..., tN)
            template <typename F, typename T0, typename ...Ts>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            typename std::enable_if<
                !std::is_base_of<C, typename std::decay<T0>::type>::value,
                R
            >::type
            operator()(F f, T0&& v0, Ts&&... vs)
            {
                return hpx::util::void_guard<R>(),
                    (*this)(f, *std::forward<T0>(v0), std::forward<Ts>(vs)...);
            }
        };

        template <typename R, typename RR, typename C, typename ...Ps>
        struct invoke_impl<R, RR (C::*)(Ps...) const>
          : invoke_impl<R, RR (C::*)(Ps...)>
        {};

        template <typename R,typename X>
        struct invoke_impl<R,::boost::reference_wrapper<X>>
          : invoke_impl<R,X&>
        {
            // support boost::[c]ref, which is not callable as std::[c]ref
            template <typename F, typename ...Ts>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            R operator()(F f, Ts&&... vs)
            {
                return hpx::util::void_guard<R>(), f.get()(std::forward<Ts>(vs)...);
            }
        };
    }

    template <typename F, typename ...Ts>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    typename util::result_of<F&&(Ts&&...)>::type
    invoke(F&& f, Ts&&... vs)
    {
        typedef typename util::result_of<F&&(Ts&&...)>::type R;

        return detail::invoke_impl<R,typename std::decay<F>::type>()(
            std::forward<F>(f), std::forward<Ts>(vs)...);
    }

    template <typename R, typename F, typename ...Ts>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    R invoke_r(F&& f, Ts&&... vs)
    {
        return detail::invoke_impl<R,typename std::decay<F>::type>()(
            std::forward<F>(f), std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace functional
    {
        struct invoke
        {
            template <typename F, typename... Ts>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            typename util::result_of<F&&(Ts &&...)>::type
            operator()(F && f, Ts &&... vs)
            {
                typedef typename util::result_of<F&&(Ts&&...)>::type R;

                return hpx::util::void_guard<R>(), util::invoke(std::forward<F>(f),
                    std::forward<Ts>(vs)...);
            }
        };

        template <typename R>
        struct invoke_r
        {
            template <typename F, typename... Ts>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            R operator()(F && f, Ts &&... vs)
            {
                return hpx::util::void_guard<R>(), util::invoke_r<R>(std::forward<F>(f),
                    std::forward<Ts>(vs)...);
            }
        };
    }
}}

#endif
