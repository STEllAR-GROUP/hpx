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
            HPX_HOST_DEVICE HPX_FORCEINLINE
            typename util::result_of<F&&(Ts&&...)>::type
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
            HPX_HOST_DEVICE HPX_FORCEINLINE
            typename std::enable_if<
                std::is_base_of<C, typename std::decay<T0>::type>::value,
                typename util::result_of<F&&(T0&)>::type
            >::type
            operator()(F f, T0& v0)
            {
                return (v0.*f);
            }

            template <typename F, typename T0>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            typename std::enable_if<
                std::is_base_of<C, typename std::decay<T0>::type>::value
             && !std::is_lvalue_reference<T0>::value,
                typename util::result_of<F&&(T0&&)>::type
            >::type
            operator()(F f, T0&& v0)
            {
                return std::move(v0.*f);
            }

            // (*t0).*f
            template <typename F, typename T0>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            typename std::enable_if<
                !std::is_base_of<C, typename std::decay<T0>::type>::value,
                typename util::result_of<F&&(T0&&)>::type
            >::type
            operator()(F f, T0&& v0)
            {
                return (*this)(f, *std::forward<T0>(v0));
            }
        };

        template <typename R, typename C, typename ...Ps>
        struct invoke_impl<R (C::*)(Ps...)>
        {
            // (t0.*f)(t1, ..., tN)
            template <typename F, typename T0, typename ...Ts>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            typename std::enable_if<
                std::is_base_of<C, typename std::decay<T0>::type>::value,
                typename util::result_of<F&&(T0&&, Ts&&...)>::type
            >::type
            operator()(F f, T0&& v0, Ts&&... vs)
            {
                return (std::forward<T0>(v0).*f)(std::forward<Ts>(vs)...);
            }

            // ((*t0).*f)(t1, ..., tN)
            template <typename F, typename T0, typename ...Ts>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            typename std::enable_if<
                !std::is_base_of<C, typename std::decay<T0>::type>::value,
                typename util::result_of<F&&(T0&&, Ts&&...)>::type
            >::type
            operator()(F f, T0&& v0, Ts&&... vs)
            {
                return (*this)(f, *std::forward<T0>(v0), std::forward<Ts>(vs)...);
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
            HPX_HOST_DEVICE HPX_FORCEINLINE
            typename util::result_of<F&&(Ts&&...)>::type
            operator()(F f, Ts&&... vs)
            {
                return f.get()(std::forward<Ts>(vs)...);
            }
        };
    }

    template <typename F, typename ...Ts>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    typename util::result_of<F&&(Ts&&...)>::type
    invoke(F&& f, Ts&&... vs)
    {
        return detail::invoke_impl<typename std::decay<F>::type>()(
            std::forward<F>(f), std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename R, typename FD>
        struct invoke_guard_impl
        {
            // f(t0, t1, ..., tN)
            template <typename F, typename ...Ts>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            R operator()(F&& f, Ts&&... vs)
            {
                return std::forward<F>(f)(std::forward<Ts>(vs)...);
            }
        };

        template <typename FD>
        struct invoke_guard_impl<void,FD>
        {
            // f(t0, t1, ..., tN)
            template <typename F, typename ...Ts>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            void operator()(F&& f, Ts&&... vs)
            {
                std::forward<F>(f)(std::forward<Ts>(vs)...);
            }
        };

        template <typename R, typename M, typename C>
        struct invoke_guard_impl<R, M C::*>
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
                return (v0.*f);
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
                return std::move(v0.*f);
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
                return (*this)(f, *std::forward<T0>(v0));
            }
        };

        template <typename M, typename C>
        struct invoke_guard_impl<void, M C::*>
        {
            // t0.*f
            template <typename F, typename T0>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            typename std::enable_if<
                std::is_base_of<C, typename std::decay<T0>::type>::value,
                void
            >::type
            operator()(F f, T0& v0)
            {
                v0.*f;
            }

            template <typename F, typename T0>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            typename std::enable_if<
                std::is_base_of<C, typename std::decay<T0>::type>::value
             && !std::is_lvalue_reference<T0>::value,
                void
            >::type
            operator()(F f, T0&& v0)
            {
                v0.*f;
            }

            // (*t0).*f
            template <typename F, typename T0>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            typename std::enable_if<
                !std::is_base_of<C, typename std::decay<T0>::type>::value,
                void
            >::type
            operator()(F f, T0&& v0)
            {
                (*this)(f, *std::forward<T0>(v0));
            }
        };

        template <typename R, typename RR, typename C, typename ...Ps>
        struct invoke_guard_impl<R, RR (C::*)(Ps...)>
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
                return (std::forward<T0>(v0).*f)(std::forward<Ts>(vs)...);
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
                return (*this)(f, *std::forward<T0>(v0), std::forward<Ts>(vs)...);
            }
        };

        template <typename RR, typename C, typename ...Ps>
        struct invoke_guard_impl<void, RR (C::*)(Ps...)>
        {
            // (t0.*f)(t1, ..., tN)
            template <typename F, typename T0, typename ...Ts>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            typename std::enable_if<
                std::is_base_of<C, typename std::decay<T0>::type>::value,
                void
            >::type
            operator()(F f, T0&& v0, Ts&&... vs)
            {
                (std::forward<T0>(v0).*f)(std::forward<Ts>(vs)...);
            }

            // ((*t0).*f)(t1, ..., tN)
            template <typename F, typename T0, typename ...Ts>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            typename std::enable_if<
                !std::is_base_of<C, typename std::decay<T0>::type>::value,
                void
            >::type
            operator()(F f, T0&& v0, Ts&&... vs)
            {
                (*this)(f, *std::forward<T0>(v0), std::forward<Ts>(vs)...);
            }
        };

        template <typename R, typename RR, typename C, typename ...Ps>
        struct invoke_guard_impl<R, RR (C::*)(Ps...) const>
          : invoke_guard_impl<R,RR (C::*)(Ps...)>
        {};

        template <typename R, typename X>
        struct invoke_guard_impl<R, ::boost::reference_wrapper<X> >
          : invoke_guard_impl<R,X&>
        {
            // support boost::[c]ref, which is not callable as std::[c]ref
            template <typename F, typename ...Ts>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            R operator()(F f, Ts&&... vs)
            {
                return f.get()(std::forward<Ts>(vs)...);
            }
        };

        template <typename X>
        struct invoke_guard_impl<void, ::boost::reference_wrapper<X> >
          : invoke_guard_impl<void,X&>
        {
            // support boost::[c]ref, which is not callable as std::[c]ref
            template <typename F, typename ...Ts>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            void operator()(F f, Ts&&... vs)
            {
                f.get()(std::forward<Ts>(vs)...);
            }
        };

        template <typename R>
        struct invoke_guard
        {
            template <typename F, typename ...Ts>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            R operator()(F&& f, Ts&&... vs)
            {
                return detail::invoke_guard_impl<R,typename std::decay<F>::type>()(
                    std::forward<F>(f), std::forward<Ts>(vs)...);
            }
        };

        template <>
        struct invoke_guard<void>
        {
            template <typename F, typename ...Ts>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            void operator()(F&& f, Ts&&... vs)
            {
                detail::invoke_guard_impl<void,typename std::decay<F>::type>()(
                    std::forward<F>(f), std::forward<Ts>(vs)...);
            }
        };
    }

    template <typename R, typename F, typename ...Ts>
    HPX_HOST_DEVICE HPX_FORCEINLINE
    R invoke(F&& f, Ts&&... vs)
    {
        return detail::invoke_guard<R>()(
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
                return util::invoke(std::forward<F>(f),
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
                return util::invoke<R>(std::forward<F>(f),
                    std::forward<Ts>(vs)...);
            }
        };
    }
}}

#endif
