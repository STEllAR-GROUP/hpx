//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace util {
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        // f(t0, t1, ..., tN)
        template <typename T>
        struct result_of_function_object : std::result_of<T>
        {
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T>
        struct result_of_member_pointer_impl;

        // t0.*f
        template <typename R, typename C>
        struct result_of_member_pointer_impl<R C::*>
        {
            R& operator()(C&);
            R const& operator()(C const&);
            R&& operator()(C&&);
            R const&& operator()(C const&&);
        };

        // (t0.*f)(t1, ..., tN)
        template <typename R, typename C, typename... Ps>
        struct result_of_member_pointer_impl<R (C::*)(Ps...)>
        {
            R operator()(C&, Ps...);
            R operator()(C&&, Ps...);
        };

        template <typename R, typename C, typename... Ps>
        struct result_of_member_pointer_impl<R (C::*)(Ps...) const>
        {
            R operator()(C const&, Ps...);
            R operator()(C const&&, Ps...);
        };

        ///////////////////////////////////////////////////////////////////////
        namespace has_dereference_impl {
            struct fallback
            {
                template <typename T>
                fallback(T const&)
                {
                }
            };

            fallback operator*(fallback);

            template <typename T>
            struct has_dereference
            {
                static bool const value =
                    !std::is_same<decltype(*std::declval<T>()),
                        fallback>::value;
            };
        }    // namespace has_dereference_impl
        using has_dereference_impl::has_dereference;

        template <typename C, typename T, typename Enable = void>
        struct result_of_member_pointer
        {
        };

        // t0.*f, (t0.*f)(t1, ..., tN)
        template <typename C, typename F, typename T0, typename... Ts>
        struct result_of_member_pointer<C, F(T0, Ts...),
            typename std::enable_if<
                std::is_base_of<C, typename std::decay<T0>::type>::value>::type>
          : result_of_function_object<result_of_member_pointer_impl<
                typename std::decay<F>::type>(T0, Ts...)>
        {
        };

        // (*t0).*f, ((*t0).*f)(t1, ..., tN)
        template <typename C, typename F, typename T0, typename... Ts>
        struct result_of_member_pointer<C, F(T0, Ts...),
            typename std::enable_if<std::enable_if<
                !std::is_base_of<C, typename std::decay<T0>::type>::value,
                has_dereference<T0>>::type::value>::type>
          : result_of_function_object<
                result_of_member_pointer_impl<typename std::decay<F>::type>(
                    decltype(*std::declval<T0>()), Ts...)>
        {
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename FD, typename T>
        struct invoke_result_impl : result_of_function_object<T>
        {
        };

        template <typename M, typename C, typename F, typename... Ts>
        struct invoke_result_impl<M C::*, F(Ts...)>
          : result_of_member_pointer<C, M C::*(Ts...)>
        {
        };

        template <typename R, typename C, typename... Ps, typename F,
            typename... Ts>
        struct invoke_result_impl<R (C::*)(Ps...), F(Ts...)>
          : result_of_member_pointer<C, R (C::*(Ts...))(Ps...)>
        {
        };

        template <typename R, typename C, typename... Ps, typename F,
            typename... Ts>
        struct invoke_result_impl<R (C::*)(Ps...) const, F(Ts...)>
          : result_of_member_pointer<C, R (C::*(Ts...))(Ps...) const>
        {
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct HPX_DEPRECATED_V(
        1, 5, "result_of is deprecated, use invoke_result instead.") result_of;

#if defined(HPX_GCC_VERSION)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    template <typename F, typename... Ts>
    struct result_of<F(Ts...)>
      : detail::invoke_result_impl<typename std::decay<F>::type, F(Ts...)>
    {
    };
#if defined(HPX_GCC_VERSION)
#pragma GCC diagnostic pop
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    struct invoke_result
      : detail::invoke_result_impl<typename std::decay<F>::type,
            F && (Ts && ...)>
    {
    };
}}    // namespace hpx::util
