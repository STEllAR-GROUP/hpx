//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedname:is_callable

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/detail/invoke.hpp>
#include <hpx/type_support/always_void.hpp>

#include <type_traits>
#include <utility>

namespace hpx {
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Enable = void>
        struct is_invocable_impl : std::false_type
        {
            static_assert(std::is_function<T>::value,
                "Argument must be of the form F(Ts...)");
        };

        template <typename F, typename... Ts>
        struct is_invocable_impl<F(Ts...),
            typename util::always_void<decltype(HPX_INVOKE(std::declval<F>(),
                std::declval<Ts>()...))>::type> : std::true_type
        {
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename R, typename Enable = void>
        struct is_invocable_r_impl : std::false_type
        {
        };

        template <typename F, typename... Ts, typename R>
        struct is_invocable_r_impl<F(Ts...), R,
            typename util::always_void<decltype(
                HPX_INVOKE(std::declval<F>(), std::declval<Ts>()...))>::type>
          : std::integral_constant<bool,
                std::is_void<R>::value ||
                    std::is_convertible<decltype(HPX_INVOKE(std::declval<F>(),
                                            std::declval<Ts>()...)),
                        R>::value>
        {
        };
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    struct is_invocable : hpx::detail::is_invocable_impl<F && (Ts && ...), void>
    {
    };

    template <typename R, typename F, typename... Ts>
    struct is_invocable_r
      : hpx::detail::is_invocable_r_impl<F && (Ts && ...), R>
    {
    };

    template <typename F, typename... Ts>
    HPX_INLINE_CONSTEXPR_VARIABLE bool is_invocable_v =
        is_invocable<F, Ts...>::value;

    template <typename R, typename F, typename... Ts>
    HPX_INLINE_CONSTEXPR_VARIABLE bool is_invocable_r_v =
        is_invocable_r<R, F, Ts...>::value;

    namespace traits {
        template <typename F, typename... Ts>
        using is_invocable HPX_DEPRECATED_V(1, 6,
            "hpx::traits::is_invocable is deprecated, use hpx::is_invocable "
            "instead") = hpx::is_invocable<F, Ts...>;

        template <typename R, typename F, typename... Ts>
        using is_invocable_r HPX_DEPRECATED_V(1, 6,
            "hpx::traits::is_invocable_r is deprecated, use "
            "hpx::is_invocable_r instead") = hpx::is_invocable_r<R, F, Ts...>;
    }    // namespace traits
}    // namespace hpx
