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
            std::void_t<decltype(HPX_INVOKE(
                std::declval<F>(), std::declval<Ts>()...))>> : std::true_type
        {
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename R, typename Enable = void>
        struct is_invocable_r_impl : std::false_type
        {
        };

        template <typename F, typename... Ts, typename R>
        struct is_invocable_r_impl<F(Ts...), R,
            std::void_t<decltype(
                HPX_INVOKE(std::declval<F>(), std::declval<Ts>()...))>>
          : std::integral_constant<bool,
                std::is_void_v<R> ||
                    std::is_convertible_v<decltype(HPX_INVOKE(std::declval<F>(),
                                              std::declval<Ts>()...)),
                        R>>
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
    inline constexpr bool is_invocable_v = is_invocable<F, Ts...>::value;

    template <typename R, typename F, typename... Ts>
    inline constexpr bool is_invocable_r_v = is_invocable_r<R, F, Ts...>::value;
}    // namespace hpx
