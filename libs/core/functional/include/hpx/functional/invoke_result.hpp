//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/detail/invoke.hpp>
#include <hpx/type_support/always_void.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace util {
    namespace detail {
        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Enable = void>
        struct invoke_result_impl
        {
        };

        template <typename F, typename... Ts>
        struct invoke_result_impl<F(Ts...),
            typename util::always_void<decltype(
                HPX_INVOKE(std::declval<F>(), std::declval<Ts>()...))>::type>
        {
            using type =
                decltype(HPX_INVOKE(std::declval<F>(), std::declval<Ts>()...));
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
    struct result_of<F(Ts...)> : detail::invoke_result_impl<F(Ts...)>
    {
    };
#if defined(HPX_GCC_VERSION)
#pragma GCC diagnostic pop
#endif

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    struct invoke_result : detail::invoke_result_impl<F && (Ts && ...)>
    {
    };
}}    // namespace hpx::util
