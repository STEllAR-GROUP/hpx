//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/detail/invoke.hpp>

#include <type_traits>
#include <utility>

namespace hpx::util {

    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Enable = void>
        struct invoke_result_impl
        {
        };

        template <typename F, typename... Ts>
        struct invoke_result_impl<F(Ts...),
            std::void_t<decltype(
                HPX_INVOKE(std::declval<F>(), std::declval<Ts>()...))>>
        {
            using type =
                decltype(HPX_INVOKE(std::declval<F>(), std::declval<Ts>()...));
        };

    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    struct invoke_result : detail::invoke_result_impl<F && (Ts && ...)>
    {
    };

    template <typename F, typename... Ts>
    using invoke_result_t = typename invoke_result<F, Ts...>::type;
}    // namespace hpx::util
