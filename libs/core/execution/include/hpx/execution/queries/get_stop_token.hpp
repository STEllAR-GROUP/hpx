//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/queries/read.hpp>
#include <hpx/execution_base/get_env.hpp>
#include <hpx/functional/detail/tag_fallback_invoke.hpp>
#include <hpx/synchronization/stop_token.hpp>

#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    // 1. execution::get_stop_token is used to ask an object for an
    //    associated stop token.
    //
    // 2. The name execution::get_stop_token denotes a customization
    //    point object. For some subexpression r, if the type of r is
    //    (possibly cv-qualified) no_env, then execution::get_stop_token(r)
    //    is ill-formed.
    //    Otherwise, it is expression equivalent to:
    //
    //    1. tag_invoke(execution::get_stop_token, as_const(r)), if this
    //       expression is well formed.
    //
    //          Mandates: The tag_invoke expression above is not potentially
    //          -throwing and its type satisfies stoppable_token.
    //
    //    2. Otherwise, never_stop_token{}.
    //
    // 3.  execution::get_stop_token() (with no arguments) is expression-
    //     equivalent to execution::read(execution::get_stop_token).
    //
    HPX_HOST_DEVICE_INLINE_CONSTEXPR_VARIABLE struct get_stop_token_t final
      : hpx::functional::detail::tag_fallback<get_stop_token_t>
    {
    private:
        template <typename Env>
        friend inline constexpr auto tag_fallback_invoke(
            get_stop_token_t, Env&&) noexcept
        {
            return hpx::experimental::never_stop_token();
        }

        friend inline constexpr auto tag_fallback_invoke(
            get_stop_token_t) noexcept;

    } get_stop_token{};

    constexpr auto tag_fallback_invoke(get_stop_token_t) noexcept
    {
        return hpx::execution::experimental::read(get_stop_token);
    }

    // Helper template allowing to extract the type of a stop_token extracted
    // from a receiver environment.
    template <typename T>
    using stop_token_of_t = std::remove_cv_t<
        std::remove_reference_t<decltype(get_stop_token(std::declval<T>()))>>;
}    // namespace hpx::execution::experimental
