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

namespace hpx::execution::experimental {

    // 1. execution::get_allocator is used to ask an object for its associated
    //    allocator.
    //
    // 2. The name execution::get_allocator denotes a customization point object.
    //     For some subexpression r, if the type of r is (possibly cv-qualified)
    //     no_env, then execution::get_allocator(r) is ill-formed.
    //     Otherwise, it is expression equivalent to:
    //
    //    1. tag_invoke(execution::get_allocator, as_const(r)), if this expression
    //       is well formed.
    //          Mandates: The tag_invoke expression above is not potentially-
    //          throwing and its type satisfies Allocator.
    //
    //    2. Otherwise, execution::get_allocator(r) is ill-formed.
    //
    // 3. execution::get_allocator() (with no arguments) is expression-equivalent
    //    to execution::read(execution::get_allocator).
    //
    inline constexpr struct get_allocator_t final
      : hpx::functional::detail::tag_fallback<get_allocator_t>
    {
    private:
        friend inline constexpr auto tag_fallback_invoke(
            get_allocator_t) noexcept;

    } get_allocator{};

    constexpr auto tag_fallback_invoke(get_allocator_t) noexcept
    {
        return hpx::execution::experimental::read(get_allocator);
    }
}    // namespace hpx::execution::experimental
