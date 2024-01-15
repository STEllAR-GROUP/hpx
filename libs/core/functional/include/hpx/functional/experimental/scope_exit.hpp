//  Copyright (c) 2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  Copyright (c) 2020 Martin Moene
//  This is inspired by https://github.com/martinmoene/scope-lite

#pragma once

#include <hpx/config.hpp>

#if !defined(HPX_HAVE_CXX26_EXPERIMENTAL_SCOPE)

#include <type_traits>

namespace hpx::experimental {

    namespace detail {

        template <typename F>
        struct scope_exit
        {
            explicit constexpr scope_exit(F&& f) noexcept(
                std::is_nothrow_move_constructible_v<F> ||
                std::is_nothrow_copy_constructible_v<F>)
              : f(HPX_MOVE(f))
            {
            }

            explicit constexpr scope_exit(F const& f) noexcept(
                std::is_nothrow_copy_constructible_v<F>)
              : f(f)
            {
            }

            constexpr scope_exit(scope_exit&& rhs) noexcept(
                std::is_nothrow_move_constructible_v<F> ||
                std::is_nothrow_copy_constructible_v<F>)
              : f(HPX_MOVE(rhs.f))
              , active(rhs.active)
            {
                rhs.release();
            }

            scope_exit(scope_exit const&) = delete;
            scope_exit& operator=(scope_exit const&) = delete;
            scope_exit& operator=(scope_exit&& rhs) = delete;

            HPX_CONSTEXPR_DESTRUCTOR ~scope_exit() noexcept
            {
                if (active)
                {
                    f();
                }
            }

            constexpr void release() noexcept
            {
                active = false;
            }

        private:
            F f;
            bool active = true;
        };
    }    // namespace detail

    template <typename F>
    auto scope_exit(F&& f)
    {
        return detail::scope_exit<std::decay_t<F>>(HPX_FORWARD(F, f));
    }
}    // namespace hpx::experimental

#endif
