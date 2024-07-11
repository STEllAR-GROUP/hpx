//  Copyright (c) 2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  Copyright (c) 2020 Martin Moene
//  This is inspired by https://github.com/martinmoene/scope-lite

/// \page hpx::experimental::scope_success
/// \headerfile hpx/experimental/scope.hpp

#pragma once

#include <hpx/config.hpp>

#if !defined(HPX_HAVE_CXX26_EXPERIMENTAL_SCOPE)

#include <exception>
#include <type_traits>

namespace hpx::experimental {

    namespace detail {

        template <typename F>
        struct scope_success
        {
            explicit constexpr scope_success(F&& f) noexcept(
                std::is_nothrow_move_constructible_v<F> ||
                std::is_nothrow_copy_constructible_v<F>)
              : f(HPX_MOVE(f))
              , active(std::uncaught_exceptions())
            {
            }

            explicit constexpr scope_success(F const& f) noexcept(
                std::is_nothrow_copy_constructible_v<F>)
              : f(f)
              , active(std::uncaught_exceptions())
            {
            }

            constexpr scope_success(scope_success&& rhs) noexcept(
                std::is_nothrow_move_constructible_v<F> ||
                std::is_nothrow_copy_constructible_v<F>)
              : f(HPX_MOVE(rhs.f))
              , active(rhs.active)
            {
                rhs.release();
            }

            scope_success(scope_success const&) = delete;
            scope_success& operator=(scope_success const&) = delete;
            scope_success& operator=(scope_success&& rhs) = delete;

            HPX_CONSTEXPR_DESTRUCTOR ~scope_success() noexcept(
                noexcept(this->f()))
            {
                if (active >= std::uncaught_exceptions())
                {
                    f();
                }
            }

            constexpr void release() noexcept
            {
                active = -1;
            }

        private:
            F f;
            int active;
        };
    }    // namespace detail

    /// \brief The class template scope_success is a general-purpose scope
    ///        guard intended to call its exit function when a scope is
    ///        normally exited.
    ///
    /// \tparam F type of stored exit function
    ///
    /// \param f stored exit function
    template <typename F>
    auto scope_success(F&& f)
    {
        return detail::scope_success<std::decay_t<F>>(HPX_FORWARD(F, f));
    }
}    // namespace hpx::experimental

#endif
