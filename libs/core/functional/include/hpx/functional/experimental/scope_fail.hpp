//  Copyright (c) 2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  Copyright (c) 2020 Martin Moene
//  This is inspired by https://github.com/martinmoene/scope-lite

/// \page hpx::experimental::scope_fail
/// \headerfile hpx/experimental/scope.hpp

#pragma once

#include <hpx/config.hpp>

#if !defined(HPX_HAVE_CXX26_EXPERIMENTAL_SCOPE)

#include <exception>
#include <limits>
#include <type_traits>

namespace hpx::experimental {

    namespace detail {

        template <typename F>
        struct scope_fail
        {
            explicit constexpr scope_fail(F&& f) noexcept(
                std::is_nothrow_move_constructible_v<F> ||
                std::is_nothrow_copy_constructible_v<F>)
              : f(HPX_MOVE(f))
              , active(std::uncaught_exceptions())
            {
            }

            explicit constexpr scope_fail(F const& f) noexcept(
                std::is_nothrow_copy_constructible_v<F>)
              : f(f)
              , active(std::uncaught_exceptions())
            {
            }

            constexpr scope_fail(scope_fail&& rhs) noexcept(
                std::is_nothrow_move_constructible_v<F> ||
                std::is_nothrow_copy_constructible_v<F>)
              : f(HPX_MOVE(rhs.f))
              , active(rhs.active)
            {
                rhs.release();
            }

            scope_fail(scope_fail const&) = delete;
            scope_fail& operator=(scope_fail const&) = delete;
            scope_fail& operator=(scope_fail&& rhs) = delete;

            HPX_CONSTEXPR_DESTRUCTOR ~scope_fail() noexcept
            {
                if (active < std::uncaught_exceptions())
                {
                    f();
                }
            }

            constexpr void release() noexcept
            {
                active = (std::numeric_limits<int>::max)();
            }

        private:
            F f;
            int active;
        };
    }    // namespace detail

    /// \brief The class template scope_fail is a general-purpose scope guard
    ///        intended to call its exit function when a scope is exited via
    ///        an exception.
    ///
    /// \tparam F type of stored exit function
    ///
    /// \param f stored exit function
    template <typename F>
    auto scope_fail(F&& f)
    {
        return detail::scope_fail<std::decay_t<F>>(HPX_FORWARD(F, f));
    }
}    // namespace hpx::experimental

#endif
