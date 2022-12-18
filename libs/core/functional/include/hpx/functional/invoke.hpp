//  Copyright (c) 2013-2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/detail/invoke.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/type_support/void_guard.hpp>

#include <utility>

namespace hpx {

#define HPX_INVOKE_R(R, F, ...)                                                \
    (::hpx::util::void_guard<R>(), HPX_INVOKE(F, __VA_ARGS__))

    /// Invokes the given callable object f with the content of
    /// the argument pack vs
    ///
    /// \param f Requires to be a callable object.
    ///          If f is a member function pointer, the first argument in
    ///          the pack will be treated as the callee (this object).
    ///
    /// \param vs An arbitrary pack of arguments
    ///
    /// \returns The result of the callable object when it's called with
    ///          the given argument types.
    ///
    /// \throws std::exception like objects thrown by call to object f
    ///         with the argument types vs.
    ///
    /// \note This function is similar to `std::invoke` (C++17)
    template <typename F, typename... Ts>
    constexpr HPX_HOST_DEVICE util::invoke_result_t<F, Ts&&...> invoke(
        F&& f, Ts&&... vs) noexcept(noexcept(HPX_INVOKE(HPX_FORWARD(F, f),
        HPX_FORWARD(Ts, vs)...)))
    {
        return HPX_INVOKE(HPX_FORWARD(F, f), HPX_FORWARD(Ts, vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// \copydoc invoke
    ///
    /// \tparam R The result type of the function when it's called with the
    ///           content of the given argument types vs.
    template <typename R, typename F, typename... Ts>
    constexpr HPX_HOST_DEVICE R invoke_r(F&& f, Ts&&... vs) noexcept(
        noexcept(HPX_INVOKE(HPX_FORWARD(F, f), HPX_FORWARD(Ts, vs)...)))
    {
        return HPX_INVOKE_R(R, HPX_FORWARD(F, f), HPX_FORWARD(Ts, vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace functional {

        struct invoke
        {
            template <typename F, typename... Ts>
            constexpr HPX_HOST_DEVICE util::invoke_result_t<F, Ts&&...>
            operator()(F&& f, Ts&&... vs) const noexcept(
                noexcept(HPX_INVOKE(HPX_FORWARD(F, f), HPX_FORWARD(Ts, vs)...)))
            {
                return HPX_INVOKE(HPX_FORWARD(F, f), HPX_FORWARD(Ts, vs)...);
            }
        };

        template <typename R>
        struct invoke_r
        {
            template <typename F, typename... Ts>
            constexpr HPX_HOST_DEVICE R operator()(F&& f, Ts&&... vs) const
                noexcept(noexcept(
                    HPX_INVOKE(HPX_FORWARD(F, f), HPX_FORWARD(Ts, vs)...)))
            {
                return HPX_INVOKE_R(
                    R, HPX_FORWARD(F, f), HPX_FORWARD(Ts, vs)...);
            }
        };
    }    // namespace functional
}    // namespace hpx

/// \cond NOINTERN
namespace hpx::util {

    template <typename F, typename... Ts>
    HPX_DEPRECATED_V(
        1, 9, "hpx::util::invoke is deprecated, use hpx::invoke instead")
    constexpr HPX_HOST_DEVICE decltype(auto) invoke(F&& f, Ts&&... vs) noexcept(
        noexcept(HPX_INVOKE(HPX_FORWARD(F, f), HPX_FORWARD(Ts, vs)...)))
    {
        return hpx::invoke(HPX_FORWARD(F, f), HPX_FORWARD(Ts, vs)...);
    }

    template <typename R, typename F, typename... Ts>
    HPX_DEPRECATED_V(1, 9,
        "hpx::util::invoke_r<R> is deprecated, use hpx::invoke_r<R> instead")
    constexpr HPX_HOST_DEVICE R invoke_r(F&& f, Ts&&... vs) noexcept(
        noexcept(HPX_INVOKE(HPX_FORWARD(F, f), HPX_FORWARD(Ts, vs)...)))
    {
        return hpx::invoke_r<R>(HPX_FORWARD(F, f), HPX_FORWARD(Ts, vs)...);
    }

    namespace functional {

        using invoke HPX_DEPRECATED_V(1, 9,
            "hpx::util::functional::invoke is deprecated, use "
            "hpx::functional::invoke instead") = hpx::functional::invoke;

        template <typename R>
        using invoke_r HPX_DEPRECATED_V(1, 9,
            "hpx::util::functional::invoke_r<R> is deprecated, use "
            "hpx::functional::invoke_r<R> instead") =
            hpx::functional::invoke_r<R>;
    }    // namespace functional
}    // namespace hpx::util
/// \endcond
