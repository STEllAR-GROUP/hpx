//  Copyright (c) 2026 The STE||AR-Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/traits/future_traits.hpp>
#include <hpx/futures/traits/is_future.hpp>
#include <hpx/modules/functional.hpp>

#include <concepts>
#include <exception>
#include <type_traits>
#include <utility>

namespace hpx { namespace futures {

    // ---------------------------------------------------------------
    // transform: apply f to the value inside a ready future
    // ---------------------------------------------------------------
    template <typename Future, typename F,
        typename R = typename hpx::traits::future_traits<
            std::decay_t<Future>>::result_type>
        requires(hpx::traits::is_future_v<std::decay_t<Future>> &&
            (std::is_void_v<R> ? std::invocable<std::decay_t<F>> :
                                 std::invocable<std::decay_t<F>, R>) )
    auto transform(Future&& fut, F&& f)
    {
        if constexpr (std::is_void_v<R>)
        {
            return fut.then([f = std::forward<F>(f)](auto&& f_inner) mutable {
                f_inner.get();
                return f();
            });
        }
        else
        {
            return fut.then([f = std::forward<F>(f)](auto&& f_inner) mutable {
                return HPX_INVOKE(f, f_inner.get());
            });
        }
    }

    // ---------------------------------------------------------------
    // and_then: chain a continuation that itself returns a future
    // ---------------------------------------------------------------
    namespace detail {

        // Helper: compute invoke_result while accounting for void R
        template <typename R, typename F>
        struct invoke_result_helper;

        template <typename R, typename F>
            requires(!std::is_void_v<R>)
        struct invoke_result_helper<R, F>
        {
            using type = std::invoke_result_t<F, R>;
        };

        template <typename R, typename F>
            requires(std::is_void_v<R>)
        struct invoke_result_helper<R, F>
        {
            using type = std::invoke_result_t<F>;
        };

        template <typename R, typename F>
        using invoke_result_helper_t =
            typename invoke_result_helper<R, F>::type;

    }    // namespace detail

    template <typename Future, typename F,
        typename R = typename hpx::traits::future_traits<
            std::decay_t<Future>>::result_type,
        typename InvokeResult =
            detail::invoke_result_helper_t<R, std::decay_t<F>>>
        requires(hpx::traits::is_future_v<std::decay_t<Future>> &&
            hpx::traits::is_future_v<InvokeResult>)
    auto and_then(Future&& fut, F&& f)
    {
        if constexpr (std::is_void_v<R>)
        {
            return fut.then([f = std::forward<F>(f)](auto&& f_inner) mutable {
                f_inner.get();
                return f();
            });
        }
        else
        {
            return fut.then([f = std::forward<F>(f)](auto&& f_inner) mutable {
                return HPX_INVOKE(f, f_inner.get());
            });
        }
    }

    // ---------------------------------------------------------------
    // or_else: error recovery — executes f only when the future
    //          holds an exception
    // ---------------------------------------------------------------
    namespace detail {

        // Lazy wrappers so std::conditional_t can select the struct
        // without eagerly instantiating both ::type members.
        template <typename F>
        struct or_else_result_eptr
        {
            using type =
                std::invoke_result_t<std::decay_t<F>, std::exception_ptr>;
        };

        template <typename F>
        struct or_else_result_void
        {
            using type = std::invoke_result_t<std::decay_t<F>>;
        };

    }    // namespace detail

    template <typename Future, typename F,
        typename R = typename hpx::traits::future_traits<
            std::decay_t<Future>>::result_type>
        requires(hpx::traits::is_future_v<std::decay_t<Future>>)
    auto or_else(Future&& fut, F&& f)
    {
        constexpr bool takes_eptr =
            std::invocable<std::decay_t<F>, std::exception_ptr>;

        using f_result_type = typename std::conditional_t<takes_eptr,
            detail::or_else_result_eptr<F>,
            detail::or_else_result_void<F>>::type;

        return fut.then(
            [f = std::forward<F>(f)](auto&& f_inner) mutable -> f_result_type {
                if (f_inner.has_exception())
                {
                    if constexpr (takes_eptr)
                    {
                        return HPX_INVOKE(f, f_inner.get_exception_ptr());
                    }
                    else
                    {
                        return f();
                    }
                }
                else
                {
                    if constexpr (hpx::traits::is_future_v<f_result_type>)
                    {
                        if constexpr (std::is_void_v<R>)
                        {
                            f_inner.get();
                            return hpx::make_ready_future();
                        }
                        else
                        {
                            return hpx::make_ready_future<std::decay_t<R>>(
                                f_inner.get());
                        }
                    }
                    else
                    {
                        return f_inner.get();
                    }
                }
            });
    }

}}    // namespace hpx::futures
