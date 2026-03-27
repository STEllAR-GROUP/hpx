//  Copyright (c) 2026 The STE||AR-Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/futures/traits/future_traits.hpp>
#include <hpx/futures/traits/is_future.hpp>
#include <hpx/modules/functional.hpp>

#include <exception>
#include <type_traits>
#include <utility>

namespace hpx::futures {

    // -----------------------------------------------------------------
    // transform: apply f to the value inside a future.
    //
    //   f takes T (or nothing for void) and returns any value.
    //   The result is a hpx::future of f's return type.
    //   Delegates to .then() + hpx::make_ready_future for consistency.
    //
    //   hpx::futures::transform(fut, [](int v) { return v * 2; })
    //   -> hpx::future<int>
    // -----------------------------------------------------------------
    HPX_CXX_CORE_EXPORT template <typename Future, typename F,
        typename R = typename hpx::traits::future_traits<
            std::decay_t<Future>>::result_type>
        requires(hpx::traits::is_future_v<std::decay_t<Future>>)
    auto transform(Future&& fut, F&& f)
    {
        return fut.then([f = std::forward<F>(f)](auto&& f_inner) mutable {
            if constexpr (std::is_void_v<R>)
            {
                f_inner.get();
                return hpx::make_ready_future(f());
            }
            else
            {
                return hpx::make_ready_future(HPX_INVOKE(f, f_inner.get()));
            }
        });
    }

    // -----------------------------------------------------------------
    // and_then: chain a continuation onto a future.
    //
    //   Delegates directly to .then(). hpx::future::then() already
    //   auto-unwraps if f returns a future<T>, so no manual .unwrap()
    //   is needed. f may also throw to propagate exceptions.
    //
    //   hpx::futures::and_then(fut, [](int v) { return to_string(v); })
    //   -> hpx::future<string>
    // -----------------------------------------------------------------
    HPX_CXX_CORE_EXPORT template <typename Future, typename F>
        requires(hpx::traits::is_future_v<std::decay_t<Future>>)
    auto and_then(Future&& fut, F&& f)
    {
        return fut.then(std::forward<F>(f));
    }

    // -----------------------------------------------------------------
    // or_else: error recovery -- executes f only when the future holds
    //          an exception. f returns T (the value type) directly.
    //
    //   If no exception: value passes through unchanged.
    //   If exception:    f is called and its return value is used.
    //                    f may also rethrow / throw a different error.
    //
    //   hpx::futures::or_else(fut, [](std::exception_ptr) { return -1; })
    //   -> hpx::future<int>
    // -----------------------------------------------------------------
    HPX_CXX_CORE_EXPORT template <typename Future, typename F,
        typename R = typename hpx::traits::future_traits<
            std::decay_t<Future>>::result_type>
        requires(hpx::traits::is_future_v<std::decay_t<Future>>)
    auto or_else(Future&& fut, F&& f)
    {
        constexpr bool takes_eptr =
            std::is_invocable_v<std::decay_t<F>, std::exception_ptr>;

        return fut.then([f = std::forward<F>(f)](auto&& f_inner) mutable {
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

            if constexpr (std::is_void_v<R>)
            {
                f_inner.get();
            }
            else
            {
                return f_inner.get();
            }
        });
    }

}    // namespace hpx::futures
