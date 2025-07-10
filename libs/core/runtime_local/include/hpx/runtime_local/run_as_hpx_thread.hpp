//  Copyright (c) 2016-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/datastructures/optional.hpp>
#include <hpx/functional/detail/invoke.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/threading_base/threading_base_fwd.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        // This is the overload for running functions that return a value.
        template <typename F, typename... Ts>
        util::invoke_result_t<F, Ts...> run_as_hpx_thread(
            std::false_type, F&& f, Ts&&... ts)
        {
            // NOTE: The condition variable needs be able to live past the scope
            // of this function. The mutex and boolean are guaranteed to live
            // long enough because of the lock.
            hpx::spinlock mtx;
            auto cond = std::make_shared<std::condition_variable_any>();
            bool stopping = false;

            using result_type = util::invoke_result_t<F, Ts...>;

            // Using the optional for storing the returned result value allows
            // to support non-default-constructible and move-only types.
            hpx::optional<result_type> result;
            std::exception_ptr exception;

            // Create the HPX thread
            hpx::threads::thread_init_data data(
                hpx::threads::make_thread_function_nullary([&, cond]() {
                    try
                    {
                        // Execute the given function, forward all parameters,
                        // store result.
                        result.emplace(HPX_INVOKE(
                            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...));
                    }
                    catch (...)
                    {
                        // make sure exceptions do not escape the HPX thread
                        // scheduler
                        exception = std::current_exception();
                    }

                    // Now signal to the waiting thread that we're done.
                    {
                        std::lock_guard<hpx::spinlock> lk(mtx);
                        stopping = true;
                    }

                    cond->notify_all();
                }),
                "run_as_hpx_thread (non-void)");
            hpx::threads::register_work(data);

            // wait for the HPX thread to exit
            {
                std::unique_lock<hpx::spinlock> lk(mtx);
                cond->wait(lk, [&]() -> bool { return stopping; });
            }

            // rethrow exceptions
            if (exception)
            {
                std::rethrow_exception(exception);
            }

            HPX_ASSERT(result);
            // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
            return HPX_MOVE(*result);
        }

        // This is the overload for running functions that return void.
        template <typename F, typename... Ts>
        void run_as_hpx_thread(std::true_type, F&& f, Ts&&... ts)
        {
            // NOTE: The condition variable needs be able to live past the scope
            // of this function. The mutex and boolean are guaranteed to live
            // long enough because of the lock.
            hpx::spinlock mtx;
            auto cond = std::make_shared<std::condition_variable_any>();
            bool stopping = false;

            std::exception_ptr exception;

            // Create an HPX thread
            hpx::threads::thread_init_data data(
                hpx::threads::make_thread_function_nullary([&, cond]() {
                    try
                    {
                        // Execute the given function, forward all parameters.
                        HPX_INVOKE(HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
                    }
                    catch (...)
                    {
                        // make sure exceptions do not escape the HPX thread
                        // scheduler
                        exception = std::current_exception();
                    }

                    // Now signal to the waiting thread that we're done.
                    {
                        std::lock_guard<hpx::spinlock> lk(mtx);
                        stopping = true;
                    }

                    cond->notify_all();
                }),
                "run_as_hpx_thread (void)");
            hpx::threads::register_work(data);

            // wait for the HPX thread to exit
            {
                std::unique_lock<hpx::spinlock> lk(mtx);
                cond->wait(lk, [&]() -> bool { return stopping; });
            }

            // rethrow exceptions
            if (exception)
            {
                std::rethrow_exception(exception);
            }
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    util::invoke_result_t<F, Ts...> run_as_hpx_thread(F&& f, Ts&&... vs)
    {
        // This shouldn't be used on a HPX-thread
        HPX_ASSERT(hpx::threads::get_self_ptr() == nullptr);

        using result_is_void =
            typename std::is_void<util::invoke_result_t<F, Ts...>>::type;

        return detail::run_as_hpx_thread(
            result_is_void(), HPX_FORWARD(F, f), HPX_FORWARD(Ts, vs)...);
    }
}    // namespace hpx

namespace hpx::threads {

    template <typename F, typename... Ts>
    HPX_DEPRECATED_V(1, 10,
        "hpx::threads::run_as_hpx_thread is deprecated, use "
        "hpx::run_as_hpx_thread instead")
    decltype(auto) run_as_hpx_thread(F&& f, Ts&&... ts)
    {
        return hpx::run_as_hpx_thread(
            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
    }
}    // namespace hpx::threads
