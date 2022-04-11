//  Copyright (c) 2016-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/datastructures/optional.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/functional/detail/invoke.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/threading_base/thread_helpers.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>

namespace hpx { namespace threads {
    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        // This is the overload for running functions which return a value.
        template <typename F, typename... Ts>
        typename util::invoke_result<F, Ts...>::type run_as_hpx_thread(
            std::false_type, F const& f, Ts&&... ts)
        {
            // NOTE: The condition variable needs be able to live past the scope
            // of this function. The mutex and boolean are guaranteed to live
            // long enough because of the lock.
            hpx::spinlock mtx;
            auto cond = std::make_shared<std::condition_variable_any>();
            bool stopping = false;

            typedef typename util::invoke_result<F, Ts...>::type result_type;

            // Using the optional for storing the returned result value
            // allows to support non-default-constructible and move-only
            // types.
            hpx::optional<result_type> result;
            std::exception_ptr exception;

            // Create the HPX thread
            hpx::threads::thread_init_data data(
                hpx::threads::make_thread_function_nullary([&, cond]() {
                    try
                    {
                        // Execute the given function, forward all parameters,
                        // store result.
                        result.emplace(HPX_INVOKE(f, HPX_FORWARD(Ts, ts)...));
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
            std::unique_lock<hpx::spinlock> lk(mtx);
            cond->wait(lk, [&]() -> bool { return stopping; });

            // rethrow exceptions
            if (exception)
                std::rethrow_exception(exception);

            return HPX_MOVE(*result);
        }

        // This is the overload for running functions which return void.
        template <typename F, typename... Ts>
        void run_as_hpx_thread(std::true_type, F const& f, Ts&&... ts)
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
                        HPX_INVOKE(f, HPX_FORWARD(Ts, ts)...);
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
            std::unique_lock<hpx::spinlock> lk(mtx);
            cond->wait(lk, [&]() -> bool { return stopping; });

            // rethrow exceptions
            if (exception)
                std::rethrow_exception(exception);
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    typename util::invoke_result<F, Ts...>::type run_as_hpx_thread(
        F const& f, Ts&&... vs)
    {
        // This shouldn't be used on a HPX-thread
        HPX_ASSERT(hpx::threads::get_self_ptr() == nullptr);

        typedef typename std::is_void<
            typename util::invoke_result<F, Ts...>::type>::type result_is_void;

        return detail::run_as_hpx_thread(
            result_is_void(), f, HPX_FORWARD(Ts, vs)...);
    }
}}    // namespace hpx::threads
