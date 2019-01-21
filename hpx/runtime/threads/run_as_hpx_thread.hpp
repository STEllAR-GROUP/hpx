//  Copyright (c) 2016-2018 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_THREADS_RUN_AS_HPX_THREAD_MAR_12_2016_0202PM)
#define HPX_THREADS_RUN_AS_HPX_THREAD_MAR_12_2016_0202PM

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/optional.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/tuple.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>

namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        // This is the overload for running functions which return a value.
        template <typename F, typename... Ts>
        typename util::invoke_result<F, Ts...>::type
        run_as_hpx_thread(std::false_type, F const& f, Ts &&... ts)
        {
            hpx::lcos::local::spinlock mtx;
            std::condition_variable_any cond;
            bool stopping = false;

            typedef typename util::invoke_result<F, Ts...>::type result_type;

            // Using the optional for storing the returned result value
            // allows to support non-default-constructible and move-only
            // types.
            hpx::util::optional<result_type> result;
            std::exception_ptr exception;

            // This lambda function will be scheduled to run as an HPX
            // thread
            auto && args = util::forward_as_tuple(std::forward<Ts>(ts)...);
            auto && wrapper =
                [&]() mutable
                {
                    try
                    {
                        // Execute the given function, forward all parameters,
                        // store result.
                        result.emplace(util::invoke_fused(f, std::move(args)));
                    }
                    catch (...)
                    {
                        // make sure exceptions do not escape the HPX thread
                        // scheduler
                        exception = std::current_exception();
                    }

                    // Now signal to the waiting thread that we're done.
                    {
                        std::lock_guard<hpx::lcos::local::spinlock> lk(mtx);
                        stopping = true;
                    }
                    cond.notify_all();
                };

            // Create the HPX thread
            hpx::threads::register_thread_nullary(std::ref(wrapper));

            // wait for the HPX thread to exit
            std::unique_lock<hpx::lcos::local::spinlock> lk(mtx);
            while (!stopping)
                cond.wait(lk);

            // rethrow exceptions
            if (exception)
            {
                std::rethrow_exception(exception);
            }

            return std::move(*result);
        }

        // This is the overload for running functions which return void.
        template <typename F, typename... Ts>
        void run_as_hpx_thread(std::true_type, F const& f, Ts &&... ts)
        {
            hpx::lcos::local::spinlock mtx;
            std::condition_variable_any cond;
            bool stopping = false;

            std::exception_ptr exception;

            // This lambda function will be scheduled to run as an HPX
            // thread
            auto && args = util::forward_as_tuple(std::forward<Ts>(ts)...);
            auto && wrapper =
                [&]() mutable
                {
                    try
                    {
                        // Execute the given function, forward all parameters.
                        util::invoke_fused(f, std::move(args));
                    }
                    catch (...)
                    {
                        // make sure exceptions do not escape the HPX thread
                        // scheduler
                        exception = std::current_exception();
                    }

                    // Now signal to the waiting thread that we're done.
                    {
                        std::lock_guard<hpx::lcos::local::spinlock> lk(mtx);
                        stopping = true;
                    }
                    cond.notify_all();
                };

            // Create an HPX thread
            hpx::threads::register_thread_nullary(std::ref(wrapper));

            // wait for the HPX thread to exit
            std::unique_lock<hpx::lcos::local::spinlock> lk(mtx);
            while (!stopping)
                cond.wait(lk);

            // rethrow exceptions
            if (exception)
            {
                std::rethrow_exception(exception);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename F, typename... Ts>
    typename util::invoke_result<F, Ts...>::type
    run_as_hpx_thread(F const& f, Ts &&... vs)
    {
        // This shouldn't be used on a HPX-thread
        HPX_ASSERT(hpx::threads::get_self_ptr() == nullptr);

        typedef typename std::is_void<
                typename util::invoke_result<F, Ts...>::type
            >::type result_is_void;

        return detail::run_as_hpx_thread(result_is_void(),
            f, std::forward<Ts>(vs)...);
    }
}}

#endif
