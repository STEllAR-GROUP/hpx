//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
#include <hpx/async_base/traits/is_launch_policy.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/lcos_local/packaged_task.hpp>
#include <hpx/modules/timed_execution.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace threads {
    ///////////////////////////////////////////////////////////////////////
    template <typename Executor, typename F, typename... Ts>
    typename std::enable_if<
        hpx::traits::is_threads_executor<Executor>::value>::type
    post_at(Executor&& exec, hpx::chrono::steady_time_point const& abs_time,
        F&& f, Ts&&... ts)
    {
        exec.add_at(abs_time,
            hpx::util::deferred_call(
                std::forward<F>(f), std::forward<Ts>(ts)...),
            "post_at");
    }

    template <typename Executor, typename F, typename... Ts>
    typename std::enable_if<
        hpx::traits::is_threads_executor<Executor>::value>::type
    post_after(Executor&& exec, hpx::chrono::steady_duration const& rel_time,
        F&& f, Ts&&... ts)
    {
        exec.add_after(rel_time,
            hpx::util::deferred_call(
                std::forward<F>(f), std::forward<Ts>(ts)...),
            "post_after");
    }

    ///////////////////////////////////////////////////////////////////////
    template <typename Executor, typename F, typename... Ts>
    typename std::enable_if<hpx::traits::is_threads_executor<Executor>::value,
        hpx::future<typename hpx::util::detail::invoke_deferred_result<F,
            Ts...>::type>>::type
    async_execute_at(Executor&& exec,
        hpx::chrono::steady_time_point const& abs_time, F&& f, Ts&&... ts)
    {
        typedef
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type
                result_type;

        lcos::local::packaged_task<result_type(
            typename std::decay<Ts>::type...)>
            task(std::forward<F>(f));

        hpx::future<result_type> result = task.get_future();

        exec.add_at(abs_time,
            hpx::util::deferred_call(std::move(task), std::forward<Ts>(ts)...),
            "async_execute_at");

        return result;
    }

    template <typename Executor, typename F, typename... Ts>
    typename std::enable_if<hpx::traits::is_threads_executor<Executor>::value,
        hpx::future<typename hpx::util::detail::invoke_deferred_result<F,
            Ts...>::type>>::type
    async_execute_after(Executor&& exec,
        hpx::chrono::steady_duration const& rel_time, F&& f, Ts&&... ts)
    {
        typedef
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type
                result_type;

        lcos::local::packaged_task<result_type(Ts...)> task(std::forward<F>(f));

        hpx::future<result_type> result = task.get_future();

        exec.add_after(rel_time,
            hpx::util::deferred_call(std::move(task), std::forward<Ts>(ts)...),
            "async_execute_after");

        return result;
    }

    ///////////////////////////////////////////////////////////////////////
    template <typename Executor, typename F, typename... Ts>
    typename std::enable_if<hpx::traits::is_threads_executor<Executor>::value,
        typename hpx::util::detail::invoke_deferred_result<F,
            Ts...>::type>::type
    sync_execute_at(Executor&& exec,
        hpx::chrono::steady_time_point const& abs_time, F&& f, Ts&&... ts)
    {
        return async_execute_at(std::forward<Executor>(exec), abs_time,
            std::forward<F>(f), std::forward<Ts>(ts)...)
            .get();
    }

    template <typename Executor, typename F, typename... Ts>
    typename std::enable_if<hpx::traits::is_threads_executor<Executor>::value,
        typename hpx::util::detail::invoke_deferred_result<F,
            Ts...>::type>::type
    sync_execute_after(Executor&& exec,
        hpx::chrono::steady_duration const& rel_time, F&& f, Ts&&... ts)
    {
        return async_execute_after(std::forward<Executor>(exec), rel_time,
            std::forward<F>(f), std::forward<Ts>(ts)...)
            .get();
    }
}}    // namespace hpx::threads
#endif
