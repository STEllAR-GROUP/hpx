//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/thread_execution.hpp

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
#include <hpx/assert.hpp>
#include <hpx/async_base/traits/is_launch_policy.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/executors/fused_bulk_execute.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/executors/current_executor.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/futures_factory.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/pack_traversal/unwrap.hpp>
#include <hpx/threading_base/thread_helpers.hpp>
#include <hpx/type_support/pack.hpp>
#include <hpx/type_support/unused.hpp>

#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// define customization point specializations for thread executors
namespace hpx { namespace threads {
    ///////////////////////////////////////////////////////////////////////////
    // async_execute()
    template <typename Executor, typename F, typename... Ts>
    HPX_FORCEINLINE typename std::enable_if<
        hpx::traits::is_threads_executor<Executor>::value,
        hpx::lcos::future<typename hpx::util::detail::invoke_deferred_result<F,
            Ts...>::type>>::type
    async_execute(Executor&& exec, F&& f, Ts&&... ts)
    {
        typedef typename util::detail::invoke_deferred_result<F, Ts...>::type
            result_type;

        char const* annotation = hpx::traits::get_function_annotation<
            typename std::decay<F>::type>::call(f);
        lcos::local::futures_factory<result_type()> p(
            std::forward<Executor>(exec),
            util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...));
        p.apply(annotation);
        return p.get_future();
    }

    ///////////////////////////////////////////////////////////////////////////
    // sync_execute()
    template <typename Executor, typename F, typename... Ts>
    HPX_FORCEINLINE typename std::enable_if<
        hpx::traits::is_threads_executor<Executor>::value,
        typename hpx::util::detail::invoke_deferred_result<F,
            Ts...>::type>::type
    sync_execute(Executor&& exec, F&& f, Ts&&... ts)
    {
        return async_execute(std::forward<Executor>(exec), std::forward<F>(f),
            std::forward<Ts>(ts)...)
            .get();
    }

    ///////////////////////////////////////////////////////////////////////////
    // then_execute()
    template <typename Executor, typename F, typename Future, typename... Ts>
    HPX_FORCEINLINE typename std::enable_if<
        hpx::traits::is_threads_executor<Executor>::value,
        hpx::lcos::future<typename hpx::util::detail::invoke_deferred_result<F,
            Future, Ts...>::type>>::type
    then_execute(Executor&& exec, F&& f, Future&& predecessor, Ts&&... ts)
    {
        typedef typename hpx::util::detail::invoke_deferred_result<F, Future,
            Ts...>::type result_type;

        auto func = hpx::util::one_shot(
            hpx::util::bind_back(std::forward<F>(f), std::forward<Ts>(ts)...));

        typename hpx::traits::detail::shared_state_ptr<result_type>::type p =
            hpx::lcos::detail::make_continuation_exec<result_type>(
                std::forward<Future>(predecessor), std::forward<Executor>(exec),
                std::move(func));

        return hpx::traits::future_access<
            hpx::lcos::future<result_type>>::create(std::move(p));
    }

    ///////////////////////////////////////////////////////////////////////////
    // post()
    template <typename Executor, typename F, typename... Ts>
    HPX_FORCEINLINE typename std::enable_if<
        hpx::traits::is_threads_executor<Executor>::value>::type
    post(Executor&& exec, F&& f, Ts&&... ts)
    {
        char const* annotation = hpx::traits::get_function_annotation<
            typename std::decay<F>::type>::call(f);
        exec.add(
            util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...),
            annotation, threads::thread_schedule_state::pending, true,
            exec.get_stacksize(), threads::thread_schedule_hint(), throws);
    }
    ///////////////////////////////////////////////////////////////////////////
    // post()
    template <typename Executor, typename F, typename Hint, typename... Ts>
    HPX_FORCEINLINE typename std::enable_if<
        hpx::traits::is_threads_executor<Executor>::value &&
        std::is_same<typename std::decay<Hint>::type,
            hpx::threads::thread_schedule_hint>::value>::type
    post(
        Executor&& exec, F&& f, Hint&& hint, const char* annotation, Ts&&... ts)
    {
        exec.add(
            util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...),
            annotation, threads::thread_schedule_state::pending, true,
            exec.get_stacksize(), std::forward<Hint>(hint), throws);
    }

    ///////////////////////////////////////////////////////////////////////////
    // bulk_async_execute()
    template <typename Executor, typename F, typename Shape, typename... Ts>
    typename std::enable_if<hpx::traits::is_threads_executor<Executor>::value,
        std::vector<hpx::lcos::future<typename parallel::execution::detail::
                bulk_function_result<F, Shape, Ts...>::type>>>::type
    bulk_async_execute(Executor&& exec, F&& f, Shape const& shape, Ts&&... ts)
    {
        std::vector<hpx::future<typename parallel::execution::detail::
                bulk_function_result<F, Shape, Ts...>::type>>
            results;
        results.reserve(util::size(shape));

        for (auto const& elem : shape)
        {
            results.push_back(
                async_execute(exec, std::forward<F>(f), elem, ts...));
        }

        return results;
    }

    ///////////////////////////////////////////////////////////////////////////
    // bulk_sync_execute()
    template <typename Executor, typename F, typename Shape, typename... Ts>
    typename std::enable_if<hpx::traits::is_threads_executor<Executor>::value,
        typename parallel::execution::detail::bulk_execute_result<F, Shape,
            Ts...>::type>::type
    bulk_sync_execute(Executor&& exec, F&& f, Shape const& shape, Ts&&... ts)
    {
        std::vector<hpx::future<typename parallel::execution::detail::
                bulk_function_result<F, Shape, Ts...>::type>>
            results;
        results.reserve(util::size(shape));

        for (auto const& elem : shape)
        {
            results.push_back(
                async_execute(exec, std::forward<F>(f), elem, ts...));
        }

        return hpx::util::unwrap(results);
    }

    ///////////////////////////////////////////////////////////////////////////
    // bulk_then_execute()
    template <typename Executor, typename F, typename Shape, typename Future,
        typename... Ts>
    HPX_FORCEINLINE typename std::enable_if<
        hpx::traits::is_threads_executor<Executor>::value,
        hpx::future<typename parallel::execution::detail::
                bulk_then_execute_result<F, Shape, Future, Ts...>::type>>::type
    bulk_then_execute(
        Executor&& exec, F&& f, Shape const& shape, Future&& predecessor,
        Ts&&...
#if !defined(HPX_COMPUTE_DEVICE_CODE)
        ts
#endif
    )
    {
#if defined(HPX_COMPUTE_DEVICE_CODE)
        HPX_UNUSED(exec);
        HPX_UNUSED(f);
        HPX_UNUSED(shape);
        HPX_UNUSED(predecessor);
        HPX_ASSERT(false);
        return hpx::make_ready_future();
#else
        typedef
            typename parallel::execution::detail::then_bulk_function_result<F,
                Shape, Future, Ts...>::type func_result_type;

        typedef std::vector<hpx::lcos::future<func_result_type>> result_type;

        auto func =
            parallel::execution::detail::make_fused_bulk_async_execute_helper<
                result_type>(exec, std::forward<F>(f), shape,
                hpx::make_tuple(std::forward<Ts>(ts)...));

        // void or std::vector<func_result_type>
        typedef
            typename parallel::execution::detail::bulk_then_execute_result<F,
                Shape, Future, Ts...>::type vector_result_type;

        typedef hpx::future<vector_result_type> result_future_type;

        typedef typename hpx::traits::detail::shared_state_ptr<
            result_future_type>::type shared_state_type;

        typedef typename std::decay<Future>::type future_type;

        thread_id_type id = hpx::threads::get_self_id();
        parallel::execution::current_executor exec_current =
            hpx::threads::get_executor(id);

        shared_state_type p =
            lcos::detail::make_continuation_exec<result_future_type>(
                std::forward<Future>(predecessor), std::forward<Executor>(exec),
                [func = std::move(func),
                    exec_current = std::move(exec_current)](
                    future_type&& predecessor) mutable -> result_future_type {
                    return hpx::parallel::execution::async_execute(exec_current,
                        hpx::util::functional::unwrap{},
                        func(std::move(predecessor)));
                });

        return hpx::traits::future_access<result_future_type>::create(
            std::move(p));
#endif
    }
}}    // namespace hpx::threads

#endif
