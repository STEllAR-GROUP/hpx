//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/thread_execution.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_THREAD_EXECUTION_JAN_03_2017_1145AM)
#define HPX_PARALLEL_EXECUTORS_THREAD_EXECUTION_JAN_03_2017_1145AM

#include <hpx/config.hpp>
#include <hpx/lcos/dataflow.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/futures_factory.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/traits/is_launch_policy.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/bind_back.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/range.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/unwrap.hpp>

#include <hpx/parallel/executors/execution.hpp>

#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// define customization point specializations for thread executors
namespace hpx { namespace threads
{
    ///////////////////////////////////////////////////////////////////////////
    // async_execute()
    template <typename Executor, typename F, typename ... Ts>
    HPX_FORCEINLINE
    typename std::enable_if<
        hpx::traits::is_threads_executor<Executor>::value,
        hpx::lcos::future<
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type
        >
    >::type
    async_execute(Executor && exec, F && f, Ts &&... ts)
    {
        typedef typename util::detail::invoke_deferred_result<F, Ts...>::type
            result_type;

        lcos::local::futures_factory<result_type()> p(
            std::forward<Executor>(exec),
            util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...));
        p.apply();
        return p.get_future();
    }

    ///////////////////////////////////////////////////////////////////////////
    // sync_execute()
    template <typename Executor, typename F, typename ... Ts>
    HPX_FORCEINLINE
    typename std::enable_if<
        hpx::traits::is_threads_executor<Executor>::value,
        typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type
    >::type
    sync_execute(Executor && exec, F && f, Ts &&... ts)
    {
        return async_execute(std::forward<Executor>(exec),
            std::forward<F>(f), std::forward<Ts>(ts)...).get();
    }

    ///////////////////////////////////////////////////////////////////////////
    // then_execute()
    template <typename Executor, typename F, typename Future, typename ... Ts>
    HPX_FORCEINLINE
    typename std::enable_if<
        hpx::traits::is_threads_executor<Executor>::value,
        hpx::lcos::future<
            typename hpx::util::detail::invoke_deferred_result<
                F, Future, Ts...
            >::type
        >
    >::type
    then_execute(Executor && exec, F && f, Future&& predecessor, Ts &&... ts)
    {
        typedef typename hpx::util::detail::invoke_deferred_result<
                F, Future, Ts...
            >::type result_type;

        auto func = hpx::util::bind_back(
            hpx::util::one_shot(std::forward<F>(f)),
            std::forward<Ts>(ts)...);

        typename hpx::traits::detail::shared_state_ptr<result_type>::type p =
            hpx::lcos::detail::make_continuation_exec<result_type>(
                std::forward<Future>(predecessor), std::forward<Executor>(exec),
                std::move(func));

        return hpx::traits::future_access<hpx::lcos::future<result_type> >::
            create(std::move(p));
    }

    ///////////////////////////////////////////////////////////////////////////
    // post()
    template <typename Executor, typename F, typename ... Ts>
    HPX_FORCEINLINE
    typename std::enable_if<
        hpx::traits::is_threads_executor<Executor>::value
    >::type
    post(Executor && exec, F && f, Ts &&... ts)
    {
        exec.add(
            util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...),
            threads::thread_schedule_hint_none,
            "hpx::parallel::execution::post",
            threads::pending,
            true,
            exec.get_stacksize(),
            throws);
    }
    ///////////////////////////////////////////////////////////////////////////
    // post()
    template <typename Executor, typename F, typename Hint, typename ... Ts>
    HPX_FORCEINLINE
    typename std::enable_if<
        hpx::traits::is_threads_executor<Executor>::value
    >::type
    post(Executor && exec, F && f, Ts &&... ts, Hint && hint)
    {
        exec.add(
            util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...),
            std::forward<Hint>(hint),
            "hpx::parallel::execution::post",
            threads::pending,
            true,
            exec.get_stacksize(),
            throws);
    }

    ///////////////////////////////////////////////////////////////////////////
    // bulk_async_execute()
    template <typename Executor, typename F, typename Shape, typename ... Ts>
        typename std::enable_if<
        hpx::traits::is_threads_executor<Executor>::value,
        std::vector<hpx::lcos::future<
            typename parallel::execution::detail::bulk_function_result<
                F, Shape, Ts...
            >::type
        > >
    >::type
    bulk_async_execute(Executor && exec, F && f, Shape const& shape, Ts &&... ts)
    {
        std::vector<hpx::future<
                typename parallel::execution::detail::bulk_function_result<
                    F, Shape, Ts...
                >::type
            > > results;
        results.reserve(util::size(shape));

        for (auto const& elem: shape)
        {
            results.push_back(
                async_execute(exec, std::forward<F>(f), elem, ts...));
        }

        return results;
    }

    ///////////////////////////////////////////////////////////////////////////
    // bulk_sync_execute()
    template <typename Executor, typename F, typename Shape, typename ... Ts>
    typename std::enable_if<
        hpx::traits::is_threads_executor<Executor>::value,
        typename parallel::execution::detail::bulk_execute_result<
            F, Shape, Ts...
        >::type
    >::type
    bulk_sync_execute(Executor && exec, F && f, Shape const& shape, Ts &&... ts)
    {
        std::vector<hpx::future<
                typename parallel::execution::detail::bulk_function_result<
                    F, Shape, Ts...
                >::type
            > > results;
        results.reserve(util::size(shape));

        for (auto const& elem: shape)
        {
            results.push_back(
                async_execute(exec, std::forward<F>(f), elem, ts...));
        }

        return hpx::util::unwrap(results);
    }

    ///////////////////////////////////////////////////////////////////////////
    // bulk_then_execute()
    template <typename Executor, typename F, typename Shape, typename Future,
        typename ... Ts>
    HPX_FORCEINLINE
    typename std::enable_if<
        hpx::traits::is_threads_executor<Executor>::value,
        hpx::future<
            typename parallel::execution::detail::bulk_then_execute_result<
                F, Shape, Future, Ts...
            >::type
        >
    >::type
    bulk_then_execute(Executor && exec, F && f, Shape const& shape,
        Future&& predecessor, Ts &&... ts)
    {
        typedef typename parallel::execution::detail::then_bulk_function_result<
                F, Shape, Future, Ts...
            >::type func_result_type;

        typedef std::vector<hpx::lcos::future<func_result_type> > result_type;

        auto func =
            parallel::execution::detail::make_fused_bulk_async_execute_helper<
                result_type
            >(exec, std::forward<F>(f), shape,
                hpx::util::make_tuple(std::forward<Ts>(ts)...));

        // void or std::vector<func_result_type>
        typedef typename parallel::execution::detail::bulk_then_execute_result<
                F, Shape, Future, Ts...
            >::type vector_result_type;

        typedef hpx::future<vector_result_type> result_future_type;

        typedef typename hpx::traits::detail::shared_state_ptr<
                result_future_type
            >::type shared_state_type;

        typedef typename std::decay<Future>::type future_type;

        shared_state_type p =
            lcos::detail::make_continuation_exec<result_future_type>(
                std::forward<Future>(predecessor),
                std::forward<Executor>(exec),
                [HPX_CAPTURE_MOVE(func)](future_type&& predecessor) mutable
                ->  result_future_type
                {
                    return hpx::dataflow(
                        hpx::util::functional::unwrap{},
                        func(std::move(predecessor)));
                });

        return hpx::traits::future_access<result_future_type>::create(
            std::move(p));
    }
}}

#endif

