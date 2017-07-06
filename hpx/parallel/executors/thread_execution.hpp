//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/thread_execution.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_THREAD_EXECUTION_JAN_03_2017_1145AM)
#define HPX_PARALLEL_EXECUTORS_THREAD_EXECUTION_JAN_03_2017_1145AM

#include <hpx/config.hpp>
#include <hpx/apply.hpp>
#include <hpx/async.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/traits/is_launch_policy.hpp>
#include <hpx/traits/future_access.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/detail/pack.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/range.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/unwrapped.hpp>

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
        return hpx::async(std::forward<Executor>(exec), std::forward<F>(f),
            std::forward<Ts>(ts)...);
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
        return hpx::async(std::forward<Executor>(exec), std::forward<F>(f),
            std::forward<Ts>(ts)...).get();
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
    then_execute(Executor && exec, F && f, Future& predecessor, Ts &&... ts)
    {
        typedef typename hpx::util::detail::invoke_deferred_result<
                F, Future, Ts...
            >::type result_type;

        auto func = hpx::util::bind(
            hpx::util::one_shot(std::forward<F>(f)),
            hpx::util::placeholders::_1, std::forward<Ts>(ts)...);

        typename hpx::traits::detail::shared_state_ptr<result_type>::type
            p = hpx::lcos::detail::make_continuation_thread_exec<result_type>(
                    predecessor, std::forward<Executor>(exec),
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
        hpx::apply(std::forward<Executor>(exec), std::forward<F>(f),
            std::forward<Ts>(ts)...);
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
            results.push_back(hpx::async(exec, std::forward<F>(f),
                elem, ts...));
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
            results.push_back(hpx::async(exec, std::forward<F>(f),
                elem, ts...));
        }

        return hpx::util::unwrapped(results);
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
        Future& predecessor, Ts &&... ts)
    {
        typedef typename parallel::execution::detail::then_bulk_function_result<
                F, Shape, Future, Ts...
            >::type func_result_type;

        typedef std::vector<hpx::lcos::future<func_result_type> > result_type;
        typedef hpx::lcos::future<result_type> result_future_type;

        // older versions of gcc are not able to capture parameter
        // packs (gcc < 4.9)
        auto args = hpx::util::make_tuple(std::forward<Ts>(ts)...);
        auto func =
            [exec, f, shape, args](Future predecessor) mutable
            ->  result_type
            {
                return parallel::execution::detail::fused_bulk_async_execute(
                    exec, f, shape, predecessor,
                    typename hpx::util::detail::make_index_pack<
                        sizeof...(Ts)
                    >::type(), args);
            };

        typedef typename hpx::traits::detail::shared_state_ptr<
                result_type
            >::type shared_state_type;

        shared_state_type p =
            lcos::detail::make_continuation_thread_exec<result_type>(
                predecessor, std::forward<Executor>(exec),
                std::move(func));

        return hpx::traits::future_access<result_future_type>::
            create(std::move(p));
    }
}}

#endif

