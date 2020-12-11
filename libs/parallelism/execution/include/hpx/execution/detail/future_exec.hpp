//  Copyright (c) 2007-2019 Hartmut Kaiser
//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/allocator_support/internal_allocator.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/async_base/traits/is_launch_policy.hpp>
#include <hpx/execution/detail/post_policy_dispatch.hpp>
#include <hpx/execution/traits/executor_traits.hpp>
#include <hpx/execution/traits/future_then_result_exec.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/invoke_result.hpp>
#include <hpx/futures/detail/future_data.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/futures/packaged_continuation.hpp>
#include <hpx/futures/traits/future_access.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/serialization/detail/polymorphic_nonintrusive_factory.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <exception>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace lcos { namespace detail {
    template <typename Executor, typename Future, typename F>
    inline typename hpx::traits::future_then_executor_result<Executor,
        typename std::decay<Future>::type, F>::type
    then_execute_helper(Executor&& exec, F&& f, Future&& predecessor)
    {
        // simply forward this to executor
        return parallel::execution::then_execute(
            exec, std::forward<F>(f), std::forward<Future>(predecessor));
    }

    ///////////////////////////////////////////////////////////////////////////
    // launch
    template <typename Future, typename Policy>
    struct future_then_dispatch<Future, Policy,
        typename std::enable_if<traits::is_launch_policy<Policy>::value>::type>
    {
        template <typename Policy_, typename F>
        HPX_FORCEINLINE static
            typename hpx::traits::future_then_result<Future, F>::type
            call(Future&& fut, Policy_&& policy, F&& f)
        {
            using result_type = typename hpx::traits::future_then_result<Future,
                F>::result_type;
            using continuation_result_type =
                typename hpx::util::invoke_result<F, Future>::type;

            typename hpx::traits::detail::shared_state_ptr<result_type>::type
                p = detail::make_continuation_alloc<continuation_result_type>(
                    hpx::util::internal_allocator<>{}, std::move(fut),
                    std::forward<Policy_>(policy), std::forward<F>(f));
            return hpx::traits::future_access<future<result_type>>::create(
                std::move(p));
        }

        template <typename Allocator, typename Policy_, typename F>
        HPX_FORCEINLINE static
            typename hpx::traits::future_then_result<Future, F>::type
            call_alloc(
                Allocator const& alloc, Future&& fut, Policy_&& policy, F&& f)
        {
            using result_type = typename hpx::traits::future_then_result<Future,
                F>::result_type;
            using continuation_result_type =
                typename hpx::util::invoke_result<F, Future>::type;

            typename hpx::traits::detail::shared_state_ptr<result_type>::type
                p = detail::make_continuation_alloc<continuation_result_type>(
                    alloc, std::move(fut), std::forward<Policy_>(policy),
                    std::forward<F>(f));
            return hpx::traits::future_access<future<result_type>>::create(
                std::move(p));
        }
    };

    // The overload for future::then taking an executor simply forwards to the
    // corresponding executor customization point.
    //
    // parallel executors v2
    // threads::executor
    template <typename Future, typename Executor>
    struct future_then_dispatch<Future, Executor,
        typename std::enable_if<traits::is_one_way_executor<Executor>::value ||
            traits::is_two_way_executor<Executor>::value
#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
            || traits::is_threads_executor<Executor>::value
#endif
            >::type>
    {
        template <typename Executor_, typename F>
        HPX_FORCEINLINE static
            typename hpx::traits::future_then_executor_result<Executor_, Future,
                F>::type
            call(Future&& fut, Executor_&& exec, F&& f)
        {
            // simply forward this to executor
            return detail::then_execute_helper(std::forward<Executor_>(exec),
                std::forward<F>(f), std::move(fut));
        }

        template <typename Allocator, typename Executor_, typename F>
        HPX_FORCEINLINE static
            typename hpx::traits::future_then_executor_result<Executor_, Future,
                F>::type
            call_alloc(Allocator const&, Future&& fut, Executor_&& exec, F&& f)
        {
            return call(std::forward<Future>(fut),
                std::forward<Executor_>(exec), std::forward<F>(f));
        }
    };

    // plain function, or function object
    template <typename Future, typename FD>
    struct future_then_dispatch<Future, FD,
        typename std::enable_if<!traits::is_launch_policy<FD>::value &&
            !(traits::is_one_way_executor<FD>::value ||
                traits::is_two_way_executor<FD>::value
#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
                || traits::is_threads_executor<FD>::value
#endif
                )>::type>
    {
        template <typename F>
        HPX_FORCEINLINE static auto call(Future&& fut, F&& f)
            -> decltype(future_then_dispatch<Future, launch>::call(
                std::move(fut), launch::all, std::forward<F>(f)))
        {
            return future_then_dispatch<Future, launch>::call(
                std::move(fut), launch::all, std::forward<F>(f));
        }

        template <typename Allocator, typename F>
        HPX_FORCEINLINE static auto call_alloc(
            Allocator const& alloc, Future&& fut, F&& f)
            -> decltype(future_then_dispatch<Future, launch>::call_alloc(
                alloc, std::move(fut), launch::all, std::forward<F>(f)))
        {
            return future_then_dispatch<Future, launch>::call_alloc(
                alloc, std::move(fut), launch::all, std::forward<F>(f));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    struct post_policy_spawner
    {
        template <typename F>
        void operator()(F&& f, hpx::util::thread_description desc)
        {
            parallel::execution::detail::post_policy_dispatch<
                hpx::launch::async_policy>::call(hpx::launch::async, desc,
                std::forward<F>(f));
        }
    };

    template <typename Executor>
    struct executor_spawner
    {
        Executor exec;

        template <typename F>
        void operator()(F&& f, hpx::util::thread_description)
        {
            hpx::parallel::execution::post(exec, std::forward<F>(f));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename ContResult, typename Future, typename Policy, typename F>
    inline typename traits::detail::shared_state_ptr<
        typename continuation_result<ContResult>::type>::type
    make_continuation(Future const& future, Policy&& policy, F&& f)
    {
        using result_type = typename continuation_result<ContResult>::type;
        using shared_state = detail::continuation<Future, F, result_type>;
        using init_no_addref = typename shared_state::init_no_addref;
        using spawner_type = post_policy_spawner;

        // create a continuation
        typename traits::detail::shared_state_ptr<result_type>::type p(
            new shared_state(init_no_addref{}, std::forward<F>(f)), false);
        static_cast<shared_state*>(p.get())->template attach<spawner_type>(
            future, spawner_type{}, std::forward<Policy>(policy));
        return p;
    }

    // same as above, except with allocator
    template <typename ContResult, typename Allocator, typename Future,
        typename Policy, typename F>
    inline typename traits::detail::shared_state_ptr<
        typename continuation_result<ContResult>::type>::type
    make_continuation_alloc(
        Allocator const& a, Future const& future, Policy&& policy, F&& f)
    {
        using result_type = typename continuation_result<ContResult>::type;

        using base_allocator = Allocator;
        using shared_state = typename traits::detail::shared_state_allocator<
            detail::continuation<Future, F, result_type>, base_allocator>::type;

        using other_allocator = typename std::allocator_traits<
            base_allocator>::template rebind_alloc<shared_state>;
        using traits = std::allocator_traits<other_allocator>;

        using init_no_addref = typename shared_state::init_no_addref;

        using unique_ptr = std::unique_ptr<shared_state,
            util::allocator_deleter<other_allocator>>;

        using spawner_type = post_policy_spawner;

        other_allocator alloc(a);
        unique_ptr p(traits::allocate(alloc, 1),
            util::allocator_deleter<other_allocator>{alloc});
        traits::construct(
            alloc, p.get(), init_no_addref{}, alloc, std::forward<F>(f));

        // create a continuation
        typename hpx::traits::detail::shared_state_ptr<result_type>::type r(
            p.release(), false);

        static_cast<shared_state*>(r.get())->template attach<spawner_type>(
            future, spawner_type{}, std::forward<Policy>(policy));

        return r;
    }

    // same as above, except with allocator and without unwrapping returned
    // futures
    template <typename ContResult, typename Allocator, typename Future,
        typename Policy, typename F>
    inline typename traits::detail::shared_state_ptr<ContResult>::type
    make_continuation_alloc_nounwrap(
        Allocator const& a, Future const& future, Policy&& policy, F&& f)
    {
        using result_type = ContResult;

        using base_allocator = Allocator;
        using shared_state = typename traits::detail::shared_state_allocator<
            detail::continuation<Future, F, result_type>, base_allocator>::type;

        using other_allocator = typename std::allocator_traits<
            base_allocator>::template rebind_alloc<shared_state>;
        using traits = std::allocator_traits<other_allocator>;

        using init_no_addref = typename shared_state::init_no_addref;

        using unique_ptr = std::unique_ptr<shared_state,
            util::allocator_deleter<other_allocator>>;

        using spawner_type = post_policy_spawner;

        other_allocator alloc(a);
        unique_ptr p(traits::allocate(alloc, 1),
            util::allocator_deleter<other_allocator>{alloc});
        traits::construct(
            alloc, p.get(), init_no_addref{}, alloc, std::forward<F>(f));

        // create a continuation
        typename hpx::traits::detail::shared_state_ptr<result_type>::type r(
            p.release(), false);

        static_cast<shared_state*>(r.get())
            ->template attach_nounwrap<spawner_type>(
                future, spawner_type{}, std::forward<Policy>(policy));

        return r;
    }

    template <typename ContResult, typename Future, typename Executor,
        typename F>
    inline typename traits::detail::shared_state_ptr<ContResult>::type
    make_continuation_exec(Future const& future, Executor&& exec, F&& f)
    {
        using shared_state = detail::continuation<Future, F, ContResult>;
        using init_no_addref = typename shared_state::init_no_addref;
        using spawner_type =
            executor_spawner<typename std::decay<Executor>::type>;

        // create a continuation
        typename traits::detail::shared_state_ptr<ContResult>::type p(
            new shared_state(init_no_addref{}, std::forward<F>(f)), false);
        static_cast<shared_state*>(p.get())
            ->template attach_nounwrap<spawner_type>(future,
                spawner_type{std::forward<Executor>(exec)},
                launch::async_policy{});
        return p;
    }

    template <typename ContResult, typename Future, typename Executor,
        typename Policy, typename F>
    inline typename traits::detail::shared_state_ptr<ContResult>::type
    make_continuation_exec_policy(
        Future const& future, Executor&& exec, Policy&& policy, F&& f)
    {
        using shared_state = detail::continuation<Future, F, ContResult>;
        using init_no_addref = typename shared_state::init_no_addref;
        using spawner_type =
            executor_spawner<typename std::decay<Executor>::type>;

        // create a continuation
        typename traits::detail::shared_state_ptr<ContResult>::type p(
            new shared_state(init_no_addref{}, std::forward<F>(f)), false);
        static_cast<shared_state*>(p.get())
            ->template attach_nounwrap<spawner_type>(future,
                spawner_type{std::forward<Executor>(exec)},
                std::forward<Policy>(policy));
        return p;
    }
}}}    // namespace hpx::lcos::detail
