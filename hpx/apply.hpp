//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLY_APR_16_20012_0943AM)
#define HPX_APPLY_APR_16_20012_0943AM

#include <hpx/config.hpp>
#include <hpx/async.hpp>
#include <hpx/runtime/applier/apply.hpp>
#include <hpx/runtime/applier/apply_continue.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#if defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
#include <hpx/traits/is_executor_v1.hpp>
#endif
#include <hpx/traits/is_executor.hpp>
#include <hpx/traits/is_launch_policy.hpp>
#include <hpx/util/bind_action.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/thread_description.hpp>

#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/executors/execution.hpp>

#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // Define apply() overloads for plain local functions and function objects.
    // dispatching trait for hpx::apply

    // launch a plain function/function object
    template <typename Func, typename Enable>
    struct apply_dispatch
    {
        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename std::enable_if<
            traits::detail::is_deferred_callable<F(Ts&&...)>::value,
            bool
        >::type
        call(F&& f, Ts&&... ts)
        {
            util::thread_description desc(f, "apply_dispatch::call");
            threads::register_thread_nullary(
                util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...),
                desc);
            return false;
        }
    };

    // threads::executor
    template <typename Executor>
    struct apply_dispatch<Executor,
        typename std::enable_if<
            traits::is_threads_executor<Executor>::value
        >::type>
    {
        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename std::enable_if<
            traits::detail::is_deferred_callable<F(Ts&&...)>::value,
            bool
        >::type
        call(Executor& sched, F&& f, Ts&&... ts)
        {
            sched.add(
                util::deferred_call(std::forward<F>(f), std::forward<Ts>(ts)...),
                "hpx::apply");
            return false;
        }
    };

#if defined(HPX_HAVE_EXECUTOR_COMPATIBILITY)
    // parallel::executor
    template <typename Executor>
    struct apply_dispatch<Executor,
        typename std::enable_if<
            traits::is_executor<Executor>::value
        >::type>
    {
        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename std::enable_if<
            traits::detail::is_deferred_callable<F(Ts&&...)>::value,
            bool
        >::type
        call(Executor& exec, F&& f, Ts&&... ts)
        {
            parallel::executor_traits<Executor>::apply_execute(exec,
                std::forward<F>(f), std::forward<Ts>(ts)...);
            return false;
        }
    };
#endif

    // parallel::execution::executor
    template <typename Executor>
    struct apply_dispatch<Executor,
        typename std::enable_if<
            traits::is_one_way_executor<Executor>::value ||
            traits::is_two_way_executor<Executor>::value
        >::type>
    {
        template <typename F, typename ...Ts>
        HPX_FORCEINLINE static
        typename std::enable_if<
            traits::detail::is_deferred_callable<F(Ts&&...)>::value,
            bool
        >::type
        call(Executor& exec, F && f, Ts &&... ts)
        {
            parallel::execution::post(exec,
                std::forward<F>(f), std::forward<Ts>(ts)...);
            return false;
        }
    };

    // bound action
    template <typename Bound>
    struct apply_dispatch<Bound,
        typename std::enable_if<
            traits::is_bound_action<Bound>::value
        >::type>
    {
        template <typename Action, typename BoundArgs, typename ...Ts>
        HPX_FORCEINLINE static bool
        call(hpx::util::detail::bound_action<Action, BoundArgs> const& bound,
            Ts&&... ts)
        {
            return bound.apply(std::forward<Ts>(ts)...);
        }
    };
}}

namespace hpx
{
    template <typename F, typename ...Ts>
    HPX_FORCEINLINE bool apply(F&& f, Ts&&... ts)
    {
        return detail::apply_dispatch<typename util::decay<F>::type>::call(
            std::forward<F>(f), std::forward<Ts>(ts)...);
    }
}

#endif
