//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_EXECUTION_DETAIL_POST_POLICY_DISPATCH_DEC_05_2017_0234PM)
#define HPX_PARALLEL_EXECUTION_DETAIL_POST_POLICY_DISPATCH_DEC_05_2017_0234PM

#include <hpx/config.hpp>
#include <hpx/runtime/get_worker_thread_num.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/runtime/threads/thread_enums.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/deferred_call.hpp>

#include <utility>

namespace hpx { namespace parallel { namespace execution { namespace detail
{
    ///////////////////////////////////////////////////////////////////////
    template <typename Policy>
    struct post_policy_dispatch
    {
        template <typename F, typename... Ts>
        static void call(hpx::util::thread_description const& desc,
            Policy const& policy, F && f, Ts &&... ts)
        {
            threads::register_thread_nullary(
                hpx::util::deferred_call(
                    std::forward<F>(f), std::forward<Ts>(ts)...),
                desc, threads::pending, false, policy.priority());
        }
    };

    template <>
    struct post_policy_dispatch<launch>
    {
        template <typename F, typename... Ts>
        static void call(hpx::util::thread_description const& desc,
            launch const& policy, F && f, Ts &&... ts)
        {
            if (policy == launch::sync)
            {
                hpx::util::deferred_call(
                    std::forward<F>(f), std::forward<Ts>(ts)...)();
                return;
            }

            threads::register_thread_nullary(
                hpx::util::deferred_call(
                    std::forward<F>(f), std::forward<Ts>(ts)...),
                desc, threads::pending, false, policy.priority());
        }
    };

    template <>
    struct post_policy_dispatch<launch::sync_policy>
    {
        template <typename F, typename... Ts>
        static void call(hpx::util::thread_description const& desc,
            launch::sync_policy const& policy, F && f, Ts &&... ts)
        {
            hpx::util::deferred_call(
                std::forward<F>(f), std::forward<Ts>(ts)...)();
        }
    };

    template <>
    struct post_policy_dispatch<launch::fork_policy>
    {
        template <typename F, typename... Ts>
        static void call(hpx::util::thread_description const& desc,
            launch::fork_policy const& policy, F && f, Ts &&... ts)
        {
            threads::thread_id_type tid = threads::register_thread_nullary(
                hpx::util::deferred_call(
                    std::forward<F>(f), std::forward<Ts>(ts)...),
                desc, threads::pending_do_not_schedule, true,
                policy.priority(), threads::thread_schedule_hint(get_worker_thread_num()),
                threads::thread_stacksize_current);

            // make sure this thread is executed last
            if (tid)
            {
                // yield_to(tid)
                hpx::this_thread::suspend(threads::pending, tid,
                    "hpx::parallel::execution::parallel_executor::post");
            }
        }
    };
}}}}

#endif

