// Copyright (c) 2025 Sai Charan Arvapally
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution.hpp>
#include <hpx/execution/traits/executor_traits.hpp>
#include <hpx/execution_base/completion_scheduler.hpp>
#include <hpx/execution_base/get_env.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/executors/thread_pool_scheduler.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/iterator_support.hpp>
#include <hpx/synchronization/stop_token.hpp>

#include <cstddef>
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

#ifndef HPX_EXECUTION_EXPERIMENTAL_SENDER_T
#define HPX_EXECUTION_EXPERIMENTAL_SENDER_T
    struct sender_t
    {
    };
#endif

    template <typename Scheduler>
    struct parallel_sender;

    struct parallel_scheduler
    {
        using wrapped_type =
            thread_pool_policy_scheduler<hpx::launch::async_policy>;

        parallel_scheduler() noexcept
          : wrapped_(hpx::launch::async_policy{})
        {
        }

        parallel_scheduler(const parallel_scheduler&) noexcept = default;
        parallel_scheduler(parallel_scheduler&&) noexcept = default;
        parallel_scheduler& operator=(
            const parallel_scheduler&) noexcept = default;
        parallel_scheduler& operator=(parallel_scheduler&&) noexcept = default;

        friend bool operator==(
            const parallel_scheduler&, const parallel_scheduler&) noexcept
        {
            return true;
        }

        friend forward_progress_guarantee tag_invoke(
            get_forward_progress_guarantee_t,
            const parallel_scheduler&) noexcept
        {
            return forward_progress_guarantee::parallel;
        }

        parallel_sender<parallel_scheduler> schedule() const noexcept;

        wrapped_type wrapped_;
    };

    parallel_scheduler get_parallel_scheduler();

    template <typename Scheduler>
    struct parallel_sender
    {
        using sender_concept = sender_t;

        using completion_signatures = completion_signatures<set_value_t(),
            set_stopped_t(), set_error_t(std::exception_ptr)>;

        explicit parallel_sender(Scheduler scheduler) noexcept
          : scheduler_(std::move(scheduler))
        {
        }

        friend auto tag_invoke(
            get_env_t, const parallel_sender& sender) noexcept
        {
            return hpx::execution::experimental::get_env(
                hpx::execution::experimental::schedule(
                    sender.scheduler_.wrapped_));
        }

        template <typename Receiver>
        struct operation_state
        {
            Receiver receiver_;
            parallel_scheduler::wrapped_type scheduler_;

            operation_state(
                Receiver&& r, parallel_scheduler::wrapped_type&& sched)
              : receiver_(std::forward<Receiver>(r))
              , scheduler_(std::move(sched))
            {
            }

            friend void tag_invoke(start_t, operation_state& op) noexcept
            {
                auto stop_token = get_stop_token(get_env(op.receiver_));
                if (stop_token.stop_requested())
                {
                    set_stopped(std::move(op.receiver_));
                    return;
                }
                try
                {
                    auto wrapped_sender =
                        hpx::execution::experimental::schedule(op.scheduler_);
                    auto wrapped_op = hpx::execution::experimental::connect(
                        std::move(wrapped_sender), std::move(op.receiver_));
                    start(wrapped_op);
                    if (hpx::get_num_worker_threads() == 1)
                    {
                        hpx::this_thread::yield();
                    }
                }
                catch (...)
                {
                    set_error(
                        std::move(op.receiver_), std::current_exception());
                }
            }
        };

        template <typename Receiver>
        friend auto tag_invoke(
            connect_t, parallel_sender&& sender, Receiver&& receiver)
        {
            return operation_state<Receiver>{std::forward<Receiver>(receiver),
                std::move(sender.scheduler_.wrapped_)};
        }

        template <typename Receiver>
        friend auto tag_invoke(
            connect_t, const parallel_sender& sender, Receiver&& receiver)
        {
            return operation_state<Receiver>{
                std::forward<Receiver>(receiver), sender.scheduler_.wrapped_};
        }

    private:
        Scheduler scheduler_;
    };

    inline parallel_sender<parallel_scheduler> parallel_scheduler::schedule()
        const noexcept
    {
        return parallel_sender<parallel_scheduler>(*this);
    }

    inline parallel_scheduler get_parallel_scheduler()
    {
        static parallel_scheduler instance;
        return instance;
    }

    inline parallel_sender<parallel_scheduler> tag_invoke(
        schedule_t, const parallel_scheduler& sched) noexcept
    {
        return sched.schedule();
    }

}    // namespace hpx::execution::experimental