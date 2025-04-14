// Copyright (c) 2025 Sai Charan Arvapally
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution/algorithms/then.hpp>
#include <hpx/executors/thread_pool_scheduler.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/synchronization/stop_token.hpp>

#include <cstddef>
#include <exception>
#include <utility>


// Forward declarations for execution::experimental
namespace hpx::execution::experimental {
    enum class forward_progress_guarantee;
    struct get_forward_progress_guarantee_t;
    struct schedule_t;
    struct connect_t;
    struct start_t;
    struct set_value_t;
    struct set_error_t;
    struct set_stopped_t;
    struct get_env_t;
    struct then_t;
    struct sender_t;
}    // namespace hpx::execution::experimental

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
        using wrapped_type = thread_pool_policy_scheduler<hpx::launch>;

        parallel_scheduler(hpx::launch policy) noexcept
          : wrapped_(policy)
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

    inline parallel_scheduler get_parallel_scheduler()
    {
        // Create a thread-local instance with the appropriate policy
        thread_local bool is_on_hpx_thread = false;
        thread_local bool initialized = false;
        
        if (!initialized) {
            try {
                // If this doesn't throw, we're on an HPX thread
                hpx::threads::get_self_id();
                is_on_hpx_thread = true;
            }
            catch (...) {
                // Not on an HPX thread
                is_on_hpx_thread = false;
            }
            initialized = true;
        }
        
        // Use sync policy if we're on an HPX thread, async otherwise
        static parallel_scheduler async_instance(hpx::launch::async);
        static parallel_scheduler sync_instance(hpx::launch::sync);
        
        return is_on_hpx_thread ? sync_instance : async_instance;
    }
    

    template <typename Receiver, typename Func>
    struct wrapped_receiver
    {
        Receiver receiver_;
        Func func_;

        wrapped_receiver(Receiver&& receiver, Func&& func)
          : receiver_(std::move(receiver))
          , func_(std::move(func))
        {
        }

        friend void tag_invoke(set_value_t, wrapped_receiver&& wr)
        {
            try
            {
                wr.func_();
                tag_invoke(set_value_t{}, std::move(wr.receiver_));
            }
            catch (...)
            {
                tag_invoke(set_error_t{}, std::move(wr.receiver_),
                    std::current_exception());
            }
        }

        friend void tag_invoke(
            set_error_t, wrapped_receiver&& wr, std::exception_ptr ep) noexcept
        {
            tag_invoke(set_error_t{}, std::move(wr.receiver_), ep);
        }

        friend void tag_invoke(set_stopped_t, wrapped_receiver&& wr) noexcept
        {
            tag_invoke(set_stopped_t{}, std::move(wr.receiver_));
        }
    };

    template <typename Sender, typename Func>
    struct then_sender
    {
        using sender_concept = sender_t;

        Sender sender_;
        Func func_;

        then_sender(Sender&& sender, Func&& func)
          : sender_(std::move(sender))
          , func_(std::move(func))
        {
        }

        friend auto tag_invoke(get_env_t, const then_sender& s) noexcept
        {
            return get_env(s.sender_);
        }

        template <typename Receiver>
        struct operation_state
        {
            using wrapped_receiver_t = wrapped_receiver<Receiver, Func>;
            using wrapped_op_state_t =
                decltype(tag_invoke(connect_t{}, std::declval<Sender&&>(),
                    std::declval<wrapped_receiver_t&&>()));

            wrapped_op_state_t wrapped_op_;

            operation_state(Sender&& sender, Func&& func, Receiver&& receiver)
              : wrapped_op_(tag_invoke(connect_t{}, std::move(sender),
                    wrapped_receiver_t{std::move(receiver), std::move(func)}))
            {
            }

            friend void tag_invoke(start_t, operation_state& op) noexcept
            {
                start(op.wrapped_op_);
            }
        };

        template <typename Receiver>
        friend auto tag_invoke(connect_t, then_sender&& s, Receiver&& receiver)
        {
            return operation_state<Receiver>{std::move(s.sender_),
                std::move(s.func_), std::forward<Receiver>(receiver)};
        }

        template <typename Receiver>
        friend auto tag_invoke(
            connect_t, const then_sender& s, Receiver&& receiver)
        {
            return operation_state<Receiver>{
                s.sender_, s.func_, std::forward<Receiver>(receiver)};
        }
    };

    template <typename Scheduler>
    struct parallel_sender
    {
        using sender_concept = sender_t;

        explicit parallel_sender(Scheduler scheduler) noexcept
          : scheduler_(std::move(scheduler))
        {
        }

        friend auto tag_invoke(
            get_env_t, const parallel_sender& sender) noexcept
        {
            struct env
            {
                Scheduler scheduler_;
            };
            return env{sender.scheduler_};
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

        template <typename Func>
        friend auto tag_invoke(then_t, parallel_sender&& sender, Func&& func)
        {
            return then_sender<parallel_sender, std::decay_t<Func>>{
                std::move(sender), std::forward<Func>(func)};
        }

        template <typename Func>
        friend auto tag_invoke(
            then_t, const parallel_sender& sender, Func&& func)
        {
            return then_sender<parallel_sender, std::decay_t<Func>>{
                sender, std::forward<Func>(func)};
        }

    private:
        Scheduler scheduler_;
    };

    inline parallel_sender<parallel_scheduler> parallel_scheduler::schedule()
        const noexcept
    {
        return parallel_sender<parallel_scheduler>(*this);
    }

    inline parallel_sender<parallel_scheduler> tag_invoke(
        schedule_t, const parallel_scheduler& sched) noexcept
    {
        return sched.schedule();
    }
}    // namespace hpx::execution::experimental
