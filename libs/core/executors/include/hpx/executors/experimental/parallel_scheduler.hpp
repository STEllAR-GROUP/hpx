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

#ifdef HPX_HAVE_STDEXEC
#include <hpx/execution_base/stdexec_forward.hpp>
#endif

#include <cstddef>
#include <exception>
#include <utility>

namespace hpx::execution::experimental {

#ifdef HPX_HAVE_STDEXEC
    using hpx_sender_concept_t = hpx::execution::experimental::sender_t;
#else
    using hpx_sender_concept_t = void;    // Fallback type
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

        forward_progress_guarantee get_forward_progress_guarantee()
            const noexcept
        {
            return forward_progress_guarantee::parallel;
        }

        friend forward_progress_guarantee tag_invoke(
            get_forward_progress_guarantee_t,
            const parallel_scheduler& sched) noexcept
        {
            return sched.get_forward_progress_guarantee();
        }

        parallel_sender<parallel_scheduler> schedule() const noexcept;

        wrapped_type wrapped_;
    };

    inline parallel_scheduler get_parallel_scheduler()
    {
        // Create a thread-local instance with the appropriate policy
        thread_local bool is_on_hpx_thread = false;
        thread_local bool initialized = false;

        if (!initialized)
        {
            try
            {
                // If this doesn't throw, we're on an HPX thread
                hpx::threads::get_self_id();
                is_on_hpx_thread = true;
            }
            catch (...)
            {
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

#ifdef HPX_HAVE_STDEXEC
        void set_value() &&
        {
            try
            {
                func_();
                std::move(receiver_).set_value();
            }
            catch (...)
            {
                std::move(receiver_).set_error(std::current_exception());
            }
        }

        void set_error(std::exception_ptr ep) && noexcept
        {
            std::move(receiver_).set_error(ep);
        }

        void set_stopped() && noexcept
        {
            std::move(receiver_).set_stopped();
        }
#else
        void set_value() &&
        {
            throw std::runtime_error("stdexec not enabled");
        }

        void set_error(std::exception_ptr) && noexcept
        {
            throw std::runtime_error("stdexec not enabled");
        }

        void set_stopped() && noexcept
        {
            throw std::runtime_error("stdexec not enabled");
        }
#endif
    };

    template <typename Sender, typename Func>
    struct then_sender
    {
        using sender_concept = hpx_sender_concept_t;

        Sender sender_;
        Func func_;

        then_sender(Sender&& sender, Func&& func)
          : sender_(std::move(sender))
          , func_(std::move(func))
        {
        }

#ifdef HPX_HAVE_STDEXEC
        auto get_env() const noexcept
        {
            return hpx::execution::experimental::get_env(sender_);
        }
#else
        auto get_env() const noexcept
        {
            throw std::runtime_error("stdexec not enabled");
        }
#endif

        template <typename Receiver>
        struct operation_state
        {
            using wrapped_receiver_t = wrapped_receiver<Receiver, Func>;
#ifdef HPX_HAVE_STDEXEC
            using wrapped_op_state_t =
                decltype(hpx::execution::experimental::connect(
                    std::declval<Sender&&>(),
                    std::declval<wrapped_receiver_t&&>()));

            wrapped_op_state_t wrapped_op_;

            operation_state(Sender&& sender, Func&& func, Receiver&& receiver)
              : wrapped_op_(hpx::execution::experimental::connect(
                    std::move(sender),
                    wrapped_receiver_t{std::move(receiver), std::move(func)}))
            {
            }

            void start() noexcept
            {
                hpx::execution::experimental::start(wrapped_op_);
            }
#else
            operation_state(Sender&&, Func&&, Receiver&&)
            {
                throw std::runtime_error("stdexec not enabled");
            }

            void start() noexcept
            {
                throw std::runtime_error("stdexec not enabled");
            }
#endif
        };

#ifdef HPX_HAVE_STDEXEC
        template <typename Receiver>
        auto connect(Receiver&& receiver) &&
        {
            return operation_state<Receiver>{std::move(sender_),
                std::move(func_), std::forward<Receiver>(receiver)};
        }

        template <typename Receiver>
        auto connect(Receiver&& receiver) const&
        {
            return operation_state<Receiver>{
                sender_, func_, std::forward<Receiver>(receiver)};
        }
#else
        template <typename Receiver>
        auto connect(Receiver&&) &&
        {
            throw std::runtime_error("stdexec not enabled");
        }

        template <typename Receiver>
        auto connect(Receiver&&) const&
        {
            throw std::runtime_error("stdexec not enabled");
        }
#endif
    };

    template <typename Scheduler>
    struct parallel_sender
    {
        using sender_concept = hpx_sender_concept_t;

        explicit parallel_sender(Scheduler scheduler) noexcept
          : scheduler_(std::move(scheduler))
        {
        }

        auto get_env() const noexcept
        {
            struct env
            {
                Scheduler scheduler_;
            };
            return env{scheduler_};
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

#ifdef HPX_HAVE_STDEXEC
            void start() noexcept
            {
                auto stop_token = hpx::execution::experimental::get_stop_token(
                    hpx::execution::experimental::get_env(receiver_));
                if (stop_token.stop_requested())
                {
                    std::move(receiver_).set_stopped();
                    return;
                }
                try
                {
                    auto wrapped_sender =
                        hpx::execution::experimental::schedule(scheduler_);
                    auto wrapped_op = hpx::execution::experimental::connect(
                        std::move(wrapped_sender), std::move(receiver_));
                    hpx::execution::experimental::start(wrapped_op);
                }
                catch (...)
                {
                    std::move(receiver_).set_error(std::current_exception());
                }
            }
#else
            void start() noexcept
            {
                throw std::runtime_error("stdexec not enabled");
            }
#endif
        };

#ifdef HPX_HAVE_STDEXEC
        template <typename Receiver>
        auto connect(Receiver&& receiver) &&
        {
            return operation_state<Receiver>{std::forward<Receiver>(receiver),
                std::move(scheduler_.wrapped_)};
        }

        template <typename Receiver>
        auto connect(Receiver&& receiver) const&
        {
            return operation_state<Receiver>{
                std::forward<Receiver>(receiver), scheduler_.wrapped_};
        }
#else
        template <typename Receiver>
        auto connect(Receiver&&) &&
        {
            throw std::runtime_error("stdexec not enabled");
        }

        template <typename Receiver>
        auto connect(Receiver&&) const&
        {
            throw std::runtime_error("stdexec not enabled");
        }
#endif

#ifdef HPX_HAVE_STDEXEC
        template <typename Func>
        auto then(Func&& func) &&
        {
            return then_sender<parallel_sender, std::decay_t<Func>>{
                std::move(*this), std::forward<Func>(func)};
        }

        template <typename Func>
        auto then(Func&& func) const&
        {
            return then_sender<parallel_sender, std::decay_t<Func>>{
                *this, std::forward<Func>(func)};
        }
#else
        template <typename Func>
        auto then(Func&&) &&
        {
            throw std::runtime_error("stdexec not enabled");
        }

        template <typename Func>
        auto then(Func&&) const&
        {
            throw std::runtime_error("stdexec not enabled");
        }
#endif

    private:
        Scheduler scheduler_;
    };

    inline parallel_sender<parallel_scheduler> parallel_scheduler::schedule()
        const noexcept
    {
#ifdef HPX_HAVE_STDEXEC
        return parallel_sender<parallel_scheduler>(*this);
#else
        throw std::runtime_error("stdexec not enabled");
#endif
    }

}    // namespace hpx::execution::experimental
