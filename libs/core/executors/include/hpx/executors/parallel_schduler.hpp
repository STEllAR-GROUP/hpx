// Copyright (c) 2025 Sai Charan Arvapally
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/async_base/launch_policy.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution_base/stdexec_forward.hpp>
#include <hpx/executors/thread_pool_scheduler.hpp>
#include <hpx/executors/thread_pool_scheduler_bulk.hpp>
#include <hpx/threading_base/detail/get_default_pool.hpp>
#include <exception>
#include <memory>

#if !defined(HPX_HAVE_STDEXEC)
#include <hpx/execution/queries/get_stop_token.hpp>
#include <hpx/synchronization/stop_token.hpp>
#endif

namespace hpx::execution::experimental {

    namespace detail {
        // Singleton-like shared thread pool for parallel_scheduler
        inline hpx::threads::thread_pool_base* get_default_parallel_pool()
        {
            // clang-format off
            static hpx::threads::thread_pool_base* default_pool =
                hpx::threads::detail::get_self_or_default_pool();
            // clang-format on
            return default_pool;
        }
    }    // namespace detail

    // Forward declaration for parallel_scheduler_domain
    class parallel_scheduler;

#if defined(HPX_HAVE_STDEXEC)
    // P2079R10: Domain for parallel_scheduler bulk operations.
    // The existing thread_pool_domain checks __completes_on with
    // thread_pool_policy_scheduler, but parallel_scheduler's sender
    // returns parallel_scheduler as the completion scheduler.
    // This domain bridges the gap by extracting the underlying
    // thread_pool_policy_scheduler and delegating to HPX's optimized
    // thread_pool_bulk_sender.
    struct parallel_scheduler_domain : stdexec::default_domain
    {
        template <typename OpTag, bulk_chunked_or_unchunked_sender Sender,
            typename Env>
        auto transform_sender(OpTag, Sender&& sndr, Env const& env) const
            noexcept
        {
            static_assert(
                hpx::execution::experimental::stdexec_internal::
                    __completes_on<Sender, parallel_scheduler, Env> ||
                    hpx::execution::experimental::stdexec_internal::
                        __starts_on<Sender, parallel_scheduler, Env>,
                "No parallel_scheduler instance can be found in the "
                "sender's attributes or receiver's environment "
                "on which to schedule bulk work.");

            // Extract bulk parameters using structured binding
            auto&& [tag, data, child] = sndr;
            auto&& [pol, shape, f] = data;

            // Get the parallel_scheduler based on the matching pattern:
            //   completes_on: from the child sender's completion scheduler
            //   starts_on:    from the receiver's environment
            auto par_sched = [&]() {
                if constexpr (
                    hpx::execution::experimental::stdexec_internal::
                        __completes_on<Sender, parallel_scheduler, Env>)
                {
                    return hpx::execution::experimental::
                        get_completion_scheduler<
                            hpx::execution::experimental::set_value_t>(
                            hpx::execution::experimental::get_env(child));
                }
                else
                {
                    return hpx::execution::experimental::get_scheduler(
                        env);
                }
            }();

            // Extract the underlying thread pool scheduler
            auto underlying = par_sched.get_underlying_scheduler();

            auto iota_shape =
                hpx::util::counting_shape(decltype(shape){0}, shape);

            constexpr bool is_chunked =
                !hpx::execution::experimental::stdexec_internal::
                    sender_expr_for<Sender,
                        hpx::execution::experimental::bulk_unchunked_t>;

            // Check if policy is sequential (pol is a __policy_wrapper,
            // use __get() to unwrap the actual policy type)
            bool is_seq =
                is_sequenced_policy_v<std::decay_t<decltype(pol.__get())>>;

            auto bulk_snd = hpx::execution::experimental::detail::
                thread_pool_bulk_sender<hpx::launch,
                    std::decay_t<decltype(child)>,
                    std::decay_t<decltype(iota_shape)>,
                    std::decay_t<decltype(f)>, is_chunked>{
                        HPX_MOVE(underlying),
                        HPX_FORWARD(decltype(child), child),
                        HPX_MOVE(iota_shape),
                        HPX_FORWARD(decltype(f), f)};

            // Store the policy for sequential execution handling
            bulk_snd.set_sequential(is_seq);
            return bulk_snd;
        }
    };
#endif

    // P2079R10 parallel_scheduler implementation
    class parallel_scheduler
    {
    public:
        parallel_scheduler() = delete;

        explicit parallel_scheduler(
            thread_pool_policy_scheduler<hpx::launch> sched) noexcept
          : scheduler_(sched)
        {
        }

        parallel_scheduler(parallel_scheduler const& other) noexcept
          : scheduler_(other.scheduler_)
        {
        }

        parallel_scheduler(parallel_scheduler&& other) noexcept
          : scheduler_(HPX_MOVE(other.scheduler_))
        {
        }

        parallel_scheduler& operator=(parallel_scheduler const& other) noexcept
        {
            if (this != &other)
                scheduler_ = other.scheduler_;
            return *this;
        }

        parallel_scheduler& operator=(parallel_scheduler&& other) noexcept
        {
            if (this != &other)
                scheduler_ = HPX_MOVE(other.scheduler_);
            return *this;
        }

        friend constexpr bool operator==(parallel_scheduler const& lhs,
            parallel_scheduler const& rhs) noexcept
        {
            return lhs.scheduler_ == rhs.scheduler_;
        }

        // P2079R10: query() member for forward progress guarantee
        // (modern stdexec pattern, preferred over tag_invoke)
        constexpr forward_progress_guarantee query(
            get_forward_progress_guarantee_t) const noexcept
        {
            return forward_progress_guarantee::parallel;
        }

        // P2079R10: operation_state owns the receiver and manages the
        // frontend/backend boundary. On start(), it checks the stop token
        // and then calls the backend (thread_pool_policy_scheduler::execute).
        template <typename Receiver>
        struct operation_state
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver_;
            thread_pool_policy_scheduler<hpx::launch> scheduler_;

            template <typename Receiver_>
            operation_state(
                Receiver_&& receiver,
                thread_pool_policy_scheduler<hpx::launch> const& sched)
              : receiver_(HPX_FORWARD(Receiver_, receiver))
              , scheduler_(sched)
            {
            }

            operation_state(operation_state&&) = default;
            operation_state(operation_state const&) = delete;
            operation_state& operator=(operation_state&&) = default;
            operation_state& operator=(operation_state const&) = delete;

            friend void tag_invoke(
                stdexec::start_t, operation_state& os) noexcept
            {
#if defined(HPX_HAVE_STDEXEC)
                // P2079R10 ยง4.1: if stop_token is stopped, complete
                // with set_stopped as soon as is practical.
                auto stop_token = stdexec::get_stop_token(
                    stdexec::get_env(os.receiver_));
                if (stop_token.stop_requested())
                {
                    stdexec::set_stopped(HPX_MOVE(os.receiver_));
                    return;
                }
#endif
                // Delegate to the backend (thread_pool) to schedule work.
                // Capture &os (not the receiver by move) so that if
                // execute() throws, os.receiver_ is still valid for
                // the error handler. The sender/receiver protocol
                // guarantees the operation_state outlives completion.
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        os.scheduler_.execute([&os]() mutable {
                            hpx::execution::experimental::set_value(
                                HPX_MOVE(os.receiver_));
                        });
                    },
                    [&](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(os.receiver_), HPX_MOVE(ep));
                    });
            }
        };

        // Nested sender type
        template <typename Scheduler>
        struct sender
        {
            Scheduler sched_;

            using sender_concept = stdexec::sender_t;
            using completion_signatures = stdexec::completion_signatures<
                stdexec::set_value_t(),
                stdexec::set_error_t(std::exception_ptr),
                stdexec::set_stopped_t()>;

            template <typename Receiver>
            friend operation_state<std::decay_t<Receiver>> tag_invoke(
                stdexec::connect_t, sender const& s, Receiver&& receiver)
                noexcept(std::is_nothrow_constructible_v<
                    std::decay_t<Receiver>, Receiver>)
            {
                return {HPX_FORWARD(Receiver, receiver),
                    s.sched_.get_underlying_scheduler()};
            }

            template <typename Receiver>
            friend operation_state<std::decay_t<Receiver>> tag_invoke(
                stdexec::connect_t, sender&& s, Receiver&& receiver)
                noexcept(std::is_nothrow_constructible_v<
                    std::decay_t<Receiver>, Receiver>)
            {
                return {HPX_FORWARD(Receiver, receiver),
                    s.sched_.get_underlying_scheduler()};
            }

            struct env
            {
                Scheduler const& sched_;

                // P2079R10: only expose completion scheduler for set_value_t.
                // set_stopped may fire on the calling thread (not the pool),
                // so claiming parallel_scheduler as the completion scheduler
                // for set_stopped_t would be technically inaccurate.
                auto query(stdexec::get_completion_scheduler_t<
                    stdexec::set_value_t>) const noexcept
                {
                    return sched_;
                }

#if defined(HPX_HAVE_STDEXEC)
                // Domain query
                parallel_scheduler_domain query(
                    stdexec::get_domain_t) const noexcept
                {
                    return {};
                }
#endif
            };

            friend env tag_invoke(
                stdexec::get_env_t, sender const& s) noexcept
            {
                return {s.sched_};
            }
        };

        // Direct schedule() member for modern stdexec (non-deprecated path)
        sender<parallel_scheduler> schedule() const noexcept
        {
            return {*this};
        }

#if defined(HPX_HAVE_STDEXEC)
        // Domain customization for bulk operations
        parallel_scheduler_domain query(stdexec::get_domain_t) const noexcept
        {
            return {};
        }

        // Completion domain query: stdexec resolves domains for sender
        // algorithms via get_completion_domain_t, not get_domain_t.
        parallel_scheduler_domain query(
            stdexec::get_completion_domain_t<stdexec::set_value_t>) const
            noexcept
        {
            return {};
        }
#endif

        thread_pool_policy_scheduler<hpx::launch> const&
        get_underlying_scheduler() const noexcept
        {
            return scheduler_;
        }

    private:
        thread_pool_policy_scheduler<hpx::launch> scheduler_;
    };

    // Stream output operator for parallel_scheduler
    inline std::ostream& operator<<(std::ostream& os, const parallel_scheduler&)
    {
        return os << "parallel_scheduler";
    }

    // P2079R10 get_parallel_scheduler function
    inline parallel_scheduler get_parallel_scheduler()
    {
        // Use the default thread pool with async policy for parallel execution
        auto pool = detail::get_default_parallel_pool();
        if (!pool)
        {
            // clang-format off
            std::terminate(); // As per P2079R10, terminate if backend is unavailable
            // clang-format on
        }
        return parallel_scheduler(thread_pool_policy_scheduler<hpx::launch>(
            pool, hpx::launch::async));
    }

}    // namespace hpx::execution::experimental
