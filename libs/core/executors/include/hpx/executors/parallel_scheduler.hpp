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

namespace hpx::execution::experimental {

#if defined(HPX_HAVE_STDEXEC)
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

    inline parallel_scheduler get_parallel_scheduler();

    // P2079R10: Domain for parallel_scheduler bulk operations.
    // The existing thread_pool_domain checks __completes_on with
    // thread_pool_policy_scheduler, but parallel_scheduler's sender
    // returns parallel_scheduler as the completion scheduler.
    // This domain bridges the gap by extracting the underlying
    // thread_pool_policy_scheduler and delegating to HPX's optimized
    // thread_pool_bulk_sender.
    struct parallel_scheduler_domain : stdexec::default_domain
    {
        template <bulk_chunked_or_unchunked_sender Sender, typename Env>
        auto transform_sender(hpx::execution::experimental::set_value_t,
            Sender&& sndr, Env const& env) const noexcept
        {
            if constexpr (hpx::execution::experimental::stdexec_internal::
                              __completes_on<Sender, parallel_scheduler, Env>)
            {
                // Extract bulk parameters using structured binding
                auto&& [tag, data, child] = sndr;
                auto&& [pol, shape, f] = data;

                // Get the parallel_scheduler from the child sender's
                // completion scheduler (completes_on pattern)
                auto par_sched = [&]() {
                    if constexpr (hpx::is_invocable_v<
                                      hpx::execution::experimental::get_completion_scheduler_t<
                                          hpx::execution::experimental::set_value_t>,
                                      decltype(hpx::execution::experimental::get_env(child))>)
                    {
                        return hpx::execution::experimental::get_completion_scheduler<
                            hpx::execution::experimental::set_value_t>(
                            hpx::execution::experimental::get_env(child));
                    }
                    else
                    {
                        return hpx::execution::experimental::get_parallel_scheduler();
                    }
                }();

                // Extract the underlying thread pool scheduler
                auto underlying = par_sched.get_underlying_scheduler();

                auto iota_shape =
                    hpx::util::counting_shape(decltype(shape){0}, shape);

                constexpr bool is_chunked = !stdexec::__sender_for<Sender,
                    hpx::execution::experimental::bulk_unchunked_t>;

                // Determine parallelism at compile time from policy type
                // (pol is a __policy_wrapper, use __get() to unwrap)
                constexpr bool is_parallel =
                    !is_sequenced_policy_v<std::decay_t<decltype(pol.__get())>>;

                // Pass the pre-cached PU mask so thread_pool_bulk_sender
                // skips its own full_mask() computation on every invocation.
                hpx::threads::mask_type pu_mask = par_sched.get_pu_mask();
                return hpx::execution::experimental::detail::
                    thread_pool_bulk_sender<hpx::launch,
                        std::decay_t<decltype(child)>,
                        std::decay_t<decltype(iota_shape)>,
                        std::decay_t<decltype(f)>, is_chunked, is_parallel>(
                        HPX_MOVE(underlying),
                        HPX_FORWARD(decltype(child), child),
                        HPX_MOVE(iota_shape), HPX_FORWARD(decltype(f), f),
                        HPX_MOVE(pu_mask));
            }
            else
            {
                // P2079R10: bulk operations require the parallel_scheduler
                // in the environment. Add a continues_on transition to the
                // parallel_scheduler before the bulk algorithm.
                static_assert(
                    hpx::execution::experimental::stdexec_internal::
                        __completes_on<Sender, parallel_scheduler, Env>,
                    "Cannot dispatch bulk algorithm to the parallel_scheduler: "
                    "no parallel_scheduler found in the environment. "
                    "Add a continues_on transition to the parallel_scheduler "
                    "before the bulk algorithm.");
            }
        }
    };

    // P2079R10 parallel_scheduler implementation
    class parallel_scheduler
    {
    public:
        parallel_scheduler() = delete;

        // Compute and cache the PU mask once at construction time so that
        // parallel_scheduler_domain::transform_sender can pass it directly to
        // thread_pool_bulk_sender, avoiding the expensive full_mask() call
        // (which iterates all PUs) on every bulk_chunked invocation.
        explicit parallel_scheduler(
            thread_pool_policy_scheduler<hpx::launch> sched)
          : scheduler_(sched)
          , pu_mask_(hpx::execution::experimental::detail::full_mask(
                hpx::execution::experimental::get_first_core(scheduler_),
                hpx::execution::experimental::processing_units_count(
                    hpx::execution::experimental::null_parameters, scheduler_,
                    hpx::chrono::null_duration, 0)))
        {
        }

        parallel_scheduler(parallel_scheduler const& other) noexcept
          : scheduler_(other.scheduler_)
          , pu_mask_(other.pu_mask_)
        {
        }

        parallel_scheduler(parallel_scheduler&& other) noexcept
          : scheduler_(HPX_MOVE(other.scheduler_))
          , pu_mask_(HPX_MOVE(other.pu_mask_))
        {
        }

        parallel_scheduler& operator=(parallel_scheduler const& other) noexcept
        {
            if (this != &other)
            {
                scheduler_ = other.scheduler_;
                pu_mask_ = other.pu_mask_;
            }
            return *this;
        }

        parallel_scheduler& operator=(parallel_scheduler&& other) noexcept
        {
            if (this != &other)
            {
                scheduler_ = HPX_MOVE(other.scheduler_);
                pu_mask_ = HPX_MOVE(other.pu_mask_);
            }
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
            operation_state(Receiver_&& receiver,
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
                start_t, operation_state& os) noexcept
            {
#if defined(HPX_HAVE_STDEXEC)
                // P2079R10 4.1: if stop_token is stopped, complete
                // with set_stopped as soon as is practical.
                auto stop_token =
                    stdexec::get_stop_token(stdexec::get_env(os.receiver_));
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
            using completion_signatures =
                stdexec::completion_signatures<stdexec::set_value_t(),
                    stdexec::set_error_t(std::exception_ptr),
                    stdexec::set_stopped_t()>;

            template <typename Receiver>
            friend operation_state<std::decay_t<Receiver>> tag_invoke(
                stdexec::connect_t, sender const& s,
                Receiver&& receiver) noexcept(std::
                    is_nothrow_constructible_v<std::decay_t<Receiver>,
                        Receiver>)
            {
                return {HPX_FORWARD(Receiver, receiver),
                    s.sched_.get_underlying_scheduler()};
            }

            template <typename Receiver>
            friend operation_state<std::decay_t<Receiver>> tag_invoke(
                stdexec::connect_t, sender&& s,
                Receiver&& receiver) noexcept(std::
                    is_nothrow_constructible_v<std::decay_t<Receiver>,
                        Receiver>)
            {
                return {HPX_FORWARD(Receiver, receiver),
                    s.sched_.get_underlying_scheduler()};
            }

            struct env
            {
                Scheduler const& sched_;

                // P2079R10: expose completion scheduler for set_value_t
                // and set_stopped_t
                auto query(
                    stdexec::get_completion_scheduler_t<stdexec::set_value_t>)
                    const noexcept
                {
                    return sched_;
                }

                auto query(
                    stdexec::get_completion_scheduler_t<stdexec::set_stopped_t>)
                    const noexcept
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

            friend env tag_invoke(stdexec::get_env_t, sender const& s) noexcept
            {
                return {s.sched_};
            }
        };

        // Direct schedule() member for modern stdexec
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

        // Required for stdexec domain resolution: when a bulk sender's
        // completing domain is resolved, stdexec queries the completion
        // scheduler with get_completion_domain_t<set_value_t>. Without
        // this, the resolution falls to default_domain and our
        // parallel_scheduler_domain::transform_sender is never called.
        parallel_scheduler_domain query(
            stdexec::get_completion_domain_t<stdexec::set_value_t>)
            const noexcept
        {
            return {};
        }
#endif

        thread_pool_policy_scheduler<hpx::launch> const&
        get_underlying_scheduler() const noexcept
        {
            return scheduler_;
        }

        hpx::threads::mask_type const& get_pu_mask() const noexcept
        {
            return pu_mask_;
        }

    private:
        thread_pool_policy_scheduler<hpx::launch> scheduler_;
        // Cached PU mask — computed once, reused for every bulk_chunked call.
        hpx::threads::mask_type pu_mask_;
    };

    // Stream output operator for parallel_scheduler
    inline std::ostream& operator<<(std::ostream& os, parallel_scheduler const&)
    {
        return os << "parallel_scheduler";
    }

    // P2079R10 get_parallel_scheduler function
    inline parallel_scheduler get_parallel_scheduler()
    {
        static const parallel_scheduler default_sched = []() {
            auto pool = detail::get_default_parallel_pool();
            if (!pool)
            {
                std::terminate(); // As per P2079R10, terminate if backend is unavailable
            }
            return parallel_scheduler(thread_pool_policy_scheduler<hpx::launch>(
                pool, hpx::launch::async));
        }();
        return default_sched;
    }

#endif    // HPX_HAVE_STDEXEC

}    // namespace hpx::execution::experimental
