// Copyright (c) 2025 Sai Charan Arvapally
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/execution_base/stdexec_forward.hpp>
#include <hpx/executors/thread_pool_scheduler.hpp>
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

    // Forward declarations
    class parallel_scheduler;
    struct parallel_scheduler_sender;

    // P2079R10 parallel_scheduler implementation
    class parallel_scheduler
    {
    public:
        // Deleted default constructor
        parallel_scheduler() = delete;

        // Constructor from thread_pool_policy_scheduler
        explicit parallel_scheduler(
            thread_pool_policy_scheduler<hpx::launch::async_policy>
                sched) noexcept
          : scheduler_(sched)
        {
        }

        // Copy constructor
        parallel_scheduler(parallel_scheduler const& other) noexcept
          : scheduler_(other.scheduler_)
        {
        }

        // Move constructor
        parallel_scheduler(parallel_scheduler&& other) noexcept
          : scheduler_(HPX_MOVE(other.scheduler_))
        {
        }

        // Copy assignment
        parallel_scheduler& operator=(parallel_scheduler const& other) noexcept
        {
            if (this != &other)
            {
                scheduler_ = other.scheduler_;
            }
            return *this;
        }

        // Move assignment
        parallel_scheduler& operator=(parallel_scheduler&& other) noexcept
        {
            if (this != &other)
            {
                scheduler_ = HPX_MOVE(other.scheduler_);
            }
            return *this;
        }

        // Equality comparison
        friend constexpr bool operator==(parallel_scheduler const& lhs,
            parallel_scheduler const& rhs) noexcept
        {
            return lhs.scheduler_ == rhs.scheduler_;
        }

        // Query for forward progress guarantee
        friend constexpr forward_progress_guarantee tag_invoke(
            get_forward_progress_guarantee_t,
            [[maybe_unused]] parallel_scheduler const&) noexcept
        {
            return forward_progress_guarantee::parallel;
        }

        // Schedule method returning a sender
        friend parallel_scheduler_sender tag_invoke(
            schedule_t, parallel_scheduler const& sched) noexcept;

        // Support get_completion_scheduler for scheduler concept
        template <typename CPO>
        friend auto tag_invoke(get_completion_scheduler_t<CPO>,
            [[maybe_unused]] parallel_scheduler const& sched) noexcept
            -> std::enable_if_t<hpx::meta::value<hpx::meta::one_of<CPO,
                                    set_value_t, set_stopped_t>>,
                parallel_scheduler const&>
        {
            return sched;
        }

        // Friend declaration to allow parallel_scheduler_sender access
        friend struct parallel_scheduler_sender;

    private:
        thread_pool_policy_scheduler<hpx::launch::async_policy> scheduler_;
    };

    // Sender for parallel_scheduler
    struct parallel_scheduler_sender
    {
        parallel_scheduler scheduler;
#if defined(HPX_HAVE_STDEXEC)
        using sender_concept = hpx::execution::experimental::sender_t;
        using completion_signatures =
            hpx::execution::experimental::completion_signatures<
                hpx::execution::experimental::set_value_t(),
                hpx::execution::experimental::set_error_t(std::exception_ptr),
                hpx::execution::experimental::set_stopped_t()>;

        template <typename Env>
        friend auto tag_invoke(
            hpx::execution::experimental::get_completion_signatures_t,
            parallel_scheduler_sender const&, Env) noexcept
            -> completion_signatures;
#else
        // Fallback types when stdexec is not available
        using sender_concept = void;    // Minimal fallback to allow compilation
        struct completion_signatures
        {
        };    // Empty struct as placeholder
#endif

        template <typename Receiver>
        friend auto tag_invoke(
            connect_t, parallel_scheduler_sender&& s, Receiver&& receiver)
        {
            // clang-format off
            return thread_pool_policy_scheduler<hpx::launch::async_policy>::
                operation_state<
                    thread_pool_policy_scheduler<hpx::launch::async_policy>,
                    Receiver>{
                    s.scheduler.scheduler_, HPX_FORWARD(Receiver, receiver)};
            // clang-format on
        }

        template <typename Receiver>
        friend auto tag_invoke(
            connect_t, parallel_scheduler_sender& s, Receiver&& receiver)
        {
            // clang-format off
            return thread_pool_policy_scheduler<hpx::launch::async_policy>::
                operation_state<
                    thread_pool_policy_scheduler<hpx::launch::async_policy>,
                    Receiver>{
                    s.scheduler.scheduler_, HPX_FORWARD(Receiver, receiver)};
            // clang-format on
        }

        struct env
        {
            parallel_scheduler const& sched;
            template <typename CPO>
            friend auto tag_invoke(
                hpx::execution::experimental::get_completion_scheduler_t<CPO>,
                env const& e) noexcept
                -> std::enable_if_t<hpx::meta::value<hpx::meta::one_of<CPO,
                                        set_value_t, set_stopped_t>>,
                    parallel_scheduler const&>
            {
                return e.sched;
            }

#if defined(HPX_HAVE_STDEXEC)
            friend auto tag_invoke(
                hpx::execution::experimental::get_stop_token_t,
                [[maybe_unused]] env const& e) noexcept
            {
                return hpx::execution::experimental::inplace_stop_token{};
            }
#else
            friend auto tag_invoke(hpx::execution::queries::get_stop_token_t,
                [[maybe_unused]] env const& e) noexcept
            {
                return hpx::execution::experimental::in_place_stop_token{};
            }
#endif
        };

        friend env tag_invoke(hpx::execution::experimental::get_env_t,
            parallel_scheduler_sender const& s) noexcept
        {
            return {s.scheduler};
        }
    };

    // Define schedule_t tag_invoke after parallel_scheduler_sender
    inline parallel_scheduler_sender tag_invoke(
        schedule_t, parallel_scheduler const& sched) noexcept
    {
        return {sched};
    }

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
        return parallel_scheduler(
            thread_pool_policy_scheduler<hpx::launch::async_policy>(pool));
    }

}    // namespace hpx::execution::experimental

namespace hpx::execution::experimental::system_context_replaceability {
    struct receiver_proxy;
    struct bulk_item_receiver_proxy;
    struct parallel_scheduler_backend;

    std::shared_ptr<parallel_scheduler_backend>
    query_parallel_scheduler_backend();
}    // namespace hpx::execution::experimental::system_context_replaceability
