//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/async_base/launch_policy.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution/detail/post_policy_dispatch.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution_base/completion_scheduler.hpp>
#include <hpx/execution_base/completion_signatures.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <hpx/threading_base/register_thread.hpp>

#include <cstddef>
#include <exception>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    namespace detail {
        template <typename Policy>
        struct get_default_scheduler_policy
        {
            static constexpr Policy call() noexcept
            {
                return Policy{};
            }
        };

        template <>
        struct get_default_scheduler_policy<hpx::launch>
        {
            static constexpr hpx::launch::async_policy call() noexcept
            {
                return hpx::launch::async_policy{};
            }
        };
    }    // namespace detail

    template <typename Policy>
    struct thread_pool_policy_scheduler
    {
        // Associate the parallel_execution_tag tag type as a default with this
        // scheduler, except if the given launch policy is synch.
        using execution_category =
            std::conditional_t<std::is_same_v<Policy, launch::sync_policy>,
                sequenced_execution_tag, parallel_execution_tag>;

        constexpr thread_pool_policy_scheduler(
            Policy l = experimental::detail::get_default_scheduler_policy<
                Policy>::call())
          : policy_(l)
        {
        }

        explicit thread_pool_policy_scheduler(
            hpx::threads::thread_pool_base* pool,
            Policy l = experimental::detail::get_default_scheduler_policy<
                Policy>::call())
          : pool_(pool)
          , policy_(l)
        {
        }

        /// \cond NOINTERNAL
        bool operator==(thread_pool_policy_scheduler const& rhs) const noexcept
        {
            return pool_ == rhs.pool_ && policy_ == rhs.policy_;
        }

        bool operator!=(thread_pool_policy_scheduler const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        hpx::threads::thread_pool_base* get_thread_pool()
        {
            HPX_ASSERT(pool_);
            return pool_;
        }

        // support with_priority property
        friend thread_pool_policy_scheduler tag_invoke(
            hpx::execution::experimental::with_priority_t,
            thread_pool_policy_scheduler const& scheduler,
            hpx::threads::thread_priority priority)
        {
            auto sched_with_priority = scheduler;
            sched_with_priority.policy_ =
                hpx::execution::experimental::with_priority(
                    sched_with_priority.policy_, priority);
            return sched_with_priority;
        }

        friend hpx::threads::thread_priority tag_invoke(
            hpx::execution::experimental::get_priority_t,
            thread_pool_policy_scheduler const& scheduler)
        {
            return hpx::execution::experimental::get_priority(
                scheduler.policy_);
        }

        // support with_stacksize property
        friend thread_pool_policy_scheduler tag_invoke(
            hpx::execution::experimental::with_stacksize_t,
            thread_pool_policy_scheduler const& scheduler,
            hpx::threads::thread_stacksize stacksize)
        {
            auto sched_with_stacksize = scheduler;
            sched_with_stacksize.policy_ =
                hpx::execution::experimental::with_stacksize(
                    sched_with_stacksize.policy_, stacksize);
            return sched_with_stacksize;
        }

        friend hpx::threads::thread_stacksize tag_invoke(
            hpx::execution::experimental::get_stacksize_t,
            thread_pool_policy_scheduler const& scheduler)
        {
            return hpx::execution::experimental::get_stacksize(
                scheduler.policy_);
        }

        // support with_hint property
        friend thread_pool_policy_scheduler tag_invoke(
            hpx::execution::experimental::with_hint_t,
            thread_pool_policy_scheduler const& scheduler,
            hpx::threads::thread_schedule_hint hint)
        {
            auto sched_with_hint = scheduler;
            sched_with_hint.policy_ = hpx::execution::experimental::with_hint(
                sched_with_hint.policy_, hint);
            return sched_with_hint;
        }

        friend hpx::threads::thread_schedule_hint tag_invoke(
            hpx::execution::experimental::get_hint_t,
            thread_pool_policy_scheduler const& scheduler)
        {
            return hpx::execution::experimental::get_hint(scheduler.policy_);
        }

        // support with_annotation property
        friend constexpr thread_pool_policy_scheduler tag_invoke(
            hpx::execution::experimental::with_annotation_t,
            thread_pool_policy_scheduler const& scheduler,
            char const* annotation)
        {
            auto sched_with_annotation = scheduler;
            sched_with_annotation.annotation_ = annotation;
            return sched_with_annotation;
        }

        friend thread_pool_policy_scheduler tag_invoke(
            hpx::execution::experimental::with_annotation_t,
            thread_pool_policy_scheduler const& scheduler,
            std::string annotation)
        {
            auto sched_with_annotation = scheduler;
            sched_with_annotation.annotation_ =
                hpx::detail::store_function_annotation(HPX_MOVE(annotation));
            return sched_with_annotation;
        }

        // support get_annotation property
        friend constexpr char const* tag_invoke(
            hpx::execution::experimental::get_annotation_t,
            thread_pool_policy_scheduler const& scheduler) noexcept
        {
            return scheduler.annotation_;
        }

        template <typename F>
        void execute(F&& f) const
        {
            hpx::util::thread_description desc(f, annotation_);
            auto pool =
                pool_ ? pool_ : threads::detail::get_self_or_default_pool();

            hpx::detail::post_policy_dispatch<Policy>::call(
                policy_, desc, pool, HPX_FORWARD(F, f));
        }

        template <typename Scheduler, typename Receiver>
        struct operation_state
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Scheduler> scheduler;
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Receiver> receiver;

            template <typename Scheduler_, typename Receiver_>
            operation_state(Scheduler_&& scheduler, Receiver_&& receiver)
              : scheduler(HPX_FORWARD(Scheduler_, scheduler))
              , receiver(HPX_FORWARD(Receiver_, receiver))
            {
            }

            operation_state(operation_state&&) = delete;
            operation_state(operation_state const&) = delete;
            operation_state& operator=(operation_state&&) = delete;
            operation_state& operator=(operation_state const&) = delete;

            friend void tag_invoke(start_t, operation_state& os) noexcept
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        os.scheduler.execute(
                            [receiver = HPX_MOVE(os.receiver)]() mutable {
                                hpx::execution::experimental::set_value(
                                    HPX_MOVE(receiver));
                            });
                    },
                    [&](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(os.receiver), HPX_MOVE(ep));
                    });
            }
        };

        template <typename Scheduler>
        struct sender
        {
            HPX_NO_UNIQUE_ADDRESS std::decay_t<Scheduler> scheduler;

            template <typename Env>
            struct generate_completion_signatures
            {
                template <template <typename...> typename Tuple,
                    template <typename...> typename Variant>
                using value_types = Variant<Tuple<>>;

                template <template <typename...> typename Variant>
                using error_types = Variant<std::exception_ptr>;

                static constexpr bool sends_stopped = false;
            };

            template <typename Env>
            friend auto tag_invoke(get_completion_signatures_t, sender const&,
                Env) noexcept -> generate_completion_signatures<Env>;

            template <typename Receiver>
            friend operation_state<Scheduler, Receiver> tag_invoke(
                connect_t, sender&& s, Receiver&& receiver)
            {
                return {HPX_MOVE(s.scheduler), HPX_FORWARD(Receiver, receiver)};
            }

            template <typename Receiver>
            friend operation_state<Scheduler, Receiver> tag_invoke(
                connect_t, sender& s, Receiver&& receiver)
            {
                return {s.scheduler, HPX_FORWARD(Receiver, receiver)};
            }

            template <typename CPO,
                HPX_CONCEPT_REQUIRES_(std::is_same_v<CPO,
                    hpx::execution::experimental::set_value_t>)>
            friend constexpr auto tag_invoke(
                hpx::execution::experimental::get_completion_scheduler_t<CPO>,
                sender const& s)
            {
                return s.scheduler;
            }
        };

        friend constexpr sender<thread_pool_policy_scheduler> tag_invoke(
            schedule_t, thread_pool_policy_scheduler&& sched)
        {
            return {HPX_MOVE(sched)};
        }

        friend constexpr sender<thread_pool_policy_scheduler> tag_invoke(
            schedule_t, thread_pool_policy_scheduler const& sched)
        {
            return {sched};
        }

        Policy policy() const
        {
            return policy_;
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        hpx::threads::thread_pool_base* pool_ =
            hpx::threads::detail::get_self_or_default_pool();
        Policy policy_;
        char const* annotation_ = nullptr;
        /// \endcond
    };

    using thread_pool_scheduler = thread_pool_policy_scheduler<hpx::launch>;
}    // namespace hpx::execution::experimental
