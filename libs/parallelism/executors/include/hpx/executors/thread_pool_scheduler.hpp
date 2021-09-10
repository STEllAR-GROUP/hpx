//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <hpx/threading_base/register_thread.hpp>

#include <cstddef>
#include <exception>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx { namespace execution { namespace experimental {
    struct thread_pool_scheduler
    {
        constexpr thread_pool_scheduler() = default;
        explicit thread_pool_scheduler(hpx::threads::thread_pool_base* pool)
          : pool_(pool)
        {
        }

        /// \cond NOINTERNAL
        bool operator==(thread_pool_scheduler const& rhs) const noexcept
        {
            return pool_ == rhs.pool_ && priority_ == rhs.priority_ &&
                stacksize_ == rhs.stacksize_ &&
                schedulehint_ == rhs.schedulehint_;
        }

        bool operator!=(thread_pool_scheduler const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        hpx::threads::thread_pool_base* get_thread_pool()
        {
            HPX_ASSERT(pool_);
            return pool_;
        }

        // support with_priority property
        friend thread_pool_scheduler tag_dispatch(
            hpx::execution::experimental::with_priority_t,
            thread_pool_scheduler const& scheduler,
            hpx::threads::thread_priority priority)
        {
            auto sched_with_priority = scheduler;
            sched_with_priority.priority_ = priority;
            return sched_with_priority;
        }

        friend hpx::threads::thread_priority tag_dispatch(
            hpx::execution::experimental::get_priority_t,
            thread_pool_scheduler const& scheduler)
        {
            return scheduler.priority_;
        }

        // support with_stacksize property
        friend thread_pool_scheduler tag_dispatch(
            hpx::execution::experimental::with_stacksize_t,
            thread_pool_scheduler const& scheduler,
            hpx::threads::thread_stacksize stacksize)
        {
            auto sched_with_stacksize = scheduler;
            sched_with_stacksize.stacksize_ = stacksize;
            return sched_with_stacksize;
        }

        friend hpx::threads::thread_stacksize tag_dispatch(
            hpx::execution::experimental::get_stacksize_t,
            thread_pool_scheduler const& scheduler)
        {
            return scheduler.stacksize_;
        }

        // support with_hint property
        friend thread_pool_scheduler tag_dispatch(
            hpx::execution::experimental::with_hint_t,
            thread_pool_scheduler const& scheduler,
            hpx::threads::thread_schedule_hint hint)
        {
            auto sched_with_hint = scheduler;
            sched_with_hint.schedulehint_ = hint;
            return sched_with_hint;
        }

        friend hpx::threads::thread_schedule_hint tag_dispatch(
            hpx::execution::experimental::get_hint_t,
            thread_pool_scheduler const& scheduler)
        {
            return scheduler.schedulehint_;
        }

        // support with_annotation property
        friend constexpr thread_pool_scheduler tag_dispatch(
            hpx::execution::experimental::with_annotation_t,
            thread_pool_scheduler const& scheduler, char const* annotation)
        {
            auto sched_with_annotation = scheduler;
            sched_with_annotation.annotation_ = annotation;
            return sched_with_annotation;
        }

        friend thread_pool_scheduler tag_dispatch(
            hpx::execution::experimental::with_annotation_t,
            thread_pool_scheduler const& scheduler, std::string annotation)
        {
            auto sched_with_annotation = scheduler;
            sched_with_annotation.annotation_ =
                hpx::util::detail::store_function_annotation(
                    std::move(annotation));
            return sched_with_annotation;
        }

        // support get_annotation property
        friend constexpr char const* tag_dispatch(
            hpx::execution::experimental::get_annotation_t,
            thread_pool_scheduler const& scheduler) noexcept
        {
            return scheduler.annotation_;
        }

        template <typename F>
        void execute(F&& f) const
        {
            char const* annotation = annotation_ == nullptr ?
                traits::get_function_annotation<std::decay_t<F>>::call(f) :
                annotation_;

            threads::thread_init_data data(
                threads::make_thread_function_nullary(std::forward<F>(f)),
                annotation, priority_, schedulehint_, stacksize_);
            threads::register_work(data, pool_);
        }

        template <typename Scheduler, typename Receiver>
        struct operation_state
        {
            std::decay_t<Scheduler> scheduler;
            std::decay_t<Receiver> receiver;

            template <typename Scheduler_, typename Receiver_>
            operation_state(Scheduler_&& scheduler, Receiver_&& receiver)
              : scheduler(std::forward<Scheduler_>(scheduler))
              , receiver(std::forward<Receiver_>(receiver))
            {
            }

            operation_state(operation_state&&) = delete;
            operation_state(operation_state const&) = delete;
            operation_state& operator=(operation_state&&) = delete;
            operation_state& operator=(operation_state const&) = delete;

            void start() & noexcept
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        scheduler.execute(
                            [receiver = std::move(receiver)]() mutable {
                                hpx::execution::experimental::set_value(
                                    std::move(receiver));
                            });
                    },
                    [&](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            std::move(receiver), std::move(ep));
                    });
            }
        };

        template <typename Scheduler>
        struct sender
        {
            std::decay_t<Scheduler> scheduler;

            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types = Variant<Tuple<>>;

            template <template <typename...> class Variant>
            using error_types = Variant<std::exception_ptr>;

            static constexpr bool sends_done = false;

            template <typename Receiver>
            operation_state<Scheduler, Receiver> connect(Receiver&& receiver) &&
            {
                return {std::move(scheduler), std::forward<Receiver>(receiver)};
            }

            template <typename Receiver>
            operation_state<Scheduler, Receiver> connect(Receiver&& receiver) &
            {
                return {scheduler, std::forward<Receiver>(receiver)};
            }

            template <typename CPO,
                HPX_CONCEPT_REQUIRES_(std::is_same_v<CPO,
                    hpx::execution::experimental::set_value_t>)>
            friend constexpr auto tag_dispatch(
                hpx::execution::experimental::get_completion_scheduler_t<CPO>,
                sender const& s)
            {
                return s.scheduler;
            }
        };

        template <template <class...> class Tuple,
            template <class...> class Variant>
        using value_types = Variant<Tuple<>>;

        template <template <class...> class Variant>
        using error_types = Variant<std::exception_ptr>;

        static constexpr bool sends_done = false;

        template <typename Receiver>
        operation_state<thread_pool_scheduler, Receiver> connect(
            Receiver&& receiver) &&
        {
            return {*this, std::forward<Receiver>(receiver)};
        }

        constexpr sender<thread_pool_scheduler> schedule() const
        {
            return {*this};
        }
        /// \endcond

    private:
        /// \cond NOINTERNAL
        hpx::threads::thread_pool_base* pool_ =
            hpx::threads::detail::get_self_or_default_pool();
        hpx::threads::thread_priority priority_ =
            hpx::threads::thread_priority::normal;
        hpx::threads::thread_stacksize stacksize_ =
            hpx::threads::thread_stacksize::small_;
        hpx::threads::thread_schedule_hint schedulehint_{};
        char const* annotation_ = nullptr;
        /// \endcond
    };
}}}    // namespace hpx::execution::experimental
