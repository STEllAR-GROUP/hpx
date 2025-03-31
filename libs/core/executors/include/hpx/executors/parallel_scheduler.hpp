//  Copyright (c) 2025 Sai Charan Arvapally
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/assert.hpp>
#include <hpx/async_base/post.hpp>
#include <hpx/execution/algorithms/bulk.hpp>
#include <atomic>
#include <exception>
#include <iostream>
#include <memory>

namespace hpx::execution::experimental {

    struct parallel_scheduler
    {
        using execution_category = parallel_execution_tag;

        explicit parallel_scheduler(
            hpx::threads::thread_pool_base* pool =
                hpx::threads::detail::get_self_or_default_pool()) noexcept
          : pool_(pool)
        {
            HPX_ASSERT(pool_);
            // std::cout << "Scheduler created with pool: " << pool_ << std::endl;
        }

        parallel_scheduler(const parallel_scheduler&) noexcept = default;
        parallel_scheduler(parallel_scheduler&&) noexcept = default;
        parallel_scheduler& operator=(
            const parallel_scheduler&) noexcept = default;
        parallel_scheduler& operator=(parallel_scheduler&&) noexcept = default;

        friend constexpr bool operator==(const parallel_scheduler& lhs,
            const parallel_scheduler& rhs) noexcept
        {
            return lhs.pool_ == rhs.pool_;
        }

        hpx::threads::thread_pool_base* get_thread_pool() const noexcept
        {
            return pool_;
        }

        friend constexpr hpx::execution::experimental::
            forward_progress_guarantee
            tag_invoke(
                hpx::execution::experimental::get_forward_progress_guarantee_t,
                const parallel_scheduler&) noexcept
        {
            return hpx::execution::experimental::forward_progress_guarantee::
                parallel;
        }

    private:
        hpx::threads::thread_pool_base* pool_;
    };

    template <typename Scheduler, typename Receiver>
    struct operation_state
    {
        Scheduler scheduler;
        Receiver& receiver;

        template <typename S, typename R>
        operation_state(S&& s, R& r)
          : scheduler(HPX_FORWARD(S, s))
          , receiver(r)
        {
            // std::cout << "Operation state created" << std::endl;
        }

        friend void tag_invoke(start_t, operation_state& os) noexcept
        {
            // std::cout << "start() called" << std::endl;
            hpx::detail::try_catch_exception_ptr(
                [&]() {
                    // std::cout << "Scheduling task on pool" << std::endl;
                    thread_pool_scheduler exec{os.scheduler.get_thread_pool()};
                    exec.execute([&r = os.receiver]() mutable {
                        // std::cout << "Task executing on pool" << std::endl;
                        hpx::execution::experimental::set_value(r);
                    });
                },
                [&](std::exception_ptr ep) {
                    std::cerr << "Error occurred" << std::endl;
                    hpx::execution::experimental::set_error(
                        HPX_MOVE(os.receiver), HPX_MOVE(ep));
                });
        }
    };

    template <typename Shape, typename F>
    struct bulk_sender
    {
        parallel_scheduler scheduler;
        Shape shape;
        F f;

        bulk_sender(parallel_scheduler&& sched, Shape sh, F&& func)
          : scheduler(HPX_MOVE(sched))
          , shape(sh)
          , f(HPX_FORWARD(F, func))
        {
            // std::cout << "Bulk sender created with shape: " << shape << std::endl;
        }

#if defined(HPX_HAVE_STDEXEC)
        using sender_concept = hpx::execution::experimental::sender_t;
#endif
        using completion_signatures =
            hpx::execution::experimental::completion_signatures<
                hpx::execution::experimental::set_value_t(),
                hpx::execution::experimental::set_error_t(std::exception_ptr),
                hpx::execution::experimental::set_stopped_t()>;

        template <typename Env>
        friend auto tag_invoke(
            hpx::execution::experimental::get_completion_signatures_t,
            bulk_sender const&, Env) noexcept -> completion_signatures
        {
            return {};
        }
    };

    template <typename Receiver, typename Shape, typename F>
    struct bulk_operation_state
    {
        parallel_scheduler scheduler;
        Receiver& receiver;    // Store by reference
        Shape shape;
        F f;
        std::shared_ptr<std::atomic<int>> tasks_remaining;

        bulk_operation_state(
            parallel_scheduler&& sched, Receiver& r, Shape sh, F&& func)
          : scheduler(HPX_MOVE(sched))
          , receiver(r)
          , shape(sh)
          , f(HPX_FORWARD(F, func))
          , tasks_remaining(
                std::make_shared<std::atomic<int>>(static_cast<int>(shape)))
        {
            // std::cout << "Bulk operation state created" << std::endl;
        }

        friend void tag_invoke(start_t, bulk_operation_state& os) noexcept
        {
            // std::cout << "Bulk start() called" << std::endl;
            hpx::detail::try_catch_exception_ptr(
                [&]() {
                    thread_pool_scheduler exec{os.scheduler.get_thread_pool()};
                    for (Shape i = 0; i < os.shape; ++i)
                    {
                        exec.execute([i, &os]() mutable {
                            // std::cout << "Bulk task executing for index: " << i <<;
                            os.f(i);
                            if (--(*os.tasks_remaining) == 0)
                            {
                                // std::cout << "All bulk tasks completed" << std::endl;
                                hpx::execution::experimental::set_value(
                                    os.receiver);
                            }
                        });
                    }
                },
                [&](std::exception_ptr ep) {
                    std::cerr << "Bulk error occurred" << std::endl;
                    hpx::execution::experimental::set_error(
                        HPX_MOVE(os.receiver), HPX_MOVE(ep));
                });
        }
    };

    template <typename Shape, typename F, typename Receiver>
    auto tag_invoke(connect_t, bulk_sender<Shape, F>&& s, Receiver& r)
    {
        return bulk_operation_state<Receiver, Shape, F>{
            HPX_MOVE(s.scheduler), r, s.shape, HPX_MOVE(s.f)};
    }

    struct parallel_sender
    {
        parallel_scheduler scheduler;

#if defined(HPX_HAVE_STDEXEC)
        using sender_concept = hpx::execution::experimental::sender_t;
#endif
        using completion_signatures =
            hpx::execution::experimental::completion_signatures<
                hpx::execution::experimental::set_value_t(),
                hpx::execution::experimental::set_error_t(std::exception_ptr),
                hpx::execution::experimental::set_stopped_t()>;

        template <typename Env>
        friend auto tag_invoke(
            hpx::execution::experimental::get_completion_signatures_t,
            parallel_sender const&, Env) noexcept -> completion_signatures
        {
            return {};
        }

        template <typename Receiver>
        friend auto tag_invoke(connect_t, parallel_sender&& s, Receiver& r)
        {
            return operation_state<parallel_scheduler, Receiver>{
                HPX_MOVE(s.scheduler), r};
        }

        template <typename Receiver>
        friend auto tag_invoke(connect_t, parallel_sender& s, Receiver& r)
        {
            return operation_state<parallel_scheduler, Receiver>{
                s.scheduler, r};
        }

        template <typename Shape, typename F>
        friend auto tag_invoke(bulk_t, parallel_sender&& s, Shape shape, F&& f)
        {
            return bulk_sender<Shape, F>{
                HPX_MOVE(s.scheduler), shape, HPX_FORWARD(F, f)};
        }

#if defined(HPX_HAVE_STDEXEC)
        struct env
        {
            parallel_scheduler const& sched;

            friend constexpr auto tag_invoke(
                hpx::execution::experimental::get_completion_scheduler_t<
                    set_value_t>,
                env const& e) noexcept -> parallel_scheduler
            {
                // std::cout << "get_completion_scheduler<set_value> called" << std::endl;
                return e.sched;
            }

            friend constexpr auto tag_invoke(
                hpx::execution::experimental::get_completion_scheduler_t<
                    set_stopped_t>,
                env const& e) noexcept -> parallel_scheduler
            {
                return e.sched;
            }
        };

        friend constexpr env tag_invoke(hpx::execution::experimental::get_env_t,
            parallel_sender const& s) noexcept
        {
            return {s.scheduler};
        }
#endif
    };

    inline parallel_sender tag_invoke(hpx::execution::experimental::schedule_t,
        parallel_scheduler&& sched) noexcept
    {
        // std::cout << "schedule() called" << std::endl;
        return {HPX_MOVE(sched)};
    }

    inline parallel_sender tag_invoke(hpx::execution::experimental::schedule_t,
        const parallel_scheduler& sched) noexcept
    {
        // std::cout << "schedule() called (const)" << std::endl;
        return {sched};
    }

    inline parallel_scheduler get_system_scheduler() noexcept
    {
        return parallel_scheduler{};
    }

}    // namespace hpx::execution::experimental
