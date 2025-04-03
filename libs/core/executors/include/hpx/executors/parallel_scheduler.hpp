// Copyright (c) 2025 Sai Charan Arvapally
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution/algorithms/bulk.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/executors/thread_pool_scheduler.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/threading_base/detail/get_default_pool.hpp>
#include <hpx/threading_base/thread_pool_base.hpp>

#include <atomic>
#include <exception>
#include <memory>
#include <type_traits>
#include <utility>

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
        }

        parallel_scheduler(parallel_scheduler const&) noexcept = default;
        parallel_scheduler(parallel_scheduler&&) noexcept = default;
        parallel_scheduler& operator=(parallel_scheduler const&) noexcept = default;
        parallel_scheduler& operator=(parallel_scheduler&&) noexcept = default;

        friend bool operator==(parallel_scheduler const& lhs,
            parallel_scheduler const& rhs) noexcept
        {
            return lhs.pool_ == rhs.pool_;
        }

        constexpr hpx::threads::thread_pool_base* get_thread_pool() const noexcept
        {
            return pool_;
        }

        friend hpx::execution::experimental::forward_progress_guarantee
            tag_invoke(
                hpx::execution::experimental::get_forward_progress_guarantee_t,
                parallel_scheduler const&) noexcept
        {
            return hpx::execution::experimental::forward_progress_guarantee::parallel;
        }

    private:
        hpx::threads::thread_pool_base* pool_;
    };

    namespace detail {

        template <typename Scheduler, typename Receiver>
        struct parallel_operation_state
        {
            Scheduler scheduler;
            Receiver& receiver;

            template <typename S, typename R>
            parallel_operation_state(S&& s, R& r)
              : scheduler(HPX_FORWARD(S, s))
              , receiver(r)
            {
            }

            friend void tag_invoke(
                hpx::execution::experimental::start_t,
                parallel_operation_state& os) noexcept
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        thread_pool_scheduler exec{os.scheduler.get_thread_pool()};
                        exec.execute([&r = os.receiver]() mutable {
                            hpx::execution::experimental::set_value(r);
                        });
                    },
                    [&](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(os.receiver), HPX_MOVE(ep));
                    });
            }
        };

        // Helper functions for determining shape size
        template <typename S>
        std::enable_if_t<std::is_integral_v<S>, std::size_t> get_shape_size(S const& sh)
        {
            return static_cast<std::size_t>(sh);
        }

        template <typename S>
        std::enable_if_t<!std::is_integral_v<S>, std::size_t> get_shape_size(S const& sh)
        {
            return std::distance(hpx::util::begin(sh), hpx::util::end(sh));
        }

        template <typename Sender, typename Shape, typename F>
        struct parallel_bulk_sender
        {
            parallel_scheduler scheduler;
            Shape shape;
            F f;

            parallel_bulk_sender(parallel_scheduler&& sched, Shape sh, F&& func)
              : scheduler(HPX_MOVE(sched))
              , shape(sh)
              , f(HPX_FORWARD(F, func))
            {
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
            friend constexpr auto tag_invoke(
                hpx::execution::experimental::get_completion_signatures_t,
                parallel_bulk_sender const&, Env) noexcept -> completion_signatures
            {
                return {};
            }
        };

        template <typename Receiver, typename Shape, typename F>
        struct parallel_bulk_operation_state
        {
            parallel_scheduler scheduler;
            Receiver& receiver;
            Shape shape;
            F f;
            std::size_t size;
            std::shared_ptr<std::atomic<int>> tasks_remaining;

            template <typename R, typename S>
            parallel_bulk_operation_state(parallel_scheduler&& sched, R& r, S sh, F&& func)
              : scheduler(HPX_MOVE(sched))
              , receiver(r)
              , shape(sh)
              , f(HPX_FORWARD(F, func))
            {
                static_assert(std::is_integral_v<S> || hpx::traits::is_range_v<S>,
                    "Shape must be an integral type or a range");
                size = get_shape_size(sh);
                std::size_t num_threads = scheduler.get_thread_pool()->get_os_thread_count();
                std::size_t chunk_size = (size + num_threads - 1) / num_threads;
                std::size_t num_chunks = (size + chunk_size - 1) / chunk_size;
                tasks_remaining = std::make_shared<std::atomic<int>>(static_cast<int>(num_chunks));
            }

            friend void tag_invoke(
                hpx::execution::experimental::start_t,
                parallel_bulk_operation_state& os) noexcept
            {
                hpx::detail::try_catch_exception_ptr(
                    [&]() {
                        thread_pool_scheduler exec{os.scheduler.get_thread_pool()};
                        std::size_t num_threads = os.scheduler.get_thread_pool()->get_os_thread_count();
                        std::size_t chunk_size = (os.size + num_threads - 1) / num_threads;

                        for (std::size_t t = 0; t < num_threads; ++t)
                        {
                            std::size_t start = t * chunk_size;
                            std::size_t end = (std::min)(start + chunk_size, os.size);
                            if (start >= os.size) break;

                            exec.execute([start, end, &os]() mutable {
                                if constexpr (std::is_integral_v<Shape>)
                                {
                                    for (std::size_t i = start; i < end; ++i)
                                    {
                                        HPX_INVOKE(os.f, static_cast<Shape>(i));
                                    }
                                }
                                else
                                {
                                    auto it = std::next(hpx::util::begin(os.shape), start);
                                    for (std::size_t i = start; i < end; ++i, ++it)
                                    {
                                        HPX_INVOKE(os.f, *it);
                                    }
                                }
                                if (--(*os.tasks_remaining) == 0)
                                {
                                    hpx::execution::experimental::set_value(os.receiver);
                                }
                            });
                        }
                    },
                    [&](std::exception_ptr ep) {
                        hpx::execution::experimental::set_error(
                            HPX_MOVE(os.receiver), HPX_MOVE(ep));
                    });
            }
        };

    } // namespace detail

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
        friend constexpr auto tag_invoke(
            hpx::execution::experimental::get_completion_signatures_t,
            parallel_sender const&, Env) noexcept -> completion_signatures
        {
            return {};
        }

        template <typename Receiver>
        friend auto tag_invoke(
            hpx::execution::experimental::connect_t,
            parallel_sender&& s,
            Receiver& r)
        {
            return detail::parallel_operation_state<parallel_scheduler, Receiver>{
                HPX_MOVE(s.scheduler), r};
        }

        template <typename Receiver>
        friend auto tag_invoke(
            hpx::execution::experimental::connect_t,
            parallel_sender& s,
            Receiver& r)
        {
            return detail::parallel_operation_state<parallel_scheduler, Receiver>{s.scheduler, r};
        }

        template <typename Shape, typename F>
        friend auto tag_invoke(
            hpx::execution::experimental::bulk_t,
            parallel_sender&& s,
            Shape shape,
            F&& f)
        {
            return detail::parallel_bulk_sender<parallel_sender, Shape, F>{
                HPX_MOVE(s.scheduler), shape, HPX_FORWARD(F, f)};
        }

        template <typename Shape, typename F, typename Receiver>
        friend auto tag_invoke(
            hpx::execution::experimental::connect_t,
            detail::parallel_bulk_sender<parallel_sender, Shape, F>&& s,
            Receiver& r)
        {
            return detail::parallel_bulk_operation_state<Receiver, Shape, F>{
                HPX_MOVE(s.scheduler), r, s.shape, HPX_MOVE(s.f)};
        }

#if defined(HPX_HAVE_STDEXEC)
        struct env
        {
            parallel_scheduler const& sched;

            friend auto tag_invoke(
                hpx::execution::experimental::get_completion_scheduler_t<set_value_t>,
                env const& e) const noexcept -> parallel_scheduler
            {
                return e.sched;
            }

            friend auto tag_invoke(
                hpx::execution::experimental::get_completion_scheduler_t<set_stopped_t>,
                env const& e) const noexcept -> parallel_scheduler
            {
                return e.sched;
            }
        };

        friend env tag_invoke(
            hpx::execution::experimental::get_env_t,
            parallel_sender const& s) const noexcept
        {
            return {s.scheduler};
        }
#endif
    };

    inline parallel_sender tag_invoke(
        hpx::execution::experimental::schedule_t,
        parallel_scheduler&& sched) noexcept
    {
        return {HPX_MOVE(sched)};
    }

    inline parallel_sender tag_invoke(
        hpx::execution::experimental::schedule_t,
        parallel_scheduler const& sched) noexcept
    {
        return {sched};
    }

    inline parallel_scheduler get_system_scheduler() noexcept
    {
        return parallel_scheduler{};
    }

}    // namespace hpx::execution::experimental
