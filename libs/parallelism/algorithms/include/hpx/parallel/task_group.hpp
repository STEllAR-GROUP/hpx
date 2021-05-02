//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file task_group.hpp

#pragma once

#include <hpx/config.hpp>

#include <hpx/assert.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/executors/parallel_executor.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/synchronization/latch.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace execution { namespace experimental {

    ///////////////////////////////////////////////////////////////////////////
    class task_group
    {
    public:
        task_group()
          : latch_(1)
        {
        }

        ~task_group()
        {
            // wait() must have been called
            HPX_ASSERT(latch_.is_ready());
        }

    private:
        struct on_exit
        {
            on_exit(hpx::lcos::local::latch& latch)
              : latch_(latch)
            {
            }

            ~on_exit()
            {
                latch_.count_down(1);
            }

            hpx::lcos::local::latch& latch_;
        };

    public:
        /// Spawns a task to compute f() and returns immediately.
        // clang-format off
        template <typename F, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                !hpx::traits::is_executor_any<std::decay_t<F>>::value
            )>
        // clang-format on
        void run(F&& f, Ts&&... ts)
        {
            latch_.count_up(1);
            hpx::parallel::execution::post(execution::parallel_executor{},
                [this, f = std::forward<F>(f),
                    t = hpx::forward_as_tuple(
                        std::forward<Ts>(ts)...)]() mutable {
                    on_exit _(latch_);
                    hpx::util::invoke_fused(std::move(f), std::move(t));
                });
        }

        // clang-format off
        template <typename Executor, typename F, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any<std::decay_t<Executor>>::value
            )>
        // clang-format on
        void run(Executor&& exec, F&& f, Ts&&... ts)
        {
            latch_.count_up(1);
            hpx::parallel::execution::post(std::forward<Executor>(exec),
                [this, f = std::forward<F>(f),
                    t = hpx::forward_as_tuple(
                        std::forward<Ts>(ts)...)]() mutable {
                    on_exit _(latch_);
                    hpx::util::invoke_fused(std::move(f), std::move(t));
                });
        }

        /// Waits for all tasks in the group to complete.
        void wait()
        {
            latch_.arrive_and_wait();
        }

    private:
        hpx::lcos::local::latch latch_;
    };
}}}    // namespace hpx::execution::experimental
