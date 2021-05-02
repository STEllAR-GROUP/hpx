//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file task_group.hpp

#pragma once

#include <hpx/config.hpp>

#include <hpx/assert.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/executors/parallel_executor.hpp>
#include <hpx/synchronization/latch.hpp>

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
        template <typename F>
        void run(F&& f)
        {
            latch_.count_up(1);
            hpx::parallel::execution::post(execution::parallel_executor{},
                [this, f = std::forward<F>(f)]() mutable {
                    on_exit _(latch_);
                    f();
                });
        }

        template <typename Executor, typename F>
        void run(Executor&& exec, F&& f)
        {
            latch_.count_up(1);
            hpx::parallel::execution::post(std::forward<Executor>(exec),
                [this, f = std::forward<F>(f)]() mutable {
                    on_exit _(latch_);
                    f();
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
