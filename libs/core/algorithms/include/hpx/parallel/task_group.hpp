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
#include <hpx/errors/exception_list.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/executors/parallel_executor.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/synchronization/latch.hpp>

#include <exception>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace hpx { namespace execution { namespace experimental {

    ///////////////////////////////////////////////////////////////////////////
    class task_group
    {
    public:
        task_group()
          : latch_(1)
          , has_arrived_(false)
        {
        }

#if defined(HPX_DEBUG)
        ~task_group()
        {
            // wait() must have been called
            HPX_ASSERT(latch_.is_ready());
        }
#else
        ~task_group() = default;
#endif

    private:
        struct on_exit
        {
            explicit on_exit(hpx::lcos::local::latch& l)
              : latch_(&l)
            {
                latch_->count_up(1);
            }

            ~on_exit()
            {
                if (latch_)
                {
                    latch_->count_down(1);
                }
            }

            on_exit(on_exit const& rhs) = delete;
            on_exit& operator=(on_exit const& rhs) = delete;

            on_exit(on_exit&& rhs)
              : latch_(rhs.latch_)
            {
                rhs.latch_ = nullptr;
            }
            on_exit& operator=(on_exit&& rhs)
            {
                latch_ = rhs.latch_;
                rhs.latch_ = nullptr;
                return *this;
            }

            hpx::lcos::local::latch* latch_;
        };

    public:
        /// Spawns a task to compute f() and returns immediately.
        // clang-format off
        template <typename Executor, typename F, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any<std::decay_t<Executor>>::value
            )>
        // clang-format on
        void run(Executor&& exec, F&& f, Ts&&... ts)
        {
            // make sure exceptions don't leave the latch in the wrong state
            on_exit l(latch_);

            hpx::parallel::execution::post(HPX_FORWARD(Executor, exec),
                [this, l = HPX_MOVE(l), f = HPX_FORWARD(F, f),
                    t = hpx::make_tuple(HPX_FORWARD(Ts, ts)...)]() mutable {
                    // latch needs to be released before the lambda exits
                    on_exit _(HPX_MOVE(l));
                    std::exception_ptr p;
                    try
                    {
                        hpx::util::invoke_fused(HPX_MOVE(f), HPX_MOVE(t));
                        return;
                    }
                    catch (...)
                    {
                        p = std::current_exception();
                    }

                    // The exception is set outside the catch block since
                    // set_exception may yield. Ending the catch block on a
                    // different worker thread than where it was started may
                    // lead to segfaults.
                    add_exception(HPX_MOVE(p));
                });
        }

        // clang-format off
        template <typename F, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                !hpx::traits::is_executor_any<std::decay_t<F>>::value
            )>
        // clang-format on
        void run(F&& f, Ts&&... ts)
        {
            run(execution::parallel_executor{}, HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
        }

        /// Waits for all tasks in the group to complete.
        void wait()
        {
            if (!has_arrived_)
            {
                latch_.arrive_and_wait();
                has_arrived_ = true;
                if (errors_.size() != 0)
                {
                    throw errors_;
                }
            }
        }

        void add_exception(std::exception_ptr p)
        {
            errors_.add(HPX_MOVE(p));
        }

    private:
        hpx::lcos::local::latch latch_;
        hpx::exception_list errors_;
        bool has_arrived_;
    };
}}}    // namespace hpx::execution::experimental
