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
#include <hpx/futures/detail/future_data.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/synchronization/latch.hpp>
#include <hpx/type_support/unused.hpp>

#include <atomic>
#include <exception>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace hpx { namespace execution { namespace experimental {

    ///////////////////////////////////////////////////////////////////////////
    class task_group
    {
    public:
        HPX_CORE_EXPORT task_group();
        HPX_CORE_EXPORT ~task_group();

    private:
        struct on_exit
        {
            HPX_CORE_EXPORT explicit on_exit(task_group& tg);
            HPX_CORE_EXPORT ~on_exit();

            on_exit(on_exit const& rhs) = delete;
            on_exit& operator=(on_exit const& rhs) = delete;

            HPX_CORE_EXPORT on_exit(on_exit&& rhs) noexcept;
            HPX_CORE_EXPORT on_exit& operator=(on_exit&& rhs) noexcept;

            hpx::lcos::local::latch* latch_;
        };

    public:
        // Spawns a task to compute f() and returns immediately.
        // clang-format off
        template <typename Executor, typename F, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_any_v<std::decay_t<Executor>>
            )>
        // clang-format on
        void run(Executor&& exec, F&& f, Ts&&... ts)
        {
            // make sure exceptions don't leave the latch in the wrong state
            on_exit l(*this);

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
                !hpx::traits::is_executor_any_v<std::decay_t<F>>
            )>
        // clang-format on
        void run(F&& f, Ts&&... ts)
        {
            run(execution::parallel_executor{}, HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
        }

        // Waits for all tasks in the group to complete.
        HPX_CORE_EXPORT void wait();

        // Add an exception to this task_group
        HPX_CORE_EXPORT void add_exception(std::exception_ptr p);

    private:
        friend class serialization::access;

        HPX_CORE_EXPORT void serialize(
            serialization::input_archive&, unsigned const);
        HPX_CORE_EXPORT void serialize(
            serialization::output_archive&, unsigned const);

    private:
        using shared_state_type = lcos::detail::future_data<void>;

        hpx::lcos::local::latch latch_;
        hpx::intrusive_ptr<shared_state_type> state_;
        hpx::exception_list errors_;
        std::atomic<bool> has_arrived_;
    };
}}}    // namespace hpx::execution::experimental
