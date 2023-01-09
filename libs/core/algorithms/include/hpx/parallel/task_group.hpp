//  Copyright (c) 2021-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file task_group.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/errors/exception_list.hpp>
#include <hpx/errors/try_catch_exception_ptr.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/executors/parallel_executor.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/futures/detail/future_data.hpp>
#include <hpx/modules/memory.hpp>
#include <hpx/serialization/serialization_fwd.hpp>
#include <hpx/synchronization/latch.hpp>

#include <atomic>
#include <exception>
#include <type_traits>
#include <utility>

/// Top-level namespace
namespace hpx::experimental {

    /// A \c task_group represents concurrent execution of a group of tasks.
    /// Tasks can be dynamically added to the group while it is executing.
    class task_group
    {
    public:
        HPX_CORE_EXPORT task_group();
        HPX_CORE_EXPORT ~task_group();

        task_group(task_group const&) = delete;
        task_group(task_group&&) = delete;

        task_group& operator=(task_group const&) = delete;
        task_group& operator=(task_group&&) = delete;

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
        /// \brief Adds a task to compute \c f() and returns immediately.
        ///
        /// \tparam Executor  The type of the executor to associate with this
        ///                   execution policy.
        /// \tparam F         The type of the user defined function to invoke.
        /// \tparam Ts        The type of additional arguments used to invoke \c f().
        ///
        /// \param exec       The executor to use for the execution of the
        ///                   parallel algorithm the returned execution
        ///                   policy is used with.
        /// \param f          The user defined function to invoke inside the task
        ///                   group.
        /// \param ts         Additional arguments to use to invoke \c f().
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

                    hpx::detail::try_catch_exception_ptr(
                        [&]() { hpx::invoke_fused(HPX_MOVE(f), HPX_MOVE(t)); },
                        [this](std::exception_ptr e) {
                            add_exception(HPX_MOVE(e));
                        });
                });
        }

        /// \brief Adds a task to compute \c f() and returns immediately.
        ///
        /// \tparam F  The type of the user defined function to invoke.
        /// \tparam Ts The type of additional arguments used to invoke \c f().
        ///
        /// \param f   The user defined function to invoke inside the task
        ///            group.
        /// \param ts  Additional arguments to use to invoke \c f().
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

        /// \brief Waits for all tasks in the group to complete or be cancelled.
        HPX_CORE_EXPORT void wait();

        /// \brief Adds an exception to this \c task_group
        HPX_CORE_EXPORT void add_exception(std::exception_ptr p);

    private:
        friend class serialization::access;

        static constexpr void serialize(
            serialization::input_archive&, unsigned const) noexcept
        {
        }
        HPX_CORE_EXPORT void serialize(
            serialization::output_archive&, unsigned const);

    private:
        using shared_state_type = lcos::detail::future_data<void>;

        hpx::lcos::local::latch latch_;
        hpx::intrusive_ptr<shared_state_type> state_;
        hpx::exception_list errors_;
        std::atomic<bool> has_arrived_;
    };
}    // namespace hpx::experimental

namespace hpx::execution::experimental {

    using task_group HPX_DEPRECATED_V(1, 9,
        "hpx::execution:experimental::task_group is deprecated, use "
        "hpx::experimental::task_group instead") =
        hpx::experimental::task_group;
}    // namespace hpx::execution::experimental
