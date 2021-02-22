//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/execution/detail/post_policy_dispatch.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution_base/receiver.hpp>
#include <hpx/execution_base/sender.hpp>

#include <cstddef>
#include <exception>
#include <type_traits>
#include <utility>

namespace hpx { namespace execution { namespace experimental {
    struct executor
    {
        constexpr executor() = default;

        /// \cond NOINTERNAL
        bool operator==(executor const& rhs) const noexcept
        {
            return pool_ == rhs.pool_ && priority_ == rhs.priority_ &&
                stacksize_ == rhs.stacksize_ &&
                schedulehint_ == rhs.schedulehint_;
        }

        bool operator!=(executor const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        friend executor tag_invoke(
            hpx::execution::experimental::make_with_priority_t,
            executor const& exec, hpx::threads::thread_priority priority)
        {
            auto exec_with_priority = exec;
            exec_with_priority.priority_ = priority;
            return exec_with_priority;
        }

        friend hpx::threads::thread_priority tag_invoke(
            hpx::execution::experimental::get_priority_t, executor const& exec)
        {
            return exec.priority_;
        }

        friend executor tag_invoke(
            hpx::execution::experimental::make_with_stacksize_t,
            executor const& exec, hpx::threads::thread_stacksize stacksize)
        {
            auto exec_with_stacksize = exec;
            exec_with_stacksize.stacksize_ = stacksize;
            return exec_with_stacksize;
        }

        friend hpx::threads::thread_stacksize tag_invoke(
            hpx::execution::experimental::get_stacksize_t, executor const& exec)
        {
            return exec.stacksize_;
        }

        friend executor tag_invoke(
            hpx::execution::experimental::make_with_hint_t,
            executor const& exec, hpx::threads::thread_schedule_hint hint)
        {
            auto exec_with_hint = exec;
            exec_with_hint.schedulehint_ = hint;
            return exec_with_hint;
        }

        friend hpx::threads::thread_schedule_hint tag_invoke(
            hpx::execution::experimental::get_hint_t, executor const& exec)
        {
            return exec.schedulehint_;
        }

        template <typename F>
        void execute(F&& f) const
        {
            hpx::util::thread_description desc(f);

            hpx::parallel::execution::detail::post_policy_dispatch<
                hpx::launch::async_policy>::call(hpx::launch::async, desc,
                pool_, priority_, stacksize_, schedulehint_,
                std::forward<F>(f));
        }

        template <typename Executor, typename R>
        struct operation_state
        {
            std::decay_t<Executor> exec;
            std::decay_t<R> r;

            void start() noexcept
            {
                try
                {
                    hpx::execution::experimental::execute(
                        exec, [r = std::move(r)]() mutable {
                            hpx::execution::experimental::set_value(
                                std::move(r));
                        });
                }
                catch (...)
                {
                    hpx::execution::experimental::set_error(
                        std::move(r), std::current_exception());
                }
            }
        };

        template <typename Executor>
        struct sender
        {
            std::decay_t<Executor> exec;

            template <template <typename...> class Tuple,
                template <typename...> class Variant>
            using value_types = Variant<Tuple<>>;

            template <template <typename...> class Variant>
            using error_types = Variant<std::exception_ptr>;

            static constexpr bool sends_done = false;

            template <typename R>
            operation_state<Executor, R> connect(R&& r) &&
            {
                return {std::move(exec), std::forward<R>(r)};
            }
        };

        template <template <class...> class Tuple,
            template <class...> class Variant>
        using value_types = Variant<Tuple<>>;

        template <template <class...> class Variant>
        using error_types = Variant<std::exception_ptr>;

        static constexpr bool sends_done = false;

        template <typename R>
        operation_state<executor, R> connect(R&& r) &&
        {
            return {*this, std::forward<R>(r)};
        }

        constexpr sender<executor> schedule() const
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
        /// \endcond
    };
}}}    // namespace hpx::execution::experimental
