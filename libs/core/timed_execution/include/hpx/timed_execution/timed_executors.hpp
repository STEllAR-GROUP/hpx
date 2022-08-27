//  Copyright (c) 2017-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/execute_at_executor.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/traits/is_launch_policy.hpp>
#include <hpx/execution/traits/executor_traits.hpp>
#include <hpx/functional/bind.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/threading.hpp>
#include <hpx/timing/steady_clock.hpp>
#include <hpx/type_support/detail/wrap_int.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/executors/parallel_executor.hpp>
#include <hpx/executors/sequenced_executor.hpp>

#include <chrono>
#include <functional>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace execution {
    namespace detail {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        template <typename Tag, typename Executor, typename F>
        struct then_execute_helper
        {
        public:
            template <typename Executor_, typename F_>
            then_execute_helper(Executor_&& exec, F_&& call)
              : exec_(HPX_FORWARD(Executor_, exec))
              , call_(HPX_FORWARD(F_, call))
            {
            }

            auto operator()(hpx::future<void>&& fut) -> decltype(Tag()(
                HPX_MOVE(fut), std::declval<Executor>(), std::declval<F>()))
            {
                return Tag()(HPX_MOVE(fut), HPX_MOVE(exec_), HPX_MOVE(call_));
            }

        private:
            Executor exec_;
            F call_;
        };

        template <typename Tag, typename Executor, typename F>
        then_execute_helper<Tag, typename std::decay<Executor>::type,
            typename std::decay<F>::type>
        make_then_execute_helper(Executor&& exec, F&& call)
        {
            return {HPX_FORWARD(Executor, exec), HPX_FORWARD(F, call)};
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Tag>
        struct sync_execute_at_helper
        {
            template <typename Executor, typename F>
            auto operator()(
                hpx::future<void>&& fut, Executor&& exec, F&& f) const
                -> decltype(execution::sync_execute(
                    HPX_FORWARD(Executor, exec), HPX_FORWARD(F, f)))
            {
                fut.get();    // rethrow exceptions
                return execution::sync_execute(
                    HPX_FORWARD(Executor, exec), HPX_FORWARD(F, f));
            }

            template <typename Executor, typename F, typename... Ts>
            static auto call(hpx::traits::detail::wrap_int, Executor&& exec,
                std::chrono::steady_clock::time_point const& abs_time, F&& f,
                Ts&&... ts)
                -> decltype(
                    execution::async_execute(HPX_FORWARD(Executor, exec),
                        HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...)
                        .get())
            {
                auto predecessor = make_ready_future_at(abs_time);
                return execution::then_execute(
                    hpx::execution::sequenced_executor(),
                    make_then_execute_helper<sync_execute_at_helper>(
                        HPX_FORWARD(Executor, exec),
                        hpx::util::deferred_call(
                            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...)),
                    predecessor)
                    .get();
            }

            template <typename Executor, typename F, typename... Ts>
            static auto call(int, Executor&& exec,
                std::chrono::steady_clock::time_point const& abs_time, F&& f,
                Ts&&... ts) -> decltype(exec.sync_execute_at(abs_time,
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...))
            {
                return exec.sync_execute_at(
                    abs_time, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
            }
        };

        template <>
        struct sync_execute_at_helper<hpx::execution::sequenced_execution_tag>
        {
            template <typename Executor, typename F, typename... Ts>
            static auto call(hpx::traits::detail::wrap_int, Executor&& exec,
                std::chrono::steady_clock::time_point const& abs_time, F&& f,
                Ts&&... ts)
                -> decltype(execution::sync_execute(HPX_FORWARD(Executor, exec),
                    HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...))
            {
                this_thread::sleep_until(abs_time);
                return execution::sync_execute(HPX_FORWARD(Executor, exec),
                    HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
            }

            template <typename Executor, typename F, typename... Ts>
            static auto call(int, Executor&& exec,
                std::chrono::steady_clock::time_point const& abs_time, F&& f,
                Ts&&... ts) -> decltype(exec.sync_execute_at(abs_time,
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...))
            {
                return exec.sync_execute_at(
                    abs_time, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
            }
        };

        template <typename Executor, typename F, typename... Ts>
        auto call_sync_execute_at(Executor&& exec,
            std::chrono::steady_clock::time_point const& abs_time, F&& f,
            Ts&&... ts)
            -> decltype(sync_execute_at_helper<
                typename hpx::traits::executor_execution_category<
                    typename std::decay<Executor>::type>::type>::call(0,
                HPX_FORWARD(Executor, exec), abs_time, HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...))
        {
            typedef typename hpx::traits::executor_execution_category<
                typename std::decay<Executor>::type>::type tag;

            return sync_execute_at_helper<tag>::call(0,
                HPX_FORWARD(Executor, exec), abs_time, HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Tag>
        struct async_execute_at_helper
        {
            template <typename Executor, typename F>
            auto operator()(
                hpx::future<void>&& fut, Executor&& exec, F&& f) const
                -> decltype(execution::async_execute(
                    HPX_FORWARD(Executor, exec), HPX_FORWARD(F, f)))
            {
                fut.get();    // rethrow exceptions
                return execution::async_execute(
                    HPX_FORWARD(Executor, exec), HPX_FORWARD(F, f));
            }

            template <typename Executor, typename F, typename... Ts>
            static auto call(hpx::traits::detail::wrap_int, Executor&& exec,
                std::chrono::steady_clock::time_point const& abs_time, F&& f,
                Ts&&... ts)
                -> decltype(
                    execution::async_execute(HPX_FORWARD(Executor, exec),
                        HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...))
            {
                auto predecessor = make_ready_future_at(abs_time);
                return execution::then_execute(
                    hpx::execution::sequenced_executor(),
                    make_then_execute_helper<async_execute_at_helper>(
                        HPX_FORWARD(Executor, exec),
                        hpx::util::deferred_call(
                            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...)),
                    predecessor);
            }

            template <typename Executor, typename F, typename... Ts>
            static auto call(int, Executor&& exec,
                std::chrono::steady_clock::time_point const& abs_time, F&& f,
                Ts&&... ts) -> decltype(exec.async_execute_at(abs_time,
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...))
            {
                return exec.async_execute_at(
                    abs_time, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
            }
        };

        template <>
        struct async_execute_at_helper<hpx::execution::sequenced_execution_tag>
        {
            template <typename Executor, typename F, typename... Ts>
            static auto call(hpx::traits::detail::wrap_int, Executor&& exec,
                std::chrono::steady_clock::time_point const& abs_time, F&& f,
                Ts&&... ts)
                -> decltype(
                    execution::async_execute(HPX_FORWARD(Executor, exec),
                        HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...))
            {
                this_thread::sleep_until(abs_time);
                return execution::async_execute(HPX_FORWARD(Executor, exec),
                    HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
            }

            template <typename Executor, typename F, typename... Ts>
            static auto call(int, Executor&& exec,
                std::chrono::steady_clock::time_point const& abs_time, F&& f,
                Ts&&... ts) -> decltype(exec.async_execute_at(abs_time,
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...))
            {
                return exec.async_execute_at(
                    abs_time, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
            }
        };

        template <typename Executor, typename F, typename... Ts>
        auto call_async_execute_at(Executor&& exec,
            std::chrono::steady_clock::time_point const& abs_time, F&& f,
            Ts&&... ts)
            -> decltype(async_execute_at_helper<
                typename hpx::traits::executor_execution_category<
                    typename std::decay<Executor>::type>::type>::call(0,
                HPX_FORWARD(Executor, exec), abs_time, HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...))
        {
            typedef typename hpx::traits::executor_execution_category<
                typename std::decay<Executor>::type>::type tag;
            return async_execute_at_helper<tag>::call(0,
                HPX_FORWARD(Executor, exec), abs_time, HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Tag>
        struct post_at_helper
        {
            template <typename Executor, typename F>
            void operator()(
                hpx::future<void>&& fut, Executor&& exec, F&& f) const
            {
                fut.get();    // rethrow exceptions
                execution::post(HPX_FORWARD(Executor, exec), HPX_FORWARD(F, f));
            }

            template <typename Executor, typename F, typename... Ts>
            static void call(hpx::traits::detail::wrap_int, Executor&& exec,
                std::chrono::steady_clock::time_point const& abs_time, F&& f,
                Ts&&... ts)
            {
                auto predecessor = make_ready_future_at(abs_time);
                execution::then_execute(hpx::execution::sequenced_executor(),
                    make_then_execute_helper<post_at_helper>(
                        HPX_FORWARD(Executor, exec),
                        hpx::util::deferred_call(
                            HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...)),
                    predecessor);
            }

            template <typename Executor, typename F, typename... Ts>
            static auto call(int, Executor&& exec,
                std::chrono::steady_clock::time_point const& abs_time, F&& f,
                Ts&&... ts) -> decltype(exec.post_at(abs_time,
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...))
            {
                return exec.post_at(
                    abs_time, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
            }
        };

        template <>
        struct post_at_helper<hpx::execution::sequenced_execution_tag>
        {
            template <typename Executor, typename F, typename... Ts>
            static void call(hpx::traits::detail::wrap_int, Executor&& exec,
                std::chrono::steady_clock::time_point const& abs_time, F&& f,
                Ts&&... ts)
            {
                this_thread::sleep_until(abs_time);
                execution::post(HPX_FORWARD(Executor, exec), HPX_FORWARD(F, f),
                    HPX_FORWARD(Ts, ts)...);
            }

            template <typename Executor, typename F, typename... Ts>
            static auto call(int, Executor&& exec,
                std::chrono::steady_clock::time_point const& abs_time, F&& f,
                Ts&&... ts) -> decltype(exec.post_at(abs_time,
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...))
            {
                exec.post_at(
                    abs_time, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
            }
        };

        template <typename Executor, typename F, typename... Ts>
        void call_post_at(Executor&& exec,
            std::chrono::steady_clock::time_point const& abs_time, F&& f,
            Ts&&... ts)
        {
            typedef typename hpx::traits::executor_execution_category<
                typename std::decay<Executor>::type>::type tag;

            return post_at_helper<tag>::call(0, HPX_FORWARD(Executor, exec),
                abs_time, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }
        /// \endcond
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // Executor allowing to run things at a given point in time
    template <typename BaseExecutor>
    struct timed_executor
    {
        typedef typename std::decay<BaseExecutor>::type base_executor_type;

        typedef typename hpx::traits::executor_execution_category<
            base_executor_type>::type execution_category;

        typedef typename hpx::traits::executor_parameters_type<
            base_executor_type>::type parameters_type;

        timed_executor(hpx::chrono::steady_time_point const& abs_time)
          : exec_(BaseExecutor())
          , execute_at_(abs_time.value())
        {
        }

        timed_executor(hpx::chrono::steady_duration const& rel_time)
          : exec_(BaseExecutor())
          , execute_at_(rel_time.from_now())
        {
        }

        template <typename Executor>
        timed_executor(
            Executor&& exec, hpx::chrono::steady_time_point const& abs_time)
          : exec_(HPX_FORWARD(Executor, exec))
          , execute_at_(abs_time.value())
        {
        }

        template <typename Executor>
        timed_executor(
            Executor&& exec, hpx::chrono::steady_duration const& rel_time)
          : exec_(HPX_FORWARD(Executor, exec))
          , execute_at_(rel_time.from_now())
        {
        }

        /// \cond NOINTERNAL
        bool operator==(timed_executor const& rhs) const noexcept
        {
            return exec_ == rhs.exec_ && execute_at_ == rhs.execute_at_;
        }

        bool operator!=(timed_executor const& rhs) const noexcept
        {
            return !(*this == rhs);
        }

        timed_executor const& context() const noexcept
        {
            return *this;
        }
        /// \endcond

        // OneWayExecutor interface
        template <typename F, typename... Ts>
        typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type
        sync_execute(F&& f, Ts&&... ts)
        {
            return detail::call_sync_execute_at(
                exec_, execute_at_, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        // TwoWayExecutor interface
        template <typename F, typename... Ts>
        hpx::future<
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
        async_execute(F&& f, Ts&&... ts)
        {
            return detail::call_async_execute_at(
                exec_, execute_at_, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        // NonBlockingOneWayExecutor (adapted) interface
        template <typename F, typename... Ts>
        void post(F&& f, Ts&&... ts)
        {
            detail::call_post_at(
                exec_, execute_at_, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        BaseExecutor exec_;
        std::chrono::steady_clock::time_point execute_at_;
    };

    ///////////////////////////////////////////////////////////////////////////
    using sequenced_timed_executor =
        timed_executor<hpx::execution::sequenced_executor>;

    using parallel_timed_executor =
        timed_executor<hpx::execution::parallel_executor>;
}}}    // namespace hpx::parallel::execution

namespace hpx { namespace parallel { namespace execution {
    /// \cond NOINTERNAL

    ///////////////////////////////////////////////////////////////////////////
    template <typename BaseExecutor>
    struct is_one_way_executor<
        parallel::execution::timed_executor<BaseExecutor>>
      : is_one_way_executor<typename std::decay<BaseExecutor>::type>
    {
    };

    template <typename BaseExecutor>
    struct is_two_way_executor<
        parallel::execution::timed_executor<BaseExecutor>>
      : is_two_way_executor<typename std::decay<BaseExecutor>::type>
    {
    };

    template <typename BaseExecutor>
    struct is_never_blocking_one_way_executor<
        parallel::execution::timed_executor<BaseExecutor>>
      : is_never_blocking_one_way_executor<
            typename std::decay<BaseExecutor>::type>
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    template <>
    struct is_one_way_executor<parallel::execution::sequenced_timed_executor>
      : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<parallel::execution::parallel_timed_executor>
      : std::true_type
    {
    };
    /// \endcond
}}}    // namespace hpx::parallel::execution
