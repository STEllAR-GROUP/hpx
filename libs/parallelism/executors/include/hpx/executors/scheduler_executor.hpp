//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/scheduler_executor.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/algorithms/bulk.hpp>
#include <hpx/execution/algorithms/detach.hpp>
#include <hpx/execution/algorithms/keep_future.hpp>
#include <hpx/execution/algorithms/make_future.hpp>
#include <hpx/execution/algorithms/on.hpp>
#include <hpx/execution/algorithms/sync_wait.hpp>
#include <hpx/execution/algorithms/transform.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/invoke_fused.hpp>

#include <exception>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace execution { namespace experimental {

    namespace detail {
#if defined(HPX_HAVE_CXX20_PERFECT_PACK_CAPTURE)
        template <typename F, typename... Ts>
        auto captured_args_then(F&& f, Ts&&... ts)
        {
            return [f = std::forward<F>(f), ... ts = std::forward<Ts>(ts)](
                       auto i, auto&& predecessor, auto& v) mutable {
                v[i] = HPX_INVOKE(std::forward<F>(f), i,
                    std::forward<decltype(predecessor)>(predecessor),
                    std::forward<Ts>(ts)...);
            };
        }
#else
        template <typename F, typename... Ts>
        auto captured_args_then(F&& f, Ts&&... ts)
        {
            return [f = std::forward<F>(f),
                       t = hpx::make_tuple(std::forward<Ts>(ts)...)](
                       auto i, auto&& predecessor, auto& v) mutable {
                v[i] = hpx::util::invoke_fused(
                    hpx::util::bind_front(std::forward<F>(f), i,
                        std::forward<decltype(predecessor)>(predecessor)),
                    std::move(t));
            };
        }
#endif
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // A scheduler_executor wraps any P2300 scheduler and implements the
    // executor functionalities for those.
    template <typename BaseScheduler>
    struct scheduler_executor
    {
        template <typename Scheduler,
            typename Enable = std::enable_if_t<
                hpx::execution::experimental::is_scheduler_v<Scheduler>>>
        constexpr explicit scheduler_executor(Scheduler&& sched)
          : sched_(std::forward<Scheduler>(sched))
        {
        }

        /// \cond NOINTERNAL
        constexpr bool operator==(scheduler_executor const& rhs) const noexcept
        {
            return sched_ == rhs.sched_;
        }

        constexpr bool operator!=(scheduler_executor const& rhs) const noexcept
        {
            return sched_ != rhs.sched_;
        }

        constexpr auto const& context() const noexcept
        {
            return *this;
        }
        /// \endcond

        /// \cond NOINTERNAL
        // Associate the parallel_execution_tag executor tag type as a default
        // with this executor.
        using execution_category = parallel_execution_tag;

        // Associate the static_chunk_size executor parameters type as a default
        // with this executor.
        using executor_parameters_type = static_chunk_size;

        template <typename T, typename... Ts>
        using future_type = hpx::future<T>;

        // NonBlockingOneWayExecutor interface
        template <typename F, typename... Ts>
        void post(F&& f, Ts&&... ts)
        {
            detach(transform(schedule(sched_),
                hpx::util::deferred_call(
                    std::forward<F>(f), std::forward<Ts>(ts)...)));
        }

        // OneWayExecutor interface
        template <typename F, typename... Ts>
        decltype(auto) sync_execute(F&& f, Ts&&... ts)
        {
            return sync_wait(transform(schedule(sched_),
                hpx::util::deferred_call(
                    std::forward<F>(f), std::forward<Ts>(ts)...)));
        }

        // TwoWayExecutor interface
        template <typename F, typename... Ts>
        decltype(auto) async_execute(F&& f, Ts&&... ts)
        {
            return make_future(transform(schedule(sched_),
                hpx::util::deferred_call(
                    std::forward<F>(f), std::forward<Ts>(ts)...)));
        }

        template <typename F, typename Future, typename... Ts>
        decltype(auto) then_execute(F&& f, Future&& predecessor, Ts&&... ts)
        {
            auto&& predecessor_on_sched =
                on(keep_future(std::forward<Future>(predecessor)), sched_);

            return make_future(transform(std::move(predecessor_on_sched),
                hpx::util::bind_back(
                    std::forward<F>(f), std::forward<Ts>(ts)...)));
        }

        // BulkTwoWayExecutor interface
        template <typename F, typename S, typename... Ts>
        decltype(auto) bulk_async_execute(F&& f, S const& shape, Ts&&... ts)
        {
            using shape_element =
                typename hpx::traits::range_traits<S>::value_type;
            using result_type = hpx::util::detail::invoke_deferred_result_t<F,
                shape_element, Ts...>;

            std::vector<hpx::future<result_type>> results;
            results.reserve(hpx::util::size(shape));

            for (auto const& s : shape)
            {
                results.push_back(async_execute(f, s, ts...));
            }

            return results;
        }

        template <typename F, typename S, typename... Ts>
        decltype(auto) bulk_sync_execute(F&& f, S const& shape, Ts&&... ts)
        {
            return sync_wait(bulk(schedule(sched_), shape,
                hpx::util::bind_back(
                    std::forward<F>(f), std::forward<Ts>(ts)...)));
        }

        template <typename F, typename S, typename Future, typename... Ts>
        decltype(auto) bulk_then_execute(
            F&& f, S const& shape, Future&& predecessor, Ts&&... ts)
        {
            using result_type =
                parallel::execution::detail::then_bulk_function_result_t<F, S,
                    Future, Ts...>;

            if constexpr (std::is_void<result_type>::value)
            {
                // the overall return value is future<void>
                auto prereq =
                    when_all(keep_future(std::forward<Future>(predecessor)));

                auto loop = bulk(on(std::move(prereq), sched_), shape,
                    hpx::util::bind_back(
                        std::forward<F>(f), std::forward<Ts>(ts)...));

                return make_future(std::move(loop));
            }
            else
            {
                // the overall return value is future<std::vector<result_type>>
                auto prereq =
                    when_all(keep_future(std::forward<Future>(predecessor)),
                        just(std::vector<result_type>(hpx::util::size(shape))));

                auto loop = bulk(on(std::move(prereq), sched_), shape,
                    detail::captured_args_then(
                        std::forward<F>(f), std::forward<Ts>(ts)...));

                return make_future(transform(
                    std::move(loop), [](auto&&, std::vector<result_type>&& v) {
                        return std::move(v);
                    }));
            }
        }

    private:
        BaseScheduler sched_;
        /// \endcond
    };
}}}    // namespace hpx::execution::experimental

namespace hpx { namespace parallel { namespace execution {

    /// \cond NOINTERNAL
    template <typename BaseScheduler>
    struct is_one_way_executor<
        hpx::execution::experimental::scheduler_executor<BaseScheduler>>
      : std::true_type
    {
    };

    template <typename BaseScheduler>
    struct is_never_blocking_one_way_executor<
        hpx::execution::experimental::scheduler_executor<BaseScheduler>>
      : std::true_type
    {
    };

    template <typename BaseScheduler>
    struct is_bulk_one_way_executor<
        hpx::execution::experimental::scheduler_executor<BaseScheduler>>
      : std::true_type
    {
    };

    template <typename BaseScheduler>
    struct is_two_way_executor<
        hpx::execution::experimental::scheduler_executor<BaseScheduler>>
      : std::true_type
    {
    };

    template <typename BaseScheduler>
    struct is_bulk_two_way_executor<
        hpx::execution::experimental::scheduler_executor<BaseScheduler>>
      : std::true_type
    {
    };
    /// \endcond
}}}    // namespace hpx::parallel::execution
