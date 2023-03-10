//  Copyright (c) 2021-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/scheduler_executor.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/algorithms/bulk.hpp>
#include <hpx/execution/algorithms/keep_future.hpp>
#include <hpx/execution/algorithms/make_future.hpp>
#include <hpx/execution/algorithms/schedule_from.hpp>
#include <hpx/execution/algorithms/start_detached.hpp>
#include <hpx/execution/algorithms/sync_wait.hpp>
#include <hpx/execution/algorithms/then.hpp>
#include <hpx/execution/algorithms/transfer.hpp>
#include <hpx/execution/algorithms/transfer_just.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/functional/bind_back.hpp>
#include <hpx/functional/bind_front.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/invoke_fused.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/modules/concepts.hpp>
#include <hpx/modules/topology.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <cstddef>
#include <exception>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx::execution::experimental {

    namespace detail {
#if defined(HPX_HAVE_CXX20_PERFECT_PACK_CAPTURE)
        template <typename F, typename... Ts>
        auto captured_args_then(F&& f, Ts&&... ts)
        {
            return [f = HPX_FORWARD(F, f), ... ts = HPX_FORWARD(Ts, ts)](
                       auto i, auto&& predecessor, auto& v) mutable {
                v[i] = HPX_INVOKE(HPX_FORWARD(F, f), i,
                    HPX_FORWARD(decltype(predecessor), predecessor),
                    HPX_FORWARD(Ts, ts)...);
            };
        }
#else
        template <typename F, typename... Ts>
        auto captured_args_then(F&& f, Ts&&... ts)
        {
            return [f = HPX_FORWARD(F, f),
                       t = hpx::make_tuple(HPX_FORWARD(Ts, ts)...)](
                       auto i, auto&& predecessor, auto& v) mutable {
                v[i] = hpx::invoke_fused(
                    hpx::bind_front(HPX_FORWARD(F, f), i,
                        HPX_FORWARD(decltype(predecessor), predecessor)),
                    HPX_MOVE(t));
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
        static_assert(hpx::execution::experimental::is_scheduler_v<
                          std::decay_t<BaseScheduler>>,
            "scheduler_executor requires a scheduler");

        constexpr scheduler_executor() = default;

        template <typename Scheduler,
            typename Enable = std::enable_if_t<
                hpx::execution::experimental::is_scheduler_v<Scheduler> &&
                !std::is_same_v<std::decay_t<Scheduler>, scheduler_executor>>>
        constexpr explicit scheduler_executor(Scheduler&& sched)
          : sched_(HPX_FORWARD(Scheduler, sched))
        {
        }

        constexpr scheduler_executor(scheduler_executor&&) = default;
        constexpr scheduler_executor& operator=(scheduler_executor&&) = default;
        constexpr scheduler_executor(scheduler_executor const&) = default;
        constexpr scheduler_executor& operator=(
            scheduler_executor const&) = default;

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

        constexpr BaseScheduler const& sched() const noexcept
        {
            return sched_;
        }

        // Associate the parallel_execution_tag executor tag type as a default
        // with this executor.
        using execution_category = parallel_execution_tag;

        // Associate the static_chunk_size executor parameters type as a default
        // with this executor.
        using executor_parameters_type = static_chunk_size;

        template <typename T, typename... Ts>
        using future_type = hpx::future<T>;

    private:
        friend auto tag_invoke(
            hpx::parallel::execution::processing_units_count_t tag,
            scheduler_executor const& exec,
            hpx::chrono::steady_duration const& = hpx::chrono::null_duration,
            std::size_t = 0)
            -> decltype(std::declval<
                hpx::parallel::execution::processing_units_count_t>()(
                std::declval<BaseScheduler>(),
                std::declval<hpx::chrono::steady_duration>(), 0))
        {
            return tag(exec.sched_);
        }

        // NonBlockingOneWayExecutor interface
        template <typename F, typename... Ts>
        friend decltype(auto) tag_invoke(hpx::parallel::execution::post_t,
            scheduler_executor const& exec, F&& f, Ts&&... ts)
        {
            start_detached(then(schedule(exec.sched_),
                hpx::util::deferred_call(
                    HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...)));
        }

        // OneWayExecutor interface
        template <typename F, typename... Ts>
        friend auto tag_invoke(hpx::parallel::execution::sync_execute_t,
            scheduler_executor const& exec, F&& f, Ts&&... ts)
        {
            return *hpx::this_thread::experimental::sync_wait(
                then(schedule(exec.sched_),
                    hpx::util::deferred_call(
                        HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...)));
        }

        // TwoWayExecutor interface
        template <typename F, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::async_execute_t,
            scheduler_executor const& exec, F&& f, Ts&&... ts)
        {
            return make_future(then(schedule(exec.sched_),
                hpx::util::deferred_call(
                    HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...)));
        }

        template <typename F, typename Future, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::then_execute_t,
            scheduler_executor const& exec, F&& f, Future&& predecessor,
            Ts&&... ts)
        {
            auto&& predecessor_transfer_sched = transfer(
                keep_future(HPX_FORWARD(Future, predecessor)), exec.sched_);

            return make_future(then(HPX_MOVE(predecessor_transfer_sched),
                hpx::bind_back(HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...)));
        }

        // BulkTwoWayExecutor interface
        // clang-format off
        template <typename F, typename S, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                !std::is_integral_v<S>
            )>
        // clang-format on
        friend auto tag_invoke(hpx::parallel::execution::bulk_async_execute_t,
            scheduler_executor const& exec, F&& f, S const& shape, Ts&&... ts)
        {
            using shape_element =
                typename hpx::traits::range_traits<S>::value_type;
            using result_type = hpx::util::detail::invoke_deferred_result_t<F,
                shape_element, Ts...>;

            if constexpr (std::is_void_v<result_type>)
            {
                return make_future(bulk(schedule(exec.sched_), shape,
                    hpx::bind_back(HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...)));
            }
            else
            {
                using promise_vector_type =
                    std::vector<hpx::promise<result_type>>;
                using result_vector_type =
                    std::vector<hpx::future<result_type>>;

                using size_type = decltype(hpx::util::size(shape));
                size_type const n = hpx::util::size(shape);

                promise_vector_type promises(n);
                result_vector_type results;
                results.reserve(n);

                for (size_type i = 0; i < n; ++i)
                {
                    results.emplace_back(promises[i].get_future());
                }

                auto f_helper = [](size_type const i,
                                    promise_vector_type& promises, F& f,
                                    S const& shape, Ts&... ts) {
                    hpx::detail::try_catch_exception_ptr(
                        [&]() mutable {
                            auto it = hpx::util::begin(shape);
                            std::advance(it, i);
                            promises[i].set_value(HPX_INVOKE(f, *it, ts...));
                        },
                        [&](std::exception_ptr&& ep) {
                            promises[i].set_exception(HPX_MOVE(ep));
                        });
                };

                start_detached(
                    bulk(transfer_just(exec.sched_, HPX_MOVE(promises),
                             HPX_FORWARD(F, f), shape, HPX_FORWARD(Ts, ts)...),
                        n, HPX_MOVE(f_helper)));

                return results;
            }
        }

        // clang-format off
        template <typename F, typename S, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                !std::is_integral_v<S>
            )>
        // clang-format on
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::bulk_sync_execute_t,
            scheduler_executor const& exec, F&& f, S const& shape, Ts&&... ts)
        {
            hpx::this_thread::experimental::sync_wait(
                bulk(schedule(exec.sched_), shape,
                    hpx::bind_back(HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...)));
        }

        // clang-format off
        template <typename F, typename S, typename Future, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                !std::is_integral_v<S>
            )>
        // clang-format on
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::bulk_then_execute_t,
            scheduler_executor const& exec, F&& f, S const& shape,
            Future&& predecessor, Ts&&... ts)
        {
            using result_type =
                parallel::execution::detail::then_bulk_function_result_t<F, S,
                    Future, Ts...>;

            if constexpr (std::is_void_v<result_type>)
            {
                // the overall return value is future<void>
                auto prereq =
                    when_all(keep_future(HPX_FORWARD(Future, predecessor)));

                auto loop = bulk(transfer(HPX_MOVE(prereq), exec.sched_), shape,
                    hpx::bind_back(HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...));

                return make_future(HPX_MOVE(loop));
            }
            else
            {
                // the overall return value is future<std::vector<result_type>>
                auto prereq =
                    when_all(keep_future(HPX_FORWARD(Future, predecessor)),
                        just(std::vector<result_type>(hpx::util::size(shape))));

                auto loop = bulk(transfer(HPX_MOVE(prereq), exec.sched_), shape,
                    detail::captured_args_then(
                        HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...));

                return make_future(then(
                    HPX_MOVE(loop), [](auto&&, std::vector<result_type>&& v) {
                        return HPX_MOVE(v);
                    }));
            }
        }

    private:
        std::decay_t<BaseScheduler> sched_;
        /// \endcond
    };

    template <typename BaseScheduler>
    explicit scheduler_executor(BaseScheduler&& sched)
        -> scheduler_executor<std::decay_t<BaseScheduler>>;

    // support all properties exposed by the wrapped scheduler
    // clang-format off
    template <typename Tag, typename BaseScheduler, typename Property,
        HPX_CONCEPT_REQUIRES_(
            hpx::execution::experimental::is_scheduling_property_v<Tag>
        )>
    // clang-format on
    auto tag_invoke(
        Tag tag, scheduler_executor<BaseScheduler> const& exec, Property&& prop)
        -> decltype(scheduler_executor<BaseScheduler>(std::declval<Tag>()(
            std::declval<BaseScheduler>(), std::declval<Property>())))
    {
        return scheduler_executor<BaseScheduler>(
            tag(exec.sched(), HPX_FORWARD(Property, prop)));
    }

    // clang-format off
    template <typename Tag, typename BaseScheduler,
        HPX_CONCEPT_REQUIRES_(
            hpx::execution::experimental::is_scheduling_property_v<Tag>
        )>
    // clang-format on
    auto tag_invoke(Tag tag, scheduler_executor<BaseScheduler> const& exec)
        -> decltype(std::declval<Tag>()(std::declval<BaseScheduler>()))
    {
        return tag(exec.sched());
    }
}    // namespace hpx::execution::experimental

namespace hpx::parallel::execution {

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
}    // namespace hpx::parallel::execution
