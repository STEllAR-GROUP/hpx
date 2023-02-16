//  Copyright (c) 2021-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/executors/annotating_executor.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/modules/concepts.hpp>
#include <hpx/modules/topology.hpp>
#include <hpx/threading_base/annotated_function.hpp>

#include <string>
#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    ///////////////////////////////////////////////////////////////////////////
    /// A \a annotating_executor wraps any other executor and adds the
    /// capability to add annotations to the launched threads.
    template <typename BaseExecutor>
    struct annotating_executor
    {
        static_assert(
            hpx::traits::is_executor_any_v<std::decay_t<BaseExecutor>>,
            "annotating_executor requires an executor");

        template <typename Executor,
            typename Enable = std::enable_if_t<
                hpx::traits::is_executor_any_v<Executor> &&
                !std::is_same_v<std::decay_t<Executor>, annotating_executor>>>
        constexpr explicit annotating_executor(
            Executor&& exec, char const* annotation = nullptr)
          : exec_(HPX_FORWARD(Executor, exec))
          , annotation_(annotation)
        {
        }

        template <typename Executor,
            typename Enable =
                std::enable_if_t<hpx::traits::is_executor_any_v<Executor>>>
        explicit annotating_executor(Executor&& exec, std::string annotation)
          : exec_(HPX_FORWARD(Executor, exec))
          , annotation_(
                hpx::detail::store_function_annotation(HPX_MOVE(annotation)))
        {
        }

        /// \cond NOINTERNAL
        constexpr bool operator==(annotating_executor const& rhs) const noexcept
        {
            return exec_ == rhs.exec_;
        }

        constexpr bool operator!=(annotating_executor const& rhs) const noexcept
        {
            return exec_ != rhs.exec_;
        }

        [[nodiscard]] constexpr auto const& context() const noexcept
        {
            return exec_.context();
        }

        [[nodiscard]] constexpr std::decay_t<BaseExecutor> const& get_executor()
            const noexcept
        {
            return exec_;
        }

        using execution_category =
            hpx::traits::executor_execution_category_t<BaseExecutor>;

        using parameters_type =
            hpx::traits::executor_parameters_type_t<BaseExecutor>;

        template <typename T, typename... Ts>
        using future_type =
            hpx::traits::executor_future_t<BaseExecutor, T, Ts...>;

    private:
        // NonBlockingOneWayExecutor interface
        template <typename F, typename... Ts>
        friend decltype(auto) tag_invoke(hpx::parallel::execution::post_t,
            annotating_executor const& exec, F&& f, Ts&&... ts)
        {
            return parallel::execution::post(exec.exec_,
                hpx::annotated_function(HPX_FORWARD(F, f), exec.annotation_),
                HPX_FORWARD(Ts, ts)...);
        }

        // OneWayExecutor interface
        template <typename F, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::sync_execute_t,
            annotating_executor const& exec, F&& f, Ts&&... ts)
        {
            return parallel::execution::sync_execute(exec.exec_,
                hpx::annotated_function(HPX_FORWARD(F, f), exec.annotation_),
                HPX_FORWARD(Ts, ts)...);
        }

        // TwoWayExecutor interface
        template <typename F, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::async_execute_t,
            annotating_executor const& exec, F&& f, Ts&&... ts)
        {
            return parallel::execution::async_execute(exec.exec_,
                hpx::annotated_function(HPX_FORWARD(F, f), exec.annotation_),
                HPX_FORWARD(Ts, ts)...);
        }

        template <typename F, typename Future, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::then_execute_t,
            annotating_executor const& exec, F&& f, Future&& predecessor,
            Ts&&... ts)
        {
            return parallel::execution::then_execute(exec.exec_,
                hpx::annotated_function(HPX_FORWARD(F, f), exec.annotation_),
                HPX_FORWARD(Future, predecessor), HPX_FORWARD(Ts, ts)...);
        }

        // BulkTwoWayExecutor interface
        template <typename F, typename S, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::bulk_async_execute_t,
            annotating_executor const& exec, F&& f, S const& shape, Ts&&... ts)
        {
            return parallel::execution::bulk_async_execute(exec.exec_,
                hpx::annotated_function(HPX_FORWARD(F, f), exec.annotation_),
                shape, HPX_FORWARD(Ts, ts)...);
        }

        template <typename F, typename S, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::bulk_sync_execute_t,
            annotating_executor const& exec, F&& f, S const& shape, Ts&&... ts)
        {
            return parallel::execution::bulk_sync_execute(exec.exec_,
                hpx::annotated_function(HPX_FORWARD(F, f), exec.annotation_),
                shape, HPX_FORWARD(Ts, ts)...);
        }

        template <typename F, typename S, typename Future, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::bulk_then_execute_t,
            annotating_executor const& exec, F&& f, S const& shape,
            Future&& predecessor, Ts&&... ts)
        {
            return parallel::execution::bulk_then_execute(exec.exec_,
                hpx::annotated_function(HPX_FORWARD(F, f), exec.annotation_),
                shape, HPX_FORWARD(Future, predecessor),
                HPX_FORWARD(Ts, ts)...);
        }

        // support with_annotation property
        friend constexpr annotating_executor tag_invoke(
            hpx::execution::experimental::with_annotation_t,
            annotating_executor const& exec, char const* annotation)
        {
            auto exec_with_annotation = exec;
            exec_with_annotation.annotation_ = annotation;
            return exec_with_annotation;
        }

        friend annotating_executor tag_invoke(
            hpx::execution::experimental::with_annotation_t,
            annotating_executor const& exec, std::string annotation)
        {
            auto exec_with_annotation = exec;
            exec_with_annotation.annotation_ =
                hpx::detail::store_function_annotation(HPX_MOVE(annotation));
            return exec_with_annotation;
        }

        // support get_annotation property
        friend constexpr char const* tag_invoke(
            hpx::execution::experimental::get_annotation_t,
            annotating_executor const& exec) noexcept
        {
            return exec.annotation_;
        }

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned int const /* version */)
        {
            // clang-format off
            ar & exec_;
            // clang-format on
        }

        std::decay_t<BaseExecutor> exec_;
        char const* const annotation_ = nullptr;
        /// \endcond
    };

    // support all properties exposed by the wrapped executor
    // clang-format off
    template <typename Tag, typename BaseExecutor,typename Property,
        HPX_CONCEPT_REQUIRES_(
            hpx::execution::experimental::is_scheduling_property_v<Tag>
            )>
    // clang-format on
    auto tag_invoke(
        Tag tag, annotating_executor<BaseExecutor> const& exec, Property&& prop)
        -> decltype(annotating_executor<BaseExecutor>(std::declval<Tag>()(
            std::declval<BaseExecutor>(), std::declval<Property>())))
    {
        return annotating_executor<BaseExecutor>(
            tag(exec.get_executor(), HPX_FORWARD(Property, prop)));
    }

    // clang-format off
    template <typename Tag, typename BaseExecutor,
        HPX_CONCEPT_REQUIRES_(
            hpx::execution::experimental::is_scheduling_property_v<Tag>
            )>
    // clang-format on
    auto tag_invoke(Tag tag, annotating_executor<BaseExecutor> const& exec)
        -> decltype(std::declval<Tag>()(std::declval<BaseExecutor>()))
    {
        return tag(exec.get_executor());
    }

    ///////////////////////////////////////////////////////////////////////////
#if !defined(DOXYGEN)    // doxygen gets confused by the deduction guides
    template <typename BaseExecutor>
    explicit annotating_executor(BaseExecutor&& sched, std::string annotation)
        -> annotating_executor<std::decay_t<BaseExecutor>>;

    template <typename BaseExecutor>
    explicit annotating_executor(
        BaseExecutor&& sched, char const* annotation = nullptr)
        -> annotating_executor<std::decay_t<BaseExecutor>>;
#endif

    ///////////////////////////////////////////////////////////////////////////
    // if the given executor does not support annotations, wrap it into
    // a annotating_executor
    //
    // The functions below are used for executors that do not directly support
    // annotations. Those are wrapped into an annotating_executor if passed
    // to `with_annotation`.
    //
    // clang-format off
    template <typename Executor,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_executor_any_v<Executor>
            )>
    // clang-format on
    constexpr auto tag_fallback_invoke(
        with_annotation_t, Executor&& exec, char const* annotation)
    {
        return annotating_executor<std::decay_t<Executor>>(
            HPX_FORWARD(Executor, exec), annotation);
    }

    // clang-format off
    template <typename Executor,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_executor_any_v<Executor>
            )>
    // clang-format on
    auto tag_fallback_invoke(
        with_annotation_t, Executor&& exec, std::string annotation)
    {
        return annotating_executor<std::decay_t<Executor>>(
            HPX_FORWARD(Executor, exec), HPX_MOVE(annotation));
    }
}    // namespace hpx::execution::experimental

namespace hpx::parallel::execution {

    // The annotating executor exposes the same executor categories as its
    // underlying (wrapped) executor.

    /// \cond NOINTERNAL
    template <typename BaseExecutor>
    struct is_one_way_executor<
        hpx::execution::experimental::annotating_executor<BaseExecutor>>
      : is_one_way_executor<BaseExecutor>
    {
    };

    template <typename BaseExecutor>
    struct is_never_blocking_one_way_executor<
        hpx::execution::experimental::annotating_executor<BaseExecutor>>
      : is_never_blocking_one_way_executor<BaseExecutor>
    {
    };

    template <typename BaseExecutor>
    struct is_bulk_one_way_executor<
        hpx::execution::experimental::annotating_executor<BaseExecutor>>
      : is_bulk_one_way_executor<BaseExecutor>
    {
    };

    template <typename BaseExecutor>
    struct is_two_way_executor<
        hpx::execution::experimental::annotating_executor<BaseExecutor>>
      : is_two_way_executor<BaseExecutor>
    {
    };

    template <typename BaseExecutor>
    struct is_bulk_two_way_executor<
        hpx::execution::experimental::annotating_executor<BaseExecutor>>
      : is_bulk_two_way_executor<BaseExecutor>
    {
    };

    template <typename BaseExecutor>
    struct is_scheduler_executor<
        hpx::execution::experimental::annotating_executor<BaseExecutor>>
      : is_scheduler_executor<BaseExecutor>
    {
    };
    /// \endcond
}    // namespace hpx::parallel::execution
