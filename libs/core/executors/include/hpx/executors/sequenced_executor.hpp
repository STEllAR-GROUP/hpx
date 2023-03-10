//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/sequenced_executor.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/errors/exception_list.hpp>
#include <hpx/execution/detail/async_launch_policy_dispatch.hpp>
#include <hpx/execution/detail/sync_launch_policy_dispatch.hpp>
#include <hpx/execution/executors/execution.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/executors/execution_policy_mappings.hpp>
#include <hpx/executors/parallel_executor.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/modules/topology.hpp>
#include <hpx/pack_traversal/unwrap.hpp>
#include <hpx/serialization/serialize.hpp>
#include <hpx/threading_base/annotated_function.hpp>
#include <hpx/threading_base/detail/get_default_pool.hpp>
#include <hpx/threading_base/thread_description.hpp>
#include <hpx/threading_base/thread_num_tss.hpp>
#include <hpx/timing/steady_clock.hpp>

#include <cstddef>
#include <iterator>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx::execution {

    ///////////////////////////////////////////////////////////////////////////
    /// A \a sequential_executor creates groups of sequential execution agents
    /// which execute in the calling thread. The sequential order is given by
    /// the lexicographical order of indices in the index space.
    ///
    struct sequenced_executor
    {
        /// \cond NOINTERNAL
        bool operator==(sequenced_executor const& /*rhs*/) const noexcept
        {
            return true;
        }

        bool operator!=(sequenced_executor const& /*rhs*/) const noexcept
        {
            return false;
        }

        sequenced_executor const& context() const noexcept
        {
            return *this;
        }
        /// \endcond

        /// \cond NOINTERNAL
        using execution_category = sequenced_execution_tag;

    private:
        // OneWayExecutor interface
        template <typename F, typename... Ts>
        static decltype(auto) sync_execute_impl(F&& f, Ts&&... ts)
        {
            return hpx::detail::sync_launch_policy_dispatch<
                launch::sync_policy>::call(launch::sync, HPX_FORWARD(F, f),
                HPX_FORWARD(Ts, ts)...);
        }

        template <typename F, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::sync_execute_t,
            [[maybe_unused]] sequenced_executor const& exec, F&& f, Ts&&... ts)
        {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            hpx::scoped_annotation annotate(exec.annotation_ ?
                    exec.annotation_ :
                    "parallel_policy_executor::sync_execute");
#endif
            return sequenced_executor::sync_execute_impl(
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        // TwoWayExecutor interface
        template <typename F, typename... Ts>
        static decltype(auto) async_execute_impl(
            hpx::threads::thread_description const& desc, F&& f, Ts&&... ts)
        {
            return hpx::detail::async_launch_policy_dispatch<
                launch::deferred_policy>::call(launch::deferred, desc,
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        template <typename F, typename... Ts>
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::async_execute_t,
            [[maybe_unused]] sequenced_executor const& exec, F&& f, Ts&&... ts)
        {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            hpx::threads::thread_description desc(f, exec.annotation_);
#else
            hpx::threads::thread_description desc(f);
#endif
            return sequenced_executor::async_execute_impl(
                desc, HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        // NonBlockingOneWayExecutor (adapted) interface
        template <typename F, typename... Ts>
        friend void tag_invoke(hpx::parallel::execution::post_t,
            [[maybe_unused]] sequenced_executor const& exec, F&& f, Ts&&... ts)
        {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            hpx::scoped_annotation annotate(exec.annotation_ ?
                    exec.annotation_ :
                    "parallel_policy_executor::sync_execute");
#endif
            sequenced_executor::sync_execute_impl(
                HPX_FORWARD(F, f), HPX_FORWARD(Ts, ts)...);
        }

        // BulkTwoWayExecutor interface
        // clang-format off
        template <typename F, typename S, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                !std::is_integral_v<S>
            )>
        // clang-format on
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::bulk_async_execute_t,
            [[maybe_unused]] sequenced_executor const& exec, F&& f,
            S const& shape, Ts&&... ts)
        {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            hpx::threads::thread_description desc(f, exec.annotation_);
#else
            hpx::threads::thread_description desc(f);
#endif

            using result_type =
                parallel::execution::detail::bulk_function_result_t<F, S,
                    Ts...>;
            std::vector<hpx::future<result_type>> results;

            try
            {
                for (auto const& elem : shape)
                {
                    results.push_back(sequenced_executor::async_execute_impl(
                        desc, f, elem, ts...));
                }
            }
            catch (std::bad_alloc const& ba)
            {
                throw ba;
            }
            catch (...)
            {
                throw exception_list(std::current_exception());
            }

            return results;
        }

        // clang-format off
        template <typename F, typename S, typename... Ts,
            HPX_CONCEPT_REQUIRES_(
                !std::is_integral_v<S>
            )>
        // clang-format on
        friend decltype(auto) tag_invoke(
            hpx::parallel::execution::bulk_sync_execute_t,
            sequenced_executor const& exec, F&& f, S const& shape, Ts&&... ts)
        {
            return hpx::unwrap(hpx::parallel::execution::bulk_async_execute(
                exec, HPX_FORWARD(F, f), shape, HPX_FORWARD(Ts, ts)...));
        }

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
        friend constexpr sequenced_executor tag_invoke(
            hpx::execution::experimental::with_annotation_t,
            sequenced_executor const& exec, char const* annotation)
        {
            auto exec_with_annotation = exec;
            exec_with_annotation.annotation_ = annotation;
            return exec_with_annotation;
        }

        friend sequenced_executor tag_invoke(
            hpx::execution::experimental::with_annotation_t,
            sequenced_executor const& exec, std::string annotation)
        {
            auto exec_with_annotation = exec;
            exec_with_annotation.annotation_ =
                hpx::detail::store_function_annotation(HPX_MOVE(annotation));
            return exec_with_annotation;
        }

        friend constexpr char const* tag_invoke(
            hpx::execution::experimental::get_annotation_t,
            sequenced_executor const& exec) noexcept
        {
            return exec.annotation_;
        }
#endif

        friend constexpr std::size_t tag_invoke(
            hpx::parallel::execution::processing_units_count_t,
            sequenced_executor const&,
            hpx::chrono::steady_duration const& = hpx::chrono::null_duration,
            std::size_t = 0)
        {
            return 1;
        }

        friend auto tag_invoke(
            hpx::execution::experimental::get_processing_units_mask_t,
            sequenced_executor const&)
        {
            return threads::detail::get_self_or_default_pool()
                ->get_used_processing_unit(hpx::get_worker_thread_num(), false);
        }

        friend auto tag_invoke(hpx::execution::experimental::get_cores_mask_t,
            sequenced_executor const&)
        {
            return threads::detail::get_self_or_default_pool()
                ->get_used_processing_unit(hpx::get_worker_thread_num(), true);
        }

        friend decltype(auto) tag_invoke(hpx::execution::experimental::to_par_t,
            [[maybe_unused]] sequenced_executor const& exec)
        {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            return hpx::execution::experimental::with_annotation(
                parallel_executor(), exec.annotation_);
#else
            return parallel_executor();
#endif
        }

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& /* ar */, unsigned int const /* version */)
        {
        }

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
        char const* annotation_ = nullptr;
#endif
        /// \endcond
    };
}    // namespace hpx::execution

namespace hpx::parallel::execution {

    /// \cond NOINTERNAL
    template <>
    struct is_one_way_executor<hpx::execution::sequenced_executor>
      : std::true_type
    {
    };

    template <>
    struct is_bulk_one_way_executor<hpx::execution::sequenced_executor>
      : std::true_type
    {
    };

    template <>
    struct is_never_blocking_one_way_executor<
        hpx::execution::sequenced_executor> : std::true_type
    {
    };

    template <>
    struct is_two_way_executor<hpx::execution::sequenced_executor>
      : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<hpx::execution::sequenced_executor>
      : std::true_type
    {
    };
    /// \endcond
}    // namespace hpx::parallel::execution
