//  Copyright (c) 2007-2017 Hartmut Kaiser
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
#include <hpx/execution/traits/is_executor.hpp>
#include <hpx/functional/deferred_call.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/futures/future.hpp>
#include <hpx/pack_traversal/unwrap.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace execution {
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
        typedef sequenced_execution_tag execution_category;

        // OneWayExecutor interface
        template <typename F, typename... Ts>
        static
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type
            sync_execute(F&& f, Ts&&... ts)
        {
            return hpx::detail::sync_launch_policy_dispatch<
                launch::sync_policy>::call(launch::sync, std::forward<F>(f),
                std::forward<Ts>(ts)...);
        }

        // TwoWayExecutor interface
        template <typename F, typename... Ts>
        static hpx::future<
            typename hpx::util::detail::invoke_deferred_result<F, Ts...>::type>
        async_execute(F&& f, Ts&&... ts)
        {
            return hpx::detail::async_launch_policy_dispatch<
                launch::deferred_policy>::call(launch::deferred,
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        // NonBlockingOneWayExecutor (adapted) interface
        template <typename F, typename... Ts>
        static void post(F&& f, Ts&&... ts)
        {
            sync_execute(std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        // BulkTwoWayExecutor interface
        template <typename F, typename S, typename... Ts>
        static std::vector<hpx::future<typename parallel::execution::detail::
                bulk_function_result<F, S, Ts...>::type>>
        bulk_async_execute(F&& f, S const& shape, Ts&&... ts)
        {
            typedef
                typename parallel::execution::detail::bulk_function_result<F, S,
                    Ts...>::type result_type;
            std::vector<hpx::future<result_type>> results;

            try
            {
                for (auto const& elem : shape)
                {
                    results.push_back(async_execute(f, elem, ts...));
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

        template <typename F, typename S, typename... Ts>
        static typename parallel::execution::detail::bulk_execute_result<F, S,
            Ts...>::type
        bulk_sync_execute(F&& f, S const& shape, Ts&&... ts)
        {
            return hpx::util::unwrap(bulk_async_execute(
                std::forward<F>(f), shape, std::forward<Ts>(ts)...));
        }

        std::size_t processing_units_count()
        {
            return 1;
        }

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& /* ar */, const unsigned int /* version */)
        {
        }
        /// \endcond
    };
}}    // namespace hpx::execution

namespace hpx { namespace parallel { namespace execution {
    using sequenced_executor HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::execution::sequenced_executor is deprecated. Use "
        "hpx::execution::sequenced_executor instead.") =
        hpx::execution::sequenced_executor;
}}}    // namespace hpx::parallel::execution

namespace hpx { namespace parallel { namespace execution {
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
}}}    // namespace hpx::parallel::execution
