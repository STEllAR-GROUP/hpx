//  Copyright (c) 2016-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/async_base/traits/is_launch_policy.hpp>
#include <hpx/execution/traits/executor_traits.hpp>
#include <hpx/execution_base/execution.hpp>

#include <type_traits>
#include <utility>

namespace hpx::parallel::execution {

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {

        /// \cond NOINTERNAL
        template <typename Category1, typename Category2>
        struct is_not_weaker : std::false_type
        {
        };

        template <typename Category>
        struct is_not_weaker<Category, Category> : std::true_type
        {
        };

        template <>
        struct is_not_weaker<hpx::execution::parallel_execution_tag,
            hpx::execution::unsequenced_execution_tag> : std::true_type
        {
        };

        template <>
        struct is_not_weaker<hpx::execution::sequenced_execution_tag,
            hpx::execution::unsequenced_execution_tag> : std::true_type
        {
        };

        template <>
        struct is_not_weaker<hpx::execution::sequenced_execution_tag,
            hpx::execution::parallel_execution_tag> : std::true_type
        {
        };

        template <typename Category1, typename Category2>
        inline constexpr bool is_not_weaker_v =
            is_not_weaker<Category1, Category2>::value;
        /// \endcond
    }    // namespace detail

    /// Rebind the type of executor used by an execution policy. The execution
    /// category of Executor shall not be weaker than that of ExecutionPolicy.
    template <typename ExPolicy, typename Executor, typename Parameters>
    struct rebind_executor
    {
        /// \cond NOINTERNAL
        using policy_type = std::decay_t<ExPolicy>;
        using executor_type = std::decay_t<Executor>;
        using parameters_type = std::decay_t<Parameters>;

        using category1 = typename policy_type::execution_category;
        using category2 =
            hpx::traits::executor_execution_category_t<executor_type>;

        static_assert(detail::is_not_weaker_v<category2, category1>,
            "detail::is_not_weaker_v<category2, category1>");
        /// \endcond

        /// The type of the rebound execution policy
        using type = typename policy_type::template rebind<executor_type,
            parameters_type>::type;
    };

    template <typename ExPolicy, typename Executor, typename Parameters>
    using rebind_executor_t =
        typename rebind_executor<ExPolicy, Executor, Parameters>::type;

    //////////////////////////////////////////////////////////////////////////
    inline constexpr struct create_rebound_policy_t final
    {
        template <typename ExPolicy, typename Executor, typename Parameters>
        constexpr decltype(auto) operator()(
            ExPolicy&&, Executor&& exec, Parameters&& parameters) const
        {
            using rebound_type =
                rebind_executor_t<ExPolicy, Executor, Parameters>;

            return rebound_type(HPX_FORWARD(Executor, exec),
                HPX_FORWARD(Parameters, parameters));
        }
    } create_rebound_policy{};
}    // namespace hpx::parallel::execution
