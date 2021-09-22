//  Copyright (c) ETH Zurich 2021
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/execution/algorithms/detail/partial_algorithm.hpp>
#include <hpx/execution/algorithms/let_value.hpp>
#include <hpx/execution/algorithms/transform.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/execution_base/sender.hpp>
#include <hpx/executors/execution_policy.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace detail {
    // This is a lighter-weight alternative to bind for use in parallel
    // algorithm overloads, where one needs to bind an execution policy to an
    // algorithm for use in execution::transform. Typically used together with
    // transform_with_bound_algorithm.
    template <typename Tag, typename ExPolicy>
    struct bound_algorithm
    {
        std::decay_t<ExPolicy> policy;

        template <typename T1, typename... Ts>
        auto operator()(T1&& t1, Ts&&... ts) -> decltype(Tag{}(
            std::move(policy), std::forward<T1>(t1), std::forward<Ts>(ts)...))
        {
            return Tag{}(std::move(policy), std::forward<T1>(t1),
                std::forward<Ts>(ts)...);
        }
    };

    // Detects if the given type is a bound_algorithm.
    template <typename Bound>
    struct is_bound_algorithm : std::false_type
    {
    };

    template <typename Tag, typename ExPolicy>
    struct is_bound_algorithm<bound_algorithm<Tag, ExPolicy>> : std::true_type
    {
    };

    // Helper function for use in creating overloads of parallel algorithms that
    // take senders. Takes an execution policy, a predecessor sender, and an
    // "algorithm" (i.e. a tag) and applies transform with the predecessor
    // sender and the execution policy bound to the algorithm.
    template <typename Tag, typename ExPolicy, typename Predecessor>
    auto transform_with_bound_algorithm(
        Predecessor&& predecessor, ExPolicy&& policy)
    {
        // If the given execution policy can has a task policy, i.e. the
        // algorithm can return a future, we use the task policy since we can
        // then directly return the future as a sender and avoid potential
        // blocking that may happen internally.
        if constexpr (hpx::execution::detail::has_async_execution_policy_v<
                          ExPolicy>)
        {
            auto task_policy =
                std::forward<ExPolicy>(policy)(hpx::execution::task);
            return hpx::execution::experimental::let_value(
                std::forward<Predecessor>(predecessor),
                bound_algorithm<Tag, decltype(task_policy)>{
                    std::move(task_policy)});
        }
        // If the policy does not have a task policy, the algorithm can only be
        // called synchronously. In this case we only use transform to chain the
        // algorithm after the predecessor sender.
        else
        {
            return hpx::execution::experimental::transform(
                std::forward<Predecessor>(predecessor),
                bound_algorithm<Tag, ExPolicy>{std::forward<ExPolicy>(policy)});
        }
    }

    // Helper base class for implementing parallel algorithm DPOs. See
    // tag_fallback documentation for details. Compared to tag_fallback this
    // adds two tag_fallback_dispatch overloads that are generic for all
    // parallel algorithms:
    //
    //   1. An overload taking a predecessor sender which sends all arguments
    //      for the regular parallel algorithm overload, except an execution
    //      policy; and an execution policy.
    //   2. An overload taking only an execution policy. This overload returns a
    //      partially applied parallel algorithm, and needs to be supplied a
    //      predecessor sender. The partially applied algorithm is callable with
    //      a predecessor sender: partially_applied_algorithm(predecessor). The
    //      predecessor can also be supplied  using the operator| overload:
    //      predecessor | partially_applied_parallel_algorithm.
    template <typename Tag>
    struct tag_parallel_algorithm : hpx::functional::tag_fallback<Tag>
    {
        // clang-format off
        template <typename Predecessor, typename ExPolicy,
            HPX_CONCEPT_REQUIRES_(
                std::conjunction_v<
                    hpx::is_execution_policy<ExPolicy>,
                    std::negation<detail::is_bound_algorithm<Predecessor>>,
                    hpx::execution::experimental::is_sender<
                        std::decay_t<Predecessor>>>
            )>
        // clang-format on
        friend auto tag_fallback_dispatch(
            Tag, Predecessor&& predecessor, ExPolicy&& policy)
        {
            return detail::transform_with_bound_algorithm<Tag>(
                std::forward<Predecessor>(predecessor),
                std::forward<ExPolicy>(policy));
        }

        template <typename ExPolicy,
            HPX_CONCEPT_REQUIRES_(hpx::is_execution_policy<ExPolicy>::value)>
        friend auto tag_fallback_dispatch(Tag, ExPolicy&& policy)
        {
            return hpx::execution::experimental::detail::partial_algorithm<Tag,
                ExPolicy>{std::forward<ExPolicy>(policy)};
        }
    };
}}    // namespace hpx::detail
