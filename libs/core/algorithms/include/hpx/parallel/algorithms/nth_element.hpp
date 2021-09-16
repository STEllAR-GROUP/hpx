//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2017 Hartmut Kaiser
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/nth_element.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>

#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/algorithms/mismatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // nth_element
    namespace detail {
        template <typename Iter>
        struct nth_element : public detail::algorithm<nth_element<Iter>, Iter>
        {
            nth_element()
              : nth_element::algorithm("nth_element")
            {
            }

            template <typename ExPolicy, typename RandomIt, typename Sent,
                typename Pred, typename Proj>
            static Iter sequential(ExPolicy, RandomIt first, RandomIt nth,
                Sent last, Pred&& pred, Proj&& proj)
            {
                util::invoke_projected<Pred, Proj> pred_projected{
                    std::forward<Pred>(pred), std::forward<Proj>(proj)};

                std::nth_element(first, nth,
                    detail::advance_to_sentinel(first, last),
                    std::move(pred_projected));

                return first;
            }

            template <typename ExPolicy, typename RandomIt, typename Sent,
                typename Pred, typename Proj>
            static typename util::detail::algorithm_result<ExPolicy, Iter>::type
            parallel(ExPolicy&& policy, RandomIt first, RandomIt nth, Sent last,
                Pred&& pred, Proj&& proj2)
            {
                return first;
            }
        };
        /// \endcond
    }    // namespace detail
}}}      // namespace hpx::parallel::v1

namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::nth_element
    HPX_INLINE_CONSTEXPR_VARIABLE struct nth_element_t final
      : hpx::detail::tag_parallel_algorithm<nth_element_t>
    {
        // clang-format off
        template <typename RandomIt,
            typename Pred = hpx::parallel::v1::detail::less,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<RandomIt>::value
            )>
        // clang-format on
        friend void tag_fallback_dispatch(hpx::nth_element_t, RandomIt first,
            RandomIt nth, RandomIt last, Pred&& pred = Pred())
        {
            static_assert(
                hpx::traits::is_random_access_iterator<RandomIt>::value,
                "Requires at least random iterator.");

            hpx::parallel::v1::detail::nth_element<RandomIt>().call(
                hpx::execution::seq, first, nth, last, std::forward<Pred>(pred),
                hpx::parallel::util::projection_identity{});
        }

        // clang-format off
        template <typename ExPolicy, typename RandomIt,
            typename Pred = hpx::parallel::v1::detail::less,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<RandomIt>::value
            )>
        // clang-format on
        friend void tag_fallback_dispatch(hpx::nth_element_t, ExPolicy&& policy,
            RandomIt first, RandomIt nth, RandomIt last, Pred&& pred = Pred())
        {
            static_assert(
                hpx::traits::is_random_access_iterator<RandomIt>::value,
                "Requires at least random iterator.");

            using result_type =
                typename hpx::parallel::util::detail::algorithm_result<
                    ExPolicy>::type;

            return hpx::util::void_guard<result_type>(),
                   hpx::parallel::v1::detail::nth_element<RandomIt>().call(
                       std::forward<ExPolicy>(policy), first, nth, last,
                       std::forward<Pred>(pred),
                       hpx::parallel::util::projection_identity{});
        }
    } nth_element{};
}    // namespace hpx
