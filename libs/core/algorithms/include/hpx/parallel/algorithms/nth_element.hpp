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
#include <hpx/parallel/algorithms/partition.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>

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
            static RandomIt sequential(ExPolicy, RandomIt first, RandomIt nth,
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
            static typename util::detail::algorithm_result<ExPolicy,
                RandomIt>::type
            parallel(ExPolicy&& policy, RandomIt first, RandomIt nth, Sent last,
                Pred&& pred, Proj&& proj)
            {
                using value_type = std::iterator_traits<RandomIt>::value_type;
                using result_type =
                    util::detail::algorithm_result<ExPolicy, RandomIt>;

                RandomIt partitionIter, return_last;

                if (first == last)
                {
                    return result_type::get(std::move(first));
                }

                if (nth == last)
                {
                    return result_type::get(std::move(nth));
                }

                try
                {
                    RandomIt last_iter =
                        detail::advance_to_sentinel(first, last);
                    return_last = last_iter;

                    while (first != last_iter)
                    {
                        auto n = detail::distance(first, last_iter);

                        // get random pivot index
                        auto pivotIndex = std::rand() % n;
                        // swap first and pivot element
#if defined(HPX_HAVE_CXX20_STD_RANGES_ITER_SWAP)
                        std::ranges::iter_swap(first, first + pivotIndex);
#else
                        std::iter_swap(first, first + pivotIndex);
#endif

                        auto partitionResult =
                            hpx::parallel::v1::detail::partition<RandomIt>()
                                .call(
                                    std::forward<ExPolicy>(policy), first + 1,
                                    last_iter,
                                    [val = HPX_INVOKE(proj, *first), &proj](
                                        value_type const& elem) {
                                        return elem <= val;
                                    },
                                    proj);

                        if constexpr (std::is_same_v<decltype(partitionResult),
                                          RandomIt>)
                        {
                            partitionIter = partitionResult;
                        }
                        else
                        {
                            partitionIter = partitionResult.get();
                        }

                        --partitionIter;
                        // swap first element and partitionIter(ending element of first group)
#if defined(HPX_HAVE_CXX20_STD_RANGES_ITER_SWAP)
                        std::ranges::iter_swap(first, partitionIter);
#else
                        std::iter_swap(first, partitionIter);
#endif

                        // if nth element < partitioned index, it lies in [first, partitionIter)
                        if (partitionIter < nth)
                        {
                            first = partitionIter + 1;
                        }
                        // else it lies in [partitionIter + 1, last)
                        else if (partitionIter > nth)
                        {
                            last_iter = partitionIter;
                        }
                        // partitionIter == nth
                        else
                        {
                            break;
                        }
                    }
                }
                catch (...)
                {
                    return result_type::get(
                        detail::handle_exception<ExPolicy, RandomIt>::call(
                            std::current_exception()));
                }

                return result_type::get(std::move(return_last));
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
                hpx::traits::is_iterator_v<RandomIt> &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<RandomIt>::value_type,
                    typename std::iterator_traits<RandomIt>::value_type
                >
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
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator_v<RandomIt> &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<RandomIt>::value_type,
                    typename std::iterator_traits<RandomIt>::value_type
                >
            )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy>
        tag_fallback_dispatch(hpx::nth_element_t, ExPolicy&& policy,
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
