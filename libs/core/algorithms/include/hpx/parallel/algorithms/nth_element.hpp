//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2017 Hartmut Kaiser
//  Copyright (c) 2020 Francisco Jose Tapia
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
#include <hpx/parallel/algorithms/minmax.hpp>
#include <hpx/parallel/algorithms/partial_sort.hpp>
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

        ///////////////////////////////////////////////////////////////////////
        ///
        /// \brief : The element placed in the nth position is exactly the
        ///          element that would occur in this position if the range
        ///          was fully sorted. All of the elements before this new nth
        ///          element are less than or equal to the elements after the
        ///          new nth element.
        ///
        /// \param first : iterator to the first element
        /// \param nth : iterator defining the sort partition point
        /// \param end : iterator to the element after the last in the range
        /// \param level : level of depth in the call from the root
        /// \param comp : object for to Compare elements
        /// \param proj : projection
        ///
        template <class RandomIt, typename Compare, typename Proj>
        void nth_element_seq(RandomIt first, RandomIt nth, RandomIt end,
            uint32_t level, Compare&& comp, Proj&& proj)
        {
            const uint32_t nmin_sort = 24;

            // Check if the iterators are corrects
            auto nelem = end - first;
            if (nelem == 0)
                return;
            auto n_nth = nth - first + 1;
            HPX_ASSERT(nelem >= 0 and n_nth > 0 and n_nth <= nelem);

            // Check  the special conditions
            if (nth == first)
            {
                RandomIt it = detail::min_element<RandomIt>().call(
                    hpx::execution::seq, first, end, comp, proj);
                if (it != first)
                    std::swap(*it, *first);
                return;
            };

            if (nelem < nmin_sort)
            {
                detail::sort<RandomIt>().call(
                    hpx::execution::seq, first, end, comp, proj);
                return;
            }
            if (level == 0)
            {
                std::make_heap(first, end, comp);
                std::sort_heap(first, nth, comp);
                return;
            };

            // Filter the range and check which part contains the nth element
            RandomIt c_last = filter(first, end, comp);

            if (c_last == nth)
                return;

            if (nth < c_last)
                nth_element_seq(first, nth, c_last, level - 1, comp, proj);
            else
                nth_element_seq(c_last + 1, nth, end, level - 1, comp, proj);

            return;
        };

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
                auto end = detail::advance_to_sentinel(first, last);

                // Check if the iterators are corrects
                auto nelem = end - first;
                if (nelem == 0)
                    return end;
                auto n_nth = nth - first + 1;
                HPX_ASSERT(nelem > 0 and n_nth > 0 and n_nth <= nelem);

                uint32_t level = detail::nbits64(nelem) * 2;
                detail::nth_element_seq(first, nth, end, level,
                    std::forward<Pred>(pred), std::forward<Proj>(proj));

                return end;
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
