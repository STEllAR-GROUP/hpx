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

#if defined(DOXYGEN)

namespace hpx {
    // clang-format off

    /// nth_element is a partial sorting algorithm that rearranges elements in
    /// [first, last) such that the element pointed at by nth is changed to
    /// whatever element would occur in that position if [first, last) were
    /// sorted and all of the elements before this new nth element are less
    /// than or equal to the elements after the new nth element.
    ///
    /// \note   Complexity: Linear in std::distance(first, last) on average.
    ///         O(N) applications of the predicate, and O(N log N) swaps,
    ///         where N = last - first.
    ///
    /// \tparam RandomIt    The type of the source begin, nth, and end
    ///                     iterators used (deduced). This iterator type must
    ///                     meet the requirements of a random access iterator.
    /// \tparam Pred        Comparison function object which returns true if
    ///                     the first argument is less than the second. This defaults
    ///                     to std::less<>.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param nth          Refers to the iterator defining the sort partition
    ///                     point
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param pred         Specifies the comparison function object which
    ///                     returns true if the first argument is less than
    ///                     (i.e. is ordered before) the second.
    ///                     The signature of this
    ///                     comparison function should be equivalent to:
    ///                     \code
    ///                     bool cmp(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type must be such that an object of
    ///                     type \a randomIt can be dereferenced and then
    ///                     implicitly converted to Type. This defaults
    ///                     to std::less<>.
    ///
    /// The comparison operations in the parallel \a nth_element
    /// algorithm invoked without an execution policy object execute in
    /// sequential order in the calling thread.
    ///
    /// \returns  The \a nth_element algorithms returns nothing.
    ///
    template <typename RandomIt, typename Pred>
    void nth_element(RandomIt first, RandomIt nth, RandomIt last, Pred&& pred);

    /// nth_element is a partial sorting algorithm that rearranges elements in
    /// [first, last) such that the element pointed at by nth is changed to
    /// whatever element would occur in that position if [first, last) were
    /// sorted and all of the elements before this new nth element are less
    /// than or equal to the elements after the new nth element.
    ///
    /// \note   Complexity: Linear in std::distance(first, last) on average.
    ///         O(N) applications of the predicate, and O(N log N) swaps,
    ///         where N = last - first.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam RandomIt    The type of the source begin, nth, and end
    ///                     iterators used (deduced). This iterator type must
    ///                     meet the requirements of a random access iterator.
    /// \tparam Pred        Comparison function object which returns true if
    ///                     the first argument is less than the second. This
    ///                     defaults to std::less<>.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param nth          Refers to the iterator defining the sort partition
    ///                     point
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param pred         Specifies the comparison function object which
    ///                     returns true if the first argument is less than
    ///                     (i.e. is ordered before) the second.
    ///                     The signature of this
    ///                     comparison function should be equivalent to:
    ///                     \code
    ///                     bool cmp(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type must be such that an object of
    ///                     type \a randomIt can be dereferenced and then
    ///                     implicitly converted to Type. This defaults
    ///                     to std::less<>.
    ///
    /// The comparison operations in the parallel \a nth_element invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a nth_element algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a nth_element algorithms returns nothing.
    ///
    template <typename ExPolicy, typename RandomIt, typename Pred>
    void nth_element(ExPolicy&& policy, RandomIt first, RandomIt nth,
        RandomIt last, Pred&& pred);

    // clang-format on
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>

#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/pivot.hpp>
#include <hpx/parallel/algorithms/minmax.hpp>
#include <hpx/parallel/algorithms/partial_sort.hpp>
#include <hpx/parallel/algorithms/partition.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
        static constexpr void nth_element_seq(RandomIt first, RandomIt nth,
            RandomIt end, std::uint32_t level, Compare&& comp, Proj&& proj)
        {
            std::uint32_t const nmin_sort = 24;
            auto nelem = end - first;

            // Check  the special conditions
            if (nth == first)
            {
                RandomIt it = detail::min_element<RandomIt>().call(
                    hpx::execution::seq, first, end, HPX_FORWARD(Compare, comp),
                    HPX_FORWARD(Proj, proj));

                if (it != first)
                {
#if defined(HPX_HAVE_CXX20_STD_RANGES_ITER_SWAP)
                    std::ranges::iter_swap(it, first);
#else
                    std::iter_swap(it, first);
#endif
                }

                return;
            };

            if (nelem < nmin_sort)
            {
                detail::sort<RandomIt>().call(hpx::execution::seq, first, end,
                    HPX_FORWARD(Compare, comp), HPX_FORWARD(Proj, proj));
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
                nth_element_seq(first, nth, c_last, level - 1,
                    HPX_FORWARD(Compare, comp), HPX_FORWARD(Proj, proj));
            else
                nth_element_seq(c_last + 1, nth, end, level - 1,
                    HPX_FORWARD(Compare, comp), HPX_FORWARD(Proj, proj));

            return;
        }

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

                auto nelem = end - first;
                if (nelem == 0)
                    return first;
                HPX_ASSERT(nelem >= 0 && (nth - first + 1) > 0 &&
                    (nth - first + 1) <= nelem);

                uint32_t level = detail::nbits64(nelem) * 2;
                detail::nth_element_seq(first, nth, end, level,
                    HPX_FORWARD(Pred, pred), HPX_FORWARD(Proj, proj));

                return end;
            }

            template <typename ExPolicy, typename RandomIt, typename Sent,
                typename Pred, typename Proj>
            static util::detail::algorithm_result_t<ExPolicy, RandomIt>
            parallel(ExPolicy&& policy, RandomIt first, RandomIt nth, Sent last,
                Pred&& pred, Proj&& proj)
            {
                using value_type =
                    typename std::iterator_traits<RandomIt>::value_type;

                RandomIt partitionIter, return_last;

                if (first == last)
                {
                    return util::detail::algorithm_result<ExPolicy,
                        RandomIt>::get(HPX_MOVE(first));
                }

                if (nth == last)
                {
                    return util::detail::algorithm_result<ExPolicy,
                        RandomIt>::get(HPX_MOVE(nth));
                }

                try
                {
                    RandomIt last_iter =
                        detail::advance_to_sentinel(first, last);
                    return_last = last_iter;

                    while (first != last_iter)
                    {
                        detail::pivot9(first, last_iter, pred);

                        partitionIter =
                            hpx::parallel::v1::detail::partition<RandomIt>()
                                .call(
                                    policy(hpx::execution::non_task), first + 1,
                                    last_iter,
                                    [val = HPX_INVOKE(proj, *first), &pred](
                                        value_type const& elem) {
                                        return HPX_INVOKE(pred, elem, val);
                                    },
                                    proj);

                        --partitionIter;
                        // swap first element and partitionIter
                        // (ending element of first group)
#if defined(HPX_HAVE_CXX20_STD_RANGES_ITER_SWAP)
                        std::ranges::iter_swap(first, partitionIter);
#else
                        std::iter_swap(first, partitionIter);
#endif

                        // if nth element < partitioned index,
                        // it lies in [first, partitionIter)
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
                    return util::detail::algorithm_result<ExPolicy,
                        RandomIt>::get(detail::handle_exception<ExPolicy,
                        RandomIt>::call(std::current_exception()));
                }

                return util::detail::algorithm_result<ExPolicy, RandomIt>::get(
                    HPX_MOVE(return_last));
            }
        };
        /// \endcond
    }    // namespace detail
}}}      // namespace hpx::parallel::v1

namespace hpx {
    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::nth_element
    inline constexpr struct nth_element_t final
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
        friend void tag_fallback_invoke(hpx::nth_element_t, RandomIt first,
            RandomIt nth, RandomIt last, Pred&& pred = Pred())
        {
            static_assert(hpx::traits::is_random_access_iterator_v<RandomIt>,
                "Requires at least random iterator.");

            hpx::parallel::v1::detail::nth_element<RandomIt>().call(
                hpx::execution::seq, first, nth, last, HPX_FORWARD(Pred, pred),
                hpx::parallel::util::projection_identity{});
        }

        // clang-format off
        template <typename ExPolicy, typename RandomIt,
            typename Pred = hpx::parallel::v1::detail::less,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<RandomIt> &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits<RandomIt>::value_type,
                    typename std::iterator_traits<RandomIt>::value_type
                >
            )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy>
        tag_fallback_invoke(hpx::nth_element_t, ExPolicy&& policy,
            RandomIt first, RandomIt nth, RandomIt last, Pred&& pred = Pred())
        {
            static_assert(hpx::traits::is_random_access_iterator_v<RandomIt>,
                "Requires at least random iterator.");

            using result_type =
                hpx::parallel::util::detail::algorithm_result_t<ExPolicy>;

            return hpx::util::void_guard<result_type>(),
                   hpx::parallel::v1::detail::nth_element<RandomIt>().call(
                       HPX_FORWARD(ExPolicy, policy), first, nth, last,
                       HPX_FORWARD(Pred, pred),
                       hpx::parallel::util::projection_identity{});
        }
    } nth_element{};
}    // namespace hpx

#endif
