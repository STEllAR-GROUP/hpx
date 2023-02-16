//  Copyright (c) 2007-2023 Hartmut Kaiser
//  Copyright (c)      2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/reverse.hpp

#pragma once

#if defined(DOXYGEN)

namespace hpx {

    /// Reverses the order of the elements in the range [first, last).
    /// Behaves as if applying \a std::iter_swap to every pair of iterators
    /// first+i, (\a last-i) - 1 for each non-negative i < (last-first)/2.
    ///
    /// \note   Complexity: Linear in the distance between \a first and \a last.
    ///
    /// \tparam BidirIter   The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     bidirectional iterator.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    ///
    /// The assignments in the parallel \a reverse algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a reverse algorithm returns \a void.
    ///
    template <typename BidirIter>
    void reverse(BidirIter first, BidirIter last);

    /// Reverses the order of the elements in the range [first, last).
    /// Behaves as if applying \a std::iter_swap to every pair of iterators
    /// first+i, (last-i) - 1 for each non-negative i < (last-first)/2.
    /// Executed according to the policy.
    ///
    /// \note   Complexity: Linear in the distance between \a first and \a last.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam BidirIter   The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     bidirectional iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    ///
    /// The assignments in the parallel \a reverse algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a reverse algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a reverse algorithm returns a \a hpx::future<void>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a void otherwise.
    ///
    template <typename ExPolicy, typename BidirIter>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, void> reverse(
        ExPolicy&& policy, BidirIter first, BidirIter last);

    ///////////////////////////////////////////////////////////////////////////
    /// Copies the elements from the range [first, last) to another range
    /// beginning at dest in such a way that the elements in the new
    /// range are in reverse order.
    /// Behaves as if by executing the assignment
    /// *(dest + (last - first) - 1 - i) = *(first + i) once for each
    /// non-negative i < (last - first)
    /// If the source and destination ranges (that is, [first, last) and
    /// [dest, dest+(last-first)) respectively) overlap, the
    /// behavior is undefined.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam BidirIter   The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     bidirectional iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the begin of the destination range.
    ///
    /// The assignments in the parallel \a reverse_copy algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a reverse_copy algorithm returns an
    ///           \a OutIter.
    ///           The \a reverse_copy algorithm returns the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename BidirIter, typename OutIter>
    OutIter reverse_copy(BidirIter first, BidirIter last, OutIter dest);

    /// Copies the elements from the range [first, last) to another range
    /// beginning at dest in such a way that the elements in the new
    /// range are in reverse order.
    /// Behaves as if by executing the assignment
    /// *(dest + (last - first) - 1 - i) = *(first + i) once for each
    /// non-negative i < (last - first)
    /// If the source and destination ranges (that is, [first, last) and
    /// [dest, dest+(last-first)) respectively) overlap, the
    /// behavior is undefined. Executed according to the policy.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam BidirIter   The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     bidirectional iterator.
    /// \tparam FwdIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param dest         Refers to the begin of the destination range.
    ///
    /// The assignments in the parallel \a reverse_copy algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a reverse_copy algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a reverse_copy algorithm returns a
    ///           \a hpx::future<FwdIter>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter
    ///           otherwise.
    ///           The \a reverse_copy algorithm returns the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename BidirIter, typename FwdIter>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, FwdIter>
    reverse_copy(
        ExPolicy&& policy, BidirIter first, BidirIter last, FwdIter dest);

}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/parallel/algorithms/detail/advance_and_get_distance.hpp>
#include <hpx/parallel/algorithms/detail/advance_to_sentinel.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/ranges_facilities.hpp>
#include <hpx/parallel/util/result_types.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>
#include <hpx/type_support/identity.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx::parallel {

    ///////////////////////////////////////////////////////////////////////////
    // reverse
    namespace detail {

        /// \cond NOINTERNAL
        template <typename Iter>
        struct reverse : public algorithm<reverse<Iter>, Iter>
        {
            constexpr reverse() noexcept
              : algorithm<reverse, Iter>("reverse")
            {
            }

            template <typename ExPolicy, typename BidirIter, typename Sent>
            constexpr static BidirIter sequential(
                ExPolicy, BidirIter first, Sent last)
            {
                auto last2 = detail::advance_to_sentinel(first, last);
                for (auto tail = last2; !(first == tail || first == --tail);
                     ++first)
                {
#if defined(HPX_HAVE_CXX20_STD_RANGES_ITER_SWAP)
                    std::ranges::iter_swap(first, tail);
#else
                    std::iter_swap(first, tail);
#endif
                }
                return last2;
            }

            template <typename ExPolicy, typename BidirIter, typename Sent>
            static decltype(auto) parallel(
                ExPolicy&& policy, BidirIter first, Sent last)
            {
                using destination_iterator = std::reverse_iterator<BidirIter>;
                using zip_iterator =
                    hpx::util::zip_iterator<BidirIter, destination_iterator>;
                using reference = typename zip_iterator::reference;

                auto last2 = first;
                auto size = detail::advance_and_get_distance(last2, last);

                return util::detail::convert_to_result(
                    for_each_n<zip_iterator>().call(
                        HPX_FORWARD(ExPolicy, policy),
                        hpx::util::zip_iterator(
                            first, destination_iterator(last2)),
                        size / 2,
                        [](reference t) -> void {
                            using hpx::get;
                            std::swap(get<0>(t), get<1>(t));
                        },
                        hpx::identity_v),
                    [last2](auto) -> BidirIter { return last2; });
            }
        };
        /// \endcond
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // reverse_copy
    namespace detail {
        /// \cond NOINTERNAL

        // sequential reverse_copy
        template <typename BidirIt, typename Sent, typename OutIter>
        constexpr util::in_out_result<BidirIt, OutIter> sequential_reverse_copy(
            BidirIt first, Sent last, OutIter dest)
        {
            auto iter{hpx::ranges::next(first, last)};
            while (first != iter)
            {
                *dest++ = *--iter;
            }
            return util::in_out_result<BidirIt, OutIter>{iter, dest};
        }

        template <typename IterPair>
        struct reverse_copy : public algorithm<reverse_copy<IterPair>, IterPair>
        {
            constexpr reverse_copy() noexcept
              : algorithm<reverse_copy, IterPair>("reverse_copy")
            {
            }

            template <typename ExPolicy, typename BidirIter, typename Sent,
                typename OutIter>
            constexpr static util::in_out_result<BidirIter, OutIter> sequential(
                ExPolicy, BidirIter first, Sent last, OutIter dest_first)
            {
                return sequential_reverse_copy(first, last, dest_first);
            }

            template <typename ExPolicy, typename BidirIter, typename Sent,
                typename FwdIter>
            static util::detail::algorithm_result_t<ExPolicy,
                util::in_out_result<BidirIter, FwdIter>>
            parallel(ExPolicy&& policy, BidirIter first, Sent last,
                FwdIter dest_first)
            {
                auto last2{hpx::ranges::next(first, last)};
                typedef std::reverse_iterator<BidirIter> iterator;

                return util::detail::convert_to_result(
                    detail::copy<util::in_out_result<iterator, FwdIter>>().call(
                        HPX_FORWARD(ExPolicy, policy), iterator(last2),
                        iterator(first), dest_first),
                    [](util::in_out_result<iterator, FwdIter> const& p)
                        -> util::in_out_result<BidirIter, FwdIter> {
                        return util::in_out_result<BidirIter, FwdIter>{
                            p.in.base(), p.out};
                    });
            }
        };
        /// \endcond
    }    // namespace detail
}    // namespace hpx::parallel

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::reverse
    inline constexpr struct reverse_t final
      : hpx::detail::tag_parallel_algorithm<reverse_t>
    {
    private:
        // clang-format off
        template <typename BidirIter,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<BidirIter>
            )>
        // clang-format on
        friend void tag_fallback_invoke(
            hpx::reverse_t, BidirIter first, BidirIter last)
        {
            static_assert(hpx::traits::is_bidirectional_iterator_v<BidirIter>,
                "Requires at least bidirectional iterator.");

            hpx::parallel::detail::reverse<BidirIter>().call(
                hpx::execution::sequenced_policy{}, first, last);
        }

        // clang-format off
        template <typename ExPolicy, typename BidirIter,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<BidirIter> &&
                hpx::is_execution_policy_v<ExPolicy>
            )>
        // clang-format on
        friend decltype(auto) tag_fallback_invoke(
            hpx::reverse_t, ExPolicy&& policy, BidirIter first, BidirIter last)
        {
            static_assert(hpx::traits::is_bidirectional_iterator_v<BidirIter>,
                "Requires at least bidirectional iterator.");

            return parallel::util::detail::algorithm_result<ExPolicy>::get(
                hpx::parallel::detail::reverse<BidirIter>().call(
                    HPX_FORWARD(ExPolicy, policy), first, last));
        }
    } reverse{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::reverse_copy
    inline constexpr struct reverse_copy_t final
      : hpx::detail::tag_parallel_algorithm<reverse_copy_t>
    {
    private:
        // clang-format off
        template <typename BidirIter, typename OutIter,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<BidirIter> &&
                hpx::traits::is_iterator_v<OutIter>
            )>
        // clang-format on
        friend OutIter tag_fallback_invoke(
            hpx::reverse_copy_t, BidirIter first, BidirIter last, OutIter dest)
        {
            static_assert(hpx::traits::is_bidirectional_iterator_v<BidirIter>,
                "Requires at least bidirectional iterator.");

            static_assert(hpx::traits::is_output_iterator_v<OutIter>,
                "Requires at least output iterator.");

            return parallel::util::get_second_element(
                parallel::detail::reverse_copy<
                    hpx::parallel::util::in_out_result<BidirIter, OutIter>>()
                    .call(
                        hpx::execution::sequenced_policy{}, first, last, dest));
        }

        // clang-format off
        template <typename ExPolicy, typename BidirIter, typename FwdIter,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<BidirIter> &&
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<FwdIter>
            )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy, FwdIter>
        tag_fallback_invoke(hpx::reverse_copy_t, ExPolicy&& policy,
            BidirIter first, BidirIter last, FwdIter dest)
        {
            static_assert(hpx::traits::is_bidirectional_iterator_v<BidirIter>,
                "Requires at least bidirectional iterator.");

            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return parallel::util::get_second_element(
                parallel::detail::reverse_copy<
                    hpx::parallel::util::in_out_result<BidirIter, FwdIter>>()
                    .call(HPX_FORWARD(ExPolicy, policy), first, last, dest));
        }
    } reverse_copy{};
}    // namespace hpx

#endif    // DOXYGEN
