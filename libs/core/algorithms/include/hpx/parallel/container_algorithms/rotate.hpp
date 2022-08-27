//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2021 Chuanqiu He
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/rotate.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx { namespace ranges {

    ///////////////////////////////////////////////////////////////////////////
    /// Performs a left rotation on a range of elements. Specifically,
    /// \a rotate swaps the elements in the range [first, last) in such a way
    /// that the element middle becomes the first element of the new range
    /// and middle - 1 becomes the last element.
    ///
    /// \note   Complexity: Linear in the distance between \a first and \a last.
    ///
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the end iterators used (deduced).
    ///                     This sentinel type must be a sentinel for FwdIter.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param middle       Refers to the element that should appear at the
    ///                     beginning of the rotated range.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    ///
    /// The assignments in the parallel \a rotate algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \note The type of dereferenced \a FwdIter must meet the requirements
    ///       of \a MoveAssignable and \a MoveConstructible.
    ///
    /// \returns  The \a rotate algorithm returns a \a
    ///           subrange_t<FwdIter, Sent>.
    ///           The \a rotate algorithm returns the iterator equal to
    ///           pair(first + (last - middle), last).
    ///
    template <typename FwdIter, typename Sent>
    subrange_t<FwdIter, Sent> rotate(FwdIter first, FwdIter middle, Sent last);

    ///////////////////////////////////////////////////////////////////////////
    /// Performs a left rotation on a range of elements. Specifically,
    /// \a rotate swaps the elements in the range [first, last) in such a way
    /// that the element middle becomes the first element of the new range
    /// and middle - 1 becomes the last element.
    ///
    /// \note   Complexity: Linear in the distance between \a first and \a last.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the end iterators used (deduced).
    ///                     This sentinel type must be a sentinel for FwdIter.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param middle       Refers to the element that should appear at the
    ///                     beginning of the rotated range.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    ///
    /// The assignments in the parallel \a rotate algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a rotate algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note The type of dereferenced \a FwdIter must meet the requirements
    ///       of \a MoveAssignable and \a MoveConstructible.
    ///
    /// \returns  The \a rotate algorithm returns a
    ///           \a hpx::future<subrange_t<FwdIter, Sent>>
    ///           if the execution policy is of type
    ///           \a parallel_task_policy and
    ///           returns a \a subrange_t<FwdIter, Sent>
    ///           otherwise.
    ///           The \a rotate algorithm returns the iterator equal to
    ///           pair(first + (last - middle), last).
    ///
    template <typename ExPolicy, typename FwdIter, typename Sent>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        subrange_t<FwdIter, Sent>>::type
    rotate(ExPolicy&& policy, FwdIter first, FwdIter middle, Sent last);

    ///////////////////////////////////////////////////////////////////////////
    /// Uses \a rng as the source range, as if using \a util::begin(rng) as
    /// \a first and \a ranges::end(rng) as \a last.
    /// Performs a left rotation on a range of elements. Specifically,
    /// \a rotate swaps the elements in the range [first, last) in such a way
    /// that the element middle becomes the first element of the new range
    /// and middle - 1 becomes the last element.
    ///
    /// \note   Complexity: Linear in the distance between \a first and \a last.
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param middle       Refers to the element that should appear at the
    ///                     beginning of the rotated range.
    ///
    /// The assignments in the parallel \a rotate algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \note The type of dereferenced \a FwdIter must meet the requirements
    ///       of \a MoveAssignable and \a MoveConstructible.
    ///
    /// \returns  The \a rotate algorithm returns a
    ///           \a subrange_t<hpx::traits::range_iterator_t<Rng>,
    ///           hpx::traits::range_iterator_t<Rng>>.
    ///           The \a rotate algorithm returns the iterator equal to
    ///           pair(first + (last - middle), last).
    ///
    template <typename Rng>
    subrange_t<hpx::traits::range_iterator_t<Rng>,
        hpx::traits::range_iterator_t<Rng>>
    rotate(Rng&& rng, hpx::traits::range_iterator_t<Rng> middle);

    ///////////////////////////////////////////////////////////////////////////
    /// Uses \a rng as the source range, as if using \a util::begin(rng) as
    /// \a first and \a ranges::end(rng) as \a last.
    /// Performs a left rotation on a range of elements. Specifically,
    /// \a rotate swaps the elements in the range [first, last) in such a way
    /// that the element middle becomes the first element of the new range
    /// and middle - 1 becomes the last element.
    ///
    /// \note   Complexity: Linear in the distance between \a first and \a last.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param middle       Refers to the element that should appear at the
    ///                     beginning of the rotated range.
    ///
    /// The assignments in the parallel \a rotate algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a rotate algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note The type of dereferenced \a FwdIter must meet the requirements
    ///       of \a MoveAssignable and \a MoveConstructible.
    ///
    /// \returns  The \a rotate algorithm returns a \a hpx::future
    ///           <subrange_t<hpx::traits::range_iterator_t<Rng>,
    ///           hpx::traits::range_iterator_t<Rng>>>
    ///           if the execution policy is of type \a sequenced_task_policy
    ///           or \a parallel_task_policy and returns
    ///           \a subrange_t<hpx::traits::range_iterator_t<Rng>,
    ///           hpx::traits::range_iterator_t<Rng>>.
    ///           otherwise.
    ///           The \a rotate algorithm returns the iterator equal to
    ///           pair(first + (last - middle), last).
    ///
    template <typename ExPolicy, typename Rng>
    typename util::detail::algorithm_result<ExPolicy,
        subrange_t<hpx::traits::range_iterator_t<Rng>,
            hpx::traits::range_iterator_t<Rng>>>::type
    rotate(ExPolicy&& policy, Rng&& rng,
        hpx::traits::range_iterator_t<Rng> middle);

    ///////////////////////////////////////////////////////////////////////////
    /// Copies the elements from the range [first, last), to another range
    /// beginning at \a dest_first in such a way, that the element
    /// \a middle becomes the first element of the new range and
    /// \a middle - 1 becomes the last element.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the end iterators used (deduced).
    ///                     This sentinel type must be a sentinel for FwdIter.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param middle       Refers to the element that should appear at the
    ///                     beginning of the rotated range.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    ///
    /// The assignments in the parallel \a rotate_copy algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a rotate_copy algorithm returns a \a
    ///           rotate_copy_result<FwdIter, OutIter>.
    ///           The \a rotate_copy algorithm returns the output iterator to
    ///           the element past the last element copied.
    ///
    template <typename FwdIter, typename Sent, typename OutIter>
    rotate_copy_result<FwdIter, OutIter> rotate_copy(
        FwdIter first, FwdIter middle, Sent last, OutIter dest_first);

    ///////////////////////////////////////////////////////////////////////////
    /// Copies the elements from the range [first, last), to another range
    /// beginning at \a dest_first in such a way, that the element
    /// \a middle becomes the first element of the new range and
    /// \a middle - 1 becomes the last element.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the end iterators used (deduced).
    ///                     This sentinel type must be a sentinel for FwdIter.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param middle       Refers to the element that should appear at the
    ///                     beginning of the rotated range.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    ///
    /// The assignments in the parallel \a rotate_copy algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a rotate_copy algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a rotate_copy algorithm returns areturns hpx::future<
    ///           rotate_copy_result<FwdIter1, FwdIter2>> if the
    ///           execution policy is of type \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a
    ///           rotate_copy_result<FwdIter1, FwdIter2> otherwise.
    ///           The \a rotate_copy algorithm returns the output iterator to
    ///           the element past the last element copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename Sent,
        typename FwdIter2>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        rotate_copy_result<FwdIter1, FwdIter2>>::type
    rotate_copy(ExPolicy&& policy, FwdIter1 first, FwdIter1 middle, Sent last,
        FwdIter2 dest_first);

    ///////////////////////////////////////////////////////////////////////////
    /// Uses \a rng as the source range, as if using \a util::begin(rng) as
    /// \a first and \a ranges::end(rng) as \a last.
    /// Copies the elements from the range [first, last), to another range
    /// beginning at \a dest_first in such a way, that the element
    /// \a middle becomes the first element of the new range and
    /// \a middle - 1 becomes the last element.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param middle       Refers to the element that should appear at the
    ///                     beginning of the rotated range.
    /// \param dest_first   Refers to the begin of the destination range.
    ///
    /// The assignments in the parallel \a rotate_copy algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a rotate algorithm returns a \a
    ///           rotate_copy_result<hpx::traits::range_iterator_t<Rng>,
    ///           OutIter>.
    ///           The \a rotate_copy algorithm returns the output iterator to
    ///           the element past the last element copied.
    ///
    template <typename Rng, typename OutIter>
    typename rotate_copy_result<hpx::traits::range_iterator_t<Rng>, OutIter>
    rotate_copy(Rng&& rng, hpx::traits::range_iterator_t<Rng> middle,
        OutIter dest_first);

    ///////////////////////////////////////////////////////////////////////////
    /// Uses \a rng as the source range, as if using \a util::begin(rng) as
    /// \a first and \a ranges::end(rng) as \a last.
    /// Copies the elements from the range [first, last), to another range
    /// beginning at \a dest_first in such a way, that the element
    /// \a new_first becomes the first element of the new range and
    /// \a new_first - 1 becomes the last element.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param middle       Refers to the element that should appear at the
    ///                     beginning of the rotated range.
    /// \param dest_first   Refers to the begin of the destination range.
    ///
    /// The assignments in the parallel \a rotate_copy algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a rotate_copy algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a rotate_copy algorithm returns a
    ///           \a hpx::future<otate_copy_result<
    ///           hpx::traits::range_iterator_t<Rng>, OutIter>>
    ///           if the execution policy is of type
    ///           \a parallel_task_policy and
    ///           returns \a rotate_copy_result<
    ///           hpx::traits::range_iterator_t<Rng>, OutIter>
    ///           otherwise.
    ///           The \a rotate_copy algorithm returns the output iterator to
    ///           the element past the last element copied.
    ///
    template <typename ExPolicy, typename Rng, typename OutIter>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        rotate_copy_result<hpx::traits::range_iterator_t<Rng>, OutIter>>::type
    rotate_copy(ExPolicy&& policy, Rng&& rng,
        hpx::traits::range_iterator_t<Rng> middle, OutIter dest_first);

}}    // namespace hpx::ranges

#else

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>

#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/iterator_support/iterator_range.hpp>
#include <hpx/parallel/algorithms/rotate.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {

    // clang-format off
    template <typename ExPolicy, typename Rng,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_range<Rng>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::rotate is deprecated, use hpx::ranges::rotate instead")
        typename util::detail::algorithm_result<ExPolicy,
            util::in_out_result<hpx::traits::range_iterator_t<Rng>,
                hpx::traits::range_iterator_t<Rng>>>::type
        rotate(ExPolicy&& policy, Rng&& rng,
            hpx::traits::range_iterator_t<Rng> middle)
    {
        return rotate(HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
            middle, hpx::util::end(rng));
    }

    ///////////////////////////////////////////////////////////////////////////
    // clang-format off
    template <typename ExPolicy, typename Rng, typename OutIter,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_range<Rng>::value &&
            hpx::traits::is_iterator_v<OutIter>
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::rotate_copy is deprecated, use "
        "hpx::ranges::rotate_copy instead")
        typename util::detail::algorithm_result<ExPolicy,
            util::in_out_result<hpx::traits::range_iterator_t<Rng>,
                OutIter>>::type rotate_copy(ExPolicy&& policy, Rng&& rng,
            hpx::traits::range_iterator_t<Rng> middle, OutIter dest_first)
    {
        return rotate_copy(HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
            middle, hpx::util::end(rng), dest_first);
    }
}}}    // namespace hpx::parallel::v1

namespace hpx { namespace ranges {
    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::ranges::rotate
    inline constexpr struct rotate_t final
      : hpx::detail::tag_parallel_algorithm<rotate_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Sent,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<FwdIter> &&
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value
            )>
        // clang-format on
        friend subrange_t<FwdIter, Sent> tag_fallback_invoke(
            hpx::ranges::rotate_t, FwdIter first, FwdIter middle, Sent last)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return hpx::parallel::util::get_subrange<FwdIter, Sent>(
                hpx::parallel::v1::detail::rotate<
                    parallel::util::in_out_result<FwdIter, Sent>>()
                    .call(hpx::execution::seq, first, middle, last));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<FwdIter> &&
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            subrange_t<FwdIter, Sent>>::type
        tag_fallback_invoke(hpx::ranges::rotate_t, ExPolicy&& policy,
            FwdIter first, FwdIter middle, Sent last)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            using is_seq = std::integral_constant<bool,
                hpx::is_sequenced_execution_policy_v<ExPolicy> ||
                    !hpx::traits::is_bidirectional_iterator_v<FwdIter>>;

            return hpx::parallel::util::get_subrange<FwdIter, Sent>(
                hpx::parallel::v1::detail::rotate<
                    parallel::util::in_out_result<FwdIter, Sent>>()
                    .call2(HPX_FORWARD(ExPolicy, policy), is_seq(), first,
                        middle, last));
        }

        // clang-format off
        template <typename Rng,
            HPX_CONCEPT_REQUIRES_(hpx::traits::is_range<Rng>::value)>
        // clang-format on
        friend subrange_t<hpx::traits::range_iterator_t<Rng>,
            hpx::traits::range_iterator_t<Rng>>
        tag_fallback_invoke(hpx::ranges::rotate_t, Rng&& rng,
            hpx::traits::range_iterator_t<Rng> middle)
        {
            return hpx::parallel::util::get_subrange<
                hpx::traits::range_iterator_t<Rng>,
                typename hpx::traits::range_sentinel<Rng>::type>(
                hpx::parallel::v1::detail::rotate<parallel::util::in_out_result<
                    hpx::traits::range_iterator_t<Rng>,
                    typename hpx::traits::range_sentinel<Rng>::type>>()
                    .call(hpx::execution::seq, hpx::util::begin(rng), middle,
                        hpx::util::end(rng)));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            subrange_t<hpx::traits::range_iterator_t<Rng>,
                hpx::traits::range_iterator_t<Rng>>>::type
        tag_fallback_invoke(hpx::ranges::rotate_t, ExPolicy&& policy, Rng&& rng,
            hpx::traits::range_iterator_t<Rng> middle)
        {
            using is_seq = std::integral_constant<bool,
                hpx::is_sequenced_execution_policy_v<ExPolicy> ||
                    !hpx::traits::is_bidirectional_iterator_v<
                        hpx::traits::range_iterator_t<Rng>>>;

            return hpx::parallel::util::get_subrange<
                hpx::traits::range_iterator_t<Rng>,
                typename hpx::traits::range_sentinel<Rng>::type>(
                hpx::parallel::v1::detail::rotate<parallel::util::in_out_result<
                    hpx::traits::range_iterator_t<Rng>,
                    typename hpx::traits::range_sentinel<Rng>::type>>()
                    .call2(HPX_FORWARD(ExPolicy, policy), is_seq(),
                        hpx::util::begin(rng), middle, hpx::util::end(rng)));
        }
    } rotate{};

    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::ranges::rotate_copy
    template <typename I, typename O>
    using rotate_copy_result = hpx::parallel::util::in_out_result<I, O>;

    inline constexpr struct rotate_copy_t final
      : hpx::detail::tag_parallel_algorithm<rotate_copy_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Sent, typename OutIter,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<FwdIter> &&
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value &&
                hpx::traits::is_iterator_v<OutIter>
            )>
        // clang-format on
        friend rotate_copy_result<FwdIter, OutIter> tag_fallback_invoke(
            hpx::ranges::rotate_copy_t, FwdIter first, FwdIter middle,
            Sent last, OutIter dest_first)
        {
            static_assert((hpx::traits::is_forward_iterator_v<FwdIter>),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_output_iterator_v<OutIter>),
                "Requires at least output iterator.");

            return hpx::parallel::v1::detail::rotate_copy<
                rotate_copy_result<FwdIter, OutIter>>()
                .call(hpx::execution::seq, first, middle, last, dest_first);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename Sent,
            typename FwdIter2, HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<FwdIter1> &&
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_sentinel_for<Sent, FwdIter1>::value &&
                hpx::traits::is_iterator_v<FwdIter2>
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            rotate_copy_result<FwdIter1, FwdIter2>>::type
        tag_fallback_invoke(hpx::ranges::rotate_copy_t, ExPolicy&& policy,
            FwdIter1 first, FwdIter1 middle, Sent last, FwdIter2 dest_first)
        {
            static_assert((hpx::traits::is_forward_iterator_v<FwdIter1>),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator_v<FwdIter2>),
                "Requires at least forward iterator.");

            using is_seq = std::integral_constant<bool,
                hpx::is_sequenced_execution_policy_v<ExPolicy> ||
                    !hpx::traits::is_bidirectional_iterator_v<FwdIter1>>;

            return hpx::parallel::v1::detail::rotate_copy<
                rotate_copy_result<FwdIter1, FwdIter2>>()
                .call2(HPX_FORWARD(ExPolicy, policy), is_seq(), first, middle,
                    last, dest_first);
        }

        // clang-format off
        template <typename Rng, typename OutIter,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::traits::is_iterator_v<OutIter>
            )>
        // clang-format on
        friend rotate_copy_result<hpx::traits::range_iterator_t<Rng>, OutIter>
        tag_fallback_invoke(hpx::ranges::rotate_copy_t, Rng&& rng,
            hpx::traits::range_iterator_t<Rng> middle, OutIter dest_first)
        {
            return hpx::parallel::v1::detail::rotate_copy<rotate_copy_result<
                hpx::traits::range_iterator_t<Rng>, OutIter>>()
                .call(hpx::execution::seq, hpx::util::begin(rng), middle,
                    hpx::util::end(rng), dest_first);
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename OutIter,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_range<Rng>::value &&
                hpx::traits::is_iterator_v<OutIter>
                )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            rotate_copy_result<hpx::traits::range_iterator_t<Rng>,
                OutIter>>::type
        tag_fallback_invoke(hpx::ranges::rotate_copy_t, ExPolicy&& policy,
            Rng&& rng, hpx::traits::range_iterator_t<Rng> middle,
            OutIter dest_first)
        {
            using is_seq = std::integral_constant<bool,
                hpx::is_sequenced_execution_policy_v<ExPolicy> ||
                    !hpx::traits::is_bidirectional_iterator_v<
                        hpx::traits::range_iterator_t<Rng>>>;

            return hpx::parallel::v1::detail::rotate_copy<rotate_copy_result<
                hpx::traits::range_iterator_t<Rng>, OutIter>>()
                .call2(HPX_FORWARD(ExPolicy, policy), is_seq(),
                    hpx::util::begin(rng), middle, hpx::util::end(rng),
                    dest_first);
        }
    } rotate_copy{};

}}    // namespace hpx::ranges

#endif    //DOXYGEN
