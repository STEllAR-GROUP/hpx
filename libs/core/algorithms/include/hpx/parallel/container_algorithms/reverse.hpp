//  Copyright (c) 2007-2023 Hartmut Kaiser
//  Copyright (c)      2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/reverse.hpp
/// \page hpx::ranges::reverse, hpx::ranges::reverse_copy
/// \headerfile hpx/algorithm.hpp

#pragma once

#if defined(DOXYGEN)

namespace hpx { namespace ranges {

    ///////////////////////////////////////////////////////////////////////////
    /// Reverses the order of the elements in the range [first, last).
    /// Behaves as if applying std::iter_swap to every pair of iterators
    /// first+i, (last-i) - 1 for each non-negative i < (last-first)/2.
    ///
    /// \note   Complexity: Linear in the distance between \a first and \a last.
    ///
    /// \tparam Iter        The type of the source iterator used (deduced).
    ///                     The iterator type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Sent        The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for Iter.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param sent         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    ///
    /// The assignments in the parallel \a reverse algorithm
    /// execute in sequential order in the calling thread.
    ///
    ///
    /// \returns  The \a reverse algorithm returns a \a Iter.
    ///           It returns \a last.
    ///
    template <typename Iter, typename Sent>
    Iter reverse(Iter first, Sent sent);

    /// Uses \a rng as the source range, as if using \a util::begin(rng) as
    /// \a first and \a ranges::end(rng) as \a last.
    /// Reverses the order of the elements in the range [first, last).
    /// Behaves as if applying std::iter_swap to every pair of iterators
    /// first+i, (last-i) - 1 for each non-negative i < (last-first)/2.
    ///
    /// \note   Complexity: Linear in the distance between \a first and \a last.
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a bidirectional iterator.
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    ///
    /// The assignments in the parallel \a reverse algorithm
    /// execute in sequential order in the calling thread.
    ///
    ///
    /// \returns  The \a reverse algorithm returns a
    ///           \a hpx::traits::range_iterator<Rng>::type.
    ///           It returns \a last.
    ///
    template <typename Rng>
    hpx::traits::range_iterator_t<Rng> reverse(Rng&& rng);

    /// Reverses the order of the elements in the range [first, last).
    /// Behaves as if applying std::iter_swap to every pair of iterators
    /// first+i, (last-i) - 1 for each non-negative i < (last-first)/2.
    ///
    /// \note   Complexity: Linear in the distance between \a first and \a last.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Iter        The type of the source iterator used (deduced).
    ///                     The iterator type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Sent        The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for Iter.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param sent         Refers to the end of the sequence of elements the
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
    /// \returns  The \a reverse algorithm returns a \a hpx::future<Iter>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a Iter otherwise.
    ///           It returns \a last.
    ///
    template <typename ExPolicy, typename Iter, typename Sent>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, Iter> reverse(
        ExPolicy&& policy, Iter first, Sent sent);

    /// Uses \a rng as the source range, as if using \a util::begin(rng) as
    /// \a first and \a ranges::end(rng) as \a last.
    /// Reverses the order of the elements in the range [first, last).
    /// Behaves as if applying std::iter_swap to every pair of iterators
    /// first+i, (last-i) - 1 for each non-negative i < (last-first)/2.
    ///
    /// \note   Complexity: Linear in the distance between \a first and \a last.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a bidirectional iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
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
    /// \returns  The \a reverse algorithm returns a
    ///           \a hpx::future<hpx::traits::range_iterator_t<Rng>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a hpx::future<
    ///             hpx::traits::range_iterator_t<Rng>> otherwise.
    ///           It returns \a last.
    ///
    template <typename ExPolicy, typename Rng>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        hpx::traits::range_iterator_t<Rng>>
    reverse(ExPolicy&& policy, Rng&& rng);

    ///////////////////////////////////////////////////////////////////////////
    /// Copies the elements from the range [first, last) to another range
    /// beginning at result in such a way that the elements in the new
    /// range are in reverse order.
    /// Behaves as if by executing the assignment
    /// *(result + (last - first) - 1 - i) = *(first + i) once for each
    /// non-negative i < (last - first)
    /// If the source and destination ranges (that is, [first, last) and
    /// [result, result+(last-first)) respectively) overlap, the
    /// behavior is undefined.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam Iter        The type of the source iterator used (deduced).
    ///                     The iterator type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Sent        The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for Iter.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param result   Refers to the begin of the destination range.
    ///
    /// The assignments in the parallel \a reverse_copy algorithm
    /// execute in sequential order in the calling thread.
    ///
    ///
    /// \returns  The \a reverse_copy algorithm returns a
    ///           \a reverse_copy_result<Iter, OutIter>.
    ///           The \a reverse_copy algorithm returns the pair of the input iterator
    ///           forwarded to the first element after the last in the input
    ///           sequence and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename Iter, typename Sent, typename OutIter>
    reverse_copy_result<Iter, OutIter> reverse_copy(
        Iter first, Sent last, OutIter result);

    /// Uses \a rng as the source range, as if using \a util::begin(rng) as
    /// \a first and \a ranges::end(rng) as \a last.
    /// Copies the elements from the range [first, last) to another range
    /// beginning at result in such a way that the elements in the new
    /// range are in reverse order.
    /// Behaves as if by executing the assignment
    /// *(result + (last - first) - 1 - i) = *(first + i) once for each
    /// non-negative i < (last - first)
    /// If the source and destination ranges (that is, [first, last) and
    /// [result, result+(last-first)) respectively) overlap, the
    /// behavior is undefined.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a bidirectional iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param result       Refers to the begin of the destination range.
    ///
    /// The assignments in the parallel \a reverse_copy algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a reverse_copy algorithm returns a
    ///           \a ranges::reverse_copy_result<
    ///           hpx::traits::range_iterator_t<Rng>, OutIter>.
    ///           The \a reverse_copy algorithm returns
    ///           an object equal to {last, result + N} where N = last - first
    ///
    template <typename Rng, typename OutIter>
    reverse_copy_result<hpx::traits::range_iterator_t<Rng>, OutIter>
    reverse_copy(Rng&& rng, OutIter result);

    /// Copies the elements from the range [first, last) to another range
    /// beginning at result in such a way that the elements in the new
    /// range are in reverse order.
    /// Behaves as if by executing the assignment
    /// *(result + (last - first) - 1 - i) = *(first + i) once for each
    /// non-negative i < (last - first)
    /// If the source and destination ranges (that is, [first, last) and
    /// [result, result+(last-first)) respectively) overlap, the
    /// behavior is undefined.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Iter        The type of the source iterator used (deduced).
    ///                     The iterator type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Sent        The type of the end iterators used (deduced). This
    ///                     sentinel type must be a sentinel for Iter.
    /// \tparam FwdIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param result   Refers to the begin of the destination range.
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
    ///           \a hpx::future<reverse_copy_result<Iter, FwdIter> >
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a reverse_copy_result<Iter, FwdIter>
    ///           otherwise.
    ///           The \a reverse_copy algorithm returns the pair of the input iterator
    ///           forwarded to the first element after the last in the input
    ///           sequence and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename Iter, typename Sent, typename FwdIter>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        reverse_copy_result<Iter, FwdIter>>::type
    reverse_copy(ExPolicy&& policy, Iter first, Sent last, FwdIter result);

    /// Uses \a rng as the source range, as if using \a util::begin(rng) as
    /// \a first and \a ranges::end(rng) as \a last.
    /// Copies the elements from the range [first, last) to another range
    /// beginning at result in such a way that the elements in the new
    /// range are in reverse order.
    /// Behaves as if by executing the assignment
    /// *(result + (last - first) - 1 - i) = *(first + i) once for each
    /// non-negative i < (last - first)
    /// If the source and destination ranges (that is, [first, last) and
    /// [result, result+(last-first)) respectively) overlap, the
    /// behavior is undefined.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a bidirectional iterator.
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param result   Refers to the begin of the destination range.
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
    ///           \a hpx::future<ranges::reverse_copy_result<
    ///            hpx::traits::range_iterator_t<Rng>, OutIter>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a ranges::reverse_copy_result<
    ///            hpx::traits::range_iterator_t<Rng>, OutIter>
    ///           otherwise.
    ///           The \a reverse_copy algorithm returns
    ///           an object equal to {last, result + N} where N = last - first
    ///
    template <typename ExPolicy, typename Rng, typename OutIter>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        reverse_copy_result<hpx::traits::range_iterator_t<Rng>, OutIter>>::type
    reverse_copy(ExPolicy&& policy, Rng&& rng, OutIter result);
}}    // namespace hpx::ranges

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/parallel/algorithms/reverse.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <type_traits>
#include <utility>

namespace hpx::ranges {

    /// `reverse_copy_result` is equivalent to
    /// `hpx::parallel::util::in_out_result`
    template <typename I, typename O>
    using reverse_copy_result = hpx::parallel::util::in_out_result<I, O>;

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::reverse
    inline constexpr struct reverse_t final
      : hpx::detail::tag_parallel_algorithm<reverse_t>
    {
    private:
        // clang-format off
        template <typename Iter, typename Sent,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_iterator_v<Iter> &&
            hpx::traits::is_sentinel_for_v<Sent, Iter>
        )>
        // clang-format on
        friend Iter tag_fallback_invoke(
            hpx::ranges::reverse_t, Iter first, Sent sent)
        {
            static_assert(hpx::traits::is_bidirectional_iterator<Iter>::value,
                "Required at least bidirectional iterator.");

            return parallel::detail::reverse<Iter>().call(
                hpx::execution::sequenced_policy{}, first, sent);
        }

        // clang-format off
        template <typename Rng,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_range_v<Rng>
        )>
        // clang-format on
        friend hpx::traits::range_iterator_t<Rng> tag_fallback_invoke(
            hpx::ranges::reverse_t, Rng&& rng)
        {
            static_assert(hpx::traits::is_bidirectional_iterator<
                              hpx::traits::range_iterator_t<Rng>>::value,
                "Required at least bidirectional iterator.");

            return parallel::detail::reverse<
                hpx::traits::range_iterator_t<Rng>>()
                .call(hpx::execution::sequenced_policy{}, hpx::util::begin(rng),
                    hpx::util::end(rng));
        }

        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_iterator_v<Iter> &&
            hpx::traits::is_sentinel_for_v<Sent, Iter>
        )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            Iter>::type
        tag_fallback_invoke(
            hpx::ranges::reverse_t, ExPolicy&& policy, Iter first, Sent sent)
        {
            static_assert(hpx::traits::is_bidirectional_iterator<Iter>::value,
                "Required at least bidirectional iterator.");

            return parallel::detail::reverse<Iter>().call(
                HPX_FORWARD(ExPolicy, policy), first, sent);
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_range_v<Rng>
        )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy,
            hpx::traits::range_iterator_t<Rng>>
        tag_fallback_invoke(
            hpx::ranges::reverse_t, ExPolicy&& policy, Rng&& rng)
        {
            static_assert(hpx::traits::is_bidirectional_iterator<
                              hpx::traits::range_iterator_t<Rng>>::value,
                "Required at least bidirectional iterator.");

            return parallel::detail::reverse<
                hpx::traits::range_iterator_t<Rng>>()
                .call(HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
                    hpx::util::end(rng));
        }
    } reverse{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::reverse_copy
    inline constexpr struct reverse_copy_t final
      : hpx::detail::tag_parallel_algorithm<reverse_copy_t>
    {
    private:
        // clang-format off
        template <typename Iter, typename Sent, typename OutIter,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_iterator_v<Iter> &&
            hpx::traits::is_sentinel_for_v<Sent, Iter> &&
            hpx::traits::is_iterator_v<OutIter>
        )>
        // clang-format on
        friend reverse_copy_result<Iter, OutIter> tag_fallback_invoke(
            hpx::ranges::reverse_copy_t, Iter first, Sent last, OutIter result)
        {
            static_assert(hpx::traits::is_bidirectional_iterator<Iter>::value,
                "Required at least bidirectional iterator.");

            static_assert(hpx::traits::is_output_iterator_v<OutIter>,
                "Required at least output iterator.");

            return parallel::detail::reverse_copy<
                hpx::parallel::util::in_out_result<Iter, OutIter>>()
                .call(hpx::execution::sequenced_policy{}, first, last, result);
        }

        // clang-format off
        template <typename Rng, typename OutIter,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_range_v<Rng> &&
            hpx::traits::is_iterator_v<OutIter>
        )>
        // clang-format on
        friend reverse_copy_result<hpx::traits::range_iterator_t<Rng>, OutIter>
        tag_fallback_invoke(
            hpx::ranges::reverse_copy_t, Rng&& rng, OutIter result)
        {
            static_assert(hpx::traits::is_bidirectional_iterator<
                              hpx::traits::range_iterator_t<Rng>>::value,
                "Required at least bidirectional iterator.");

            static_assert(hpx::traits::is_output_iterator_v<OutIter>,
                "Required at least output iterator.");

            return parallel::detail::reverse_copy<
                hpx::parallel::util::in_out_result<
                    hpx::traits::range_iterator_t<Rng>, OutIter>>()
                .call(hpx::execution::sequenced_policy{}, hpx::util::begin(rng),
                    hpx::util::end(rng), result);
        }

        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent, typename FwdIter,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_iterator_v<Iter> &&
            hpx::traits::is_sentinel_for_v<Sent, Iter> &&
            hpx::traits::is_iterator_v<FwdIter>
        )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy,
            reverse_copy_result<Iter, FwdIter>>
        tag_fallback_invoke(hpx::ranges::reverse_copy_t, ExPolicy&& policy,
            Iter first, Sent last, FwdIter result)
        {
            static_assert(hpx::traits::is_bidirectional_iterator<Iter>::value,
                "Required at least bidirectional iterator.");

            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Required at least forward iterator.");

            return parallel::detail::reverse_copy<
                hpx::parallel::util::in_out_result<Iter, FwdIter>>()
                .call(HPX_FORWARD(ExPolicy, policy), first, last, result);
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename OutIter,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_range_v<Rng> &&
            hpx::traits::is_iterator_v<OutIter>
        )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy,
            reverse_copy_result<hpx::traits::range_iterator_t<Rng>, OutIter>>
        tag_fallback_invoke(hpx::ranges::reverse_copy_t, ExPolicy&& policy,
            Rng&& rng, OutIter result)
        {
            static_assert(hpx::traits::is_bidirectional_iterator<
                              hpx::traits::range_iterator_t<Rng>>::value,
                "Required at least bidirectional iterator.");

            static_assert(hpx::traits::is_output_iterator_v<OutIter>,
                "Required at least output iterator.");

            return parallel::detail::reverse_copy<
                hpx::parallel::util::in_out_result<
                    hpx::traits::range_iterator_t<Rng>, OutIter>>()
                .call(HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
                    hpx::util::end(rng), result);
        }
    } reverse_copy{};
}    // namespace hpx::ranges

#endif    // DOXYGEN
