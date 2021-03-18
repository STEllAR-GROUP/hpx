//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/reverse.hpp

#pragma once

#if defined(DOXYGEN)

namespace hpx { namespace ranges {

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
    typename hpx::traits::range_iterator<Rng>::type reverse(Rng&& rng);

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
    ///           \a hpx::future<typename hpx::traits::range_iterator<Rng>::type>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a hpx::future<
    ///             typename hpx::traits::range_iterator<Rng>::type> otherwise.
    ///           It returns \a last.
    ///
    template <typename ExPolicy, typename Rng>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        typename hpx::traits::range_iterator<Rng>::type>::type
    reverse(ExPolicy&& policy, Rng&& rng);

    ///////////////////////////////////////////////////////////////////////////
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
    /// \tparam OutputIter  The type of the iterator representing the
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
    ///           typename hpx::traits::range_iterator<Rng>::type, OutIter>>::type.
    ///           The \a reverse_copy algorithm returns
    ///           an object equal to {last, result + N} where N = last - first
    ///
    template <typename Rng, typename OutIter>
    typename ranges::reverse_copy_result<
        typename hpx::traits::range_iterator<Rng>::type, OutIter>
    reverse_copy(Rng&& rng, OutIter result);

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
    /// \tparam OutputIter  The type of the iterator representing the
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
    ///            typename hpx::traits::range_iterator<Rng>::type, OutIter>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a ranges::reverse_copy_result<
    ///            typename hpx::traits::range_iterator<Rng>::type, OutIter>
    ///           otherwise.
    ///           The \a reverse_copy algorithm returns
    ///           an object equal to {last, result + N} where N = last - first
    ///
    template <typename ExPolicy, typename Rng, typename OutIter>
    typename util::detail::algorithm_result<ExPolicy,
        ranges::reverse_copy_result<
            typename hpx::traits::range_iterator<Rng>::type, OutIter>>::type
    reverse_copy(ExPolicy&& policy, Rng&& rng, OutIter result);

}}    // namespace hpx::ranges

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/parallel/algorithms/reverse.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {

    // clang-format off
    template <typename ExPolicy, typename Rng,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_range<Rng>::value)>
    // clang-format on
    HPX_DEPRECATED_V(1, 7,
        "hpx::parallel::reverse is deprecated, use hpx::ranges::reverse "
        "instead") typename util::detail::algorithm_result<ExPolicy,
        typename hpx::traits::range_iterator<Rng>::type>::type
        reverse(ExPolicy&& policy, Rng&& rng)
    {
        typedef hpx::is_sequenced_execution_policy<ExPolicy> is_seq;

        return detail::reverse<
            typename hpx::traits::range_iterator<Rng>::type>()
            .call(std::forward<ExPolicy>(policy), is_seq(),
                hpx::util::begin(rng), hpx::util::end(rng));
    }

    // clang-format off
    template <typename ExPolicy, typename Rng, typename OutIter,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_range<Rng>::value &&
            hpx::traits::is_iterator<OutIter>::value
        )>
    // clang-format on
    typename util::detail::algorithm_result<ExPolicy,
        util::in_out_result<typename hpx::traits::range_iterator<Rng>::type,
            OutIter>>::type
    reverse_copy(ExPolicy&& policy, Rng&& rng, OutIter dest_first)
    {
        typedef hpx::is_sequenced_execution_policy<ExPolicy> is_seq;

        return detail::reverse_copy<util::in_out_result<
            typename hpx::traits::range_iterator<Rng>::type, OutIter>>()
            .call(std::forward<ExPolicy>(policy), is_seq(),
                hpx::util::begin(rng), hpx::util::end(rng), dest_first);
    }
}}}    // namespace hpx::parallel::v1

namespace hpx { namespace ranges {
    /// `reverse_copy_result` is equivalent to
    /// `hpx::parallel::util::in_out_result`
    template <typename I, typename O>
    using reverse_copy_result = hpx::parallel::util::in_out_result<I, O>;

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::reverse
    HPX_INLINE_CONSTEXPR_VARIABLE struct reverse_t final
      : hpx::functional::tag_fallback<reverse_t>
    {
    private:
        // clang-format off
        template <typename Iter, typename Sent,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_iterator<Iter>::value &&
            hpx::traits::is_sentinel_for<Sent, Iter>::value
        )>
        // clang-format on
        friend Iter tag_fallback_invoke(
            hpx::ranges::reverse_t, Iter first, Sent sent)
        {
            static_assert((hpx::traits::is_bidirectional_iterator<Iter>::value),
                "Required at least biderectional iterator.");

            return parallel::v1::detail::reverse<Iter>().call(
                hpx::execution::sequenced_policy{}, std::true_type{}, first,
                sent);
        }

        // clang-format off
        template <typename Rng,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_range<Rng>::value
        )>
        // clang-format on
        friend typename hpx::traits::range_iterator<Rng>::type
        tag_fallback_invoke(hpx::ranges::reverse_t, Rng&& rng)
        {
            static_assert(
                (hpx::traits::is_bidirectional_iterator<
                    typename hpx::traits::range_iterator<Rng>::type>::value),
                "Required at least biderectional iterator.");

            return parallel::v1::detail::reverse<
                typename hpx::traits::range_iterator<Rng>::type>()
                .call(hpx::execution::sequenced_policy{}, std::true_type{},
                    hpx::util::begin(rng), hpx::util::end(rng));
        }

        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<Iter>::value &&
            hpx::traits::is_sentinel_for<Sent, Iter>::value
        )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            Iter>::type
        tag_fallback_invoke(
            hpx::ranges::reverse_t, ExPolicy&& policy, Iter first, Sent sent)
        {
            static_assert((hpx::traits::is_bidirectional_iterator<Iter>::value),
                "Required at least biderectional iterator.");
            typedef hpx::is_sequenced_execution_policy<ExPolicy> is_seq;

            return parallel::v1::detail::reverse<Iter>().call(
                std::forward<ExPolicy>(policy), is_seq(), first, sent);
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_range<Rng>::value
        )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            typename hpx::traits::range_iterator<Rng>::type>::type
        tag_fallback_invoke(
            hpx::ranges::reverse_t, ExPolicy&& policy, Rng&& rng)
        {
            static_assert(
                (hpx::traits::is_bidirectional_iterator<
                    typename hpx::traits::range_iterator<Rng>::type>::value),
                "Required at least biderectional iterator.");

            typedef hpx::is_sequenced_execution_policy<ExPolicy> is_seq;

            return parallel::v1::detail::reverse<
                typename hpx::traits::range_iterator<Rng>::type>()
                .call(std::forward<ExPolicy>(policy), is_seq(),
                    hpx::util::begin(rng), hpx::util::end(rng));
        }
    } reverse{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::reverse_copy
    HPX_INLINE_CONSTEXPR_VARIABLE struct reverse_copy_t final
      : hpx::functional::tag_fallback<reverse_copy_t>
    {
    private:
        // clang-format off
        template <typename Iter, typename Sent, typename OutIter,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_iterator<Iter>::value &&
            hpx::traits::is_sentinel_for<Sent, Iter>::value &&
            hpx::traits::is_iterator<OutIter>::value
        )>
        // clang-format on
        friend reverse_copy_result<Iter, OutIter> tag_fallback_invoke(
            hpx::ranges::reverse_copy_t, Iter first, Sent last, OutIter result)
        {
            static_assert((hpx::traits::is_bidirectional_iterator<Iter>::value),
                "Required at least biderectional iterator.");

            static_assert((hpx::traits::is_output_iterator<OutIter>::value),
                "Required at least output iterator.");

            return parallel::v1::detail::reverse_copy<
                hpx::parallel::util::in_out_result<Iter, OutIter>>()
                .call(hpx::execution::sequenced_policy{}, std::true_type{},
                    first, last, result);
        }

        // clang-format off
        template <typename Rng, typename OutIter,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_range<Rng>::value &&
            hpx::traits::is_iterator<OutIter>::value
        )>
        // clang-format on
        friend reverse_copy_result<
            typename hpx::traits::range_iterator<Rng>::type, OutIter>
        tag_fallback_invoke(
            hpx::ranges::reverse_copy_t, Rng&& rng, OutIter result)
        {
            static_assert(
                (hpx::traits::is_bidirectional_iterator<
                    typename hpx::traits::range_iterator<Rng>::type>::value),
                "Required at least biderectional iterator.");

            static_assert((hpx::traits::is_output_iterator<OutIter>::value),
                "Required at least output iterator.");

            return parallel::v1::detail::reverse_copy<
                hpx::parallel::util::in_out_result<
                    typename hpx::traits::range_iterator<Rng>::type, OutIter>>()
                .call(hpx::execution::sequenced_policy{}, std::true_type{},
                    hpx::util::begin(rng), hpx::util::end(rng), result);
        }

        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent, typename OutIter,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<Iter>::value &&
            hpx::traits::is_sentinel_for<Sent, Iter>::value &&
            hpx::traits::is_iterator<OutIter>::value
        )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            reverse_copy_result<Iter, OutIter>>::type
        tag_fallback_invoke(hpx::ranges::reverse_copy_t, ExPolicy&& policy,
            Iter first, Sent last, OutIter result)
        {
            static_assert((hpx::traits::is_bidirectional_iterator<Iter>::value),
                "Required at least biderectional iterator.");

            static_assert((hpx::traits::is_output_iterator<OutIter>::value),
                "Required at least output iterator.");

            typedef hpx::is_sequenced_execution_policy<ExPolicy> is_seq;

            return parallel::v1::detail::reverse_copy<
                hpx::parallel::util::in_out_result<Iter, OutIter>>()
                .call(std::forward<ExPolicy>(policy), is_seq(), first, last,
                    result);
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename OutIter,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_range<Rng>::value &&
            hpx::traits::is_iterator<OutIter>::value
        )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            reverse_copy_result<typename hpx::traits::range_iterator<Rng>::type,
                OutIter>>::type
        tag_fallback_invoke(hpx::ranges::reverse_copy_t, ExPolicy&& policy,
            Rng&& rng, OutIter result)
        {
            static_assert(
                (hpx::traits::is_bidirectional_iterator<
                    typename hpx::traits::range_iterator<Rng>::type>::value),
                "Required at least biderectional iterator.");

            static_assert((hpx::traits::is_output_iterator<OutIter>::value),
                "Required at least output iterator.");

            typedef hpx::is_sequenced_execution_policy<ExPolicy> is_seq;

            return parallel::v1::detail::reverse_copy<
                hpx::parallel::util::in_out_result<
                    typename hpx::traits::range_iterator<Rng>::type, OutIter>>()
                .call(std::forward<ExPolicy>(policy), is_seq(),
                    hpx::util::begin(rng), hpx::util::end(rng), result);
        }
    } reverse_copy{};

}}    // namespace hpx::ranges

#endif    // DOXYGEN
