//  Copyright (c) 2015-2023 Hartmut Kaiser
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/shift_right.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx { namespace ranges {
    // clang-format off

    ///////////////////////////////////////////////////////////////////////////
    /// Shifts the elements in the range [first, last) by n positions towards
    /// the end of the range. For every integer i in [0, last - first - n),
    /// moves the element originally at position first + i to position first
    /// + n + i.
    ///
    /// \note   Complexity: At most (last - first) - n assignments.
    ///
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     positions to shift by.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param n            Refers to the number of positions to shift.
    ///
    /// The assignment operations in the parallel \a shift_right algorithm
    /// invoked without an execution policy object will execute in sequential
    /// order in the calling thread.
    ///
    /// \note The type of dereferenced \a FwdIter must meet the requirements
    ///       of \a MoveAssignable.
    ///
    /// \returns  The \a shift_right algorithm returns \a FwdIter.
    ///           The \a shift_right algorithm returns an iterator to the
    ///           end of the resulting range.
    ///
    template <typename FwdIter, typename Sent, typename Size>
    FwdIter shift_right(FwdIter first, Sent last, Size n);

    ///////////////////////////////////////////////////////////////////////////
    /// Shifts the elements in the range [first, last) by n positions towards
    /// the end of the range. For every integer i in [0, last - first - n),
    /// moves the element originally at position first + i to position first
    /// + n + i.
    ///
    /// \note   Complexity: At most (last - first) - n assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     positions to shift by.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param n            Refers to the number of positions to shift.
    ///
    /// The assignment operations in the parallel \a shift_right algorithm
    /// invoked with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignment operations in the parallel \a shift_right algorithm
    /// invoked with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note The type of dereferenced \a FwdIter must meet the requirements
    ///       of \a MoveAssignable.
    ///
    /// \returns  The \a shift_right algorithm returns a
    ///           \a hpx::future<FwdIter> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a shift_right algorithm returns an iterator to the
    ///           end of the resulting range.
    ///
    template <typename ExPolicy, typename FwdIter, typename Sent,
        typename Size>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, FwdIter>
    shift_right(ExPolicy&& policy, FwdIter first, Sent last, Size n);

    ///////////////////////////////////////////////////////////////////////////
    /// Shifts the elements in the range [first, last) by n positions towards
    /// the end of the range. For every integer i in [0, last - first - n),
    /// moves the element originally at position first + i to position first
    /// + n + i.
    ///
    /// \note   Complexity: At most (last - first) - n assignments.
    ///
    /// \tparam Rng         The type of the range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     positions to shift by.
    ///
    /// \param rng          Refers to the range in which the elements
    ///                     will be shifted.
    /// \param n            Refers to the number of positions to shift.
    ///
    /// The assignment operations in the parallel \a shift_right algorithm
    /// invoked without an execution policy object will execute in sequential
    /// order in the calling thread.
    ///
    /// \note The type of dereferenced \a hpx::traits::range_iterator_t<Rng>
    ///       must meet the requirements of \a MoveAssignable.
    ///
    /// \returns  The \a shift_right algorithm returns \a
    ///           hpx::traits::range_iterator_t<Rng>.
    ///           The \a shift_right algorithm returns an iterator to the
    ///           end of the resulting range.
    ///
    template <typename Rng, typename Size>
    hpx::traits::range_iterator_t<Rng> shift_right(Rng&& rng, Size n);

    ///////////////////////////////////////////////////////////////////////////
    /// Shifts the elements in the range [first, last) by n positions towards
    /// the end of the range. For every integer i in [0, last - first - n),
    /// moves the element originally at position first + i to position first
    /// + n + i.
    ///
    /// \note   Complexity: At most (last - first) - n assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     positions to shift by.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the range in which the elements
    ///                     will be shifted.
    /// \param n            Refers to the number of positions to shift.
    ///
    /// The assignment operations in the parallel \a shift_right algorithm
    /// invoked with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignment operations in the parallel \a shift_right algorithm
    /// invoked with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note The type of dereferenced \a hpx::traits::range_iterator_t<Rng>
    ///       must meet the requirements of \a MoveAssignable.
    ///
    /// \returns  The \a shift_right algorithm returns a
    ///           \a hpx::future<hpx::traits::range_iterator_t<Rng>> if
    ///           the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a hpx::traits::range_iterator_t<Rng> otherwise.
    ///           The \a shift_right algorithm returns an iterator to the
    ///           end of the resulting range.
    ///
    template <typename ExPolicy, typename Rng, typename Size>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        hpx::traits::range_iterator_t<Rng>>
    shift_right(ExPolicy&& policy, Rng&& rng, Size n);

    // clang-format on
}}    // namespace hpx::ranges

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/parallel/algorithms/shift_right.hpp>

#include <type_traits>
#include <utility>

namespace hpx::ranges {

    inline constexpr struct shift_right_t final
      : hpx::functional::detail::tag_fallback<shift_right_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Sent, typename Size,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<FwdIter> &&
                hpx::traits::is_sentinel_for_v<Sent, FwdIter> &&
                std::is_integral_v<Size>
            )>
        // clang-format on
        friend FwdIter tag_fallback_invoke(
            hpx::ranges::shift_right_t, FwdIter first, Sent last, Size n)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::shift_right<FwdIter>().call(
                hpx::execution::seq, first, last, n);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename Size,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<FwdIter> &&
                hpx::traits::is_sentinel_for_v<Sent, FwdIter> &&
                std::is_integral_v<Size>
            )>
        // clang-format on
        friend hpx::parallel::util::detail::algorithm_result_t<ExPolicy,
            FwdIter>
        tag_fallback_invoke(hpx::ranges::shift_right_t, ExPolicy&& policy,
            FwdIter first, Sent last, Size n)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::shift_right<FwdIter>().call(
                HPX_FORWARD(ExPolicy, policy), first, last, n);
        }

        // clang-format off
        template <typename Rng, typename Size,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range_v<Rng> &&
                std::is_integral_v<Size>
            )>
        // clang-format on
        friend hpx::traits::range_iterator_t<Rng> tag_fallback_invoke(
            hpx::ranges::shift_right_t, Rng&& rng, Size n)
        {
            static_assert(hpx::traits::is_forward_iterator_v<
                              hpx::traits::range_iterator_t<Rng>>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::shift_right<
                hpx::traits::range_iterator_t<Rng>>()
                .call(hpx::execution::seq, std::begin(rng), std::end(rng), n);
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,  typename Size,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_range_v<Rng> &&
                std::is_integral_v<Size>
            )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy,
            hpx::traits::range_iterator_t<Rng>>
        tag_fallback_invoke(
            hpx::ranges::shift_right_t, ExPolicy&& policy, Rng&& rng, Size n)
        {
            static_assert(hpx::traits::is_forward_iterator_v<
                              hpx::traits::range_iterator_t<Rng>>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::shift_right<
                hpx::traits::range_iterator_t<Rng>>()
                .call(HPX_FORWARD(ExPolicy, policy), std::begin(rng),
                    std::end(rng), n);
        }
    } shift_right{};
}    // namespace hpx::ranges

#endif    // DOXYGEN
