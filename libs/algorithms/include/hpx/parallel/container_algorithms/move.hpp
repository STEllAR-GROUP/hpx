//  Copyright (c) 2017 Bruno Pitrus
//  Copyright (c) 2017-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/move.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    // clang-format off

    /// Moves the elements in the range \a rng to another range beginning
    /// at \a dest. After this operation the elements in the moved-from
    /// range will still contain valid values of the appropriate type,
    /// but not necessarily the same values as before the move.
    ///
    /// \note   Complexity: Performs exactly
    ///         std::distance(begin(rng), end(rng)) assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the begin source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent1       The type of the end source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for FwdIter1.
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
    /// \param dest         Refers to the beginning of the destination range.
    ///
    /// The assignments in the parallel \a copy algorithm invoked with an
    /// execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a copy algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a move algorithm returns a
    ///           \a hpx::future<ranges::move_result<iterator_t<Rng>, FwdIter2>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or \a parallel_task_policy and
    ///           returns \a ranges::move_result<iterator_t<Rng>, FwdIter2>
    ///           otherwise.
    ///           The \a move algorithm returns the pair of the input iterator
    ///           \a last and the output iterator to the element in the
    ///           destination range, one past the last element moved.
    ///
    template <typename ExPolicy, typename FwdIter1, typename Sent1,
        typename FwdIter>
    typename util::detail::algorithm_result<
        ExPolicy, ranges::move_result<FwdIter1, FwdIter>>::type
    move(ExPolicy&& policy, FwdIter1 iter, Sent1 sent, FwdIter dest);

    /// Moves the elements in the range \a rng to another range beginning
    /// at \a dest. After this operation the elements in the moved-from
    /// range will still contain valid values of the appropriate type,
    /// but not necessarily the same values as before the move.
    ///
    /// \note   Complexity: Performs exactly
    ///         std::distance(begin(rng), end(rng)) assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam FwdIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    ///
    /// The assignments in the parallel \a copy algorithm invoked with an
    /// execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a copy algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a move algorithm returns a
    ///           \a hpx::future<ranges::move_result<iterator_t<Rng>, FwdIter2>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or \a parallel_task_policy and
    ///           returns \a ranges::move_result<iterator_t<Rng>, FwdIter2>
    ///           otherwise.
    ///           The \a move algorithm returns the pair of the input iterator
    ///           \a last and the output iterator to the element in the
    ///           destination range, one past the last element moved.
    ///
    template <typename ExPolicy, typename Rng, typename FwdIter>
    typename util::detail::algorithm_result<
        ExPolicy, ranges::move_result<
            typename hpx::traits::range_traits<Rng>::iterator_type, FwdIter>
    >::type
    move(ExPolicy&& policy, Rng&& rng, FwdIter dest);

    // clang-format on
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <hpx/parallel/algorithms/move.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace ranges {

    template <typename I, typename O>
    using move_result = parallel::util::in_out_result<I, O>;

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::move
    HPX_INLINE_CONSTEXPR_VARIABLE struct move_t final
      : hpx::functional::tag<move_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename Iter1, typename Sent1,
            typename Iter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_sentinel_for<Sent1, Iter1>::value &&
                hpx::traits::is_iterator<Iter2>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            move_result<Iter1, Iter2>>::type
        tag_invoke(
            move_t, ExPolicy&& policy, Iter1 first, Sent1 last, Iter2 dest)
        {
            return hpx::parallel::v1::detail::transfer<
                hpx::parallel::v1::detail::move<Iter1, Iter2>>(
                std::forward<ExPolicy>(policy), first, last, dest);
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename Iter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::traits::is_iterator<Iter2>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            move_result<typename hpx::traits::range_iterator<Rng>::type,
                Iter2>>::type
        tag_invoke(move_t, ExPolicy&& policy, Rng&& rng, Iter2 dest)
        {
            using iterator_type =
                typename hpx::traits::range_iterator<Rng>::type;

            return hpx::parallel::v1::detail::transfer<
                hpx::parallel::v1::detail::move<iterator_type, Iter2>>(
                std::forward<ExPolicy>(policy), hpx::util::begin(rng),
                hpx::util::end(rng), dest);
        }

        // clang-format off
        template <typename Iter1, typename Sent1, typename Iter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_sentinel_for<Sent1, Iter1>::value &&
                hpx::traits::is_iterator<Iter2>::value
            )>
        // clang-format on
        friend move_result<Iter1, Iter2> tag_invoke(
            move_t, Iter1 first, Sent1 last, Iter2 dest)
        {
            return hpx::parallel::v1::detail::transfer<
                hpx::parallel::v1::detail::move<Iter1, Iter2>>(
                hpx::parallel::execution::seq, first, last, dest);
        }

        // clang-format off
        template <typename Rng, typename Iter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::traits::is_iterator<Iter2>::value
            )>
        // clang-format on
        friend move_result<typename hpx::traits::range_iterator<Rng>::type,
            Iter2>
        tag_invoke(move_t, Rng&& rng, Iter2 dest)
        {
            using iterator_type =
                typename hpx::traits::range_iterator<Rng>::type;

            return hpx::parallel::v1::detail::transfer<
                hpx::parallel::v1::detail::move<iterator_type, Iter2>>(
                hpx::parallel::execution::seq, hpx::util::begin(rng),
                hpx::util::end(rng), dest);
        }
    } move;

}}    // namespace hpx::ranges

namespace hpx { namespace parallel { inline namespace v1 {

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename Sent1,
        typename FwdIter,
        HPX_CONCEPT_REQUIRES_(
            execution::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter1>::value &&
            hpx::traits::is_sentinel_for<Sent1, FwdIter1>::value &&
            hpx::traits::is_iterator<FwdIter>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::move is deprecated, use hpx::ranges::move instead")
        typename util::detail::algorithm_result<ExPolicy,
            ranges::move_result<FwdIter1, FwdIter>>::type
        move(ExPolicy&& policy, FwdIter1 iter, Sent1 sent, FwdIter dest)
    {
        using move_iter_t = detail::move<FwdIter1, FwdIter>;
        return detail::transfer<move_iter_t>(
            std::forward<ExPolicy>(policy), iter, sent, dest);
    }

    // clang-format off
    template <typename ExPolicy, typename Rng, typename FwdIter,
        HPX_CONCEPT_REQUIRES_(
            execution::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_range<Rng>::value &&
            hpx::traits::is_iterator<FwdIter>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::move is deprecated, use hpx::ranges::move instead")
        typename util::detail::algorithm_result<ExPolicy,
            ranges::move_result<
                typename hpx::traits::range_traits<Rng>::iterator_type,
                FwdIter>>::type move(ExPolicy&& policy, Rng&& rng, FwdIter dest)
    {
        using move_iter_t =
            detail::move<typename hpx::traits::range_traits<Rng>::iterator_type,
                FwdIter>;

        return detail::transfer<move_iter_t>(std::forward<ExPolicy>(policy),
            hpx::util::begin(rng), hpx::util::end(rng), dest);
    }
}}}    // namespace hpx::parallel::v1

#endif    // DOXYGEN
