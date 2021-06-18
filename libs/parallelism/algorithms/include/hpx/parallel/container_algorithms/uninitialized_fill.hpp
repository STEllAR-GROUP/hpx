//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/uninitialized_fill.hpp

#pragma once

#if defined(DOXYGEN)

namespace hpx { namespace ranges {

    /// Copies the elements in the range, defined by [first, last), to an
    /// uninitialized memory area beginning at \a dest. If an exception is
    /// thrown during the copy operation, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Sent1       The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter.
    /// \tparam FwdIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Sent2       The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter2.
    ///
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     that will be copied from
    /// \param last1        Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied
    /// \param first2       Refers to the beginning of the destination range.
    /// \param last2        Refers to sentinel value denoting the end of the
    ///                     second range the algorithm will be applied to.
    ///
    /// The assignments in the parallel \a uninitialized_fill algorithm invoked
    /// without an execution policy object will execute in sequential order in
    /// the calling thread.
    ///
    /// \returns  The \a uninitialized_fill algorithm returns an
    ///           \a in_out_result<InIter, FwdIter>.
    ///           The \a uninitialized_fill algorithm returns an input iterator
    ///           to one past the last element copied from and the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    template <typename InIter, typename Sent1, typename FwdIter, typename Sent2>
    hpx::parallel::util::in_out_result<InIter, FwdIter> uninitialized_fill(
        InIter first1, Sent1 last1, FwdIter first2, Sent2 last2);

    /// Copies the elements in the range, defined by [first, last), to an
    /// uninitialized memory area beginning at \a dest. If an exception is
    /// thrown during the copy operation, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Sent1       The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Sent2       The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter2.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     that will be copied from
    /// \param last1        Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param first2       Refers to the beginning of the destination range.
    /// \param last2        Refers to sentinel value denoting the end of the
    ///                     second range the algorithm will be applied to.
    ///
    /// The assignments in the parallel \a uninitialized_fill algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_fill algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_fill algorithm returns a
    ///           \a hpx::future<in_out_result<InIter, FwdIter>>, if the
    ///           execution policy is of type \a sequenced_task_policy
    ///           or \a parallel_task_policy and
    ///           returns \a in_out_result<InIter, FwdIter> otherwise.
    ///           The \a uninitialized_fill algorithm returns an input iterator
    ///           to one past the last element copied from and the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename Sent1,
        typename FwdIter2, typename Sent2>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        parallel::util::in_out_result<FwdIter1, FwdIter2>>::type
    uninitialized_fill(ExPolicy&& policy, FwdIter1 first1, Sent1 last1,
        FwdIter2 first2, Sent2 last2);

    /// Copies the elements in the range, defined by [first, last), to an
    /// uninitialized memory area beginning at \a dest. If an exception is
    /// thrown during the copy operation, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam Rng1        The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Rng2        The type of the destination range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    ///
    /// \param rng1         Refers to the range from which the elements
    ///                     will be copied from
    /// \param rng2         Refers to the range to which the elements
    ///                     will be copied to
    ///
    /// The assignments in the parallel \a uninitialized_fill algorithm invoked
    /// without an execution policy object will execute in sequential order in
    /// the calling thread.
    ///
    /// \returns  The \a uninitialized_fill algorithm returns an
    ///           \a in_out_result<typename hpx::traits::range_traits<Rng1>
    ///           ::iterator_type, typename hpx::traits::range_traits<Rng2>
    ///           ::iterator_type>.
    ///           The \a uninitialized_fill algorithm returns an input iterator
    ///           to one past the last element copied from and the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    template <typename Rng1, typename Rng2>
    hpx::parallel::util::in_out_result<
        typename hpx::traits::range_traits<Rng1>::iterator_type,
        typename hpx::traits::range_traits<Rng2>::iterator_type>
    uninitialized_fill(Rng1&& rng1, Rng2&& rng2);

    /// Copies the elements in the range, defined by [first, last), to an
    /// uninitialized memory area beginning at \a dest. If an exception is
    /// thrown during the copy operation, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng1        The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Rng2        The type of the destination range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng1         Refers to the range from which the elements
    ///                     will be copied from
    /// \param rng2         Refers to the range to which the elements
    ///                     will be copied to
    ///
    /// The assignments in the parallel \a uninitialized_fill algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_fill algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_fill algorithm returns a
    ///           \a hpx::future<in_out_result<InIter, FwdIter>>, if the
    ///           execution policy is of type \a sequenced_task_policy
    ///           or \a parallel_task_policy and
    ///           returns \a in_out_result<
    ///             typename hpx::traits::range_traits<Rng1>::iterator_type
    ///           , typename hpx::traits::range_traits<Rng2>::iterator_type>
    ///           otherwise. The \a uninitialized_fill algorithm returns the
    ///           input iterator to one past the last element copied from and
    ///           the output iterator to the element in the destination range,
    ///           one past the last element copied.
    ///
    template <typename ExPolicy, typename Rng1, typename Rng2>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        hpx::parallel::util::in_out_result<
            typename hpx::traits::range_traits<Rng1>::iterator_type,
            typename hpx::traits::range_traits<Rng2>::iterator_type>>::type
    uninitialized_fill(ExPolicy&& policy, Rng1&& rng1, Rng2&& rng2);

    /// Copies the elements in the range [first, first + count), starting from
    /// first and proceeding to first + count - 1., to another range beginning
    /// at dest. If an exception is thrown during the copy operation, the
    /// function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam FwdIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Sent2       The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    ///
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     that will be copied from
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the destination range.
    /// \param last2        Refers to sentinel value denoting the end of the
    ///                     second range the algorithm will be applied to.
    ///
    /// The assignments in the parallel \a uninitialized_fill_n algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// \returns  The \a uninitialized_fill_n algorithm returns
    ///           \a in_out_result<InIter, FwdIter>.
    ///           The \a uninitialized_fill_n algorithm returns the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    template <typename InIter, typename Size, typename FwdIter, typename Sent2>
    hpx::parallel::util::in_out_result<InIter, FwdIter> uninitialized_fill_n(
        InIter first1, Size count, FwdIter first2, Sent2 last2);

    /// Copies the elements in the range [first, first + count), starting from
    /// first and proceeding to first + count - 1., to another range beginning
    /// at dest. If an exception is thrown during the copy operation, the
    /// function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Sent2       The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter2.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     that will be copied from
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the destination range.
    /// \param last1        Refers to sentinel value denoting the end of the
    ///                     second range the algorithm will be applied to.
    ///
    /// The assignments in the parallel \a uninitialized_fill_n algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_fill_n algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_fill_n algorithm returns a
    ///           \a hpx::future<in_out_result<FwdIter1, FwdIter2>> if the
    ///           execution policy is of type \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter2 otherwise.
    ///           The \a uninitialized_fill_n algorithm returns the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename Size,
        typename FwdIter2, typename Sent2>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        parallel::util::in_out_result<FwdIter1, FwdIter2>>::type
    uninitialized_fill_n(ExPolicy&& policy, FwdIter1 first1, Size count,
        FwdIter2 first2, Sent2 last2);
}}    // namespace hpx::ranges
#else

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/algorithms/uninitialized_fill.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace ranges {
    HPX_INLINE_CONSTEXPR_VARIABLE struct uninitialized_fill_t final
      : hpx::functional::tag_fallback<uninitialized_fill_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Sent, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_forward_iterator<FwdIter>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value
            )>
        // clang-format on
        friend FwdIter tag_fallback_dispatch(hpx::ranges::uninitialized_fill_t,
            FwdIter first, Sent last, T const& value)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::uninitialized_fill<FwdIter>()
                .call(hpx::execution::seq, first, last, value);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter,
            typename Sent, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_forward_iterator<FwdIter>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_dispatch(hpx::ranges::uninitialized_fill_t,
            ExPolicy&& policy, FwdIter first, Sent last, T const& value)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::uninitialized_fill<FwdIter>()
                .call(std::forward<ExPolicy>(policy), first, last, value);
        }

        // clang-format off
        template <typename Rng, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend typename hpx::traits::range_traits<Rng>::iterator_type
        tag_fallback_dispatch(
            hpx::ranges::uninitialized_fill_t, Rng&& rng, T const& value)
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            static_assert(
                hpx::traits::is_forward_iterator<iterator_type>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::uninitialized_fill<
                iterator_type>()
                .call(
                    hpx::execution::seq, std::begin(rng), std::end(rng), value);
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            typename hpx::traits::range_traits<Rng>::iterator_type>::type
        tag_fallback_dispatch(hpx::ranges::uninitialized_fill_t,
            ExPolicy&& policy, Rng&& rng, T const& value)
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            static_assert(
                hpx::traits::is_forward_iterator<iterator_type>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::uninitialized_fill<
                iterator_type>()
                .call(std::forward<ExPolicy>(policy), std::begin(rng),
                    std::end(rng), value);
        }
    } uninitialized_fill{};

    HPX_INLINE_CONSTEXPR_VARIABLE struct uninitialized_fill_n_t final
      : hpx::functional::tag_fallback<uninitialized_fill_n_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Size, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_forward_iterator<FwdIter>::value
            )>
        // clang-format on
        friend FwdIter tag_fallback_dispatch(
            hpx::ranges::uninitialized_fill_n_t, FwdIter first, Size count,
            T const& value)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::uninitialized_fill_n<FwdIter>()
                .call(hpx::execution::seq, first, count, value);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Size,
            typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_forward_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_dispatch(hpx::ranges::uninitialized_fill_n_t,
            ExPolicy&& policy, FwdIter first, Size count, T const& value)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::uninitialized_fill_n<FwdIter>()
                .call(std::forward<ExPolicy>(policy), first, count, value);
        }
    } uninitialized_fill_n{};
}}    // namespace hpx::ranges

#endif
