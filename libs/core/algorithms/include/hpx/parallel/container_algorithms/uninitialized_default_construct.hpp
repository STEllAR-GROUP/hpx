//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/uninitialized_default_construct.hpp

#pragma once

#if defined(DOXYGEN)

namespace hpx { namespace ranges {
    // clang-format off

    /// Constructs objects of type typename iterator_traits<ForwardIt>
    /// ::value_type in the uninitialized storage designated by the range
    /// by default-initialization. If an exception is thrown during the
    /// initialization, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    ///
    /// The assignments in the parallel \a uninitialized_default_construct
    /// algorithm invoked without an execution policy object will execute in
    /// sequential order in the calling thread.
    ///
    /// \returns  The \a uninitialized_default_construct algorithm returns a
    ///           returns \a FwdIter.
    ///           The \a uninitialized_default_construct algorithm returns the
    ///           output iterator to the element in the range, one past
    ///           the last element constructed.
    ///
    template <typename FwdIter, typename Sent>
    FwdIter uninitialized_default_construct(
        FwdIter first, Sent last);

    /// Constructs objects of type typename iterator_traits<ForwardIt>
    /// ::value_type in the uninitialized storage designated by the range
    /// by default-initialization. If an exception is thrown during the
    /// initialization, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
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
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    ///
    /// The assignments in the parallel \a uninitialized_default_construct
    /// algorithm invoked with an execution policy object of type \a
    /// sequenced_policy execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_default_construct
    /// algorithm invoked with an execution policy object of type \a
    /// parallel_policy or \a parallel_task_policy are permitted to execute
    /// in an unordered fashion in unspecified threads, and indeterminately
    /// sequenced within each thread.
    ///
    /// \returns  The \a uninitialized_default_construct algorithm returns a
    ///           \a hpx::future<FwdIter> if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a uninitialized_default_construct algorithm returns
    ///           the iterator to the element in the source range, one past
    ///           the last element constructed.
    ///
    template <typename ExPolicy, typename FwdIter, typename Sent>
    typename parallel::util::detail::algorithm_result<ExPolicy, FwdIter>::type
    uninitialized_default_construct(
        ExPolicy&& policy, FwdIter first, Sent last);

    /// Constructs objects of type typename iterator_traits<ForwardIt>
    /// ::value_type in the uninitialized storage designated by the range
    /// by default-initialization. If an exception is thrown during the
    /// initialization, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    ///
    /// \param rng          Refers to the range to which will be default
    ///                     constructed.
    ///
    /// The assignments in the parallel \a uninitialized_default_construct
    /// algorithm invoked without an execution policy object will execute in
    /// sequential order in the calling thread.
    ///
    /// \returns  The \a uninitialized_default_construct algorithm returns a
    ///           returns \a hpx::traits::range_traits<Rng>
    ///           ::iterator_type.
    ///           The \a uninitialized_default_construct algorithm returns
    ///           the output iterator to the element in the range, one past
    ///           the last element constructed.
    ///
    template <typename Rng>
    typename hpx::traits::range_traits<Rng>::iterator_type
    uninitialized_default_construct(Rng&& rng);

    /// Constructs objects of type typename iterator_traits<ForwardIt>
    /// ::value_type in the uninitialized storage designated by the range
    /// by default-initialization. If an exception is thrown during the
    /// initialization, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the range to which the value
    ///                     will be default consutrcted
    ///
    /// The assignments in the parallel \a uninitialized_default_construct
    /// algorithm invoked with an execution policy object of type \a
    /// sequenced_policy execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_default_construct
    /// algorithm invoked with an execution policy object of type \a
    /// parallel_policy or \a parallel_task_policy are permitted to execute
    /// in an unordered fashion in unspecified threads, and indeterminately
    /// sequenced within each thread.
    ///
    /// \returns  The \a uninitialized_default_construct algorithm returns a
    ///           \a hpx::future<typename hpx::traits::range_traits<Rng>
    ///           ::iterator_type>, if the
    ///           execution policy is of type \a sequenced_task_policy
    ///           or \a parallel_task_policy and returns \a typename
    ///           hpx::traits::range_traits<Rng>::iterator_type otherwise.
    ///           The \a uninitialized_default_construct algorithm returns
    ///           the output iterator to the element in the range, one past
    ///           the last element constructed.
    ///
    template <typename ExPolicy, typename Rng>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        typename hpx::traits::range_traits<Rng>::iterator_type>::type
    uninitialized_default_construct(ExPolicy&& policy, Rng&& rng);

    /// Constructs objects of type typename iterator_traits<ForwardIt>
    /// ::value_type in the uninitialized storage designated by the range
    /// [first, first + count) by default-initialization. If an exception
    /// is thrown during the initialization, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a count assignments, if
    ///         count > 0, no assignments otherwise.
    ///
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    ///
    /// The assignments in the parallel \a uninitialized_default_construct_n
    /// algorithm invoked without an execution policy object execute in
    /// sequential order in the calling thread.
    ///
    /// \returns  The \a uninitialized_default_construct_n algorithm returns a
    ///           returns \a FwdIter.
    ///           The \a uninitialized_default_construct_n algorithm returns
    ///           the iterator to the element in the source range, one past
    ///           the last element constructed.
    ///
    template <typename FwdIter, typename Size>
    FwdIter uninitialized_default_construct_n(FwdIter first, Size count);

    /// Constructs objects of type typename iterator_traits<ForwardIt>
    /// ::value_type in the uninitialized storage designated by the range
    /// [first, first + count) by default-initialization. If an exception
    /// is thrown during the initialization, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a count assignments, if
    ///         count > 0, no assignments otherwise.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    ///
    /// The assignments in the parallel \a uninitialized_default_construct_n
    /// algorithm invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_default_construct_n
    /// algorithm invoked with an execution policy object of type
    /// \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_default_construct_n algorithm returns a
    ///           \a hpx::future<FwdIter> if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a uninitialized_default_construct_n algorithm returns
    ///           the iterator to the element in the source range, one past
    ///           the last element constructed.
    ///
    template <typename ExPolicy, typename FwdIter, typename Size>
    typename typename parallel::util::detail::algorithm_result<ExPolicy,
        FwdIter>::type
    uninitialized_default_construct_n(
        ExPolicy&& policy, FwdIter first, Size count);

    // clang-format on
}}    // namespace hpx::ranges
#else

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/algorithms/uninitialized_default_construct.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace ranges {
    HPX_INLINE_CONSTEXPR_VARIABLE struct uninitialized_default_construct_t final
      : hpx::detail::tag_parallel_algorithm<uninitialized_default_construct_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Sent,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_forward_iterator<FwdIter>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value
            )>
        // clang-format on
        friend FwdIter tag_fallback_dispatch(
            hpx::ranges::uninitialized_default_construct_t, FwdIter first,
            Sent last)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::uninitialized_default_construct<
                FwdIter>()
                .call(hpx::execution::seq, first, last);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_forward_iterator<FwdIter>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_dispatch(hpx::ranges::uninitialized_default_construct_t,
            ExPolicy&& policy, FwdIter first, Sent last)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::uninitialized_default_construct<
                FwdIter>()
                .call(std::forward<ExPolicy>(policy), first, last);
        }

        // clang-format off
        template <typename Rng,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend typename hpx::traits::range_traits<Rng>::iterator_type
        tag_fallback_dispatch(
            hpx::ranges::uninitialized_default_construct_t, Rng&& rng)
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            static_assert(
                hpx::traits::is_forward_iterator<iterator_type>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::uninitialized_default_construct<
                iterator_type>()
                .call(hpx::execution::seq, std::begin(rng), std::end(rng));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            typename hpx::traits::range_traits<Rng>::iterator_type>::type
        tag_fallback_dispatch(hpx::ranges::uninitialized_default_construct_t,
            ExPolicy&& policy, Rng&& rng)
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            static_assert(
                hpx::traits::is_forward_iterator<iterator_type>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::uninitialized_default_construct<
                iterator_type>()
                .call(std::forward<ExPolicy>(policy), std::begin(rng),
                    std::end(rng));
        }
    } uninitialized_default_construct{};

    HPX_INLINE_CONSTEXPR_VARIABLE struct uninitialized_default_construct_n_t
        final
      : hpx::detail::tag_parallel_algorithm<uninitialized_default_construct_n_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Size,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_forward_iterator<FwdIter>::value
            )>
        // clang-format on
        friend FwdIter tag_fallback_dispatch(
            hpx::ranges::uninitialized_default_construct_n_t, FwdIter first,
            Size count)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::uninitialized_default_construct_n<
                FwdIter>()
                .call(hpx::execution::seq, first, count);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Size,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_forward_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_dispatch(hpx::ranges::uninitialized_default_construct_n_t,
            ExPolicy&& policy, FwdIter first, Size count)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::uninitialized_default_construct_n<
                FwdIter>()
                .call(std::forward<ExPolicy>(policy), first, count);
        }
    } uninitialized_default_construct_n{};
}}    // namespace hpx::ranges

#endif
