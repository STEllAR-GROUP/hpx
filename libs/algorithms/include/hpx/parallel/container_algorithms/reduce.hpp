//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/reduce.hpp

#pragma once

#if defined(DOXYGEN)

namespace hpx { namespace ranges {

    // clang-format off

    /// Returns GENERALIZED_SUM(f, init, *first, ..., *(first + (last - first) - 1)).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a f.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source begin iterator used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a copy_if requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is a
    ///                     binary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&.
    ///                     The types \a Type1 \a Ret must be
    ///                     such that an object of type \a FwdIterB can be
    ///                     dereferenced and then implicitly converted to any
    ///                     of those types.
    /// \param init         The initial value for the generalized sum.
    ///
    /// The reduce operations in the parallel \a reduce algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a copy_if algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a reduce algorithm returns a \a hpx::future<T> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a T otherwise.
    ///           The \a reduce algorithm returns the result of the
    ///           generalized sum over the elements given by the input range
    ///           [first, last).
    ///
    /// \note   GENERALIZED_SUM(op, a1, ..., aN) is defined as follows:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_SUM(op, b1, ..., bK), GENERALIZED_SUM(op, bM, ..., bN)),
    ///           where:
    ///           * b1, ..., bN may be any permutation of a1, ..., aN and
    ///           * 1 < K+1 = M <= N.
    ///
    /// The difference between \a reduce and \a accumulate is
    /// that the behavior of reduce may be non-deterministic for
    /// non-associative or non-commutative binary predicate.
    ///
    template <typename ExPolicy, typename FwdIter, typename Sent, typename T,
        typename F>
    typename util::detail::algorithm_result<ExPolicy, T>::type reduce(
        ExPolicy&& policy, FwdIter first, Sent last, T init, F&& f);

    /// Returns GENERALIZED_SUM(+, init, *first, ..., *(first + (last - first) - 1)).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         operator+().
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source begin iterator used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    /// \param init         The initial value for the generalized sum.
    ///
    /// The reduce operations in the parallel \a reduce algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a copy_if algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a reduce algorithm returns a \a hpx::future<T> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a T otherwise.
    ///           The \a reduce algorithm returns the result of the
    ///           generalized sum (applying operator+()) over the elements given
    ///           by the input range [first, last).
    ///
    /// \note   GENERALIZED_SUM(+, a1, ..., aN) is defined as follows:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_SUM(+, b1, ..., bK), GENERALIZED_SUM(+, bM, ..., bN)),
    ///           where:
    ///           * b1, ..., bN may be any permutation of a1, ..., aN and
    ///           * 1 < K+1 = M <= N.
    ///
    /// The difference between \a reduce and \a accumulate is
    /// that the behavior of reduce may be non-deterministic for
    /// non-associative or non-commutative binary predicate.
    ///
    template <typename ExPolicy, typename FwdIter, typename Sent, typename T>
    typename util::detail::algorithm_result<ExPolicy, T>::type reduce(
        ExPolicy&& policy, FwdIter first, Sent last, T init);

    /// Returns GENERALIZED_SUM(+, T(), *first, ..., *(first + (last - first) - 1)).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         operator+().
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source begin iterator used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements the
    ///                     algorithm will be applied to.
    ///
    /// The reduce operations in the parallel \a reduce algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a copy_if algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a reduce algorithm returns a \a hpx::future<T> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns T otherwise (where T is the value_type of
    ///           \a FwdIterB).
    ///           The \a reduce algorithm returns the result of the
    ///           generalized sum (applying operator+()) over the elements given
    ///           by the input range [first, last).
    ///
    /// \note   The type of the initial value (and the result type) \a T is
    ///         determined from the value_type of the used \a FwdIterB.
    ///
    /// \note   GENERALIZED_SUM(+, a1, ..., aN) is defined as follows:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_SUM(+, b1, ..., bK), GENERALIZED_SUM(+, bM, ..., bN)),
    ///           where:
    ///           * b1, ..., bN may be any permutation of a1, ..., aN and
    ///           * 1 < K+1 = M <= N.
    ///
    /// The difference between \a reduce and \a accumulate is
    /// that the behavior of reduce may be non-deterministic for
    /// non-associative or non-commutative binary predicate.
    ///
    template <typename ExPolicy, typename FwdIter, typename Sent>
    typename util::detail::algorithm_result<ExPolicy,
        typename std::iterator_traits<FwdIter>::value_type>::type
    reduce(ExPolicy&& policy, FwdIter first, Sent last);

    /// Returns GENERALIZED_SUM(f, init, *first, ..., *(first + (last - first) - 1)).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         predicate \a f.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a copy_if requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is a
    ///                     binary predicate. The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     Ret fun(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&.
    ///                     The types \a Type1 \a Ret must be
    ///                     such that an object of type \a FwdIterB can be
    ///                     dereferenced and then implicitly converted to any
    ///                     of those types.
    /// \param init         The initial value for the generalized sum.
    ///
    /// The reduce operations in the parallel \a reduce algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a copy_if algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a reduce algorithm returns a \a hpx::future<T> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a T otherwise.
    ///           The \a reduce algorithm returns the result of the
    ///           generalized sum over the elements given by the input range
    ///           [first, last).
    ///
    /// \note   GENERALIZED_SUM(op, a1, ..., aN) is defined as follows:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_SUM(op, b1, ..., bK), GENERALIZED_SUM(op, bM, ..., bN)),
    ///           where:
    ///           * b1, ..., bN may be any permutation of a1, ..., aN and
    ///           * 1 < K+1 = M <= N.
    ///
    /// The difference between \a reduce and \a accumulate is
    /// that the behavior of reduce may be non-deterministic for
    /// non-associative or non-commutative binary predicate.
    ///
    template <typename ExPolicy, typename Rng, typename T, typename F>
    typename util::detail::algorithm_result<ExPolicy, T>::type
    reduce(ExPolicy&& policy, Rng&& rng, T init, F&& f);

    /// Returns GENERALIZED_SUM(+, init, *first, ..., *(first + (last - first) - 1)).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         operator+().
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam T           The type of the value to be used as initial (and
    ///                     intermediate) values (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param init         The initial value for the generalized sum.
    ///
    /// The reduce operations in the parallel \a reduce algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a copy_if algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a reduce algorithm returns a \a hpx::future<T> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a T otherwise.
    ///           The \a reduce algorithm returns the result of the
    ///           generalized sum (applying operator+()) over the elements given
    ///           by the input range [first, last).
    ///
    /// \note   GENERALIZED_SUM(+, a1, ..., aN) is defined as follows:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_SUM(+, b1, ..., bK), GENERALIZED_SUM(+, bM, ..., bN)),
    ///           where:
    ///           * b1, ..., bN may be any permutation of a1, ..., aN and
    ///           * 1 < K+1 = M <= N.
    ///
    /// The difference between \a reduce and \a accumulate is
    /// that the behavior of reduce may be non-deterministic for
    /// non-associative or non-commutative binary predicate.
    ///
    template <typename ExPolicy, typename Rng, typename T>
    typename util::detail::algorithm_result<ExPolicy, T>::type
    reduce(ExPolicy&& policy, Rng&& rng, T init);

    /// Returns GENERALIZED_SUM(+, T(), *first, ..., *(first + (last - first) - 1)).
    ///
    /// \note   Complexity: O(\a last - \a first) applications of the
    ///         operator+().
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
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    ///
    /// The reduce operations in the parallel \a reduce algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The reduce operations in the parallel \a copy_if algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a reduce algorithm returns a \a hpx::future<T> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns T otherwise (where T is the value_type of
    ///           \a FwdIterB).
    ///           The \a reduce algorithm returns the result of the
    ///           generalized sum (applying operator+()) over the elements given
    ///           by the input range [first, last).
    ///
    /// \note   The type of the initial value (and the result type) \a T is
    ///         determined from the value_type of the used \a FwdIterB.
    ///
    /// \note   GENERALIZED_SUM(+, a1, ..., aN) is defined as follows:
    ///         * a1 when N is 1
    ///         * op(GENERALIZED_SUM(+, b1, ..., bK), GENERALIZED_SUM(+, bM, ..., bN)),
    ///           where:
    ///           * b1, ..., bN may be any permutation of a1, ..., aN and
    ///           * 1 < K+1 = M <= N.
    ///
    /// The difference between \a reduce and \a accumulate is
    /// that the behavior of reduce may be non-deterministic for
    /// non-associative or non-commutative binary predicate.
    ///
    template <typename ExPolicy, typename Rng>
    typename util::detail::algorithm_result<ExPolicy,
        typename std::iterator_traits<
            typename hpx::traits::range_traits<Rng>::iterator_type
        >::value_type
    >::type
    reduce(ExPolicy&& policy, Rng&& rng);

    // clang-format on
}}    // namespace hpx::ranges

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/iterator_support/traits/is_sentinel_for.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/reduce.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace ranges {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::reduce
    HPX_INLINE_CONSTEXPR_VARIABLE struct reduce_t final
      : hpx::functional::tag<reduce_t>
    {
        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename T, typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            T>::type
        tag_invoke(hpx::ranges::reduce_t, ExPolicy&& policy, FwdIter first,
            Sent last, T init, F&& f)
        {
            using is_segmented = hpx::traits::is_segmented_iterator<FwdIter>;

            return hpx::parallel::v1::detail::reduce_(
                std::forward<ExPolicy>(policy), first, last, std::move(init),
                std::forward<F>(f), is_segmented{});
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename T, typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            T>::type
        tag_invoke(
            hpx::ranges::reduce_t, ExPolicy&& policy, Rng&& rng, T init, F&& f)
        {
            using is_segmented = hpx::traits::is_segmented_iterator<
                typename hpx::traits::range_traits<Rng>::iterator_type>;

            return hpx::parallel::v1::detail::reduce_(
                std::forward<ExPolicy>(policy), hpx::util::begin(rng),
                hpx::util::end(rng), std::move(init), std::forward<F>(f),
                is_segmented{});
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            T>::type
        tag_invoke(hpx::ranges::reduce_t, ExPolicy&& policy, FwdIter first,
            Sent last, T init)
        {
            using is_segmented = hpx::traits::is_segmented_iterator<FwdIter>;

            return hpx::parallel::v1::detail::reduce_(
                std::forward<ExPolicy>(policy), first, last, std::move(init),
                std::plus<T>{}, is_segmented{});
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            T>::type
        tag_invoke(hpx::ranges::reduce_t, ExPolicy&& policy, Rng&& rng, T init)
        {
            using is_segmented = hpx::traits::is_segmented_iterator<
                typename hpx::traits::range_traits<Rng>::iterator_type>;

            return hpx::parallel::v1::detail::reduce_(
                std::forward<ExPolicy>(policy), hpx::util::begin(rng),
                hpx::util::end(rng), std::move(init), std::plus<T>{},
                is_segmented{});
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<FwdIter>::value_type>::type
        tag_invoke(
            hpx::ranges::reduce_t, ExPolicy&& policy, FwdIter first, Sent last)
        {
            using value_type =
                typename std::iterator_traits<FwdIter>::value_type;

            using is_segmented = hpx::traits::is_segmented_iterator<FwdIter>;

            return hpx::parallel::v1::detail::reduce_(
                std::forward<ExPolicy>(policy), first, last, value_type{},
                std::plus<value_type>{}, is_segmented{});
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            typename std::iterator_traits<typename hpx::traits::range_traits<
                Rng>::iterator_type>::value_type>::type
        tag_invoke(hpx::ranges::reduce_t, ExPolicy&& policy, Rng&& rng)
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;
            using value_type =
                typename std::iterator_traits<iterator_type>::value_type;
            using is_segmented =
                hpx::traits::is_segmented_iterator<iterator_type>;

            return hpx::parallel::v1::detail::reduce_(
                std::forward<ExPolicy>(policy), hpx::util::begin(rng),
                hpx::util::end(rng), value_type{}, std::plus<value_type>{},
                is_segmented{});
        }

        ////////////////////////////////////////////////////////////////////////
        // clang-format off
        template <typename FwdIter, typename Sent, typename T, typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value
            )>
        // clang-format on
        friend T tag_invoke(
            hpx::ranges::reduce_t, FwdIter first, Sent last, T init, F&& f)
        {
            using is_segmented = hpx::traits::is_segmented_iterator<FwdIter>;

            return hpx::parallel::v1::detail::reduce_(
                hpx::parallel::execution::seq, first, last, std::move(init),
                std::forward<F>(f), is_segmented{});
        }

        // clang-format off
        template <typename Rng, typename T, typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend T tag_invoke(hpx::ranges::reduce_t, Rng&& rng, T init, F&& f)
        {
            using is_segmented = hpx::traits::is_segmented_iterator<
                typename hpx::traits::range_traits<Rng>::iterator_type>;

            return hpx::parallel::v1::detail::reduce_(
                hpx::parallel::execution::seq, hpx::util::begin(rng),
                hpx::util::end(rng), std::move(init), std::forward<F>(f),
                is_segmented{});
        }

        // clang-format off
        template <typename FwdIter, typename Sent,
            typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value
            )>
        // clang-format on
        friend T tag_invoke(
            hpx::ranges::reduce_t, FwdIter first, Sent last, T init)
        {
            using is_segmented = hpx::traits::is_segmented_iterator<FwdIter>;

            return hpx::parallel::v1::detail::reduce_(
                hpx::parallel::execution::seq, first, last, std::move(init),
                std::plus<T>{}, is_segmented{});
        }

        // clang-format off
        template <typename Rng, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend T tag_invoke(hpx::ranges::reduce_t, Rng&& rng, T init)
        {
            using is_segmented = hpx::traits::is_segmented_iterator<
                typename hpx::traits::range_traits<Rng>::iterator_type>;

            return hpx::parallel::v1::detail::reduce_(
                hpx::parallel::execution::seq, hpx::util::begin(rng),
                hpx::util::end(rng), std::move(init), std::plus<T>{},
                is_segmented{});
        }

        // clang-format off
        template <typename FwdIter, typename Sent,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value
            )>
        // clang-format on
        friend typename std::iterator_traits<FwdIter>::value_type tag_invoke(
            hpx::ranges::reduce_t, FwdIter first, Sent last)
        {
            using value_type =
                typename std::iterator_traits<FwdIter>::value_type;

            using is_segmented = hpx::traits::is_segmented_iterator<FwdIter>;

            return hpx::parallel::v1::detail::reduce_(
                hpx::parallel::execution::seq, first, last, value_type{},
                std::plus<value_type>{}, is_segmented{});
        }

        // clang-format off
        template <typename Rng,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend typename std::iterator_traits<
            typename hpx::traits::range_traits<Rng>::iterator_type>::value_type
        tag_invoke(hpx::ranges::reduce_t, Rng&& rng)
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;
            using value_type =
                typename std::iterator_traits<iterator_type>::value_type;
            using is_segmented =
                hpx::traits::is_segmented_iterator<iterator_type>;

            return hpx::parallel::v1::detail::reduce_(
                hpx::parallel::execution::seq, hpx::util::begin(rng),
                hpx::util::end(rng), value_type{}, std::plus<value_type>{},
                is_segmented{});
        }
    } reduce{};
}}    // namespace hpx::ranges

#endif    // DOXYGEN
