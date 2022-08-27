//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if defined(DOXYGEN)

namespace hpx { namespace ranges {
    /// Determines if the range [first, last) is sorted. Uses pred to
    /// compare elements.
    ///
    /// \note   Complexity: at most (N+S-1) comparisons where
    ///         \a N = distance(first, last).
    ///         \a S = number of partitions
    ///
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of that the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     that the algorithm will be applied to.
    /// \param pred         Refers to the binary predicate which returns true
    ///                     if the first argument should be treated as less than
    ///                     the second argument. The signature of the function
    ///                     should be equivalent to
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that objects of
    ///                     types \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparison operations in the parallel \a is_sorted algorithm
    /// executes in sequential order in the calling thread.
    ///
    /// \returns  The \a is_sorted algorithm returns a \a bool.
    ///           The \a is_sorted algorithm returns true if each element in
    ///           the sequence [first, last) satisfies the predicate passed.
    ///           If the range [first, last) contains less than two elements,
    ///           the function always returns true.
    ///
    template <typename FwdIter, typename Sent,
        typename Pred = hpx::parallel::v1::detail::less,
        typename Proj = hpx::parallel::util::projection_identity>
    bool is_sorted(
        FwdIter first, Sent last, Pred&& pred = Pred(), Proj&& proj = Proj());

    /// Determines if the range [first, last) is sorted. Uses pred to
    /// compare elements.
    ///
    /// \note   Complexity: at most (N+S-1) comparisons where
    ///         \a N = distance(first, last).
    ///         \a S = number of partitions
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a is_sorted requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of that the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     that the algorithm will be applied to.
    /// \param pred         Refers to the binary predicate which returns true
    ///                     if the first argument should be treated as less than
    ///                     the second argument. The signature of the function
    ///                     should be equivalent to
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that objects of
    ///                     types \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparison operations in the parallel \a is_sorted algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// executes in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a is_sorted algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a is_sorted algorithm returns a \a hpx::future<bool>
    ///           if the execution policy is of type \a task_execution_policy
    ///           and returns \a bool otherwise.
    ///           The \a is_sorted algorithm returns a bool if each element in
    ///           the sequence [first, last) satisfies the predicate passed.
    ///           If the range [first, last) contains less than two elements,
    ///           the function always returns true.
    ///
    template <typename ExPolicy, typename FwdIter, typename Sent,
        typename Pred = hpx::parallel::v1::detail::less,
        typename Proj = hpx::parallel::util::projection_identity>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy, bool>::type
    is_sorted(ExPolicy&& policy, FwdIter first, Sent last, Pred&& pred = Pred(),
        Proj&& proj = Proj());

    /// Determines if the range rng is sorted. Uses pred to
    /// compare elements.
    ///
    /// \note   Complexity: at most (N+S-1) comparisons where
    ///         \a N = size(rng).
    ///         \a S = number of partitions
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param pred         Refers to the binary predicate which returns true
    ///                     if the first argument should be treated as less than
    ///                     the second argument. The signature of the function
    ///                     should be equivalent to
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that objects of
    ///                     types \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparison operations in the parallel \a is_sorted algorithm
    /// executes in sequential order in the calling thread.
    ///
    /// \returns  The \a is_sorted algorithm returns a \a bool.
    ///           The \a is_sorted algorithm returns true if each element in
    ///           the rng satisfies the predicate passed.
    ///           If the range rng contains less than two elements,
    ///           the function always returns true.
    ///
    template <typename Rng, typename Pred = hpx::parallel::v1::detail::less,
        typename Proj = hpx::parallel::util::projection_identity>
    bool is_sorted(ExPolicy&& policy, Rng&& rng, Pred&& pred = Pred(),
        Proj&& proj = Proj());

    /// Determines if the range rng is sorted. Uses pred to
    /// compare elements.
    ///
    /// \note   Complexity: at most (N+S-1) comparisons where
    ///         \a N = size(rng).
    ///         \a S = number of partitions
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a is_sorted requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param pred         Refers to the binary predicate which returns true
    ///                     if the first argument should be treated as less than
    ///                     the second argument. The signature of the function
    ///                     should be equivalent to
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that objects of
    ///                     types \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparison operations in the parallel \a is_sorted algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// executes in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a is_sorted algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a is_sorted algorithm returns a \a hpx::future<bool>
    ///           if the execution policy is of type \a task_execution_policy
    ///           and returns \a bool otherwise.
    ///           The \a is_sorted algorithm returns a bool if each element in
    ///           the range rng satisfies the predicate passed.
    ///           If the range rng contains less than two elements,
    ///           the function always returns true.
    ///
    template <typename ExPolicy, typename Rng,
        typename Pred = hpx::parallel::v1::detail::less,
        typename Proj = hpx::parallel::util::projection_identity>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy, bool>::type
    is_sorted(ExPolicy&& policy, Rng&& rng, Pred&& pred = Pred(),
        Proj&& proj = Proj());

    /// Returns the first element in the range [first, last) that is not sorted.
    /// Uses a predicate to compare elements or the less than operator.
    ///
    /// \note   Complexity: at most (N+S-1) comparisons where
    ///         \a N = distance(first, last).
    ///         \a S = number of partitions
    ///
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of that the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     that the algorithm will be applied to.
    /// \param pred         Refers to the binary predicate which returns true
    ///                     if the first argument should be treated as less than
    ///                     the second argument. The signature of the function
    ///                     should be equivalent to
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that objects of
    ///                     types \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparison operations in the parallel \a is_sorted_until algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a is_sorted_until algorithm returns a \a FwdIter.
    ///           The \a is_sorted_until algorithm returns the first unsorted
    ///           element. If the sequence has less than two elements or the
    ///           sequence is sorted, last is returned.
    ///
    template <typename FwdIter, typename Sent,
        typename Pred = hpx::parallel::v1::detail::less,
        typename Proj = hpx::parallel::util::projection_identity>
    FwdIter is_sorted_until(
        FwdIter first, Sent last, Pred&& pred = Pred(), Proj&& proj = Proj());

    /// Returns the first element in the range [first, last) that is not sorted.
    /// Uses a predicate to compare elements or the less than operator.
    ///
    /// \note   Complexity: at most (N+S-1) comparisons where
    ///         \a N = distance(first, last).
    ///         \a S = number of partitions
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a is_sorted_until requires \a Pred to meet
    ///                     the requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of that the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     that the algorithm will be applied to.
    /// \param pred         Refers to the binary predicate which returns true
    ///                     if the first argument should be treated as less than
    ///                     the second argument. The signature of the function
    ///                     should be equivalent to
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that objects of
    ///                     types \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparison operations in the parallel \a is_sorted_until algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy executes in sequential order in the
    /// calling thread.
    ///
    /// The comparison operations in the parallel \a is_sorted_until algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified threads,
    /// and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a is_sorted_until algorithm returns a \a hpx::future<FwdIter>
    ///           if the execution policy is of type \a task_execution_policy
    ///           and returns \a FwdIter otherwise.
    ///           The \a is_sorted_until algorithm returns the first unsorted
    ///           element. If the sequence has less than two elements or the
    ///           sequence is sorted, last is returned.
    ///
    template <typename ExPolicy, typename FwdIter, typename Sent,
        typename Pred = hpx::parallel::v1::detail::less,
        typename Proj = hpx::parallel::util::projection_identity>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        FwdIter>::type
    is_sorted_until(ExPolicy&& policy, FwdIter first, Sent last,
        Pred&& pred = Pred(), Proj&& proj = Proj());

    /// Returns the first element in the range rng that is not sorted.
    /// Uses a predicate to compare elements or the less than operator.
    ///
    /// \note   Complexity: at most (N+S-1) comparisons where
    ///         \a N = size(rng).
    ///         \a S = number of partitions
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param pred         Refers to the binary predicate which returns true
    ///                     if the first argument should be treated as less than
    ///                     the second argument. The signature of the function
    ///                     should be equivalent to
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that objects of
    ///                     types \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparison operations in the parallel \a is_sorted_until algorithm
    /// execute in sequential order in the calling thread.
    ///
    /// \returns  The \a is_sorted_until algorithm returns a \a FwdIter.
    ///           The \a is_sorted_until algorithm returns the first unsorted
    ///           element. If the sequence has less than two elements or the
    ///           sequence is sorted, last is returned.
    ///
    template <typename Rng, typename Pred = hpx::parallel::v1::detail::less,
        typename Proj = hpx::parallel::util::projection_identity>
    typename hpx::traits::range_iterator<Rng>::type is_sorted_until(
        ExPolicy&& policy, Rng&& rng, Pred&& pred = Pred(),
        Proj&& proj = Proj());

    /// Returns the first element in the range rng that is not sorted.
    /// Uses a predicate to compare elements or the less than operator.
    ///
    /// \note   Complexity: at most (N+S-1) comparisons where
    ///         \a N = size(rng).
    ///         \a S = number of partitions
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a is_sorted_until requires \a Pred to meet
    ///                     the requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param pred         Refers to the binary predicate which returns true
    ///                     if the first argument should be treated as less than
    ///                     the second argument. The signature of the function
    ///                     should be equivalent to
    ///                     \code
    ///                     bool pred(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that objects of
    ///                     types \a FwdIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparison operations in the parallel \a is_sorted_until algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy executes in sequential order in the
    /// calling thread.
    ///
    /// The comparison operations in the parallel \a is_sorted_until algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified threads,
    /// and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a is_sorted_until algorithm returns a \a hpx::future<FwdIter>
    ///           if the execution policy is of type \a task_execution_policy
    ///           and returns \a FwdIter otherwise.
    ///           The \a is_sorted_until algorithm returns the first unsorted
    ///           element. If the sequence has less than two elements or the
    ///           sequence is sorted, last is returned.
    ///
    template <typename ExPolicy, typename Rng,
        typename Pred = hpx::parallel::v1::detail::less,
        typename Proj = hpx::parallel::util::projection_identity>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        typename hpx::traits::range_iterator<Rng>::type>::type
    is_sorted_until(ExPolicy&& policy, Rng&& rng, Pred&& pred = Pred(),
        Proj&& proj = Proj());
}}    // namespace hpx::ranges

#else

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>

#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/is_sorted.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace ranges {
    inline constexpr struct is_sorted_t final
      : hpx::detail::tag_parallel_algorithm<is_sorted_t>
    {
    private:
        template <typename FwdIter, typename Sent,
            typename Pred = hpx::parallel::v1::detail::less,
            typename Proj = hpx::parallel::util::projection_identity,
            // clang-format off
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_forward_iterator<FwdIter>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value &&
                hpx::parallel::traits::is_projected<Proj, FwdIter>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected<Proj, FwdIter>,
                    hpx::parallel::traits::projected<Proj, FwdIter>
                >::value
            )>
        // clang-format on
        friend bool tag_fallback_invoke(hpx::ranges::is_sorted_t, FwdIter first,
            Sent last, Pred&& pred = Pred(), Proj&& proj = Proj())
        {
            return hpx::parallel::v1::detail::is_sorted<FwdIter, Sent>().call(
                hpx::execution::seq, first, last, HPX_FORWARD(Pred, pred),
                HPX_FORWARD(Proj, proj));
        }

        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename Pred = hpx::parallel::v1::detail::less,
            typename Proj = hpx::parallel::util::projection_identity,
            // clang-format off
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_forward_iterator<FwdIter>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value &&
                hpx::parallel::traits::is_projected<Proj, FwdIter>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected<Proj, FwdIter>,
                    hpx::parallel::traits::projected<Proj, FwdIter>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_invoke(hpx::ranges::is_sorted_t, ExPolicy&& policy,
            FwdIter first, Sent last, Pred&& pred = Pred(),
            Proj&& proj = Proj())
        {
            return hpx::parallel::v1::detail::is_sorted<FwdIter, Sent>().call(
                HPX_FORWARD(ExPolicy, policy), first, last,
                HPX_FORWARD(Pred, pred), HPX_FORWARD(Proj, proj));
        }

        template <typename Rng, typename Pred = hpx::parallel::v1::detail::less,
            typename Proj = hpx::parallel::util::projection_identity,
            // clang-format off
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected_range<Proj, Rng>,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend bool tag_fallback_invoke(hpx::ranges::is_sorted_t, Rng&& rng,
            Pred&& pred = Pred(), Proj&& proj = Proj())
        {
            return hpx::parallel::v1::detail::is_sorted<
                typename hpx::traits::range_iterator<Rng>::type,
                typename hpx::traits::range_iterator<Rng>::type>()
                .call(hpx::execution::seq, hpx::util::begin(rng),
                    hpx::util::end(rng), HPX_FORWARD(Pred, pred),
                    HPX_FORWARD(Proj, proj));
        }

        template <typename ExPolicy, typename Rng,
            typename Pred = hpx::parallel::v1::detail::less,
            typename Proj = hpx::parallel::util::projection_identity,
            // clang-format off
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected_range<Proj, Rng>,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_invoke(hpx::ranges::is_sorted_t, ExPolicy&& policy,
            Rng&& rng, Pred&& pred = Pred(), Proj&& proj = Proj())
        {
            return hpx::parallel::v1::detail::is_sorted<
                typename hpx::traits::range_iterator<Rng>::type,
                typename hpx::traits::range_iterator<Rng>::type>()
                .call(HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
                    hpx::util::end(rng), HPX_FORWARD(Pred, pred),
                    HPX_FORWARD(Proj, proj));
        }
    } is_sorted{};

    inline constexpr struct is_sorted_until_t final
      : hpx::detail::tag_parallel_algorithm<is_sorted_until_t>
    {
    private:
        template <typename FwdIter, typename Sent,
            typename Pred = hpx::parallel::v1::detail::less,
            typename Proj = hpx::parallel::util::projection_identity,
            // clang-format off
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_forward_iterator<FwdIter>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value &&
                hpx::parallel::traits::is_projected<Proj, FwdIter>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected<Proj, FwdIter>,
                    hpx::parallel::traits::projected<Proj, FwdIter>
                >::value
            )>
        // clang-format on
        friend FwdIter tag_fallback_invoke(hpx::ranges::is_sorted_until_t,
            FwdIter first, Sent last, Pred&& pred = Pred(),
            Proj&& proj = Proj())
        {
            return hpx::parallel::v1::detail::is_sorted_until<FwdIter, Sent>()
                .call(hpx::execution::seq, first, last, HPX_FORWARD(Pred, pred),
                    HPX_FORWARD(Proj, proj));
        }

        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename Pred = hpx::parallel::v1::detail::less,
            typename Proj = hpx::parallel::util::projection_identity,
            // clang-format off
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_forward_iterator<FwdIter>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value &&
                hpx::parallel::traits::is_projected<Proj, FwdIter>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected<Proj, FwdIter>,
                    hpx::parallel::traits::projected<Proj, FwdIter>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_invoke(hpx::ranges::is_sorted_until_t, ExPolicy&& policy,
            FwdIter first, Sent last, Pred&& pred = Pred(),
            Proj&& proj = Proj())
        {
            return hpx::parallel::v1::detail::is_sorted_until<FwdIter, Sent>()
                .call(HPX_FORWARD(ExPolicy, policy), first, last,
                    HPX_FORWARD(Pred, pred), HPX_FORWARD(Proj, proj));
        }

        template <typename Rng, typename Pred = hpx::parallel::v1::detail::less,
            typename Proj = hpx::parallel::util::projection_identity,
            // clang-format off
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected_range<Proj, Rng>,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend typename hpx::traits::range_iterator<Rng>::type
        tag_fallback_invoke(hpx::ranges::is_sorted_until_t, Rng&& rng,
            Pred&& pred = Pred(), Proj&& proj = Proj())
        {
            return hpx::parallel::v1::detail::is_sorted_until<
                typename hpx::traits::range_iterator<Rng>::type,
                typename hpx::traits::range_iterator<Rng>::type>()
                .call(hpx::execution::seq, hpx::util::begin(rng),
                    hpx::util::end(rng), HPX_FORWARD(Pred, pred),
                    HPX_FORWARD(Proj, proj));
        }

        template <typename ExPolicy, typename Rng,
            typename Pred = hpx::parallel::v1::detail::less,
            typename Proj = hpx::parallel::util::projection_identity,
            // clang-format off
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected_range<Proj, Rng>,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            typename hpx::traits::range_iterator<Rng>::type>::type
        tag_fallback_invoke(hpx::ranges::is_sorted_until_t, ExPolicy&& policy,
            Rng&& rng, Pred&& pred = Pred(), Proj&& proj = Proj())
        {
            return hpx::parallel::v1::detail::is_sorted_until<
                typename hpx::traits::range_iterator<Rng>::type,
                typename hpx::traits::range_iterator<Rng>::type>()
                .call(HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
                    hpx::util::end(rng), HPX_FORWARD(Pred, pred),
                    HPX_FORWARD(Proj, proj));
        }
    } is_sorted_until{};
}}    // namespace hpx::ranges

#endif
