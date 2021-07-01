//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/exclusive_scan.hpp

#pragma once

#if defined(DOXYGEN)

namespace hpx { namespace ranges {

    /// Checks if the first range [first1, last1) is lexicographically less than
    /// the second range [first2, last2). uses a provided predicate to compare
    /// elements.
    ///
    /// \note   Complexity: At most 2 * min(N1, N2) applications of the comparison
    ///         operation, where N1 = std::distance(first1, last)
    ///         and N2 = std::distance(first2, last2).
    ///
    /// \tparam InIter1     The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Sent1       The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter1.
    /// \tparam InIter2     The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Sent2       The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter2.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a exclusive_scan requires \a Pred to
    ///                     meet the requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    /// \tparam Proj1       The type of an optional projection function for FwdIter1. This
    ///                     defaults to \a util::projection_identity
    /// \tparam Proj2       The type of an optional projection function for FwdIter2. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last1        Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the sequence of elements
    ///                     of the second range the algorithm will be applied to.
    /// \param last2        Refers to the end of the sequence of elements of
    ///                     the second range the algorithm will be applied to.
    /// \param pred         Refers to the comparison function that the first
    ///                     and second ranges will be applied to
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the first range
    ///                     as a projection operation before the actual predicate
    ///                     \a is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the second range
    ///                     as a projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparison operations in the parallel \a exclusive_scan
    /// algorithm invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The comparison operations in the parallel \a exclusive_scan
    /// algorithm invoked with an execution policy object of type
    /// \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note     Lexicographical comparison is an operation with the
    ///           following properties
    ///             - Two ranges are compared element by element
    ///             - The first mismatching element defines which range
    ///               is lexicographically
    ///               \a less or \a greater than the other
    ///             - If one range is a prefix of another, the shorter range is
    ///               lexicographically \a less than the other
    ///             - If two ranges have equivalent elements and are of the same length,
    ///               then the ranges are lexicographically \a equal
    ///             - An empty range is lexicographically \a less than any non-empty
    ///               range
    ///             - Two empty ranges are lexicographically \a equal
    ///
    /// \returns  The \a lexicographically_compare algorithm returns a
    ///           \a hpx::future<bool> if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a bool otherwise.
    ///           The \a lexicographically_compare algorithm returns true
    ///           if the first range is lexicographically less, otherwise
    ///           it returns false.
    ///           range [first2, last2), it returns false.
    template <typename InIter1, typename Sent1, typename InIter2,
        typename Sent2,
        typename Proj1 = hpx::parallel::util::projection_identity,
        typename Proj2 = hpx::parallel::util::projection_identity,
        typename Pred = detail::less>
    bool exclusive_scan(InIter1 first1, Sent1 last1, InIter2 first2,
        Sent2 last2, Pred&& pred = Pred(), Proj1&& proj1 = Proj1(),
        Proj2&& proj2 = Proj2());

    /// Checks if the first range [first1, last1) is lexicographically less than
    /// the second range [first2, last2). uses a provided predicate to compare
    /// elements.
    ///
    /// \note   Complexity: At most 2 * min(N1, N2) applications of the comparison
    ///         operation, where N1 = std::distance(first1, last)
    ///         and N2 = std::distance(first2, last2).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent1       The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter1.
    /// \tparam FwdIter2    The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent2       The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter2.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a exclusive_scan requires \a Pred to
    ///                     meet the requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    /// \tparam Proj1       The type of an optional projection function for FwdIter1. This
    ///                     defaults to \a util::projection_identity
    /// \tparam Proj2       The type of an optional projection function for FwdIter2. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last1        Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the sequence of elements
    ///                     of the second range the algorithm will be applied to.
    /// \param last2        Refers to the end of the sequence of elements of
    ///                     the second range the algorithm will be applied to.
    /// \param pred         Refers to the comparison function that the first
    ///                     and second ranges will be applied to
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the first range
    ///                     as a projection operation before the actual predicate
    ///                     \a is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the second range
    ///                     as a projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparison operations in the parallel \a exclusive_scan
    /// algorithm invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The comparison operations in the parallel \a exclusive_scan
    /// algorithm invoked with an execution policy object of type
    /// \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note     Lexicographical comparison is an operation with the
    ///           following properties
    ///             - Two ranges are compared element by element
    ///             - The first mismatching element defines which range
    ///               is lexicographically
    ///               \a less or \a greater than the other
    ///             - If one range is a prefix of another, the shorter range is
    ///               lexicographically \a less than the other
    ///             - If two ranges have equivalent elements and are of the same length,
    ///               then the ranges are lexicographically \a equal
    ///             - An empty range is lexicographically \a less than any non-empty
    ///               range
    ///             - Two empty ranges are lexicographically \a equal
    ///
    /// \returns  The \a lexicographically_compare algorithm returns a
    ///           \a hpx::future<bool> if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a bool otherwise.
    ///           The \a lexicographically_compare algorithm returns true
    ///           if the first range is lexicographically less, otherwise
    ///           it returns false.
    ///           range [first2, last2), it returns false.

    template <typename ExPolicy, typename FwdIter1, typename Sent1,
        typename FwdIter2, typename Sent2,
        typename Proj1 = hpx::parallel::util::projection_identity,
        typename Proj2 = hpx::parallel::util::projection_identity,
        typename Pred = detail::less>
    typename parallel::util::detail::algorithm_result<ExPolicy, bool>::type
    exclusive_scan(ExPolicy&& policy, FwdIter1 first1, Sent1 last1,
        FwdIter2 first2, Sent2 last2, Pred&& pred = Pred(),
        Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2());

    /// Checks if the first range rng1 is lexicographically less than
    /// the second range rng2. uses a provided predicate to compare
    /// elements.
    ///
    /// \note   Complexity: At most 2 * min(N1, N2) applications of the comparison
    ///         operation, where N1 = std::distance(std::begin(rng1), std::end(rng1))
    ///         and N2 = std::distance(std::begin(rng2), std::end(rng2)).
    ///
    /// \tparam Rng1        The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Rng2        The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a exclusive_scan requires \a Pred to
    ///                     meet the requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    /// \tparam Proj1       The type of an optional projection function for elements of the first range.
    ///                     This defaults to \a util::projection_identity
    /// \tparam Proj2       The type of an optional projection function for elements of the second range.
    ///                     This defaults to \a util::projection_identity
    ///
    /// \param rng1         Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param rng2         Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param pred         Refers to the comparison function that the first
    ///                     and second ranges will be applied to
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the first range
    ///                     as a projection operation before the actual predicate
    ///                     \a is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the second range
    ///                     as a projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparison operations in the parallel \a exclusive_scan
    /// algorithm invoked without an execution policy object  execute in sequential
    /// order in the calling thread.
    ///
    /// \note     Lexicographical comparison is an operation with the
    ///           following properties
    ///             - Two ranges are compared element by element
    ///             - The first mismatching element defines which range
    ///               is lexicographically
    ///               \a less or \a greater than the other
    ///             - If one range is a prefix of another, the shorter range is
    ///               lexicographically \a less than the other
    ///             - If two ranges have equivalent elements and are of the same length,
    ///               then the ranges are lexicographically \a equal
    ///             - An empty range is lexicographically \a less than any non-empty
    ///               range
    ///             - Two empty ranges are lexicographically \a equal
    ///
    /// \returns  The \a lexicographically_compare algorithm returns \a bool.
    ///           The \a lexicographically_compare algorithm returns true
    ///           if the first range is lexicographically less, otherwise
    ///           it returns false.
    ///           range [first2, last2), it returns false.
    template <typename Rng1, typename Rng2,
        typename Proj1 = hpx::parallel::util::projection_identity,
        typename Proj2 = hpx::parallel::util::projection_identity,
        typename Pred = detail::less>
    bool exclusive_scan(Rng1&& rng1, Rng2&& rng2, Pred&& pred = Pred(),
        Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2());

    /// Checks if the first range rng1 is lexicographically less than
    /// the second range rng2. uses a provided predicate to compare
    /// elements.
    ///
    /// \note   Complexity: At most 2 * min(N1, N2) applications of the comparison
    ///         operation, where N1 = std::distance(std::begin(rng1), std::end(rng1))
    ///         and N2 = std::distance(std::begin(rng2), std::end(rng2)).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng1        The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Rng2        The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a exclusive_scan requires \a Pred to
    ///                     meet the requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    /// \tparam Proj1       The type of an optional projection function for elements of the first range.
    ///                     This defaults to \a util::projection_identity
    /// \tparam Proj2       The type of an optional projection function for elements of the second range.
    ///                     This defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng1         Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param rng2         Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param pred         Refers to the comparison function that the first
    ///                     and second ranges will be applied to
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the first range
    ///                     as a projection operation before the actual predicate
    ///                     \a is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the second range
    ///                     as a projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The comparison operations in the parallel \a exclusive_scan
    /// algorithm invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The comparison operations in the parallel \a exclusive_scan
    /// algorithm invoked with an execution policy object of type
    /// \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \note     Lexicographical comparison is an operation with the
    ///           following properties
    ///             - Two ranges are compared element by element
    ///             - The first mismatching element defines which range
    ///               is lexicographically
    ///               \a less or \a greater than the other
    ///             - If one range is a prefix of another, the shorter range is
    ///               lexicographically \a less than the other
    ///             - If two ranges have equivalent elements and are of the same length,
    ///               then the ranges are lexicographically \a equal
    ///             - An empty range is lexicographically \a less than any non-empty
    ///               range
    ///             - Two empty ranges are lexicographically \a equal
    ///
    /// \returns  The \a lexicographically_compare algorithm returns a
    ///           \a hpx::future<bool> if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a bool otherwise.
    ///           The \a lexicographically_compare algorithm returns true
    ///           if the first range is lexicographically less, otherwise
    ///           it returns false.
    ///           range [first2, last2), it returns false.

    template <typename ExPolicy, typename Rng1, typename Rng2,
        typename Proj1 = hpx::parallel::util::projection_identity,
        typename Proj2 = hpx::parallel::util::projection_identity,
        typename Pred = detail::less>
    typename parallel::util::detail::algorithm_result<ExPolicy, bool>::type
    exclusive_scan(ExPolicy&& policy, Rng1&& rng1, Rng2&& rng2,
        Pred&& pred = Pred(), Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2());
}}    // namespace hpx::ranges
#else

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/algorithms/exclusive_scan.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace ranges {
    HPX_INLINE_CONSTEXPR_VARIABLE struct exclusive_scan_t final
      : hpx::functional::tag_fallback<exclusive_scan_t>
    {
    private:
        // clang-format off
        template <typename InIter, typename Sent, typename OutIter,
             typename T, typename Op = std::plus<T>,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<InIter>::value &&
                hpx::traits::is_sentinel_for<Sent, InIter>::value &&
                hpx::traits::is_iterator<OutIter>::value
            )>
        // clang-format on
        friend OutIter tag_fallback_dispatch(hpx::ranges::exclusive_scan_t,
            InIter first, Sent last, OutIter dest, T init, Op&& op = Op())
        {
            static_assert(hpx::traits::is_input_iterator<InIter>::value,
                "Requires at least input iterator.");
            static_assert(hpx::traits::is_output_iterator<OutIter>::value,
                "Requires at least output iterator.");

            return hpx::parallel::v1::detail::exclusive_scan<OutIter>().call(
                hpx::execution::seq, first, last, dest, std::move(init),
                std::forward<Op>(op));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename Sent,
            typename FwdIter2, typename T, typename Op = std::plus<T>,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter2>::type
        tag_fallback_dispatch(hpx::ranges::exclusive_scan_t, ExPolicy&& policy,
            FwdIter1 first, Sent last, FwdIter2 dest, T init, Op&& op = Op())
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter1>::value,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator<FwdIter2>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::exclusive_scan<FwdIter2>().call(
                std::forward<ExPolicy>(policy), first, last, dest,
                std::move(init), std::forward<Op>(op));
        }

        // clang-format off
        template <typename Rng, typename O, typename T,
            typename Op = std::plus<T>,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend O tag_fallback_dispatch(hpx::ranges::exclusive_scan_t, Rng&& rng,
            O dest, T init, Op&& op = Op())
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            static_assert(hpx::traits::is_input_iterator<iterator_type>::value,
                "Requires at least input iterator.");

            return hpx::parallel::v1::detail::exclusive_scan<O>().call(
                hpx::execution::seq, std::begin(rng), std::end(rng), dest,
                std::move(init), std::forward<Op>(op));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,  typename O, typename T,
            typename Op = std::plus<T>,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend
            typename parallel::util::detail::algorithm_result<ExPolicy, O>::type
            tag_fallback_dispatch(hpx::ranges::exclusive_scan_t,
                ExPolicy&& policy, Rng&& rng, O dest, T init, Op&& op = Op())
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            static_assert(
                hpx::traits::is_forward_iterator<iterator_type>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::exclusive_scan<O>().call(
                std::forward<ExPolicy>(policy), std::begin(rng), std::end(rng),
                dest, std::move(init), std::forward<Op>(op));
        }
    } exclusive_scan{};
}}    // namespace hpx::ranges

#endif
