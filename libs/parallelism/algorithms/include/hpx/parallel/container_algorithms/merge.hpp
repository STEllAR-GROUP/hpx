//  Copyright (c) 2017 Taeguk Kwon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/merge.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx { namespace ranges {
    // clang-format off

    /// Merges two sorted ranges [first1, last1) and [first2, last2)
    /// into one sorted range beginning at \a dest. The order of
    /// equivalent elements in the each of original two ranges is preserved.
    /// For equivalent elements in the original two ranges, the elements from
    /// the first range precede the elements from the second range.
    /// The destination range cannot overlap with either of the input ranges.
    ///
    /// \note   Complexity: Performs
    ///         O(std::distance(first1, last1) + std::distance(first2, last2))
    ///         applications of the comparison \a comp and the each projection.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Iter1       The type of the source iterators used (deduced)
    ///                     representing the first sequence.
    ///                     This iterator type must meet the requirements of an
    ///                     random access iterator.
    /// \tparam Sent1       The type of the end source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter1.
    /// \tparam Iter2       The type of the source iterators used (deduced)
    ///                     representing the second sequence.
    ///                     This iterator type must meet the requirements of an
    ///                     random access iterator.
    /// \tparam Sent2       The type of the end source iterators used (deduced)
    ///                     representing the second sequence.
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter2.
    /// \tparam Iter3       The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     random access iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a merge requires \a Comp to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    /// \tparam Proj1       The type of an optional projection function to be
    ///                     used for elements of the first range. This defaults
    ///                     to \a util::projection_identity
    /// \tparam Proj2       The type of an optional projection function to be
    ///                     used for elements of the second range. This defaults
    ///                     to \a util::projection_identity
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
    /// \param dest         Refers to the beginning of the destination range.
    /// \param comp         \a comp is a callable object which returns true if
    ///                     the first argument is less than the second,
    ///                     and false otherwise. The signature of this
    ///                     comparison should be equivalent to:
    ///                     \code
    ///                     bool comp(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such that
    ///                     objects of types \a RandIter1 and \a RandIter2 can be
    ///                     dereferenced and then implicitly converted to
    ///                     both \a Type1 and \a Type2
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     first range as a projection operation before the
    ///                     actual comparison \a comp is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     second range as a projection operation before the
    ///                     actual comparison \a comp is invoked.
    ///
    /// The assignments in the parallel \a merge algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a merge algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a merge algorithm returns a
    /// \a hpx::future<merge_result<Iter1, Iter2, Iter3>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns
    ///           \a merge_result<Iter1, Iter2, Iter3> otherwise.
    ///           The \a merge algorithm returns the tuple of
    ///           the source iterator \a last1,
    ///           the source iterator \a last2,
    ///           the destination iterator to the end of the \a dest range.
    ///
    template <typename ExPolicy, typename Iter1, typename Sent, typename Iter2,
        typename Sent2, typename Iter3, typename Comp = hpx::ranges::less,
        typename Proj1 = util::projection_identity,
        typename Proj2 = util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy,
        hpx::ranges::merge_result<Iter1, Iter2, Iter3>>::type
    merge(ExPolicy&& policy, Iter1 first1, Sent1 last1, Iter2 first2,
        Sent2 last2, Iter3 dest, Comp&& comp = Comp(), Proj1&& proj1 = Proj1(),
        Proj2&& proj2 = Proj2());

    /// Merges two sorted ranges [first1, last1) and [first2, last2)
    /// into one sorted range beginning at \a dest. The order of
    /// equivalent elements in the each of original two ranges is preserved.
    /// For equivalent elements in the original two ranges, the elements from
    /// the first range precede the elements from the second range.
    /// The destination range cannot overlap with either of the input ranges.
    ///
    /// \note   Complexity: Performs
    ///         O(std::distance(first1, last1) + std::distance(first2, last2))
    ///         applications of the comparison \a comp and the each projection.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng1        The type of the first source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an random access iterator.
    /// \tparam Rng2        The type of the second source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an random access iterator.
    /// \tparam RandIter3   The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     random access iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a merge requires \a Comp to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    /// \tparam Proj1       The type of an optional projection function to be
    ///                     used for elements of the first range. This defaults
    ///                     to \a util::projection_identity
    /// \tparam Proj2       The type of an optional projection function to be
    ///                     used for elements of the second range. This defaults
    ///                     to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng1         Refers to the first range of elements the algorithm
    ///                     will be applied to.
    /// \param rng2         Refers to the second range of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param comp         \a comp is a callable object which returns true if
    ///                     the first argument is less than the second,
    ///                     and false otherwise. The signature of this
    ///                     comparison should be equivalent to:
    ///                     \code
    ///                     bool comp(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such that
    ///                     objects of types \a RandIter1 and \a RandIter2 can be
    ///                     dereferenced and then implicitly converted to
    ///                     both \a Type1 and \a Type2
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     first range as a projection operation before the
    ///                     actual comparison \a comp is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     second range as a projection operation before the
    ///                     actual comparison \a comp is invoked.
    ///
    /// The assignments in the parallel \a merge algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a merge algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a merge algorithm returns a
    /// \a hpx::future<merge_result<RandIter1, RandIter2, RandIter3>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns
    ///           \a merge_result<RandIter1, RandIter2, RandIter3> otherwise.
    ///           The \a merge algorithm returns the tuple of
    ///           the source iterator \a last1,
    ///           the source iterator \a last2,
    ///           the destination iterator to the end of the \a dest range.
    ///
    template <typename ExPolicy, typename Rng1, typename Rng2,
        typename RandIter3, typename Comp = hpx::ranges::less,
        typename Proj1 = util::projection_identity,
        typename Proj2 = util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy,
        hpx::ranges::merge_result<
            typename hpx::traits::range_iterator<Rng1>::type,
            typename hpx::traits::range_iterator<Rng2>::type,
            RandIter3
        >
    >::type
    merge(ExPolicy&& policy, Rng1&& rng1, Rng2&& rng2, RandIter3 dest,
        Comp&& comp = Comp(), Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2());

    /// Merges two consecutive sorted ranges [first, middle) and
    /// [middle, last) into one sorted range [first, last). The order of
    /// equivalent elements in the each of original two ranges is preserved.
    /// For equivalent elements in the original two ranges, the elements from
    /// the first range precede the elements from the second range.
    ///
    /// \note   Complexity: Performs O(std::distance(first, last))
    ///         applications of the comparison \a comp and the each projection.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Iter        The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     random access iterator.
    /// \tparam Sent       The type of the end source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter1.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a inplace_merge requires \a Comp
    ///                     to meet the requirements of \a CopyConstructible.
    ///                     This defaults to std::less<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the first sorted range
    ///                     the algorithm will be applied to.
    /// \param middle       Refers to the end of the first sorted range and
    ///                     the beginning of the second sorted range
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the second sorted range
    ///                     the algorithm will be applied to.
    /// \param comp         \a comp is a callable object which returns true if
    ///                     the first argument is less than the second,
    ///                     and false otherwise. The signature of this
    ///                     comparison should be equivalent to:
    ///                     \code
    ///                     bool comp(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be
    ///                     such that objects of types \a RandIter can be
    ///                     dereferenced and then implicitly converted to both
    ///                     \a Type1 and \a Type2
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a inplace_merge algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a inplace_merge algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a inplace_merge algorithm returns a
    ///           \a hpx::future<Iter> if the execution policy is of type
    ///           \a sequenced_task_policy or \a parallel_task_policy
    ///           and returns \a Iter otherwise.
    ///           The \a inplace_merge algorithm returns
    ///           the source iterator \a last
    ///
    template <typename ExPolicy, typename Iter, typename Sent,
        typename Comp = hpx::ranges::less,
        typename Proj = util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy, Iter>::type
    inplace_merge(ExPolicy&& policy, Iter first, Iter middle, Sent last,
        Comp&& comp = Comp(), Proj&& proj = Proj());

    /// Merges two consecutive sorted ranges [first, middle) and
    /// [middle, last) into one sorted range [first, last). The order of
    /// equivalent elements in the each of original two ranges is preserved.
    /// For equivalent elements in the original two ranges, the elements from
    /// the first range precede the elements from the second range.
    ///
    /// \note   Complexity: Performs O(std::distance(first, last))
    ///         applications of the comparison \a comp and the each projection.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an random access iterator.
    /// \tparam RandIter    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     random access iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a inplace_merge requires \a Comp
    ///                     to meet the requirements of \a CopyConstructible.
    ///                     This defaults to std::less<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the range of elements the algorithm
    ///                     will be applied to.
    /// \param middle       Refers to the end of the first sorted range and
    ///                     the beginning of the second sorted range
    ///                     the algorithm will be applied to.
    /// \param comp         \a comp is a callable object which returns true if
    ///                     the first argument is less than the second,
    ///                     and false otherwise. The signature of this
    ///                     comparison should be equivalent to:
    ///                     \code
    ///                     bool comp(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be
    ///                     such that objects of types \a RandIter can be
    ///                     dereferenced and then implicitly converted to both
    ///                     \a Type1 and \a Type2
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a inplace_merge algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a inplace_merge algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a inplace_merge algorithm returns a
    ///           \a hpx::future<RandIter> if the execution policy is of type
    ///           \a sequenced_task_policy or \a parallel_task_policy
    ///           and returns \a RandIter otherwise.
    ///           The \a inplace_merge algorithm returns
    ///           the source iterator \a last
    ///
    template <typename ExPolicy, typename Rng, typename RandIter,
        typename Comp = hpx::ranges::less,
        typename Proj = util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy, RandIter>::type
    inplace_merge(ExPolicy&& policy, Rng&& rng, RandIter middle,
        Comp&& comp = Comp(), Proj&& proj = Proj());

    // clang-format on
}}    // namespace hpx::ranges

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_sentinel_for.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/merge.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {

    ///////////////////////////////////////////////////////////////////////////
    // TODO: Support forward and bidirectional iterator. (#2826)
    // For now, only support random access iterator.

    // clang-format off
    template <typename ExPolicy, typename Rng1, typename Rng2,
        typename RandIter3, typename Comp = detail::less,
        typename Proj1 = util::projection_identity,
        typename Proj2 = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_range<Rng1>::value &&
            hpx::parallel::traits::is_projected_range<Proj1, Rng1>::value &&
            hpx::traits::is_range<Rng2>::value &&
            hpx::parallel::traits::is_projected_range<Proj2, Rng2>::value &&
            hpx::traits::is_iterator<RandIter3>::value &&
            hpx::parallel::traits::is_indirect_callable<ExPolicy, Comp,
                hpx::parallel::traits::projected_range<Proj1, Rng1>,
                hpx::parallel::traits::projected_range<Proj2, Rng2>
            >::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::merge is deprecated, use hpx::ranges::merge instead")
        typename util::detail::algorithm_result<ExPolicy,
            hpx::parallel::util::in_in_out_result<
                typename hpx::traits::range_iterator<Rng1>::type,
                typename hpx::traits::range_iterator<Rng2>::type,
                RandIter3>>::type merge(ExPolicy&& policy, Rng1&& rng1,
            Rng2&& rng2, RandIter3 dest, Comp&& comp = Comp(),
            Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2())
    {
        using iterator_type1 = typename hpx::traits::range_iterator<Rng1>::type;
        using iterator_type2 = typename hpx::traits::range_iterator<Rng2>::type;

        static_assert(
            hpx::traits::is_random_access_iterator<iterator_type1>::value,
            "Required at least random access iterator.");
        static_assert(
            hpx::traits::is_random_access_iterator<iterator_type2>::value,
            "Requires at least random access iterator.");
        static_assert(
            (hpx::traits::is_random_access_iterator<RandIter3>::value),
            "Requires at least random access iterator.");

        using is_seq =
            hpx::parallel::execution::is_sequenced_execution_policy<ExPolicy>;
        using result_type =
            hpx::parallel::util::in_in_out_result<iterator_type1,
                iterator_type2, RandIter3>;

        return hpx::parallel::v1::detail::merge<result_type>().call(
            std::forward<ExPolicy>(policy), is_seq(), hpx::util::begin(rng1),
            hpx::util::end(rng1), hpx::util::begin(rng2), hpx::util::end(rng2),
            dest, std::forward<Comp>(comp), std::forward<Proj1>(proj1),
            std::forward<Proj2>(proj2));
    }

    ///////////////////////////////////////////////////////////////////////////
    // TODO: Support bidirectional iterator. (#2826)
    // For now, only support random access iterator.

    // clang-format off
    template <typename ExPolicy, typename Rng, typename RandIter,
        typename Comp = detail::less, typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_range<Rng>::value &&
            hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
            hpx::traits::is_iterator<RandIter>::value &&
            hpx::parallel::traits::is_projected<Proj, RandIter>::value &&
            hpx::parallel::traits::is_indirect_callable<ExPolicy, Comp,
                hpx::parallel::traits::projected_range<Proj, Rng>,
                hpx::parallel::traits::projected_range<Proj, Rng>
            >::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::inplace_merge is deprecated, use "
        "hpx::ranges::inplace_merge instead")
        typename util::detail::algorithm_result<ExPolicy, RandIter>::type
        inplace_merge(ExPolicy&& policy, Rng&& rng, RandIter middle,
            Comp&& comp = Comp(), Proj&& proj = Proj())
    {
        using iterator_type = typename hpx::traits::range_iterator<Rng>::type;

        static_assert(
            hpx::traits::is_random_access_iterator<iterator_type>::value,
            "Required at least random access iterator.");

        using is_seq =
            hpx::parallel::execution::is_sequenced_execution_policy<ExPolicy>;

        return hpx::parallel::v1::detail::inplace_merge<RandIter>().call(
            std::forward<ExPolicy>(policy), is_seq(), hpx::util::begin(rng),
            middle, hpx::util::end(rng), std::forward<Comp>(comp),
            std::forward<Proj>(proj));
    }
}}}    // namespace hpx::parallel::v1

namespace hpx { namespace ranges {

    template <typename I1, typename I2, typename O>
    using merge_result = parallel::util::in_in_out_result<I1, I2, O>;

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::merge
    HPX_INLINE_CONSTEXPR_VARIABLE struct merge_t final
      : hpx::functional::tag<merge_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename Rng1, typename Rng2,
            typename Iter3, typename Comp = hpx::ranges::less,
            typename Proj1 = hpx::parallel::util::projection_identity,
            typename Proj2 = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng1>::value &&
                hpx::parallel::traits::is_projected_range<Proj1, Rng1>::value &&
                hpx::traits::is_range<Rng2>::value &&
                hpx::parallel::traits::is_projected_range<Proj2, Rng2>::value &&
                hpx::traits::is_iterator<Iter3>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy, Comp,
                    hpx::parallel::traits::projected_range<Proj1, Rng1>,
                    hpx::parallel::traits::projected_range<Proj2, Rng2>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            hpx::ranges::merge_result<
                typename hpx::traits::range_iterator<Rng1>::type,
                typename hpx::traits::range_iterator<Rng2>::type, Iter3>>::type
        tag_invoke(merge_t, ExPolicy&& policy, Rng1&& rng1, Rng2&& rng2,
            Iter3 dest, Comp&& comp = Comp(), Proj1&& proj1 = Proj1(),
            Proj2&& proj2 = Proj2())
        {
            using iterator_type1 =
                typename hpx::traits::range_iterator<Rng1>::type;
            using iterator_type2 =
                typename hpx::traits::range_iterator<Rng2>::type;

            static_assert(
                hpx::traits::is_random_access_iterator<iterator_type1>::value,
                "Required at least random access iterator.");
            static_assert(
                hpx::traits::is_random_access_iterator<iterator_type2>::value,
                "Requires at least random access iterator.");
            static_assert(hpx::traits::is_random_access_iterator<Iter3>::value,
                "Requires at least random access iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;
            using result_type = hpx::ranges::merge_result<iterator_type1,
                iterator_type2, Iter3>;

            return hpx::parallel::v1::detail::merge<result_type>().call(
                std::forward<ExPolicy>(policy), is_seq(),
                hpx::util::begin(rng1), hpx::util::end(rng1),
                hpx::util::begin(rng2), hpx::util::end(rng2), dest,
                std::forward<Comp>(comp), std::forward<Proj1>(proj1),
                std::forward<Proj2>(proj2));
        }

        // clang-format off
        template <typename ExPolicy, typename Iter1, typename Sent1,
            typename Iter2, typename Sent2, typename Iter3,
            typename Comp = hpx::ranges::less,
            typename Proj1 = hpx::parallel::util::projection_identity,
            typename Proj2 = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_sentinel_for<Sent1, Iter1>::value &&
                hpx::parallel::traits::is_projected<Proj1, Iter1>::value &&
                hpx::traits::is_sentinel_for<Sent2, Iter2>::value &&
                hpx::parallel::traits::is_projected<Proj2, Iter2>::value &&
                hpx::traits::is_iterator<Iter3>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy, Comp,
                    hpx::parallel::traits::projected<Proj1, Iter1>,
                    hpx::parallel::traits::projected<Proj2, Iter2>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            hpx::ranges::merge_result<Iter1, Iter2, Iter3>>::type
        tag_invoke(merge_t, ExPolicy&& policy, Iter1 first1, Sent1 last1,
            Iter2 first2, Sent2 last2, Iter3 dest, Comp&& comp = Comp(),
            Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2())
        {
            static_assert(hpx::traits::is_random_access_iterator<Iter1>::value,
                "Required at least random access iterator.");
            static_assert(hpx::traits::is_random_access_iterator<Iter2>::value,
                "Requires at least random access iterator.");
            static_assert(hpx::traits::is_random_access_iterator<Iter3>::value,
                "Requires at least random access iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;
            using result_type = hpx::ranges::merge_result<Iter1, Iter2, Iter3>;

            return hpx::parallel::v1::detail::merge<result_type>().call(
                std::forward<ExPolicy>(policy), is_seq(), first1, last1, first2,
                last2, dest, std::forward<Comp>(comp),
                std::forward<Proj1>(proj1), std::forward<Proj2>(proj2));
        }

        // clang-format off
        template <typename Rng1, typename Rng2,
            typename Iter3, typename Comp = hpx::ranges::less,
            typename Proj1 = hpx::parallel::util::projection_identity,
            typename Proj2 = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng1>::value &&
                hpx::parallel::traits::is_projected_range<Proj1, Rng1>::value &&
                hpx::traits::is_range<Rng2>::value &&
                hpx::parallel::traits::is_projected_range<Proj2, Rng2>::value &&
                hpx::traits::is_iterator<Iter3>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Comp,
                    hpx::parallel::traits::projected_range<Proj1, Rng1>,
                    hpx::parallel::traits::projected_range<Proj2, Rng2>
                >::value
            )>
        // clang-format on
        friend hpx::ranges::merge_result<
            typename hpx::traits::range_iterator<Rng1>::type,
            typename hpx::traits::range_iterator<Rng2>::type, Iter3>
        tag_invoke(merge_t, Rng1&& rng1, Rng2&& rng2, Iter3 dest,
            Comp&& comp = Comp(), Proj1&& proj1 = Proj1(),
            Proj2&& proj2 = Proj2())
        {
            using iterator_type1 =
                typename hpx::traits::range_iterator<Rng1>::type;
            using iterator_type2 =
                typename hpx::traits::range_iterator<Rng2>::type;

            static_assert(
                hpx::traits::is_random_access_iterator<iterator_type1>::value,
                "Required at least random access iterator.");
            static_assert(
                hpx::traits::is_random_access_iterator<iterator_type2>::value,
                "Requires at least random access iterator.");
            static_assert(hpx::traits::is_random_access_iterator<Iter3>::value,
                "Requires at least random access iterator.");

            using result_type = hpx::ranges::merge_result<iterator_type1,
                iterator_type2, Iter3>;

            return hpx::parallel::v1::detail::merge<result_type>().call(
                hpx::execution::seq, std::true_type(), hpx::util::begin(rng1),
                hpx::util::end(rng1), hpx::util::begin(rng2),
                hpx::util::end(rng2), dest, std::forward<Comp>(comp),
                std::forward<Proj1>(proj1), std::forward<Proj2>(proj2));
        }

        // clang-format off
        template <typename Iter1, typename Sent1,
            typename Iter2, typename Sent2, typename Iter3,
            typename Comp = hpx::ranges::less,
            typename Proj1 = hpx::parallel::util::projection_identity,
            typename Proj2 = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_sentinel_for<Sent1, Iter1>::value &&
                hpx::parallel::traits::is_projected<Proj1, Iter1>::value &&
                hpx::traits::is_sentinel_for<Sent2, Iter2>::value &&
                hpx::parallel::traits::is_projected<Proj2, Iter2>::value &&
                hpx::traits::is_iterator<Iter3>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Comp,
                    hpx::parallel::traits::projected<Proj1, Iter1>,
                    hpx::parallel::traits::projected<Proj2, Iter2>
                >::value
            )>
        // clang-format on
        friend hpx::ranges::merge_result<Iter1, Iter2, Iter3> tag_invoke(
            merge_t, Iter1 first1, Sent1 last1, Iter2 first2, Sent2 last2,
            Iter3 dest, Comp&& comp = Comp(), Proj1&& proj1 = Proj1(),
            Proj2&& proj2 = Proj2())
        {
            static_assert(hpx::traits::is_random_access_iterator<Iter1>::value,
                "Required at least random access iterator.");
            static_assert(hpx::traits::is_random_access_iterator<Iter2>::value,
                "Requires at least random access iterator.");
            static_assert(hpx::traits::is_random_access_iterator<Iter3>::value,
                "Requires at least random access iterator.");

            using result_type = hpx::ranges::merge_result<Iter1, Iter2, Iter3>;

            return hpx::parallel::v1::detail::merge<result_type>().call(
                hpx::execution::seq, std::true_type(), first1, last1, first2,
                last2, dest, std::forward<Comp>(comp),
                std::forward<Proj1>(proj1), std::forward<Proj2>(proj2));
        }
    } merge{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::inplace_merge
    HPX_INLINE_CONSTEXPR_VARIABLE struct inplace_merge_t final
      : hpx::functional::tag<inplace_merge_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename Rng, typename Iter,
            typename Comp = hpx::ranges::less,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
                hpx::traits::is_iterator<Iter>::value &&
                hpx::parallel::traits::is_projected<Proj, Iter>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy, Comp,
                    hpx::parallel::traits::projected_range<Proj, Rng>,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            Iter>::type
        tag_invoke(inplace_merge_t, ExPolicy&& policy, Rng&& rng, Iter middle,
            Comp&& comp = Comp(), Proj&& proj = Proj())
        {
            using iterator_type =
                typename hpx::traits::range_iterator<Rng>::type;

            static_assert(
                hpx::traits::is_random_access_iterator<iterator_type>::value,
                "Required at least random access iterator.");
            static_assert(hpx::traits::is_random_access_iterator<Iter>::value,
                "Required at least random access iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            return hpx::parallel::v1::detail::inplace_merge<Iter>().call(
                std::forward<ExPolicy>(policy), is_seq(), hpx::util::begin(rng),
                middle, hpx::util::end(rng), std::forward<Comp>(comp),
                std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent,
            typename Comp = hpx::ranges::less,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_sentinel_for<Sent, Iter>::value &&
                hpx::parallel::traits::is_projected<Proj, Iter>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy, Comp,
                    hpx::parallel::traits::projected<Proj, Iter>,
                    hpx::parallel::traits::projected<Proj, Iter>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            Iter>::type
        tag_invoke(inplace_merge_t, ExPolicy&& policy, Iter first, Iter middle,
            Sent last, Comp&& comp = Comp(), Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_random_access_iterator<Iter>::value,
                "Required at least random access iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            return hpx::parallel::v1::detail::inplace_merge<Iter>().call(
                std::forward<ExPolicy>(policy), is_seq(), first, middle, last,
                std::forward<Comp>(comp), std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename Rng, typename Iter,
            typename Comp = hpx::ranges::less,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_projected_range<Proj, Rng>::value &&
                hpx::traits::is_iterator<Iter>::value &&
                hpx::parallel::traits::is_projected<Proj, Iter>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Comp,
                    hpx::parallel::traits::projected_range<Proj, Rng>,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend Iter tag_invoke(inplace_merge_t, Rng&& rng, Iter middle,
            Comp&& comp = Comp(), Proj&& proj = Proj())
        {
            using iterator_type =
                typename hpx::traits::range_iterator<Rng>::type;

            static_assert(
                hpx::traits::is_random_access_iterator<iterator_type>::value,
                "Required at least random access iterator.");
            static_assert(hpx::traits::is_random_access_iterator<Iter>::value,
                "Required at least random access iterator.");

            return hpx::parallel::v1::detail::inplace_merge<Iter>().call(
                hpx::execution::seq, std::true_type(), hpx::util::begin(rng),
                middle, hpx::util::end(rng), std::forward<Comp>(comp),
                std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename Iter, typename Sent,
            typename Comp = hpx::ranges::less,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_sentinel_for<Sent, Iter>::value &&
                hpx::parallel::traits::is_projected<Proj, Iter>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Comp,
                    hpx::parallel::traits::projected<Proj, Iter>,
                    hpx::parallel::traits::projected<Proj, Iter>
                >::value
            )>
        // clang-format on
        friend Iter tag_invoke(inplace_merge_t, Iter first, Iter middle,
            Sent last, Comp&& comp = Comp(), Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_random_access_iterator<Iter>::value,
                "Required at least random access iterator.");

            return hpx::parallel::v1::detail::inplace_merge<Iter>().call(
                hpx::execution::seq, std::true_type(), first, middle, last,
                std::forward<Comp>(comp), std::forward<Proj>(proj));
        }
    } inplace_merge{};

}}    // namespace hpx::ranges
#endif    //DOXYGEN
