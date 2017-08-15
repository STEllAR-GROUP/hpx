//  Copyright (c) 2017 Taeguk Kwon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/merge.hpp

#if !defined(HPX_PARALLEL_CONTAINER_ALGORITHM_MERGE_AUG_15_2017_1045AM)
#define HPX_PARALLEL_CONTAINER_ALGORITHM_MERGE_AUG_15_2017_1045AM

#include <hpx/config.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/traits/is_range.hpp>
#include <hpx/util/range.hpp>
#include <hpx/util/tagged_tuple.hpp>

#include <hpx/parallel/algorithms/merge.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/traits/projected.hpp>
#include <hpx/parallel/traits/projected_range.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1
{
    // TODO: Support forward and bidirectional iterator. (#2826)
    // For now, only support random access iterator.
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
    /// \param comp         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last). This is an
    ///                     binary compicate which returns \a true if the first
    ///                     argument is less than the second. The signature of
    ///                     this compicate should be equivalent to:
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
    /// \a hpx::future<tagged_tuple<tag::in1(RandIter1), tag::in2(RandIter2), tag::out(RandIter3)> >
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns
    /// \a tagged_tuple<tag::in1(RandIter1), tag::in2(RandIter2), tag::out(RandIter3)>
    ///           otherwise.
    ///           The \a merge algorithm returns the tuple of
    ///           the source iterator \a last1,
    ///           the source iterator \a last2,
    ///           the destination iterator to the end of the \a dest range.
    ///
    template <typename ExPolicy,
        typename Rng1, typename Rng2, typename RandIter3,
        typename Comp = detail::less,
        typename Proj1 = util::projection_identity,
        typename Proj2 = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_range<Rng1>::value &&
        hpx::traits::is_range<Rng2>::value &&
        hpx::traits::is_iterator<RandIter3>::value &&
        traits::is_projected_range<Proj1, Rng1>::value &&
        traits::is_projected_range<Proj2, Rng2>::value &&
        traits::is_indirect_callable<
            ExPolicy, Comp,
            traits::projected_range<Proj1, Rng1>,
            traits::projected_range<Proj2, Rng2>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy,
        hpx::util::tagged_tuple<
            tag::in1(typename hpx::traits::range_iterator<Rng1>::type),
            tag::in2(typename hpx::traits::range_iterator<Rng2>::type),
            tag::out(RandIter3)>
    >::type
    merge(ExPolicy && policy, Rng1 && rng1, Rng2 && rng2,
        RandIter3 dest, Comp && comp = Comp(),
        Proj1 && proj1 = Proj1(), Proj2 && proj2 = Proj2())
    {
        return merge(std::forward<ExPolicy>(policy),
            hpx::util::begin(rng1), hpx::util::end(rng1),
            hpx::util::begin(rng2), hpx::util::end(rng2), dest,
            std::forward<Comp>(comp),
            std::forward<Proj1>(proj1),
            std::forward<Proj2>(proj2));
    }
    /// \endcond
}}}

#endif
