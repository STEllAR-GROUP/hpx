//  Copyright (c) 2017 Taeguk Kwon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/partition.hpp

#if !defined(HPX_PARALLEL_CONTAINER_ALGORITHM_PARTITION_JUL_09_2017_0501PM)
#define HPX_PARALLEL_CONTAINER_ALGORITHM_PARTITION_JUL_09_2017_0501PM

#include <hpx/config.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/traits/is_range.hpp>
#include <hpx/util/range.hpp>
#include <hpx/util/tagged_tuple.hpp>

#include <hpx/parallel/algorithms/partition.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/traits/projected.hpp>
#include <hpx/parallel/traits/projected_range.hpp>

#include <boost/range/functions.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1
{
    /// Copies the elements in the range \a rng,
    /// to two different ranges depending on the value returned by
    /// the predicate \a pred. The elements, that satisfy the predicate \a pred,
    /// are copied to the range beginning at \a dest_true. The rest of
    /// the elements are copied to the range beginning at \a dest_false.
    /// The order of the elements is preserved.
    ///
    /// \note   Complexity: Performs not more than N assignments,
    ///         exactly N applications of the predicate \a pred,
    ///         where N = std::distance(begin(rng), end(rng)).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range for the elements that satisfy
    ///                     the predicate \a pred (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter3    The type of the iterator representing the
    ///                     destination range for the elements that don't satisfy
    ///                     the predicate \a pred (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a partition_copy requires \a Pred to meet
    ///                     the requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest_true    Refers to the beginning of the destination range for
    ///                     the elements that satisfy the predicate \a pred.
    /// \param dest_false   Refers to the beginning of the destination range for
    ///                     the elements that don't satisfy the predicate \a pred.
    /// \param pred         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the sequence
    ///                     specified by the range \a rng. This is an unary predicate
    ///                     for partitioning the source iterators. The signature of
    ///                     this predicate should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
    ///
    /// The assignments in the parallel \a partition_copy algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a partition_copy algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a partition_copy algorithm returns a
    /// \a hpx::future<tagged_tuple<tag::in(InIter), tag::out1(OutIter1), tag::out2(OutIter2)> >
    ///           if the execution policy is of type \a parallel_task_policy
    ///           and returns
    /// \a tagged_tuple<tag::in(InIter), tag::out1(OutIter1), tag::out2(OutIter2)>
    ///           otherwise.
    ///           The \a partition_copy algorithm returns the tuple of
    ///           the source iterator \a last,
    ///           the destination iterator to the end of the \a dest_true range, and
    ///           the destination iterator to the end of the \a dest_false range.
    ///
    template <typename ExPolicy, typename Rng, typename FwdIter2, typename FwdIter3,
        typename Pred, typename Proj = util::projection_identity,
    HPX_CONCEPT_REQUIRES_(
        execution::is_execution_policy<ExPolicy>::value &&
        hpx::traits::is_range<Rng>::value &&
        hpx::traits::is_iterator<FwdIter2>::value &&
        hpx::traits::is_iterator<FwdIter3>::value &&
        traits::is_projected_range<Proj, Rng>::value &&
        traits::is_indirect_callable<
            ExPolicy, Pred, traits::projected_range<Proj, Rng>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy,
        hpx::util::tagged_tuple<
            tag::in(typename hpx::traits::range_traits<Rng>::iterator_type),
            tag::out1(FwdIter2), tag::out2(FwdIter3)>
    >::type
    partition_copy(ExPolicy && policy, Rng && rng,
        FwdIter2 dest_true, FwdIter3 dest_false, Pred && pred,
        Proj && proj = Proj())
    {
        return partition_copy(std::forward<ExPolicy>(policy),
            std::begin(rng), std::end(rng), dest_true, dest_false,
            std::forward<Pred>(pred),
            std::forward<Proj>(proj));
    }
}}}

#endif
