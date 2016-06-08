//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/reverse.hpp

#if !defined(HPX_PARALLEL_CONTAINER_ALGORITHMS_REVERSE_DEC_21_2015_0245PM)
#define HPX_PARALLEL_CONTAINER_ALGORITHMS_REVERSE_DEC_21_2015_0245PM

#include <hpx/config.hpp>
#include <hpx/traits/concepts.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/tagged_pair.hpp>

#include <hpx/parallel/algorithms/reverse.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/traits/is_range.hpp>
#include <hpx/parallel/traits/projected_range.hpp>
#include <hpx/parallel/traits/range_traits.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <boost/range/functions.hpp>

#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    /// Reverses the order of the elements in the range [first, last).
    /// Behaves as if applying std::iter_swap to every pair of iterators
    /// first+i, (last-i) - 1 for each non-negative i < (last-first)/2.
    ///
    /// \note   Complexity: Linear in the distance between \a first and \a last.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a bidirectional iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    ///
    /// The assignments in the parallel \a reverse algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a reverse algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a reverse algorithm returns a \a hpx::future<BidirIter>
    ///           if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a BidirIter otherwise.
    ///           It returns \a last.
    ///
    template <typename ExPolicy, typename Rng,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        traits::is_range<Rng>::value)>
    typename util::detail::algorithm_result<
        ExPolicy, typename traits::range_iterator<Rng>::type
    >::type
    reverse(ExPolicy && policy, Rng && rng)
    {
        return reverse(std::forward<ExPolicy>(policy),
            boost::begin(rng), boost::end(rng));
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Copies the elements from the range [first, last) to another range
    /// beginning at dest_first in such a way that the elements in the new
    /// range are in reverse order.
    /// Behaves as if by executing the assignment
    /// *(dest_first + (last - first) - 1 - i) = *(first + i) once for each
    /// non-negative i < (last - first)
    /// If the source and destination ranges (that is, [first, last) and
    /// [dest_first, dest_first+(last-first)) respectively) overlap, the
    /// behavior is undefined.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a bidirectional iterator.
    /// \tparam OutputIter  The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest_first   Refers to the begin of the destination range.
    ///
    /// The assignments in the parallel \a reverse_copy algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a reverse_copy algorithm invoked with
    /// an execution policy object of type \a parallel_execution_policy or
    /// \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a reverse_copy algorithm returns a
    ///           \a hpx::future<tagged_pair<tag::in(BidirIter), tag::out(OutIter)> >
    ///           if the execution policy is of type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and
    ///           returns \a tagged_pair<tag::in(BidirIter), tag::out(OutIter)>
    ///           otherwise.
    ///           The \a copy algorithm returns the pair of the input iterator
    ///           forwarded to the first element after the last in the input
    ///           sequence and the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename Rng, typename OutIter,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        traits::is_range<Rng>::value &&
        hpx::traits::is_iterator<OutIter>::value)>
    typename util::detail::algorithm_result<
        ExPolicy,
        hpx::util::tagged_pair<
            tag::in(typename traits::range_iterator<Rng>::type),
            tag::out(OutIter)
        >
    >::type
    reverse_copy(ExPolicy && policy, Rng && rng, OutIter dest_first)
    {
        return reverse_copy(std::forward<ExPolicy>(policy),
            boost::begin(rng), boost::end(rng), dest_first);
    }
}}}

#endif
