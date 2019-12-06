//  Copyright (c) 2017 Bruno Pitrus
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/move.hpp

#if !defined(HPX_PARALLEL_CONTAINER_ALGORITHM_MOVE_26_NOV_2017_1248PM)
#define HPX_PARALLEL_CONTAINER_ALGORITHM_MOVE_26_NOV_2017_1248PM

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/util/tagged_pair.hpp>

#include <hpx/parallel/algorithms/move.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {
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
    /// \tparam OutIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
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
    ///           \a hpx::future<tagged_pair<tag::in(iterator_t<Rng>),
    ///           tag::out(FwdIter2)> > if the execution policy is of type
    ///           \a sequenced_task_policy or \a parallel_task_policy and
    ///           returns \a tagged_pair<tag::in(iterator_t<Rng>),
    ///           tag::out(FwdIter2)> otherwise.
    ///           The \a move algorithm returns the pair of the input iterator
    ///           \a last and the output iterator to the element in the
    ///           destination range, one past the last element moved.
    ///
    template <typename ExPolicy, typename Rng, typename OutIter,
        HPX_CONCEPT_REQUIRES_(execution::is_execution_policy<ExPolicy>::value&&
                hpx::traits::is_range<Rng>::value&&
                    hpx::traits::is_iterator<OutIter>::value)>
    typename util::detail::algorithm_result<ExPolicy,
        hpx::util::tagged_pair<
            tag::in(typename hpx::traits::range_traits<Rng>::iterator_type),
            tag::out(OutIter)>>::type
    move(ExPolicy&& policy, Rng&& rng, OutIter dest)
    {
        return move(std::forward<ExPolicy>(policy), hpx::util::begin(rng),
            hpx::util::end(rng), dest);
    }
}}}    // namespace hpx::parallel::v1

#endif
