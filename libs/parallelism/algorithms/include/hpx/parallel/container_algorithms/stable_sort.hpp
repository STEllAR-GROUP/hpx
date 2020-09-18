//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/stable_sort.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>

#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/parallel/algorithms/stable_sort.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace rangev1 {

    /// Sorts the elements in the range [first, last) in ascending order. The
    /// relative order of equal elements is preserved. The function
    /// uses the given comparison function object comp (defaults to using
    /// operator<()).
    ///
    /// \note   Complexity: O(Nlog(N)), where N = std::distance(first, last)
    ///                     comparisons.
    ///
    /// A sequence is sorted with respect to a comparator \a comp and a
    /// projection \a proj if for every iterator i pointing to the sequence and
    /// every non-negative integer n such that i + n is a valid iterator
    /// pointing to an element of the sequence, and
    /// INVOKE(comp, INVOKE(proj, *(i + n)), INVOKE(proj, *i)) == false.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param comp         comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that comp
    ///                     will not apply any non-constant function through the
    ///                     dereferenced iterator.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each pair of elements as a
    ///                     projection operation before the actual predicate
    ///                     \a comp is invoked.
    ///
    /// \a comp has to induce a strict weak ordering on the values.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a stable_sort algorithm returns a
    ///           \a hpx::future<RandomIt> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a RandomIt
    ///           otherwise.
    ///           The algorithm returns an iterator pointing to the first
    ///           element after the last element in the input sequence.
    //-----------------------------------------------------------------------------
    // clang-format off
    template <typename ExPolicy, typename Rng,
        typename Compare = v1::detail::less,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_range<Rng>::value&&
            traits::is_projected_range<Proj, Rng>::value&&
            traits::is_indirect_callable<ExPolicy, Compare,
                traits::projected_range<Proj, Rng>,
                traits::projected_range<Proj, Rng>
            >::value
        )>
    // clang-format on
    typename util::detail::algorithm_result<ExPolicy,
        typename hpx::traits::range_iterator<Rng>::type>::type
    stable_sort(ExPolicy&& policy, Rng&& rng, Compare&& comp = Compare(),
        Proj&& proj = Proj())
    {
        return v1::stable_sort(std::forward<ExPolicy>(policy),
            hpx::util::begin(rng), hpx::util::end(rng),
            std::forward<Compare>(comp), std::forward<Proj>(proj));
    }
}}}    // namespace hpx::parallel::rangev1
