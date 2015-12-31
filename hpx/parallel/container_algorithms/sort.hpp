//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/sort.hpp

#if !defined(HPX_PARALLEL_CONTAINER_ALGORITHM_SORT_DEC_06_2015_1133AM)
#define HPX_PARALLEL_CONTAINER_ALGORITHM_SORT_DEC_06_2015_1133AM

#include <hpx/config.hpp>
#include <hpx/util/move.hpp>
#include <hpx/traits/concepts.hpp>

#include <hpx/parallel/algorithms/sort.hpp>
#include <hpx/parallel/traits/is_range.hpp>
#include <hpx/parallel/traits/projected_range.hpp>
#include <hpx/parallel/traits/range_traits.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <boost/range/functions.hpp>

#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    /// Sorts the elements in the range \a rng  in ascending order. The
    /// order of equal elements is not guaranteed to be preserved. The function
    /// uses the given comparison function object comp (defaults to using
    /// operator<()).
    ///
    /// \note   Complexity: O(Nlog(N)),
    ///             where N = std::distance(begin(rng), end(rng)) comparisons.
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
    /// \a sequential_execution_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_execution_policy or \a parallel_task_execution_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a sort algorithm returns a
    ///           \a hpx::future<Iter> if the execution policy is of
    ///           type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and returns \a Iter
    ///           otherwise.
    ///           It returns \a last.
    template <typename ExPolicy, typename Rng,
        typename Proj = util::projection_identity,
        typename Compare = std::less<
            typename std::remove_reference<
                typename traits::projected_range_result_of<Proj, Rng>::type
            >::type
        >,
    HPX_CONCEPT_REQUIRES_(
        is_execution_policy<ExPolicy>::value &&
        traits::is_range<Rng>::value &&
        traits::is_projected_range<Proj, Rng>::value &&
        traits::is_indirect_callable<
            Compare,
                traits::projected_range<Proj, Rng>,
                traits::projected_range<Proj, Rng>
        >::value)>
    typename util::detail::algorithm_result<
        ExPolicy, typename traits::range_iterator<Rng>::type
    >::type
    sort(ExPolicy && policy, Rng && rng, Compare && comp = Compare(),
        Proj && proj = Proj())
    {
        return sort(std::forward<ExPolicy>(policy),
            boost::begin(rng), boost::end(rng), std::forward<Compare>(comp),
            std::forward<Proj>(proj));
    }
}}}

#endif


