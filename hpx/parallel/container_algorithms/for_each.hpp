//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/for_each.hpp

#if !defined(HPX_PARALLEL_CONTAINER_ALGORITHM_FOR_EACH_JUL_18_2015_0959AM)
#define HPX_PARALLEL_CONTAINER_ALGORITHM_FOR_EACH_JUL_18_2015_0959AM

#include <hpx/config.hpp>
#include <hpx/traits/concepts.hpp>

#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/traits/is_range.hpp>
#include <hpx/parallel/traits/projected_range.hpp>
#include <hpx/parallel/traits/range_traits.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <boost/range/functions.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    /// Applies \a f to the result of dereferencing every iterator in the
    /// given range \a rng.
    ///
    /// \note   Complexity: Applies \a f exactly \a size(rng) times.
    ///
    /// If \a f returns a result, the result is ignored.
    ///
    /// If the type of \a first satisfies the requirements of a mutable
    /// iterator, \a f may apply non-constant functions through the
    /// dereferenced iterator.
    ///
    /// Unlike its sequential form, the parallel overload of
    /// \a for_each does not return a copy of its \a Function parameter,
    /// since parallelization may not permit efficient state
    /// accumulation.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a for_each requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param f            Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     sequence specified by [first, last).
    ///                     The signature of this predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     <ignored> pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const&. The
    ///                     type \a Type must be such that an object of
    ///                     type \a InIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
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
    /// \returns  The \a for_each algorithm returns a
    ///           \a hpx::future<InIter> if the execution policy is of
    ///           type
    ///           \a sequential_task_execution_policy or
    ///           \a parallel_task_execution_policy and returns \a InIter
    ///           otherwise.
    ///           It returns \a last.
    ///
//    template <typename ExPolicy, typename Rng, typename F,
//        typename Proj = util::projection_identity>//,
//    HPX_CONCEPT_REQUIRES_(
//        is_execution_policy<ExPolicy>::value &&
//        traits::is_range<Rng>::value &&
//        traits::is_projected_range<Proj, Rng>::value &&
//        traits::is_indirect_callable<
//            F, traits::projected_range<Proj, Rng>
//        >::value)>
//    typename util::detail::algorithm_result<
//        ExPolicy, typename traits::range_iterator<Rng>::type
//    >::type
//    for_each(ExPolicy && policy, Rng && rng, F && f, Proj && proj = Proj())
//    {
//        return for_each(std::forward<ExPolicy>(policy),
//            boost::begin(rng), boost::end(rng), std::forward<F>(f),
//            std::forward<Proj>(proj));
//    }
}}}

#endif


