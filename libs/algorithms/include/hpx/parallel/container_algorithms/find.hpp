//  Copyright (c) 2018 Bruno Pitrus
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/find.hpp

#if !defined(HPX_PARALLEL_CONTAINER_ALGORITHMS_FIND)
#define HPX_PARALLEL_CONTAINER_ALGORITHMS_FIND

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>

#include <hpx/parallel/algorithms/find.hpp>
#include <hpx/parallel/traits/projected.hpp>
#include <hpx/parallel/traits/projected_range.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // find_end

    /// Returns the last subsequence of elements \a rng2 found in the range
    /// \a rng using the given predicate \a f to compare elements.
    ///
    /// \note   Complexity: at most S*(N-S+1) comparisons where
    ///         \a S = distance(begin(rng2), end(rng2)) and
    ///         \a N = distance(begin(rng), end(rng)).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the first source range (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam Rng2        The type of the second source range (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a replace requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the first sequence of elements
    ///                     the algorithm will be applied to.
    /// \param rng2         Refers to the second sequence of elements
    ///                     the algorithm will be applied to.
    /// \param op           The binary predicate which returns \a true
    ///                     if the elements should be treated as equal. The signature
    ///                     should be equivalent to the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such
    ///                     that objects of types \a iterator_t<Rng> and \a iterator_t<Rng2>
    ///                     can be dereferenced and then implicitly converted
    ///                     to \a Type1 and \a Type2 respectively.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of type
    ///                     dereferenced \a iterator_t<Rng> and
    ///                     dereferenced \a iterator_t<Rng2>
    ///                     as a projection operation before the function \a op
    ///                     is invoked.
    ///
    /// The comparison operations in the parallel \a find_end algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find_end algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find_end algorithm returns a \a hpx::future<iterator_t<Rng> >
    ///           if the execution policy is of type \a sequenced_task_policy
    ///           or \a parallel_task_policy and returns \a iterator_t<Rng> otherwise.
    ///           The \a find_end algorithm returns an iterator to the beginning of
    ///           the last subsequence \a rng2 in range \a rng.
    ///           If the length of the subsequence \a rng2 is greater
    ///           than the length of the range \a rng, \a end(rng) is returned.
    ///           Additionally if the size of the subsequence is empty or no subsequence
    ///           is found, \a end(rng) is also returned.
    ///
    /// This overload of \a find_end is available if the user decides to provide the
    /// algorithm their own predicate \a op.
    ///
    template <typename ExPolicy, typename Rng, typename Rng2,
        typename Pred = detail::equal_to,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(execution::is_execution_policy<ExPolicy>::value&&
                hpx::traits::is_range<Rng>::value&& traits::is_projected_range<
                    Proj, Rng>::value&& hpx::traits::is_range<Rng2>::value&&
                    traits::is_projected_range<Proj, Rng2>::value&&
                        traits::is_indirect_callable<ExPolicy, Pred,
                            traits::projected_range<Proj, Rng>,
                            traits::projected_range<Proj, Rng2>>::value)>
    typename util::detail::algorithm_result<ExPolicy,
        typename hpx::traits::range_iterator<Rng>::type>::type
    find_end(ExPolicy&& policy, Rng&& rng, Rng2&& rng2, Pred&& op = Pred(),
        Proj&& proj = Proj())
    {
        return find_end(std::forward<ExPolicy>(policy), hpx::util::begin(rng),
            hpx::util::end(rng), hpx::util::begin(rng2), hpx::util::end(rng2),
            std::forward<Pred>(op), std::forward<Proj>(proj));
    }
    ///////////////////////////////////////////////////////////////////////////
    // find_first_of

    /// Searches the range \a rng1 for any elements in the range \a rng2.
    /// Uses binary predicate \a p to compare elements
    ///
    /// \note   Complexity: at most (S*N) comparisons where
    ///         \a S = distance(begin(rng2), end(rng2)) and
    ///         \a N = distance(begin(rng1), end(rng1)).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng1        The type of the first source range (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam Rng2        The type of the second source range (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of a forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a replace requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj1       The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity and is
    ///                     applied to the elements in \a rng1.
    /// \tparam Proj2       The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity and is
    ///                     applied to the elements in \a rng2.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng1         Refers to the first sequence of elements
    ///                     the algorithm will be applied to.
    /// \param rng2         Refers to the second sequence of elements
    ///                     the algorithm will be applied to.
    /// \param op           The binary predicate which returns \a true
    ///                     if the elements should be treated as equal. The signature
    ///                     should be equivalent to the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such
    ///                     that objects of types \a iterator_t<Rng1>
    ///                     and \a iterator_t<Rng2>
    ///                     can be dereferenced and then implicitly converted
    ///                     to \a Type1 and \a Type2 respectively.
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of type
    ///                     dereferenced \a iterator_t<Rng1> before the function
    ///                     \a op is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of type
    ///                     dereferenced \a iterator_t<Rng2> before the function
    ///                     \a op is invoked.
    ///
    /// The comparison operations in the parallel \a find_first_of algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find_first_of algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find_end algorithm returns a \a hpx::future<iterator_t<Rng1> >
    ///           if the execution policy is of type \a sequenced_task_policy
    ///           or \a parallel_task_policy and returns \a iterator_t<Rng1> otherwise.
    ///           The \a find_first_of algorithm returns an iterator to the first element
    ///           in the range \a rng1 that is equal to an element from the range
    ///           \a rng2.
    ///           If the length of the subsequence \a rng2 is
    ///           greater than the length of the range \a rng1,
    ///           \a end(rng1) is returned.
    ///           Additionally if the size of the subsequence is empty or no subsequence
    ///           is found, \a end(rng1) is also returned.
    ///
    /// This overload of \a find_first_of is available if the user decides to provide the
    /// algorithm their own predicate \a op.
    ///
    template <typename ExPolicy, typename Rng1, typename Rng2,
        typename Pred = detail::equal_to,
        typename Proj1 = util::projection_identity,
        typename Proj2 = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(execution::is_execution_policy<ExPolicy>::value&&
                hpx::traits::is_range<Rng1>::value&& traits::is_projected_range<
                    Proj1, Rng1>::value&& hpx::traits::is_range<Rng2>::value&&
                    traits::is_projected_range<Proj2, Rng2>::value&&
                        traits::is_indirect_callable<ExPolicy, Pred,
                            traits::projected_range<Proj1, Rng1>,
                            traits::projected_range<Proj2, Rng2>>::value)>
    typename util::detail::algorithm_result<ExPolicy,
        typename hpx::traits::range_iterator<Rng1>::type>::type
    find_first_of(ExPolicy&& policy, Rng1&& rng1, Rng2&& rng2,
        Pred&& op = Pred(), Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2())
    {
        return find_first_of(std::forward<ExPolicy>(policy),
            hpx::util::begin(rng1), hpx::util::end(rng1),
            hpx::util::begin(rng2), hpx::util::end(rng2),
            std::forward<Pred>(op), std::forward<Proj1>(proj1),
            std::forward<Proj2>(proj2));
    }

}}}    // namespace hpx::parallel::v1

#endif
