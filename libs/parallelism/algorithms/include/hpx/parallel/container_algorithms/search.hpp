//  Copyright (c) 2018 Christopher Ogle
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/search.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/parallel/algorithms/search.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {
    /// Searches the range [first, last) for any elements in the range [s_first, s_last).
    /// Uses a provided predicate to compare elements.
    ///
    /// \note   Complexity: at most (S*N) comparisons where
    ///         \a S = distance(s_first, s_last) and
    ///         \a N = distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng1        The type of the examine range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Rng2        The type of the search range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a adjacent_find requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj1       The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity and is applied
    ///                     to the elements of \a Rng1.
    /// \tparam Proj2       The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity and is applied
    ///                     to the elements of \a Rng2.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng1         Refers to the sequence of elements the algorithm
    ///                     will be examining.
    /// \param rng2         Refers to the sequence of elements the algorithm
    ///                     will be searching for.
    /// \param op           Refers to the binary predicate which returns true if the
    ///                     elements should be treated as equal. the signature of
    ///                     the function should be equivalent to
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such
    ///                     that objects of types \a FwdIter1 and \a FwdIter2 can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type1 and \a Type2 respectively
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of \a rng1
    ///                     as a projection operation before the actual
    ///                     predicate \a is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of \a rng2
    ///                     as a projection operation before the actual
    ///                     predicate \a is invoked.
    ///
    /// The comparison operations in the parallel \a search algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a search algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a search algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type \a task_execution_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a search algorithm returns an iterator to the beginning of
    ///           the first subsequence [s_first, s_last) in range [first, last).
    ///           If the length of the subsequence [s_first, s_last) is greater
    ///           than the length of the range [first, last), \a last is returned.
    ///           Additionally if the size of the subsequence is empty \a first is
    ///           returned. If no subsequence is found, \a last is returned.
    ///
    template <typename ExPolicy, typename Rng1, typename Rng2,
        typename Pred = detail::equal_to,
        typename Proj1 = util::projection_identity,
        typename Proj2 = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value&& hpx::traits::is_range<
                Rng1>::value&& traits::is_projected_range<Proj1, Rng1>::value&&
                hpx::traits::is_range<Rng2>::value&& traits::is_projected_range<
                    Proj2, Rng2>::value&& traits::is_indirect_callable<ExPolicy,
                    Pred, traits::projected_range<Proj1, Rng1>,
                    traits::projected_range<Proj2, Rng2>>::value)>
    typename util::detail::algorithm_result<ExPolicy,
        typename hpx::traits::range_iterator<Rng1>::type>::type
    search(ExPolicy&& policy, Rng1&& rng1, Rng2&& rng2, Pred&& op = Pred(),
        Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2())
    {
        return search(std::forward<ExPolicy>(policy), hpx::util::begin(rng1),
            hpx::util::end(rng1), hpx::util::begin(rng2), hpx::util::end(rng2),
            std::forward<Pred>(op), std::forward<Proj1>(proj1),
            std::forward<Proj2>(proj2));
    }

    /// Searches the range [first, last) for any elements in the range [s_first, s_last).
    /// Uses a provided predicate to compare elements.
    ///
    /// \note   Complexity: at most (S*N) comparisons where
    ///         \a S = distance(s_first, s_last) and
    ///         \a N = distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng1        The type of the examine range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Rng2        The type of the search range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a adjacent_find requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj1       The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity and is applied
    ///                     to the elements of \a Rng1.
    /// \tparam Proj2       The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity and is applied
    ///                     to the elements of \a Rng2.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng1         Refers to the sequence of elements the algorithm
    ///                     will be examining.
    /// \param count        The number of elements to apply the algorithm on.
    /// \param rng2         Refers to the sequence of elements the algorithm
    ///                     will be searching for.
    /// \param op           Refers to the binary predicate which returns true if the
    ///                     elements should be treated as equal. the signature of
    ///                     the function should be equivalent to
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such
    ///                     that objects of types \a FwdIter1 and \a FwdIter2 can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type1 and \a Type2 respectively
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of \a rng1
    ///                     as a projection operation before the actual
    ///                     predicate \a is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of \a rng2
    ///                     as a projection operation before the actual
    ///                     predicate \a is invoked.
    ///
    /// The comparison operations in the parallel \a search algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a search algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a search algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type \a task_execution_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a search algorithm returns an iterator to the beginning of
    ///           the first subsequence [s_first, s_last) in range [first, last).
    ///           If the length of the subsequence [s_first, s_last) is greater
    ///           than the length of the range [first, last), \a last is returned.
    ///           Additionally if the size of the subsequence is empty \a first is
    ///           returned. If no subsequence is found, \a last is returned.
    ///
    template <typename ExPolicy, typename Rng1, typename Rng2,
        typename Pred = detail::equal_to,
        typename Proj1 = util::projection_identity,
        typename Proj2 = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value&& hpx::traits::is_range<
                Rng1>::value&& traits::is_projected_range<Proj1, Rng1>::value&&
                hpx::traits::is_range<Rng2>::value&& traits::is_projected_range<
                    Proj2, Rng2>::value&& traits::is_indirect_callable<ExPolicy,
                    Pred, traits::projected_range<Proj1, Rng1>,
                    traits::projected_range<Proj2, Rng2>>::value)>
    typename util::detail::algorithm_result<ExPolicy,
        typename hpx::traits::range_iterator<Rng1>::type>::type
    search_n(ExPolicy&& policy, Rng1&& rng1, std::size_t count, Rng2&& rng2,
        Pred&& op = Pred(), Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2())
    {
        return search_n(std::forward<ExPolicy>(policy), hpx::util::begin(rng1),
            count, hpx::util::begin(rng2), hpx::util::end(rng2),
            std::forward<Pred>(op), std::forward<Proj1>(proj1),
            std::forward<Proj2>(proj2));
    }
}}}    // namespace hpx::parallel::v1
