//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2018 Christopher Ogle
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/search.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/parallel/algorithms/detail/search.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

#if defined(DOXYGEN)

namespace hpx { namespace ranges {

    /// Searches the range [first, last) for any elements in the range [s_first, s_last).
    /// Uses a provided predicate to compare elements.
    ///
    /// \note   Complexity: at most (S*N) comparisons where
    ///         \a S = distance(s_first, s_last) and
    ///         \a N = distance(first, last).
    ///
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel.
    /// \tparam FwdIter2    The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent2       The type of the source sentinel used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a adjacent_find requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj1       The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity and is applied
    ///                     to the elements of type dereferenced \a FwdIter.
    /// \tparam Proj2       The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity and is applied
    ///                     to the elements of type dereferenced \a FwdIter2.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param s_first      Refers to the beginning of the sequence of elements
    ///                     the algorithm will be searching for.
    /// \param s_last       Refers to the end of the sequence of elements of
    ///                     the algorithm will be searching for.
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
    ///                     will be invoked for each of the elements of type
    ///                     dereferenced \a FwdIter1 as a projection operation
    ///                     before the actual predicate \a is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of type
    ///                     dereferenced \a FwdIter2 as a projection operation
    ///                     before the actual predicate \a is invoked.
    ///
    /// The comparison operations in the parallel \a search algorithm execute
    /// in sequential order in the calling thread.
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
    template <typename FwdIter, typename Sent, typename FwdIter2,
        typename Sent2, typename Pred = hpx::ranges::equal_to,
        typename Proj1 = parallel::util::projection_identity,
        typename Proj2 = parallel::util::projection_identity>
    FwdIter search(FwdIter first, Sent last, FwdIter2 s_first, Sent2 s_last,
        Pred&& op = Pred(), Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2());

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
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent        The type of the source sentinel used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel.
    /// \tparam FwdIter2    The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent2       The type of the source sentinel used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a adjacent_find requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj1       The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity and is applied
    ///                     to the elements of type dereferenced \a FwdIter.
    /// \tparam Proj2       The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity and is applied
    ///                     to the elements of type dereferenced \a FwdIter2.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param s_first      Refers to the beginning of the sequence of elements
    ///                     the algorithm will be searching for.
    /// \param s_last       Refers to the end of the sequence of elements of
    ///                     the algorithm will be searching for.
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
    ///                     will be invoked for each of the elements of type
    ///                     dereferenced \a FwdIter1 as a projection operation
    ///                     before the actual predicate \a is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of type
    ///                     dereferenced \a FwdIter2 as a projection operation
    ///                     before the actual predicate \a is invoked.
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
    template <typename ExPolicy, typename FwdIter, typename Sent,
        typename FwdIter2, typename Sent2,
        typename Pred = hpx::ranges::equal_to,
        typename Proj1 = parallel::util::projection_identity,
        typename Proj2 = parallel::util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type search(
        ExPolicy&& policy, FwdIter first, Sent last, FwdIter2 s_first,
        Sent2 s_last, Pred&& op = Pred(), Proj1&& proj1 = Proj1(),
        Proj2&& proj2 = Proj2());

    /// Searches the range [first, last) for any elements in the range [s_first, s_last).
    /// Uses a provided predicate to compare elements.
    ///
    /// \note   Complexity: at most (S*N) comparisons where
    ///         \a S = distance(s_first, s_last) and
    ///         \a N = distance(first, last).
    ///
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
    /// The comparison operations in the parallel \a search algorithm execute
    /// in sequential order in the calling thread.
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
    template <typename Rng1, typename Rng2,
        typename Pred = hpx::ranges::equal_to,
        typename Proj1 = hpx::parallel::util::projection_identity,
        typename Proj2 = hpx::parallel::util::projection_identity>
    typename hpx::traits::range_iterator<Rng1>::type search(Rng1&& rng1,
        Rng2&& rng2, Pred&& op = Pred(), Proj1&& proj1 = Proj1(),
        Proj2&& proj2 = Proj2());

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
        typename Pred = hpx::ranges::equal_to,
        typename Proj1 = hpx::parallel::util::projection_identity,
        typename Proj2 = hpx::parallel::util::projection_identity>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        typename hpx::traits::range_iterator<Rng1>::type>::type
    search(ExPolicy&& policy, Rng1&& rng1, Rng2&& rng2, Pred&& op = Pred(),
        Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2());

    /// Searches the range [first, last) for any elements in the range [s_first, s_last).
    /// Uses a provided predicate to compare elements.
    ///
    /// \note   Complexity: at most (S*N) comparisons where
    ///         \a S = distance(s_first, s_last) and
    ///         \a N = count.
    ///
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent2       The type of the source sentinel used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a adjacent_find requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    ///
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param count        Refers to the range of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param s_first      Refers to the beginning of the sequence of elements
    ///                     the algorithm will be searching for.
    /// \param s_last       Refers to the end of the sequence of elements of
    ///                     the algorithm will be searching for.
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
    ///                     will be invoked for each of the elements of type
    ///                     dereferenced \a FwdIter1 as a projection operation
    ///                     before the actual predicate \a is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of type
    ///                     dereferenced \a FwdIter2 as a projection operation
    ///                     before the actual predicate \a is invoked.
    ///
    /// The comparison operations in the parallel \a search_n algorithm execute
    /// in sequential order in the calling thread.
    ///
    /// \returns  The \a search_n algorithm returns \a FwdIter.
    ///           The \a search_n algorithm returns an iterator to the beginning of
    ///           the last subsequence [s_first, s_last) in range [first, first+count).
    ///           If the length of the subsequence [s_first, s_last) is greater
    ///           than the length of the range [first, first+count),
    ///           \a first is returned.
    ///           Additionally if the size of the subsequence is empty or no subsequence
    ///           is found, \a first is also returned.
    ///
    template <typename FwdIter, typename FwdIter2, typename Sent2,
        typename Pred = hpx::ranges::equal_to,
        typename Proj1 = parallel::util::projection_identity,
        typename Proj2 = parallel::util::projection_identity>
    FwdIter search_n(ExPolicy&& policy, FwdIter first, std::size_t count,
        FwdIter2 s_first, Sent s_last, Pred&& op = Pred(),
        Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2());

    /// Searches the range [first, last) for any elements in the range [s_first, s_last).
    /// Uses a provided predicate to compare elements.
    ///
    /// \note   Complexity: at most (S*N) comparisons where
    ///         \a S = distance(s_first, s_last) and
    ///         \a N = count.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent2       The type of the source sentinel used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a adjacent_find requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    ///
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param count        Refers to the range of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param s_first      Refers to the beginning of the sequence of elements
    ///                     the algorithm will be searching for.
    /// \param s_last       Refers to the end of the sequence of elements of
    ///                     the algorithm will be searching for.
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
    ///                     will be invoked for each of the elements of type
    ///                     dereferenced \a FwdIter1 as a projection operation
    ///                     before the actual predicate \a is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of type
    ///                     dereferenced \a FwdIter2 as a projection operation
    ///                     before the actual predicate \a is invoked.
    ///
    /// The comparison operations in the parallel \a search_n algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a search_n algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a search_n algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type \a task_execution_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a search_n algorithm returns an iterator to the beginning of
    ///           the last subsequence [s_first, s_last) in range [first, first+count).
    ///           If the length of the subsequence [s_first, s_last) is greater
    ///           than the length of the range [first, first+count),
    ///           \a first is returned.
    ///           Additionally if the size of the subsequence is empty or no subsequence
    ///           is found, \a first is also returned.
    ///
    template <typename ExPolicy, typename FwdIter, typename FwdIter2,
        typename Sent2, typename Pred = hpx::ranges::equal_to,
        typename Proj1 = parallel::util::projection_identity,
        typename Proj2 = parallel::util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type search_n(
        ExPolicy&& policy, FwdIter first, std::size_t count, FwdIter2 s_first,
        Sent2 s_last, Pred&& op = Pred(), Proj1&& proj1 = Proj1(),
        Proj2&& proj2 = Proj2());

    /// Searches the range [first, last) for any elements in the range [s_first, s_last).
    /// Uses a provided predicate to compare elements.
    ///
    /// \note   Complexity: at most (S*N) comparisons where
    ///         \a S = distance(s_first, s_last) and
    ///         \a N = distance(first, last).
    ///
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
    /// The comparison operations in the parallel \a search algorithm execute
    /// in sequential order in the calling thread.
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
    template <typename Rng1, typename Rng2,
        typename Pred = hpx::ranges::equal_to,
        typename Proj1 = hpx::parallel::util::projection_identity,
        typename Proj2 = hpx::parallel::util::projection_identity>
    typename hpx::traits::range_iterator<Rng1>::type search_n(Rng1&& rng1,
        std::size_t count, Rng2&& rng2, Pred&& op = Pred(),
        Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2());

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
        typename Pred = hpx::ranges::equal_to,
        typename Proj1 = hpx::parallel::util::projection_identity,
        typename Proj2 = hpx::parallel::util::projection_identity>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        typename hpx::traits::range_iterator<Rng1>::type>::type
    search_n(ExPolicy&& policy, Rng1&& rng1, std::size_t count, Rng2&& rng2,
        Pred&& op = Pred(), Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2());

}}    // namespace hpx::ranges

#else

namespace hpx { namespace parallel { inline namespace v1 {

    // clang-format off
    template <typename ExPolicy, typename Rng1, typename Rng2,
        typename Pred = hpx::ranges::equal_to,
        typename Proj1 = parallel::util::projection_identity,
        typename Proj2 = parallel::util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_range<Rng1>::value &&
            traits::is_projected_range<Proj1, Rng1>::value &&
            hpx::traits::is_range<Rng2>::value &&
            traits::is_projected_range<Proj2, Rng2>::value &&
            traits::is_indirect_callable<ExPolicy,
                Pred, traits::projected_range<Proj1, Rng1>,
                traits::projected_range<Proj2, Rng2>
            >::value
        )>
    HPX_DEPRECATED_V(1, 6, "Please use hpx::ranges::search instead.")
    // clang-format on
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

    // clang-format off
    template <typename ExPolicy, typename Rng1, typename Rng2,
        typename Pred = hpx::ranges::equal_to,
        typename Proj1 = parallel::util::projection_identity,
        typename Proj2 = parallel::util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_range<Rng1>::value &&
            traits::is_projected_range<Proj1, Rng1>::value &&
            hpx::traits::is_range<Rng2>::value &&
            traits::is_projected_range<Proj2, Rng2>::value &&
            traits::is_indirect_callable<ExPolicy,
                Pred, traits::projected_range<Proj1, Rng1>,
                traits::projected_range<Proj2, Rng2>
            >::value
        )>
    HPX_DEPRECATED_V(1, 6, "Please use hpx::ranges::search_n instead.")
    // clang-format on
    typename util::detail::algorithm_result<ExPolicy,
        typename hpx::traits::range_iterator<Rng1>::type>::type
        search_n(ExPolicy&& policy, Rng1&& rng1, std::size_t count, Rng2&& rng2,
            Pred&& op = Pred(), Proj1&& proj1 = Proj1(),
            Proj2&& proj2 = Proj2())
    {
        return search_n(std::forward<ExPolicy>(policy), hpx::util::begin(rng1),
            count, hpx::util::begin(rng2), hpx::util::end(rng2),
            std::forward<Pred>(op), std::forward<Proj1>(proj1),
            std::forward<Proj2>(proj2));
    }

}}}    // namespace hpx::parallel::v1

namespace hpx { namespace ranges {

    HPX_INLINE_CONSTEXPR_VARIABLE struct search_t final
      : hpx::functional::tag<search_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Sent, typename FwdIter2,
            typename Sent2, typename Pred = hpx::ranges::equal_to,
            typename Proj1 = parallel::util::projection_identity,
            typename Proj2 = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_forward_iterator<FwdIter>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value &&
                parallel::traits::is_projected<Proj1, FwdIter>::value &&
                hpx::traits::is_forward_iterator<FwdIter2>::value &&
                hpx::traits::is_sentinel_for<Sent2, FwdIter2>::value &&
                parallel::traits::is_projected<Proj2, FwdIter2>::value &&
                parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy,
                    Pred,
                    parallel::traits::projected<Proj1, FwdIter>,
                    parallel::traits::projected<Proj2, FwdIter2>
                >::value
            )>
        // clang-format on
        friend FwdIter tag_invoke(hpx::ranges::search_t, FwdIter first,
            Sent last, FwdIter2 s_first, Sent2 s_last, Pred&& op = Pred(),
            Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2())
        {
            return hpx::parallel::v1::detail::search<FwdIter, Sent>().call(
                hpx::execution::seq, std::true_type{}, first, last, s_first,
                s_last, std::forward<Pred>(op), std::forward<Proj1>(proj1),
                std::forward<Proj2>(proj2));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent,
                 typename FwdIter2, typename Sent2,
            typename Pred = hpx::ranges::equal_to,
            typename Proj1 = parallel::util::projection_identity,
            typename Proj2 = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_forward_iterator<FwdIter>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value &&
                parallel::traits::is_projected<Proj1, FwdIter>::value &&
                hpx::traits::is_forward_iterator<FwdIter2>::value &&
                hpx::traits::is_sentinel_for<Sent2, FwdIter2>::value &&
                parallel::traits::is_projected<Proj2, FwdIter2>::value &&
                parallel::traits::is_indirect_callable<ExPolicy,
                    Pred,
                    parallel::traits::projected<Proj1, FwdIter>,
                    parallel::traits::projected<Proj2, FwdIter2>
                >::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_invoke(hpx::ranges::search_t, ExPolicy&& policy, FwdIter first,
            Sent last, FwdIter2 s_first, Sent2 s_last, Pred&& op = Pred(),
            Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2())
        {
            using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

            return hpx::parallel::v1::detail::search<FwdIter, Sent>().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last, s_first,
                s_last, std::forward<Pred>(op), std::forward<Proj1>(proj1),
                std::forward<Proj2>(proj2));
        }

        // clang-format off
        template <typename Rng1, typename Rng2,
            typename Pred = hpx::ranges::equal_to,
            typename Proj1 = hpx::parallel::util::projection_identity,
            typename Proj2 = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng1>::value &&
                hpx::parallel::traits::is_projected_range<Proj1, Rng1>::value &&
                hpx::traits::is_range<Rng2>::value &&
                hpx::parallel::traits::is_projected_range<Proj2, Rng2>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy,
                    Pred, hpx::parallel::traits::projected_range<Proj1, Rng1>,
                    hpx::parallel::traits::projected_range<Proj2, Rng2>
                >::value
            )>
        // clang-format on
        friend typename hpx::traits::range_iterator<Rng1>::type tag_invoke(
            hpx::ranges::search_t, Rng1&& rng1, Rng2&& rng2, Pred&& op = Pred(),
            Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2())
        {
            using fwditer_type =
                typename hpx::traits::range_iterator<Rng1>::type;
            using sent_type = typename hpx::traits::range_sentinel<Rng1>::type;

            return hpx::parallel::v1::detail::search<fwditer_type, sent_type>()
                .call(hpx::execution::seq, std::true_type{},
                    hpx::util::begin(rng1), hpx::util::end(rng1),
                    hpx::util::begin(rng2), hpx::util::end(rng2),
                    std::forward<Pred>(op), std::forward<Proj1>(proj1),
                    std::forward<Proj2>(proj2));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng1, typename Rng2,
            typename Pred = hpx::ranges::equal_to,
            typename Proj1 = hpx::parallel::util::projection_identity,
            typename Proj2 = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng1>::value &&
                hpx::parallel::traits::is_projected_range<Proj1, Rng1>::value &&
                hpx::traits::is_range<Rng2>::value &&
                hpx::parallel::traits::is_projected_range<Proj2, Rng2>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy,
                    Pred, hpx::parallel::traits::projected_range<Proj1, Rng1>,
                    hpx::parallel::traits::projected_range<Proj2, Rng2>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            typename hpx::traits::range_iterator<Rng1>::type>::type
        tag_invoke(hpx::ranges::search_t, ExPolicy&& policy, Rng1&& rng1,
            Rng2&& rng2, Pred&& op = Pred(), Proj1&& proj1 = Proj1(),
            Proj2&& proj2 = Proj2())
        {
            using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;
            using fwditer_type =
                typename hpx::traits::range_iterator<Rng1>::type;
            using sent_type = typename hpx::traits::range_sentinel<Rng1>::type;

            return hpx::parallel::v1::detail::search<fwditer_type, sent_type>()
                .call(std::forward<ExPolicy>(policy), is_seq(),
                    hpx::util::begin(rng1), hpx::util::end(rng1),
                    hpx::util::begin(rng2), hpx::util::end(rng2),
                    std::forward<Pred>(op), std::forward<Proj1>(proj1),
                    std::forward<Proj2>(proj2));
        }

    } search{};

    HPX_INLINE_CONSTEXPR_VARIABLE struct search_n_t final
      : hpx::functional::tag<search_n_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename FwdIter2, typename Sent2,
            typename Pred = hpx::ranges::equal_to,
            typename Proj1 = parallel::util::projection_identity,
            typename Proj2 = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_forward_iterator<FwdIter>::value &&
                parallel::traits::is_projected<Proj1, FwdIter>::value &&
                hpx::traits::is_forward_iterator<FwdIter2>::value &&
                hpx::traits::is_sentinel_for<Sent2, FwdIter2>::value &&
                parallel::traits::is_projected<Proj2, FwdIter2>::value &&
                parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Pred,
                    parallel::traits::projected<Proj1, FwdIter>,
                    parallel::traits::projected<Proj2, FwdIter2>
                >::value
            )>
        // clang-format on
        friend FwdIter tag_invoke(hpx::ranges::search_n_t, FwdIter first,
            std::size_t count, FwdIter2 s_first, Sent2 s_last,
            Pred&& op = Pred(), Proj1&& proj1 = Proj1(),
            Proj2&& proj2 = Proj2())
        {
            return hpx::parallel::v1::detail::search_n<FwdIter, FwdIter>().call(
                hpx::execution::seq, std::true_type{}, first, count, s_first,
                s_last, std::forward<Pred>(op), std::forward<Proj1>(proj1),
                std::forward<Proj2>(proj2));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename FwdIter2,
            typename Sent2,
            typename Pred = hpx::ranges::equal_to,
            typename Proj1 = parallel::util::projection_identity,
            typename Proj2 = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_forward_iterator<FwdIter>::value &&
                parallel::traits::is_projected<Proj1, FwdIter>::value &&
                hpx::traits::is_forward_iterator<FwdIter2>::value &&
                hpx::traits::is_sentinel_for<Sent2, FwdIter2>::value &&
                parallel::traits::is_projected<Proj2, FwdIter2>::value&&
                parallel::traits::is_indirect_callable<ExPolicy, Pred,
                    parallel::traits::projected<Proj1, FwdIter>,
                    parallel::traits::projected<Proj2, FwdIter2>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_invoke(hpx::ranges::search_n_t, ExPolicy&& policy, FwdIter first,
            std::size_t count, FwdIter2 s_first, Sent2 s_last,
            Pred&& op = Pred(), Proj1&& proj1 = Proj1(),
            Proj2&& proj2 = Proj2())
        {
            using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

            return hpx::parallel::v1::detail::search_n<FwdIter, FwdIter>().call(
                std::forward<ExPolicy>(policy), is_seq(), first, count, s_first,
                s_last, std::forward<Pred>(op), std::forward<Proj1>(proj1),
                std::forward<Proj2>(proj2));
        }

        // clang-format off
        template <typename Rng1, typename Rng2,
            typename Pred = hpx::ranges::equal_to,
            typename Proj1 = hpx::parallel::util::projection_identity,
            typename Proj2 = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng1>::value &&
                hpx::parallel::traits::is_projected_range<Proj1, Rng1>::value &&
                hpx::traits::is_range<Rng2>::value &&
                hpx::parallel::traits::is_projected_range<Proj2, Rng2>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy,
                    Pred, hpx::parallel::traits::projected_range<Proj1, Rng1>,
                    hpx::parallel::traits::projected_range<Proj2, Rng2>
                >::value
            )>
        // clang-format on
        friend typename hpx::traits::range_iterator<Rng1>::type tag_invoke(
            hpx::ranges::search_n_t, Rng1&& rng1, std::size_t count,
            Rng2&& rng2, Pred&& op = Pred(), Proj1&& proj1 = Proj1(),
            Proj2&& proj2 = Proj2())
        {
            using fwditer_type =
                typename hpx::traits::range_iterator<Rng1>::type;
            using sent_type = typename hpx::traits::range_sentinel<Rng1>::type;

            return hpx::parallel::v1::detail::search_n<fwditer_type,
                sent_type>()
                .call(hpx::execution::seq, std::true_type{},
                    hpx::util::begin(rng1), count, hpx::util::begin(rng2),
                    hpx::util::end(rng2), std::forward<Pred>(op),
                    std::forward<Proj1>(proj1), std::forward<Proj2>(proj2));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng1, typename Rng2,
            typename Pred = hpx::ranges::equal_to,
            typename Proj1 = hpx::parallel::util::projection_identity,
            typename Proj2 = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng1>::value &&
                hpx::parallel::traits::is_projected_range<Proj1, Rng1>::value &&
                hpx::traits::is_range<Rng2>::value &&
                hpx::parallel::traits::is_projected_range<Proj2, Rng2>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy,
                    Pred, hpx::parallel::traits::projected_range<Proj1, Rng1>,
                    hpx::parallel::traits::projected_range<Proj2, Rng2>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            typename hpx::traits::range_iterator<Rng1>::type>::type
        tag_invoke(hpx::ranges::search_n_t, ExPolicy&& policy, Rng1&& rng1,
            std::size_t count, Rng2&& rng2, Pred&& op = Pred(),
            Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2())
        {
            using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;
            using fwditer_type =
                typename hpx::traits::range_iterator<Rng1>::type;
            using sent_type = typename hpx::traits::range_sentinel<Rng1>::type;

            return hpx::parallel::v1::detail::search_n<fwditer_type,
                sent_type>()
                .call(std::forward<ExPolicy>(policy), is_seq(),
                    hpx::util::begin(rng1), count, hpx::util::begin(rng2),
                    hpx::util::end(rng2), std::forward<Pred>(op),
                    std::forward<Proj1>(proj1), std::forward<Proj2>(proj2));
        }

    } search_n{};

}}    // namespace hpx::ranges
#endif
