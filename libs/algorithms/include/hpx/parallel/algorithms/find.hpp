//  Copyright (c) 2014 Grant Mercer
//  Copyright (c) 2017-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/find.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    // clang-format off

    /// Returns the first element in the range [first, last) that is equal
    /// to value
    ///
    /// \note   Complexity: At most last - first
    ///         applications of the operator==().
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam T           The type of the value to find (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param val          the value to compare the elements to
    ///
    /// The comparison operations in the parallel \a find algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a find algorithm returns the first element in the range
    ///           [first,last) that is equal to \a val.
    ///           If no such element in the range of [first,last) is equal to
    ///           \a val, then the algorithm returns \a last.
    ///
    template <typename ExPolicy, typename FwdIter, typename T>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    find(ExPolicy&& policy, FwdIter first, FwdIter last, T const& val);

    /// Returns the first element in the range [first, last) for which
    /// predicate \a f returns true
    ///
    /// \note   Complexity: At most last - first
    ///         applications of the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param f            The unary predicate which returns true for the
    ///                     required element. The signature of the predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such
    ///                     that objects of type \a FwdIter can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type.
    ///
    /// The comparison operations in the parallel \a find_if algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find_if algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find_if algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a find_if algorithm returns the first element in the range
    ///           [first,last) that satisfies the predicate \a f.
    ///           If no such element exists that satisfies the predicate f, the
    ///           algorithm returns \a last.
    ///
    template <typename ExPolicy, typename FwdIter, typename F>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    find_if(ExPolicy&& policy, FwdIter first, FwdIter last, F&& f);

    /// Returns the first element in the range [first, last) for which
    /// predicate \a f returns false
    ///
    /// \note   Complexity: At most last - first
    ///         applications of the predicate.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam F           The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a F to meet the
    ///                     requirements of \a CopyConstructible.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param f            The unary predicate which returns false for the
    ///                     required element. The signature of the predicate
    ///                     should be equivalent to:
    ///                     \code
    ///                     bool pred(const Type &a);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such
    ///                     that objects of type \a FwdIter can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type.
    ///
    /// The comparison operations in the parallel \a find_if_not algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a find_if_not algorithm invoked
    /// with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a find_if_not algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a find_if_not algorithm returns the first element in the range
    ///           [first, last) that does \b not satisfy the predicate \a f.
    ///           If no such element exists that does not satisfy the predicate f, the
    ///           algorithm returns \a last.
    ///
    template <typename ExPolicy, typename FwdIter, typename F>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    find_if_not(ExPolicy&& policy, FwdIter first, FwdIter last, F&& f);

    /// Returns the last subsequence of elements [first2, last2) found in the range
    /// [first, last) using the given predicate \a f to compare elements.
    ///
    /// \note   Complexity: at most S*(N-S+1) comparisons where
    ///         \a S = distance(first2, last2) and
    ///         \a N = distance(first1, last1).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam FwdIter2    The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a replace requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity and is applied
    ///                     to the elements of type dereferenced \a FwdIter1
    ///                     and dereferenced \a FwdIter2.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last1        Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the sequence of elements
    ///                     the algorithm will be searching for.
    /// \param last2        Refers to the end of the sequence of elements of
    ///                     the algorithm will be searching for.
    /// \param op           The binary predicate which returns \a true
    ///                     if the elements should be treated as equal. The signature
    ///                     should be equivalent to the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such
    ///                     that objects of types \a FwdIter1 and \a FwdIter2
    ///                     can be dereferenced and then implicitly converted
    ///                     to \a Type1 and \a Type2 respectively.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of type
    ///                     dereferenced \a FwdIter1 and dereferenced \a FwdIter2
    ///                     as a projection operation before the function \a f
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
    /// \returns  The \a find_end algorithm returns a \a hpx::future<FwdIter> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a find_end algorithm returns an iterator to the beginning of
    ///           the last subsequence [first2, last2) in range [first, last).
    ///           If the length of the subsequence [first2, last2) is greater
    ///           than the length of the range [first1, last1), \a last1 is returned.
    ///           Additionally if the size of the subsequence is empty or no subsequence
    ///           is found, \a last1 is also returned.
    ///
    /// This overload of \a find_end is available if the user decides to provide the
    /// algorithm their own predicate \a f.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Pred = detail::equal_to>
    typename util::detail::algorithm_result<ExPolicy, FwdIter1>::type
    find_end(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
        FwdIter2 first2, FwdIter2 last2, Pred&& op = Pred());

    /// Searches the range [first, last) for any elements in the range [s_first, s_last).
    /// Uses binary predicate p to compare elements
    ///
    /// \note   Complexity: at most (S*N) comparisons where
    ///         \a S = distance(s_first, s_last) and
    ///         \a N = distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used for the
    ///                     first range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward  iterator.
    /// \tparam FwdIter2    The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a equal requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
    /// \tparam Proj1       The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity and is applied
    ///                     to the elements of type dereferenced \a FwdIter1.
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
    /// \param op           The binary predicate which returns \a true
    ///                     if the elements should be treated as equal. The signature
    ///                     should be equivalent to the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such
    ///                     that objects of types \a FwdIter1 and \a FwdIter2
    ///                     can be dereferenced and then implicitly converted
    ///                     to \a Type1 and \a Type2 respectively.
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of type
    ///                     dereferenced \a FwdIter1 as a projection operation
    ///                     before the function \a op is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of type
    ///                     dereferenced \a FwdIter2 as a projection operation
    ///                     before the function \a op is invoked.
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
    /// \returns  The \a find_first_of algorithm returns a \a hpx::future<FwdIter1> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter1 otherwise.
    ///           The \a find_first_of algorithm returns an iterator to the first element
    ///           in the range [first, last) that is equal to an element from the range
    ///           [s_first, s_last).
    ///           If the length of the subsequence [s_first, s_last) is
    ///           greater than the length of the range [first, last),
    ///           \a last is returned.
    ///           Additionally if the size of the subsequence is empty or no subsequence
    ///           is found, \a last is also returned.
    ///           This overload of \a find_end is available if
    ///           the user decides to provide the
    ///           algorithm their own predicate \a f.
    ///
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Pred = detail::equal_to>
    typename util::detail::algorithm_result<ExPolicy, FwdIter1>::type
    find_first_of(ExPolicy&& policy, FwdIter1 first, FwdIter1 last,
        FwdIter2 s_first, FwdIter2 s_last, Pred&& op = Pred());

    // clang-format on
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/advance_to_sentinel.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/util/compare_projected.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/invoke_projected.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 {

    ///////////////////////////////////////////////////////////////////////////
    // find
    namespace detail {

        template <typename Iter, typename Sent, typename T, typename Proj>
        constexpr Iter sequential_find(
            Iter first, Sent last, T const& value, Proj&& proj)
        {
            for (/**/; first != last; ++first)
            {
                if (hpx::util::invoke(proj, *first) == value)
                {
                    break;
                }
            }
            return first;
        }

        template <typename FwdIter>
        struct find : public detail::algorithm<find<FwdIter>, FwdIter>
        {
            find()
              : find::algorithm("find")
            {
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename T, typename Proj = util::projection_identity>
            static constexpr Iter sequential(ExPolicy, Iter first, Sent last,
                T const& val, Proj&& proj = Proj())
            {
                return sequential_find(
                    first, last, val, std::forward<Proj>(proj));
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename T, typename Proj = util::projection_identity>
            static typename util::detail::algorithm_result<ExPolicy, Iter>::type
            parallel(ExPolicy&& policy, Iter first, Sent last, T const& val,
                Proj&& proj = Proj())
            {
                typedef util::detail::algorithm_result<ExPolicy, Iter> result;
                typedef typename std::iterator_traits<Iter>::value_type type;
                typedef typename std::iterator_traits<Iter>::difference_type
                    difference_type;

                difference_type count = detail::distance(first, last);
                if (count <= 0)
                    return result::get(std::move(last));

                util::cancellation_token<std::size_t> tok(count);

                auto f1 = [val, proj = std::forward<Proj>(proj), tok](Iter it,
                              std::size_t part_size,
                              std::size_t base_idx) mutable -> void {
                    util::loop_idx_n(base_idx, it, part_size, tok,
                        [&val, &proj, &tok](type& v, std::size_t i) -> void {
                            if (hpx::util::invoke(proj, v) == val)
                            {
                                tok.cancel(i);
                            }
                        });
                };

                auto f2 =
                    [tok, count, first, last](
                        std::vector<hpx::future<void>>&&) mutable -> Iter {
                    difference_type find_res =
                        static_cast<difference_type>(tok.get_data());

                    if (find_res != count)
                    {
                        std::advance(first, find_res);
                    }
                    else
                    {
                        first = detail::advance_to_sentinel(first, last);
                    }
                    return std::move(first);
                };

                return util::partitioner<ExPolicy, Iter, void>::call_with_index(
                    std::forward<ExPolicy>(policy), first, count, 1,
                    std::move(f1), std::move(f2));
            }
        };

        template <typename ExPolicy, typename Iter, typename Sent, typename T,
            typename Proj>
        inline typename util::detail::algorithm_result<ExPolicy, Iter>::type
        find_(ExPolicy&& policy, Iter first, Sent last, T const& val,
            Proj&& proj, std::false_type)
        {
            static_assert((hpx::traits::is_forward_iterator<Iter>::value),
                "Requires at least forward iterator.");

            typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;

            return detail::find<Iter>().call(std::forward<ExPolicy>(policy),
                is_seq(), first, last, val, std::forward<Proj>(proj));
        }

        template <typename ExPolicy, typename FwdIter, typename T,
            typename Proj>
        inline typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        find_(ExPolicy&& policy, FwdIter first, FwdIter last, T const& val,
            Proj&& proj, std::true_type);
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter, typename T,
        HPX_CONCEPT_REQUIRES_(
            execution::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(
        1, 6, "hpx::parallel::find is deprecated, use hpx::find instead")
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        find(ExPolicy&& policy, FwdIter first, FwdIter last, T const& val)
    {
        using is_segmented = hpx::traits::is_segmented_iterator<FwdIter>;

        return detail::find_(std::forward<ExPolicy>(policy), first, last,
            std::move(val), util::projection_identity(), is_segmented());
    }

    ///////////////////////////////////////////////////////////////////////////
    // find_if
    namespace detail {

        template <typename Iter, typename Sent, typename Pred, typename Proj>
        constexpr Iter sequential_find_if(
            Iter first, Sent last, Pred&& pred, Proj&& proj)
        {
            for (/**/; first != last; ++first)
            {
                if (hpx::util::invoke(pred, hpx::util::invoke(proj, *first)))
                {
                    break;
                }
            }
            return first;
        }

        template <typename FwdIter>
        struct find_if : public detail::algorithm<find_if<FwdIter>, FwdIter>
        {
            find_if()
              : find_if::algorithm("find_if")
            {
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename F, typename Proj = util::projection_identity>
            static constexpr Iter sequential(
                ExPolicy, Iter first, Sent last, F&& f, Proj&& proj = Proj())
            {
                return sequential_find_if(
                    first, last, std::forward<F>(f), std::forward<Proj>(proj));
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename F, typename Proj = util::projection_identity>
            static typename util::detail::algorithm_result<ExPolicy, Iter>::type
            parallel(ExPolicy&& policy, Iter first, Sent last, F&& f,
                Proj&& proj = Proj())
            {
                typedef util::detail::algorithm_result<ExPolicy, Iter> result;
                typedef typename std::iterator_traits<Iter>::value_type type;
                typedef typename std::iterator_traits<Iter>::difference_type
                    difference_type;

                difference_type count = detail::distance(first, last);
                if (count <= 0)
                    return result::get(std::move(last));

                util::cancellation_token<std::size_t> tok(count);

                auto f1 = [f = std::forward<F>(f),
                              proj = std::forward<Proj>(proj),
                              tok](Iter it, std::size_t part_size,
                              std::size_t base_idx) mutable -> void {
                    util::loop_idx_n(base_idx, it, part_size, tok,
                        [&f, &proj, &tok](type& v, std::size_t i) -> void {
                            if (hpx::util::invoke(
                                    f, hpx::util::invoke(proj, v)))
                            {
                                tok.cancel(i);
                            }
                        });
                };

                auto f2 =
                    [tok, count, first, last](
                        std::vector<hpx::future<void>>&&) mutable -> Iter {
                    difference_type find_res =
                        static_cast<difference_type>(tok.get_data());

                    if (find_res != count)
                    {
                        std::advance(first, find_res);
                    }
                    else
                    {
                        first = detail::advance_to_sentinel(first, last);
                    }
                    return std::move(first);
                };

                return util::partitioner<ExPolicy, Iter, void>::call_with_index(
                    std::forward<ExPolicy>(policy), first, count, 1,
                    std::move(f1), std::move(f2));
            }
        };

        template <typename ExPolicy, typename Iter, typename Sent, typename F,
            typename Proj>
        inline typename util::detail::algorithm_result<ExPolicy, Iter>::type
        find_if_(ExPolicy&& policy, Iter first, Sent last, F&& f, Proj&& proj,
            std::false_type)
        {
            static_assert((hpx::traits::is_forward_iterator<Iter>::value),
                "Requires at least forward iterator.");

            typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;

            return detail::find_if<Iter>().call(std::forward<ExPolicy>(policy),
                is_seq(), first, last, std::forward<F>(f),
                std::forward<Proj>(proj));
        }

        template <typename ExPolicy, typename FwdIter, typename F,
            typename Proj>
        inline typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        find_if_(ExPolicy&& policy, FwdIter first, FwdIter last, F&& f,
            Proj&& proj, std::true_type);
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter, typename F,
        HPX_CONCEPT_REQUIRES_(
            execution::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter>::value &&
            hpx::traits::is_invocable<F,
                typename std::iterator_traits<FwdIter>::value_type
            >::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(
        1, 6, "hpx::parallel::find_if is deprecated, use hpx::find_If instead")
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        find_if(ExPolicy&& policy, FwdIter first, FwdIter last, F&& f)
    {
        using is_segmented = hpx::traits::is_segmented_iterator<FwdIter>;

        return detail::find_if_(std::forward<ExPolicy>(policy), first, last,
            std::forward<F>(f), util::projection_identity(), is_segmented());
    }

    ///////////////////////////////////////////////////////////////////////////
    // find_if_not
    namespace detail {

        template <typename Iter, typename Sent, typename Pred, typename Proj>
        constexpr Iter sequential_find_if_not(
            Iter first, Sent last, Pred&& pred, Proj&& proj)
        {
            for (/**/; first != last; ++first)
            {
                if (!hpx::util::invoke(pred, hpx::util::invoke(proj, *first)))
                {
                    break;
                }
            }
            return first;
        }

        template <typename FwdIter>
        struct find_if_not
          : public detail::algorithm<find_if_not<FwdIter>, FwdIter>
        {
            find_if_not()
              : find_if_not::algorithm("find_if_not")
            {
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename F, typename Proj = util::projection_identity>
            static constexpr Iter sequential(
                ExPolicy, Iter first, Sent last, F&& f, Proj&& proj = Proj())
            {
                return sequential_find_if_not(
                    first, last, std::forward<F>(f), std::forward<Proj>(proj));
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename F, typename Proj = util::projection_identity>
            static typename util::detail::algorithm_result<ExPolicy, Iter>::type
            parallel(ExPolicy&& policy, Iter first, Sent last, F&& f,
                Proj&& proj = Proj())
            {
                typedef util::detail::algorithm_result<ExPolicy, Iter> result;
                typedef typename std::iterator_traits<Iter>::value_type type;
                typedef typename std::iterator_traits<Iter>::difference_type
                    difference_type;

                difference_type count = detail::distance(first, last);
                if (count <= 0)
                    return result::get(std::move(last));

                util::cancellation_token<std::size_t> tok(count);

                auto f1 = [f = std::forward<F>(f),
                              proj = std::forward<Proj>(proj),
                              tok](Iter it, std::size_t part_size,
                              std::size_t base_idx) mutable -> void {
                    util::loop_idx_n(base_idx, it, part_size, tok,
                        [&f, &proj, &tok](type& v, std::size_t i) -> void {
                            if (!hpx::util::invoke(
                                    f, hpx::util::invoke(proj, v)))
                            {
                                tok.cancel(i);
                            }
                        });
                };

                auto f2 =
                    [tok, count, first, last](
                        std::vector<hpx::future<void>>&&) mutable -> Iter {
                    difference_type find_res =
                        static_cast<difference_type>(tok.get_data());

                    if (find_res != count)
                    {
                        std::advance(first, find_res);
                    }
                    else
                    {
                        first = detail::advance_to_sentinel(first, last);
                    }
                    return std::move(first);
                };

                return util::partitioner<ExPolicy, Iter, void>::call_with_index(
                    std::forward<ExPolicy>(policy), first, count, 1,
                    std::move(f1), std::move(f2));
            }
        };

        template <typename ExPolicy, typename Iter, typename Sent, typename F,
            typename Proj>
        inline typename util::detail::algorithm_result<ExPolicy, Iter>::type
        find_if_not_(ExPolicy&& policy, Iter first, Sent last, F&& f,
            Proj&& proj, std::false_type)
        {
            static_assert((hpx::traits::is_forward_iterator<Iter>::value),
                "Requires at least forward iterator.");

            using is_seq = execution::is_sequenced_execution_policy<ExPolicy>;

            return detail::find_if_not<Iter>().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last,
                std::forward<F>(f), std::forward<Proj>(proj));
        }

        template <typename ExPolicy, typename FwdIter, typename F,
            typename Proj>
        inline typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        find_if_not_(ExPolicy&& policy, FwdIter first, FwdIter last, F&& f,
            Proj&& proj, std::true_type);
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter, typename F,
        HPX_CONCEPT_REQUIRES_(
            execution::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter>::value &&
            hpx::traits::is_invocable<F,
                typename std::iterator_traits<FwdIter>::value_type
            >::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::find_if_not is deprecated, use hpx::find_if_not "
        "instead")
        typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        find_if_not(ExPolicy&& policy, FwdIter first, FwdIter last, F&& f)
    {
        using is_segmented = hpx::traits::is_segmented_iterator<FwdIter>;

        return detail::find_if_not_(std::forward<ExPolicy>(policy), first, last,
            std::forward<F>(f), util::projection_identity(), is_segmented());
    }

    ///////////////////////////////////////////////////////////////////////////
    // find_end
    namespace detail {

        template <typename Iter1, typename Sent1, typename Iter2,
            typename Sent2, typename Pred, typename Proj1, typename Proj2>
        constexpr Iter1 sequential_search(Iter1 first1, Sent1 last1,
            Iter2 first2, Sent2 last2, Pred&& op, Proj1&& proj1, Proj2&& proj2)
        {
            for (/**/; /**/; ++first1)
            {
                Iter1 it1 = first1;
                for (Iter2 it2 = first2; /**/; (void) ++it1, ++it2)
                {
                    if (it2 == last2)
                    {
                        return first1;
                    }
                    if (it1 == last1)
                    {
                        return last1;
                    }
                    if (!hpx::util::invoke(op, hpx::util::invoke(proj1, *it1),
                            hpx::util::invoke(proj2, *it2)))
                    {
                        break;
                    }
                }
            }
        }

        template <typename Iter1, typename Sent1, typename Iter2,
            typename Sent2, typename Pred, typename Proj1, typename Proj2>
        constexpr Iter1 sequential_find_end(Iter1 first1, Sent1 last1,
            Iter2 first2, Sent2 last2, Pred&& op, Proj1&& proj1, Proj2&& proj2)
        {
            if (first2 == last2)
            {
                return detail::advance_to_sentinel(first1, last1);
            }

            Iter1 result = last1;
            while (true)
            {
                Iter1 new_result = sequential_search(
                    first1, last1, first2, last2, op, proj1, proj2);

                if (new_result == last1)
                {
                    break;
                }
                else
                {
                    result = new_result;
                    first1 = result;
                    ++first1;
                }
            }
            return result;
        }

        template <typename FwdIter>
        struct find_end : public detail::algorithm<find_end<FwdIter>, FwdIter>
        {
            find_end()
              : find_end::algorithm("find_end")
            {
            }

            template <typename ExPolicy, typename Iter1, typename Sent1,
                typename Iter2, typename Sent2, typename Pred, typename Proj1,
                typename Proj2>
            static constexpr Iter1 sequential(ExPolicy, Iter1 first1,
                Sent1 last1, Iter2 first2, Sent2 last2, Pred&& op,
                Proj1&& proj1, Proj2&& proj2)
            {
                return sequential_find_end(first1, last1, first2, last2,
                    std::forward<Pred>(op), std::forward<Proj1>(proj1),
                    std::forward<Proj2>(proj2));
            }

            template <typename ExPolicy, typename Iter1, typename Sent1,
                typename Iter2, typename Sent2, typename Pred, typename Proj1,
                typename Proj2>
            static
                typename util::detail::algorithm_result<ExPolicy, Iter1>::type
                parallel(ExPolicy&& policy, Iter1 first1, Sent1 last1,
                    Iter2 first2, Sent2 last2, Pred&& op, Proj1&& proj1,
                    Proj2&& proj2)
            {
                using result_type =
                    util::detail::algorithm_result<ExPolicy, Iter1>;
                using reference =
                    typename std::iterator_traits<Iter1>::reference;
                using difference_type =
                    typename std::iterator_traits<Iter1>::difference_type;

                difference_type diff = detail::distance(first2, last2);
                if (diff <= 0)
                {
                    return result_type::get(std::move(last1));
                }

                difference_type count = detail::distance(first1, last1);
                if (diff > count)
                {
                    return result_type::get(std::move(last1));
                }

                util::cancellation_token<difference_type,
                    std::greater<difference_type>>
                    tok(-1);

                auto f1 = [count, diff, tok, first2,
                              op = std::forward<Pred>(op),
                              proj1 = std::forward<Proj1>(proj1),
                              proj2 = std::forward<Proj2>(proj2)](Iter1 it,
                              std::size_t part_size,
                              std::size_t base_idx) mutable -> void {
                    Iter1 curr = it;

                    util::loop_idx_n(base_idx, it, part_size, tok,
                        [=, &tok, &curr, &op, &proj1, &proj2](
                            reference t, std::size_t i) -> void {
                            ++curr;
                            if (hpx::util::invoke(op,
                                    hpx::util::invoke(proj1, t),
                                    hpx::util::invoke(proj2, *first2)))
                            {
                                difference_type local_count = 1;
                                FwdIter mid = curr;

                                for (difference_type len = 0;
                                     local_count != diff && len != count;
                                     (void) ++local_count, ++len, ++mid)
                                {
                                    if (!hpx::util::invoke(op,
                                            hpx::util::invoke(proj1, t),
                                            hpx::util::invoke(proj2, *first2)))
                                    {
                                        break;
                                    }
                                }

                                if (local_count == diff)
                                {
                                    tok.cancel(i);
                                }
                            }
                        });
                };

                auto f2 =
                    [tok, count, first1, last1](
                        std::vector<hpx::future<void>>&&) mutable -> Iter1 {
                    difference_type find_end_res = tok.get_data();

                    if (find_end_res >= 0 && find_end_res != count)
                    {
                        std::advance(first1, find_end_res);
                    }
                    else
                    {
                        first1 = last1;
                    }
                    return std::move(first1);
                };

                return util::partitioner<ExPolicy, Iter1,
                    void>::call_with_index(std::forward<ExPolicy>(policy),
                    first1, count - diff + 1, 1, std::move(f1), std::move(f2));
            }
        };
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Pred = detail::equal_to,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            execution::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter1>::value &&
            hpx::traits::is_iterator<FwdIter2>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::find_end is deprecated, use hpx::find_end instead")
        typename util::detail::algorithm_result<ExPolicy, FwdIter1>::type
        find_end(ExPolicy&& policy, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, FwdIter2 last2, Pred&& op = Pred(),
            Proj&& proj = Proj())
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Requires at least forward iterator.");
        static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Requires at least forward iterator.");

        using is_seq = execution::is_sequenced_execution_policy<ExPolicy>;

        return detail::find_end<FwdIter1>().call(std::forward<ExPolicy>(policy),
            is_seq(), first1, last1, first2, last2, std::forward<Pred>(op),
            std::forward<Proj>(proj));
    }

    ///////////////////////////////////////////////////////////////////////////
    // find_first_of
    namespace detail {
        template <typename FwdIter>
        struct find_first_of
          : public detail::algorithm<find_first_of<FwdIter>, FwdIter>
        {
            find_first_of()
              : find_first_of::algorithm("find_first_of")
            {
            }

            template <typename ExPolicy, typename InIter1, typename InIter2,
                typename Pred, typename Proj1, typename Proj2>
            static InIter1 sequential(ExPolicy, InIter1 first, InIter1 last,
                InIter2 s_first, InIter2 s_last, Pred&& op, Proj1&& proj1,
                Proj2&& proj2)
            {
                if (first == last)
                    return last;

                for (/* */; first != last; ++first)
                {
                    for (InIter2 iter = s_first; iter != s_last; ++iter)
                    {
                        if (hpx::util::invoke(
                                util::compare_projected<Pred, Proj1, Proj2>(
                                    std::forward<Pred>(op),
                                    std::forward<Proj1>(proj1),
                                    std::forward<Proj2>(proj2)),
                                *first, *iter))
                        {
                            return first;
                        }
                    }
                }
                return last;
            }

            template <typename ExPolicy, typename FwdIter2, typename Pred,
                typename Proj1, typename Proj2>
            static
                typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
                parallel(ExPolicy&& policy, FwdIter first, FwdIter last,
                    FwdIter2 s_first, FwdIter2 s_last, Pred&& op, Proj1&& proj1,
                    Proj2&& proj2)
            {
                typedef util::detail::algorithm_result<ExPolicy, FwdIter>
                    result;
                typedef
                    typename std::iterator_traits<FwdIter>::reference reference;
                typedef typename std::iterator_traits<FwdIter>::difference_type
                    difference_type;
                typedef typename std::iterator_traits<FwdIter2>::difference_type
                    s_difference_type;

                s_difference_type diff = std::distance(s_first, s_last);
                if (diff <= 0)
                    return result::get(std::move(last));

                difference_type count = std::distance(first, last);
                if (diff > count)
                    return result::get(std::move(last));

                util::cancellation_token<difference_type> tok(count);

                auto f1 = [s_first, s_last, tok, op = std::forward<Pred>(op),
                              proj1 = std::forward<Proj1>(proj1),
                              proj2 = std::forward<Proj2>(proj2)](FwdIter it,
                              std::size_t part_size,
                              std::size_t base_idx) mutable -> void {
                    util::loop_idx_n(base_idx, it, part_size, tok,
                        [&tok, &s_first, &s_last, &op, &proj1, &proj2](
                            reference v, std::size_t i) -> void {
                            for (FwdIter2 iter = s_first; iter != s_last;
                                 ++iter)
                            {
                                if (hpx::util::invoke(
                                        util::compare_projected<Pred, Proj1,
                                            Proj2>(std::forward<Pred>(op),
                                            std::forward<Proj1>(proj1),
                                            std::forward<Proj2>(proj2)),
                                        v, *iter))
                                {
                                    tok.cancel(i);
                                }
                            }
                        });
                };

                auto f2 =
                    [tok, count, first, last](
                        std::vector<hpx::future<void>>&&) mutable -> FwdIter {
                    difference_type find_first_of_res = tok.get_data();

                    if (find_first_of_res != count)
                    {
                        std::advance(first, find_first_of_res);
                    }
                    else
                    {
                        first = last;
                    }

                    return std::move(first);
                };

                return util::partitioner<ExPolicy, FwdIter,
                    void>::call_with_index(std::forward<ExPolicy>(policy),
                    first, count, 1, std::move(f1), std::move(f2));
            }
        };
    }    // namespace detail

    // clang-format off
    template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
        typename Pred = detail::equal_to,
        typename Proj1 = util::projection_identity,
        typename Proj2 = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            execution::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<FwdIter1>::value &&
            hpx::traits::is_iterator<FwdIter2>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::find_first_of is deprecated, use hpx::find_first_of "
        "instead")
        typename util::detail::algorithm_result<ExPolicy, FwdIter1>::type
        find_first_of(ExPolicy&& policy, FwdIter1 first, FwdIter1 last,
            FwdIter2 s_first, FwdIter2 s_last, Pred&& op = Pred(),
            Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2())
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
            "Requires at least forward iterator.");
        static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Subsequence requires at least forward iterator.");

        using is_seq = execution::is_sequenced_execution_policy<ExPolicy>;

        return detail::find_first_of<FwdIter1>().call(
            std::forward<ExPolicy>(policy), is_seq(), first, last, s_first,
            s_last, std::forward<Pred>(op), std::forward<Proj1>(proj1),
            std::forward<Proj2>(proj2));
    }
}}}    // namespace hpx::parallel::v1

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::find
    HPX_INLINE_CONSTEXPR_VARIABLE struct find_t final
      : hpx::functional::tag<find_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_invoke(find_t, ExPolicy&& policy, FwdIter first, FwdIter last,
            T const& val)
        {
            using is_segmented = hpx::traits::is_segmented_iterator<FwdIter>;

            return hpx::parallel::v1::detail::find_(
                std::forward<ExPolicy>(policy), first, last, val,
                hpx::parallel::util::projection_identity(), is_segmented());
        }

        // clang-format off
        template <typename FwdIter, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend FwdIter tag_invoke(
            find_t, FwdIter first, FwdIter last, T const& val)
        {
            return hpx::parallel::v1::detail::find_(
                hpx::parallel::execution::seq, first, last, val,
                hpx::parallel::util::projection_identity(), std::false_type());
        }

    } find;

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::find_if
    HPX_INLINE_CONSTEXPR_VARIABLE struct find_if_t final
      : hpx::functional::tag<find_if_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter>::value &&
                hpx::traits::is_invocable<F,
                    typename std::iterator_traits<FwdIter>::value_type
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_invoke(
            find_if_t, ExPolicy&& policy, FwdIter first, FwdIter last, F&& f)
        {
            using is_segmented = hpx::traits::is_segmented_iterator<FwdIter>;

            return hpx::parallel::v1::detail::find_if_(
                std::forward<ExPolicy>(policy), first, last, std::forward<F>(f),
                hpx::parallel::util::projection_identity(), is_segmented());
        }

        // clang-format off
        template <typename FwdIter, typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter>::value &&
                hpx::traits::is_invocable<F,
                    typename std::iterator_traits<FwdIter>::value_type
                >::value
            )>
        // clang-format on
        friend FwdIter tag_invoke(find_if_t, FwdIter first, FwdIter last, F&& f)
        {
            return hpx::parallel::v1::detail::find_if_(
                hpx::parallel::execution::seq, first, last, std::forward<F>(f),
                hpx::parallel::util::projection_identity(), std::false_type());
        }

    } find_if;

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::find_if_not
    HPX_INLINE_CONSTEXPR_VARIABLE struct find_if_not_t final
      : hpx::functional::tag<find_if_not_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter>::value &&
                hpx::traits::is_invocable<F,
                    typename std::iterator_traits<FwdIter>::value_type
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_invoke(find_if_not_t, ExPolicy&& policy, FwdIter first,
            FwdIter last, F&& f)
        {
            using is_segmented = hpx::traits::is_segmented_iterator<FwdIter>;

            return hpx::parallel::v1::detail::find_if_not_(
                std::forward<ExPolicy>(policy), first, last, std::forward<F>(f),
                hpx::parallel::util::projection_identity(), is_segmented());
        }

        // clang-format off
        template <typename FwdIter, typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter>::value &&
                hpx::traits::is_invocable<F,
                    typename std::iterator_traits<FwdIter>::value_type
                >::value
            )>
        // clang-format on
        friend FwdIter tag_invoke(
            find_if_not_t, FwdIter first, FwdIter last, F&& f)
        {
            return hpx::parallel::v1::detail::find_if_not_(
                hpx::parallel::execution::seq, first, last, std::forward<F>(f),
                hpx::parallel::util::projection_identity(), std::false_type());
        }

    } find_if_not;

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::find_end
    HPX_INLINE_CONSTEXPR_VARIABLE struct find_end_t final
      : hpx::functional::tag<find_end_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Pred,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::traits::is_invocable<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter1>::type
        tag_invoke(find_end_t, ExPolicy&& policy, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, FwdIter2 last2, Pred&& op)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            return hpx::parallel::v1::detail::find_end<FwdIter1>().call(
                std::forward<ExPolicy>(policy), is_seq(), first1, last1, first2,
                last2, std::forward<Pred>(op),
                hpx::parallel::util::projection_identity(),
                hpx::parallel::util::projection_identity());
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter1>::type
        tag_invoke(find_end_t, ExPolicy&& policy, FwdIter1 first1,
            FwdIter1 last1, FwdIter2 first2, FwdIter2 last2)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            return hpx::parallel::v1::detail::find_end<FwdIter1>().call(
                std::forward<ExPolicy>(policy), is_seq(), first1, last1, first2,
                last2, hpx::parallel::v1::detail::equal_to{},
                hpx::parallel::util::projection_identity(),
                hpx::parallel::util::projection_identity());
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2,
            typename Pred,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::traits::is_invocable<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                >::value
            )>
        // clang-format on
        friend FwdIter1 tag_invoke(find_end_t, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, FwdIter2 last2, Pred&& op)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::find_end<FwdIter1>().call(
                hpx::parallel::execution::seq, std::true_type(), first1, last1,
                first2, last2, std::forward<Pred>(op),
                hpx::parallel::util::projection_identity(),
                hpx::parallel::util::projection_identity());
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend FwdIter1 tag_invoke(find_end_t, FwdIter1 first1, FwdIter1 last1,
            FwdIter2 first2, FwdIter2 last2)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::find_end<FwdIter1>().call(
                hpx::parallel::execution::seq, std::true_type(), first1, last1,
                first2, last2, hpx::parallel::v1::detail::equal_to{},
                hpx::parallel::util::projection_identity(),
                hpx::parallel::util::projection_identity());
        }
    } find_end;

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::find_first_of
    HPX_INLINE_CONSTEXPR_VARIABLE struct find_first_of_t final
      : hpx::functional::tag<find_first_of_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Pred,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::traits::is_invocable<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter1>::type
        tag_invoke(find_first_of_t, ExPolicy&& policy, FwdIter1 first,
            FwdIter1 last, FwdIter2 s_first, FwdIter2 s_last, Pred&& op)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Subsequence requires at least forward iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            return hpx::parallel::v1::detail::find_first_of<FwdIter1>().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last, s_first,
                s_last, std::forward<Pred>(op),
                hpx::parallel::util::projection_identity(),
                hpx::parallel::util::projection_identity());
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter1>::type
        tag_invoke(find_first_of_t, ExPolicy&& policy, FwdIter1 first,
            FwdIter1 last, FwdIter2 s_first, FwdIter2 s_last)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Subsequence requires at least forward iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            return hpx::parallel::v1::detail::find_first_of<FwdIter1>().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last, s_first,
                s_last, hpx::parallel::v1::detail::equal_to{},
                hpx::parallel::util::projection_identity(),
                hpx::parallel::util::projection_identity());
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2, typename Pred,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::traits::is_invocable<Pred,
                    typename std::iterator_traits<FwdIter1>::value_type,
                    typename std::iterator_traits<FwdIter2>::value_type
                >::value
            )>
        // clang-format on
        friend FwdIter1 tag_invoke(find_first_of_t, FwdIter1 first,
            FwdIter1 last, FwdIter2 s_first, FwdIter2 s_last, Pred&& op)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Subsequence requires at least forward iterator.");

            return hpx::parallel::v1::detail::find_first_of<FwdIter1>().call(
                hpx::parallel::execution::seq, std::true_type(), first, last,
                s_first, s_last, std::forward<Pred>(op),
                hpx::parallel::util::projection_identity(),
                hpx::parallel::util::projection_identity());
        }

        // clang-format off
        template <typename FwdIter1, typename FwdIter2,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value
            )>
        // clang-format on
        friend FwdIter1 tag_invoke(find_first_of_t, FwdIter1 first,
            FwdIter1 last, FwdIter2 s_first, FwdIter2 s_last)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
                "Subsequence requires at least forward iterator.");

            return hpx::parallel::v1::detail::find_first_of<FwdIter1>().call(
                hpx::parallel::execution::seq, std::true_type(), first, last,
                s_first, s_last, hpx::parallel::v1::detail::equal_to{},
                hpx::parallel::util::projection_identity(),
                hpx::parallel::util::projection_identity());
        }
    } find_first_of;
}    // namespace hpx

#endif    // DOXYGEN
