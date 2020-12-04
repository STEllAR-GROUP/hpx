//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/search.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>

#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/for_each.hpp>
#include <hpx/parallel/util/compare_projected.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/zip_iterator.hpp>

#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/partitioner.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(DOXYGEN)

namespace hpx {

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
    /// \tparam FwdIter2    The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a adjacent_find requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
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
    template <typename FwdIter, typename FwdIter2,
        typename Pred = detail::equal_to>
    FwdIter search(FwdIter first, FwdIter last, FwdIter2 s_first,
        FwdIter2 s_last, Pred&& op = Pred());

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
    /// \tparam FwdIter2    The type of the source iterators used for the
    ///                     second range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a adjacent_find requires \a Pred to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::equal_to<>
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
    template <typename ExPolicy, typename FwdIter, typename FwdIter2,
        typename Pred = detail::equal_to>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type search(
        ExPolicy&& policy, FwdIter first, FwdIter last, FwdIter2 s_first,
        FwdIter2 s_last, Pred&& op = Pred());

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
    template <typename FwdIter, typename FwdIter2,
        typename Pred = detail::equal_to>
    FwdIter search_n(FwdIter first, std::size_t count, FwdIter2 s_first,
        FwdIter2 s_last, Pred&& op = Pred());

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
        typename Pred = detail::equal_to>
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type search_n(
        ExPolicy&& policy, FwdIter first, std::size_t count, FwdIter2 s_first,
        FwdIter2 s_last, Pred&& op = Pred());
}    // namespace hpx

#else

namespace hpx { namespace parallel { inline namespace v1 {

    ///////////////////////////////////////////////////////////////////////////
    // search
    namespace detail {
        /// \cond NOINTERNAL
        template <typename FwdIter, typename Sent>
        struct search : public detail::algorithm<search<FwdIter, Sent>, FwdIter>
        {
            search()
              : search::algorithm("search")
            {
            }

            template <typename ExPolicy, typename FwdIter2, typename Sent2,
                typename Pred, typename Proj1, typename Proj2>
            static FwdIter sequential(ExPolicy, FwdIter first, Sent last,
                FwdIter2 s_first, Sent2 s_last, Pred&& op, Proj1&& proj1,
                Proj2&& proj2)
            {
                return std::search(first, last, s_first, s_last,
                    util::compare_projected<Pred, Proj1, Proj2>(
                        op, proj1, proj2));
            }

            template <typename ExPolicy, typename FwdIter2, typename Sent2,
                typename Pred, typename Proj1, typename Proj2>
            static
                typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
                parallel(ExPolicy&& policy, FwdIter first, Sent last,
                    FwdIter2 s_first, Sent2 s_last, Pred&& op, Proj1&& proj1,
                    Proj2&& proj2)
            {
                using reference =
                    typename std::iterator_traits<FwdIter>::reference;

                using difference_type =
                    typename std::iterator_traits<FwdIter>::difference_type;

                using s_difference_type =
                    typename std::iterator_traits<FwdIter2>::difference_type;

                using result =
                    util::detail::algorithm_result<ExPolicy, FwdIter>;

                s_difference_type diff = std::distance(s_first, s_last);
                if (diff <= 0)
                    return result::get(std::move(first));

                difference_type count = std::distance(first, last);
                if (diff > count)
                    return result::get(std::move(last));

                using partitioner = util::partitioner<ExPolicy, FwdIter, void>;

                util::cancellation_token<difference_type> tok(count);

                auto f1 = [diff, count, tok, s_first,
                              op = std::forward<Pred>(op),
                              proj1 = std::forward<Proj1>(proj1),
                              proj2 = std::forward<Proj2>(proj2)](FwdIter it,
                              std::size_t part_size,
                              std::size_t base_idx) mutable -> void {
                    FwdIter curr = it;

                    util::loop_idx_n(base_idx, it, part_size, tok,
                        [diff, count, s_first, &tok, &curr,
                            op = std::forward<Pred>(op),
                            proj1 = std::forward<Proj1>(proj1),
                            proj2 = std::forward<Proj2>(proj2)](
                            reference v, std::size_t i) -> void {
                            ++curr;
                            if (hpx::util::invoke(op,
                                    hpx::util::invoke(proj1, v),
                                    hpx::util::invoke(proj2, *s_first)))
                            {
                                difference_type local_count = 1;
                                FwdIter2 needle = s_first;
                                FwdIter mid = curr;

                                for (difference_type len = 0;
                                     local_count != diff && len != count;
                                     ++local_count, ++len, ++mid)
                                {
                                    if (!hpx::util::invoke(op,
                                            hpx::util::invoke(proj1, *mid),
                                            hpx::util::invoke(
                                                proj2, *++needle)))
                                        break;
                                }

                                if (local_count == diff)
                                    tok.cancel(i);
                            }
                        });
                };

                auto f2 =
                    [=](std::vector<hpx::future<void>>&&) mutable -> FwdIter {
                    difference_type search_res = tok.get_data();
                    if (search_res != count)
                        std::advance(first, search_res);
                    else
                        first = last;

                    return std::move(first);
                };
                return partitioner::call_with_index(
                    std::forward<ExPolicy>(policy), first, count - (diff - 1),
                    1, std::move(f1), std::move(f2));
            }
        };
        /// \endcond
    }    // namespace detail

    template <typename ExPolicy, typename FwdIter, typename FwdIter2,
        typename Pred = detail::equal_to,
        typename Proj1 = parallel::util::projection_identity,
        typename Proj2 = parallel::util::projection_identity,
        HPX_CONCEPT_REQUIRES_(hpx::is_execution_policy<
            ExPolicy>::value&& hpx::traits::is_iterator<FwdIter>::value&&
                parallel::traits::is_projected<Proj1, FwdIter>::value&&
                    hpx::traits::is_iterator<FwdIter2>::value&&
                        parallel::traits::is_projected<Proj2, FwdIter2>::value&&
                            parallel::traits::is_indirect_callable<ExPolicy,
                                Pred,
                                parallel::traits::projected<Proj1, FwdIter>,
                                parallel::traits::projected<Proj2,
                                    FwdIter2>>::value)>
    HPX_DEPRECATED_V(1, 6, "Please use hpx::search instead.")
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
        search(ExPolicy&& policy, FwdIter first, FwdIter last, FwdIter2 s_first,
            FwdIter2 s_last, Pred&& op = Pred(), Proj1&& proj1 = Proj1(),
            Proj2&& proj2 = Proj2())
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
            "Requires at least forward iterator.");
        static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Subsequence requires at least forward iterator.");

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

        return detail::search<FwdIter, FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(), first, last, s_first,
            s_last, std::forward<Pred>(op), std::forward<Proj1>(proj1),
            std::forward<Proj2>(proj2));
    }

    ///////////////////////////////////////////////////////////////////////////
    // search_n
    namespace detail {
        /// \cond NOINTERNAL
        template <typename FwdIter, typename Sent>
        struct search_n
          : public detail::algorithm<search_n<FwdIter, Sent>, FwdIter>
        {
            search_n()
              : search_n::algorithm("search_n")
            {
            }

            template <typename ExPolicy, typename FwdIter2, typename Pred,
                typename Proj1, typename Proj2>
            static FwdIter sequential(ExPolicy, FwdIter first,
                std::size_t count, FwdIter2 s_first, FwdIter2 s_last, Pred&& op,
                Proj1&& proj1, Proj2&& proj2)
            {
                return std::search(first, std::next(first, count), s_first,
                    s_last,
                    util::compare_projected<Pred, Proj1, Proj2>(
                        op, proj1, proj2));
            }

            template <typename ExPolicy, typename FwdIter2, typename Pred,
                typename Proj1, typename Proj2>
            static
                typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
                parallel(ExPolicy&& policy, FwdIter first, std::size_t count,
                    FwdIter2 s_first, FwdIter2 s_last, Pred&& op, Proj1&& proj1,
                    Proj2&& proj2)
            {
                typedef
                    typename std::iterator_traits<FwdIter>::reference reference;
                typedef typename std::iterator_traits<FwdIter>::difference_type
                    difference_type;
                typedef typename std::iterator_traits<FwdIter2>::difference_type
                    s_difference_type;
                typedef util::detail::algorithm_result<ExPolicy, FwdIter>
                    result;

                s_difference_type diff = std::distance(s_first, s_last);
                if (diff <= 0)
                    return result::get(std::move(first));

                if (diff > s_difference_type(count))
                    return result::get(std::move(first));

                typedef util::partitioner<ExPolicy, FwdIter, void> partitioner;

                util::cancellation_token<difference_type> tok(count);

                auto f1 = [count, diff, tok, s_first,
                              op = std::forward<Pred>(op),
                              proj1 = std::forward<Proj1>(proj1),
                              proj2 = std::forward<Proj2>(proj2)](FwdIter it,
                              std::size_t part_size,
                              std::size_t base_idx) mutable -> void {
                    FwdIter curr = it;

                    util::loop_idx_n(base_idx, it, part_size, tok,
                        [count, diff, s_first, &tok, &curr,
                            op = std::forward<Pred>(op),
                            proj1 = std::forward<Proj1>(proj1),
                            proj2 = std::forward<Proj2>(proj2)](
                            reference v, std::size_t i) -> void {
                            ++curr;
                            if (hpx::util::invoke(op,
                                    hpx::util::invoke(proj1, v),
                                    hpx::util::invoke(proj2, *s_first)))
                            {
                                difference_type local_count = 1;
                                FwdIter2 needle = s_first;
                                FwdIter mid = curr;

                                for (difference_type len = 0;
                                     local_count != diff &&
                                     len != difference_type(count);
                                     ++local_count, ++len, ++mid)
                                {
                                    if (!hpx::util::invoke(op,
                                            hpx::util::invoke(proj1, *mid),
                                            hpx::util::invoke(
                                                proj2, *++needle)))
                                        break;
                                }

                                if (local_count == diff)
                                    tok.cancel(i);
                            }
                        });
                };

                auto f2 =
                    [=](std::vector<hpx::future<void>>&&) mutable -> FwdIter {
                    difference_type search_res = tok.get_data();
                    if (search_res != s_difference_type(count))
                        std::advance(first, search_res);

                    return std::move(first);
                };
                return partitioner::call_with_index(
                    std::forward<ExPolicy>(policy), first, count - (diff - 1),
                    1, std::move(f1), std::move(f2));
            }
        };
        /// \endcond
    }    // namespace detail

    template <typename ExPolicy, typename FwdIter, typename FwdIter2,
        typename Pred = parallel::v1::detail::equal_to,
        typename Proj1 = parallel::util::projection_identity,
        typename Proj2 = parallel::util::projection_identity,
        HPX_CONCEPT_REQUIRES_(hpx::is_execution_policy<
            ExPolicy>::value&& hpx::traits::is_iterator<FwdIter>::value&&
                parallel::traits::is_projected<Proj1, FwdIter>::value&&
                    hpx::traits::is_iterator<FwdIter2>::value&&
                        parallel::traits::is_projected<Proj2, FwdIter2>::value&&
                            parallel::traits::is_indirect_callable<ExPolicy,
                                Pred,
                                parallel::traits::projected<Proj1, FwdIter>,
                                parallel::traits::projected<Proj2,
                                    FwdIter2>>::value)>
    HPX_DEPRECATED_V(1, 6, "Please use hpx::search_n instead.")
    typename parallel::util::detail::algorithm_result<ExPolicy, FwdIter>::type
        search_n(ExPolicy&& policy, FwdIter first, std::size_t count,
            FwdIter2 s_first, FwdIter2 s_last, Pred&& op = Pred(),
            Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2())
    {
        static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
            "Requires at least forward iterator.");
        static_assert((hpx::traits::is_forward_iterator<FwdIter2>::value),
            "Subsequence requires at least forward iterator.");

        using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

        return detail::search_n<FwdIter, FwdIter>().call(
            std::forward<ExPolicy>(policy), is_seq(), first, count, s_first,
            s_last, std::forward<Pred>(op), std::forward<Proj1>(proj1),
            std::forward<Proj2>(proj2));
    }
}}}    // namespace hpx::parallel::v1

namespace hpx {

    HPX_INLINE_CONSTEXPR_VARIABLE struct search_t final
      : hpx::functional::tag<search_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename FwdIter2,
            typename Pred = parallel::v1::detail::equal_to,
            HPX_CONCEPT_REQUIRES_(
                traits::is_forward_iterator<FwdIter>::value &&
                traits::is_forward_iterator<FwdIter2>::value &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits <FwdIter>::value_type,
                    typename std::iterator_traits <FwdIter2>::value_type
                >
            )>
        // clang-format on
        friend FwdIter tag_invoke(hpx::search_t, FwdIter first, FwdIter last,
            FwdIter2 s_first, FwdIter2 s_last, Pred&& op = Pred())
        {
            return hpx::parallel::v1::detail::search<FwdIter, FwdIter>().call(
                hpx::execution::seq, std::true_type{}, first, last, s_first,
                s_last, std::forward<Pred>(op),
                hpx::parallel::util::projection_identity{},
                hpx::parallel::util::projection_identity{});
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename FwdIter2,
            typename Pred = parallel::v1::detail::equal_to,
            HPX_CONCEPT_REQUIRES_(
                is_execution_policy<ExPolicy>::value &&
                traits::is_forward_iterator<FwdIter>::value &&
                traits::is_forward_iterator<FwdIter2>::value &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits <FwdIter>::value_type,
                    typename std::iterator_traits <FwdIter2>::value_type
                >
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_invoke(hpx::search_t, ExPolicy&& policy, FwdIter first,
            FwdIter last, FwdIter2 s_first, FwdIter2 s_last, Pred&& op = Pred())
        {
            using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

            return hpx::parallel::v1::detail::search<FwdIter, FwdIter>().call(
                std::forward<ExPolicy>(policy), is_seq(), first, last, s_first,
                s_last, std::forward<Pred>(op),
                hpx::parallel::util::projection_identity{},
                hpx::parallel::util::projection_identity{});
        }

    } search{};

    HPX_INLINE_CONSTEXPR_VARIABLE struct search_n_t final
      : hpx::functional::tag<search_n_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename FwdIter2,
            typename Pred = parallel::v1::detail::equal_to,
            HPX_CONCEPT_REQUIRES_(
                traits::is_forward_iterator<FwdIter>::value &&
                traits::is_forward_iterator<FwdIter2>::value &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits <FwdIter>::value_type,
                    typename std::iterator_traits <FwdIter2>::value_type
                >
            )>
        // clang-format on
        friend FwdIter tag_invoke(hpx::search_n_t, FwdIter first,
            std::size_t count, FwdIter2 s_first, FwdIter2 s_last,
            Pred&& op = Pred())
        {
            return hpx::parallel::v1::detail::search_n<FwdIter, FwdIter>().call(
                hpx::execution::seq, std::true_type{}, first, count, s_first,
                s_last, std::forward<Pred>(op),
                hpx::parallel::util::projection_identity{},
                hpx::parallel::util::projection_identity{});
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename FwdIter2,
            typename Pred = parallel::v1::detail::equal_to,
            HPX_CONCEPT_REQUIRES_(
                is_execution_policy<ExPolicy>::value &&
                traits::is_forward_iterator<FwdIter>::value &&
                traits::is_forward_iterator<FwdIter2>::value &&
                hpx::is_invocable_v<Pred,
                    typename std::iterator_traits <FwdIter>::value_type,
                    typename std::iterator_traits <FwdIter2>::value_type
                >
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_invoke(hpx::search_n_t, ExPolicy&& policy, FwdIter first,
            std::size_t count, FwdIter2 s_first, FwdIter2 s_last,
            Pred&& op = Pred())
        {
            using is_seq = hpx::is_sequenced_execution_policy<ExPolicy>;

            return hpx::parallel::v1::detail::search_n<FwdIter, FwdIter>().call(
                std::forward<ExPolicy>(policy), is_seq(), first, count, s_first,
                s_last, std::forward<Pred>(op),
                hpx::parallel::util::projection_identity{},
                hpx::parallel::util::projection_identity{});
        }

    } search_n{};

}    // namespace hpx

#endif
