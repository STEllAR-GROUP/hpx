//  Copyright (c) 2017 Taeguk Kwon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    // clang-format off

    /// Merges two sorted ranges [first1, last1) and [first2, last2)
    /// into one sorted range beginning at \a dest. The order of
    /// equivalent elements in the each of original two ranges is preserved.
    /// For equivalent elements in the original two ranges, the elements from
    /// the first range precede the elements from the second range.
    /// The destination range cannot overlap with either of the input ranges.
    ///
    /// \note   Complexity: Performs
    ///         O(std::distance(first1, last1) + std::distance(first2, last2))
    ///         applications of the comparison \a comp and the each projection.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam RandIter1   The type of the source iterators used (deduced)
    ///                     representing the first sorted range.
    ///                     This iterator type must meet the requirements of an
    ///                     random access iterator.
    /// \tparam RandIter2   The type of the source iterators used (deduced)
    ///                     representing the second sorted range.
    ///                     This iterator type must meet the requirements of an
    ///                     random access iterator.
    /// \tparam RandIter3   The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     random access iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a merge requires \a Comp to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the first range of elements
    ///                     the algorithm will be applied to.
    /// \param last1        Refers to the end of the first range of elements
    ///                     the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the second range of elements
    ///                     the algorithm will be applied to.
    /// \param last2        Refers to the end of the second range of elements
    ///                     the algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param comp         \a comp is a callable object which returns true if
    ///                     the first argument is less than the second,
    ///                     and false otherwise. The signature of this
    ///                     comparison should be equivalent to:
    ///                     \code
    ///                     bool comp(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be such that
    ///                     objects of types \a RandIter1 and \a RandIter2 can be
    ///                     dereferenced and then implicitly converted to
    ///                     both \a Type1 and \a Type2
    ///
    /// The assignments in the parallel \a merge algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a merge algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a merge algorithm returns a
    /// \a hpx::future<RandIter3> >
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns
    ///           \a tagged_tuple<RandIter3> otherwise.
    ///           The \a merge algorithm returns the destination iterator to
    ///           the end of the \a dest range.
    ///
    template <typename ExPolicy, typename RandIter1, typename RandIter2,
        typename RandIter3, typename Comp = detail::less>
    typename util::detail::algorithm_result<ExPolicy, RandIter3>::type
    merge(ExPolicy&& policy, RandIter1 first1, RandIter1 last1,
        RandIter2 first2, RandIter2 last2, RandIter3 dest, Comp&& comp = Comp());

    /// Merges two consecutive sorted ranges [first, middle) and
    /// [middle, last) into one sorted range [first, last). The order of
    /// equivalent elements in the each of original two ranges is preserved.
    /// For equivalent elements in the original two ranges, the elements from
    /// the first range precede the elements from the second range.
    ///
    /// \note   Complexity: Performs O(std::distance(first, last))
    ///         applications of the comparison \a comp and the each projection.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam RandIter    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     random access iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a inplace_merge requires \a Comp
    ///                     to meet the requirements of \a CopyConstructible.
    ///                     This defaults to std::less<>
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the first sorted range
    ///                     the algorithm will be applied to.
    /// \param middle       Refers to the end of the first sorted range and
    ///                     the beginning of the second sorted range
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the second sorted range
    ///                     the algorithm will be applied to.
    /// \param comp         \a comp is a callable object which returns true if
    ///                     the first argument is less than the second,
    ///                     and false otherwise. The signature of this
    ///                     comparison should be equivalent to:
    ///                     \code
    ///                     bool comp(const Type1 &a, const Type2 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const&, but
    ///                     the function must not modify the objects passed to
    ///                     it. The types \a Type1 and \a Type2 must be
    ///                     such that objects of types \a RandIter can be
    ///                     dereferenced and then implicitly converted to both
    ///                     \a Type1 and \a Type2
    ///
    /// The assignments in the parallel \a inplace_merge algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a inplace_merge algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a inplace_merge algorithm returns a
    ///           \a hpx::future<void> if the execution policy is of type
    ///           \a sequenced_task_policy or \a parallel_task_policy
    ///           and returns void otherwise.
    ///           The \a inplace_merge algorithm returns
    ///           the source iterator \a last
    ///
    template <typename ExPolicy, typename RandIter,
        typename Comp = detail::less>
    typename util::detail::algorithm_result<ExPolicy>::type
    inplace_merge(ExPolicy&& policy, RandIter first, RandIter middle,
        RandIter last, Comp&& comp = Comp());

    // clang-format on
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/execution/algorithms/detail/is_negative.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/parallel/algorithms/detail/advance_to_sentinel.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/rotate.hpp>
#include <hpx/parallel/algorithms/detail/upper_lower_bound.hpp>
#include <hpx/parallel/util/compare_projected.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iterator>
#include <list>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 {
    /////////////////////////////////////////////////////////////////////////////
    // merge
    namespace detail {
        /// \cond NOINTERNAL

        // sequential merge with projection function.
        template <typename Iter1, typename Sent1, typename Iter2,
            typename Sent2, typename OutIter, typename Comp, typename Proj1,
            typename Proj2>
        constexpr util::in_in_out_result<Iter1, Iter2, OutIter>
        sequential_merge(Iter1 first1, Sent1 last1, Iter2 first2, Sent2 last2,
            OutIter dest, Comp&& comp, Proj1&& proj1, Proj2&& proj2)
        {
            if (first1 != last1 && first2 != last2)
            {
                while (true)
                {
                    if (hpx::util::invoke(comp,
                            hpx::util::invoke(proj2, *first2),
                            hpx::util::invoke(proj1, *first1)))
                    {
                        *dest++ = *first2++;
                        if (first2 == last2)
                        {
                            break;
                        }
                    }
                    else
                    {
                        *dest++ = *first1++;
                        if (first1 == last1)
                        {
                            break;
                        }
                    }
                }
            }

            auto copy_result1 = util::copy(first1, last1, dest);
            auto copy_result2 = util::copy(first2, last2, copy_result1.out);

            return {copy_result1.in, copy_result2.in, copy_result2.out};
        }

        ///////////////////////////////////////////////////////////////////////
        struct lower_bound_helper;

        struct upper_bound_helper
        {
            // upper_bound with projection function.
            template <typename Iter, typename Sent, typename T, typename Comp,
                typename Proj>
            static constexpr Iter call(
                Iter first, Sent last, T const& value, Comp&& comp, Proj&& proj)
            {
                return detail::upper_bound(first, last, value,
                    std::forward<Comp>(comp), std::forward<Proj>(proj));
            }

            using another_type = lower_bound_helper;
        };

        struct lower_bound_helper
        {
            // lower_bound with projection function.
            template <typename Iter, typename Sent, typename T, typename Comp,
                typename Proj>
            static constexpr Iter call(
                Iter first, Sent last, T const& value, Comp&& comp, Proj&& proj)
            {
                return detail::lower_bound(first, last, value,
                    std::forward<Comp>(comp), std::forward<Proj>(proj));
            }

            using another_type = upper_bound_helper;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename Iter1, typename Sent1,
            typename Iter2, typename Sent2, typename Iter3, typename Comp,
            typename Proj1, typename Proj2, typename BinarySearchHelper>
        void parallel_merge_helper(ExPolicy policy, Iter1 first1, Sent1 last1,
            Iter2 first2, Sent2 last2, Iter3 dest, Comp&& comp, Proj1&& proj1,
            Proj2&& proj2, bool range_reversal, BinarySearchHelper)
        {
            constexpr std::size_t threshold = 65536;

            std::size_t size1 = detail::distance(first1, last1);
            std::size_t size2 = detail::distance(first2, last2);

            // Perform sequential merge if data size is smaller than threshold.
            if (size1 + size2 <= threshold)
            {
                if (range_reversal)
                {
                    sequential_merge(first2, last2, first1, last1, dest,
                        std::forward<Comp>(comp), std::forward<Proj2>(proj2),
                        std::forward<Proj1>(proj1));
                }
                else
                {
                    sequential_merge(first1, last1, first2, last2, dest,
                        std::forward<Comp>(comp), std::forward<Proj1>(proj1),
                        std::forward<Proj2>(proj2));
                }
                return;
            }

            // Let size1 is bigger than size2 always.
            if (size1 < size2)
            {
                // For stability of algorithm, must switch binary search methods
                // when swapping size1 and size2.
                parallel_merge_helper(policy, first2, last2, first1, last1,
                    dest, std::forward<Comp>(comp), std::forward<Proj2>(proj2),
                    std::forward<Proj1>(proj1), !range_reversal,
                    typename BinarySearchHelper::another_type());
                return;
            }

            HPX_ASSERT(size1 >= size2);
            HPX_ASSERT(size1 >= 1ul);

            Iter1 mid1 = first1 + size1 / 2;
            Iter2 boundary2 = BinarySearchHelper::call(
                first2, last2, hpx::util::invoke(proj1, *mid1), comp, proj2);
            Iter3 target = dest + (mid1 - first1) + (boundary2 - first2);

            *target = *mid1;

            hpx::future<void> fut =
                execution::async_execute(policy.executor(), [&]() -> void {
                    // Process left side ranges.
                    parallel_merge_helper(policy, first1, mid1, first2,
                        boundary2, dest, comp, proj1, proj2, range_reversal,
                        BinarySearchHelper());
                });

            try
            {
                // Process right side ranges.
                parallel_merge_helper(policy, mid1 + 1, last1, boundary2, last2,
                    target + 1, std::forward<Comp>(comp),
                    std::forward<Proj1>(proj1), std::forward<Proj2>(proj2),
                    range_reversal, BinarySearchHelper());
            }
            catch (...)
            {
                fut.wait();

                std::vector<hpx::future<void>> futures;
                futures.reserve(2);
                futures.emplace_back(std::move(fut));
                futures.emplace_back(hpx::make_exceptional_future<void>(
                    std::current_exception()));

                std::list<std::exception_ptr> errors;
                util::detail::handle_local_exceptions<ExPolicy>::call(
                    futures, errors);

                // Not reachable.
                HPX_ASSERT(false);
                return;
            }

            fut.get();
        }

        template <typename ExPolicy, typename Iter1, typename Sent1,
            typename Iter2, typename Sent2, typename Iter3, typename Comp,
            typename Proj1, typename Proj2>
        hpx::future<util::in_in_out_result<Iter1, Iter2, Iter3>> parallel_merge(
            ExPolicy&& policy, Iter1 first1, Sent1 last1, Iter2 first2,
            Sent2 last2, Iter3 dest, Comp&& comp, Proj1&& proj1, Proj2&& proj2)
        {
            using result_type = util::in_in_out_result<Iter1, Iter2, Iter3>;

            auto f1 = [first1, last1, first2, last2, dest,
                          policy = std::forward<ExPolicy>(policy),
                          comp = std::forward<Comp>(comp),
                          proj1 = std::forward<Proj1>(proj1),
                          proj2 = std::forward<Proj2>(proj2)]() -> result_type {
                try
                {
                    parallel_merge_helper(std::move(policy), first1, last1,
                        first2, last2, dest, std::move(comp), std::move(proj1),
                        std::move(proj2), false, lower_bound_helper());

                    auto len1 = detail::distance(first1, last1);
                    auto len2 = detail::distance(first2, last2);
                    return {std::next(first1, len1), std::next(first2, len2),
                        std::next(dest, len1 + len2)};
                }
                catch (...)
                {
                    util::detail::handle_local_exceptions<ExPolicy>::call(
                        std::current_exception());
                }

                // Not reachable.
                HPX_ASSERT(false);
            };

            return execution::async_execute(policy.executor(), std::move(f1));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename IterTuple>
        struct merge : public detail::algorithm<merge<IterTuple>, IterTuple>
        {
            merge()
              : merge::algorithm("merge")
            {
            }

            template <typename ExPolicy, typename Iter1, typename Sent1,
                typename Iter2, typename Sent2, typename Iter3, typename Comp,
                typename Proj1, typename Proj2>
            static util::in_in_out_result<Iter1, Iter2, Iter3> sequential(
                ExPolicy, Iter1 first1, Sent1 last1, Iter2 first2, Sent2 last2,
                Iter3 dest, Comp&& comp, Proj1&& proj1, Proj2&& proj2)
            {
                return sequential_merge(first1, last1, first2, last2, dest,
                    std::forward<Comp>(comp), std::forward<Proj1>(proj1),
                    std::forward<Proj2>(proj2));
            }

            template <typename ExPolicy, typename Iter1, typename Sent1,
                typename Iter2, typename Sent2, typename Iter3, typename Comp,
                typename Proj1, typename Proj2>
            static typename util::detail::algorithm_result<ExPolicy,
                util::in_in_out_result<Iter1, Iter2, Iter3>>::type
            parallel(ExPolicy&& policy, Iter1 first1, Sent1 last1, Iter2 first2,
                Sent2 last2, Iter3 dest, Comp&& comp, Proj1&& proj1,
                Proj2&& proj2)
            {
                typedef util::in_in_out_result<Iter1, Iter2, Iter3> result_type;
                typedef util::detail::algorithm_result<ExPolicy, result_type>
                    algorithm_result;

                try
                {
                    return algorithm_result::get(parallel_merge(
                        std::forward<ExPolicy>(policy), first1, last1, first2,
                        last2, dest, std::forward<Comp>(comp),
                        std::forward<Proj1>(proj1),
                        std::forward<Proj2>(proj2)));
                }
                catch (...)
                {
                    return algorithm_result::get(
                        detail::handle_exception<ExPolicy, result_type>::call(
                            std::current_exception()));
                }
            }
        };
        /// \endcond
    }    // namespace detail

    // TODO: Support forward and bidirectional iterator. (#2826)
    // For now, only support random access iterator.

    // clang-format off
    template <typename ExPolicy, typename RandIter1, typename RandIter2,
        typename RandIter3, typename Comp = detail::less,
        typename Proj1 = util::projection_identity,
        typename Proj2 = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<RandIter1>::value &&
            hpx::traits::is_iterator<RandIter2>::value &&
            hpx::traits::is_iterator<RandIter3>::value &&
            traits::is_projected<Proj1, RandIter1>::value &&
            traits::is_projected<Proj2, RandIter2>::value &&
            traits::is_indirect_callable<ExPolicy, Comp,
                traits::projected<Proj1, RandIter1>,
                traits::projected<Proj2, RandIter2>
            >::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(
        1, 6, "hpx::parallel::merge is deprecated, use hpx::merge instead")
        typename util::detail::algorithm_result<ExPolicy,
            util::in_in_out_result<RandIter1, RandIter2, RandIter3>>::type
        merge(ExPolicy&& policy, RandIter1 first1, RandIter1 last1,
            RandIter2 first2, RandIter2 last2, RandIter3 dest,
            Comp&& comp = Comp(), Proj1&& proj1 = Proj1(),
            Proj2&& proj2 = Proj2())
    {
        static_assert(
            (hpx::traits::is_random_access_iterator<RandIter1>::value),
            "Required at least random access iterator.");
        static_assert(
            (hpx::traits::is_random_access_iterator<RandIter2>::value),
            "Requires at least random access iterator.");
        static_assert(
            (hpx::traits::is_random_access_iterator<RandIter3>::value),
            "Requires at least random access iterator.");

        using is_seq = execution::is_sequenced_execution_policy<ExPolicy>;
        using result_type =
            util::in_in_out_result<RandIter1, RandIter2, RandIter3>;

        return detail::merge<result_type>().call(std::forward<ExPolicy>(policy),
            is_seq(), first1, last1, first2, last2, dest,
            std::forward<Comp>(comp), std::forward<Proj1>(proj1),
            std::forward<Proj2>(proj2));
    }

    /////////////////////////////////////////////////////////////////////////////
    // inplace_merge
    namespace detail {

        // sequential inplace_merge with projection function.
        template <typename Iter, typename Sent, typename Comp, typename Proj>
        inline Iter sequential_inplace_merge(
            Iter first, Iter middle, Sent last, Comp&& comp, Proj&& proj)
        {
            std::inplace_merge(first, middle,
                detail::advance_to_sentinel(middle, last),
                util::compare_projected<typename std::decay<Comp>::type,
                    typename std::decay<Proj>::type>(
                    std::forward<Comp>(comp), std::forward<Proj>(proj)));
            return last;
        }

        template <typename ExPolicy, typename Iter, typename Sent,
            typename Comp, typename Proj>
        void parallel_inplace_merge_helper(ExPolicy&& policy, Iter first,
            Iter middle, Sent last, Comp&& comp, Proj&& proj)
        {
            const std::size_t threshold = 65536ul;
            HPX_ASSERT(threshold >= 5ul);

            std::size_t left_size = middle - first;
            std::size_t right_size = last - middle;

            // Perform sequential inplace_merge
            //   if data size is smaller than threshold.
            if (left_size + right_size <= threshold)
            {
                sequential_inplace_merge(first, middle, last,
                    std::forward<Comp>(comp), std::forward<Proj>(proj));
                return;
            }

            if (left_size >= right_size)
            {
                // Means that always 'pivot' < 'middle'.
                HPX_ASSERT(left_size >= 3ul);

                // Select pivot in left-side range.
                Iter pivot = first + left_size / 2;
                Iter boundary = lower_bound_helper::call(
                    middle, last, hpx::util::invoke(proj, *pivot), comp, proj);
                Iter target = pivot + (boundary - middle);

                // Swap two blocks, [pivot, middle) and [middle, boundary).
                // After this, [first, last) will be divided into three blocks,
                //   [first, target), target, and [target+1, last).
                // And all elements of [first, target) are less than
                //   the thing of target.
                // And all elements of [target+1, last) are greater or equal than
                //   the thing of target.
                detail::sequential_rotate(pivot, middle, boundary);

                hpx::future<void> fut =
                    execution::async_execute(policy.executor(), [&]() -> void {
                        // Process the range which is left-side of 'target'.
                        parallel_inplace_merge_helper(
                            policy, first, pivot, target, comp, proj);
                    });

                try
                {
                    // Process the range which is right-side of 'target'.
                    parallel_inplace_merge_helper(
                        policy, target + 1, boundary, last, comp, proj);
                }
                catch (...)
                {
                    fut.wait();

                    std::vector<hpx::future<void>> futures;
                    futures.reserve(2);
                    futures.emplace_back(std::move(fut));
                    futures.emplace_back(hpx::make_exceptional_future<void>(
                        std::current_exception()));

                    std::list<std::exception_ptr> errors;
                    util::detail::handle_local_exceptions<ExPolicy>::call(
                        futures, errors);

                    // Not reachable.
                    HPX_ASSERT(false);
                }

                if (fut.valid())    // NOLINT
                {
                    fut.get();
                }
            }
            else    // left_size < right_size
            {
                // Means that always 'pivot' < 'last'.
                HPX_ASSERT(right_size >= 3ul);

                // Select pivot in right-side range.
                Iter pivot = middle + right_size / 2;
                Iter boundary = upper_bound_helper::call(
                    first, middle, hpx::util::invoke(proj, *pivot), comp, proj);
                Iter target = boundary + (pivot - middle);

                // Swap two blocks, [boundary, middle) and [middle, pivot+1).
                // After this, [first, last) will be divided into three blocks,
                //   [first, target), target, and [target+1, last).
                // And all elements of [first, target) are less than
                //   the thing of target.
                // And all elements of [target+1, last) are greater or equal than
                //   the thing of target.
                detail::sequential_rotate(boundary, middle, pivot + 1);

                hpx::future<void> fut =
                    execution::async_execute(policy.executor(), [&]() -> void {
                        // Process the range which is left-side of 'target'.
                        parallel_inplace_merge_helper(
                            policy, first, boundary, target, comp, proj);
                    });

                try
                {
                    // Process the range which is right-side of 'target'.
                    parallel_inplace_merge_helper(
                        policy, target + 1, pivot + 1, last, comp, proj);
                }
                catch (...)
                {
                    fut.wait();

                    std::vector<hpx::future<void>> futures;
                    futures.reserve(2);
                    futures.emplace_back(std::move(fut));
                    futures.emplace_back(hpx::make_exceptional_future<void>(
                        std::current_exception()));

                    std::list<std::exception_ptr> errors;
                    util::detail::handle_local_exceptions<ExPolicy>::call(
                        futures, errors);

                    // Not reachable.
                    HPX_ASSERT(false);
                }

                if (fut.valid())    // NOLINT
                {
                    fut.get();
                }
            }
        }

        template <typename ExPolicy, typename Iter, typename Sent,
            typename Comp, typename Proj>
        inline hpx::future<Iter> parallel_inplace_merge(ExPolicy&& policy,
            Iter first, Iter middle, Sent last, Comp&& comp, Proj&& proj)
        {
            return execution::async_execute(policy.executor(),
                [policy, first, middle, last, comp = std::forward<Comp>(comp),
                    proj = std::forward<Proj>(proj)]() mutable -> Iter {
                    try
                    {
                        parallel_inplace_merge_helper(policy, first, middle,
                            last, std::move(comp), std::move(proj));
                        return last;
                    }
                    catch (...)
                    {
                        util::detail::handle_local_exceptions<ExPolicy>::call(
                            std::current_exception());
                    }

                    // Not reachable.
                    HPX_ASSERT(false);
                });
        }

        template <typename Result>
        struct inplace_merge
          : public detail::algorithm<inplace_merge<Result>, Result>
        {
            inplace_merge()
              : inplace_merge::algorithm("inplace_merge")
            {
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename Comp, typename Proj>
            static Iter sequential(ExPolicy, Iter first, Iter middle, Sent last,
                Comp&& comp, Proj&& proj)
            {
                return sequential_inplace_merge(first, middle, last,
                    std::forward<Comp>(comp), std::forward<Proj>(proj));
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename Comp, typename Proj>
            static typename util::detail::algorithm_result<ExPolicy, Iter>::type
            parallel(ExPolicy&& policy, Iter first, Iter middle, Sent last,
                Comp&& comp, Proj&& proj)
            {
                using result = util::detail::algorithm_result<ExPolicy, Iter>;

                try
                {
                    return result::get(parallel_inplace_merge(
                        std::forward<ExPolicy>(policy), first, middle, last,
                        std::forward<Comp>(comp), std::forward<Proj>(proj)));
                }
                catch (...)
                {
                    return result::get(
                        detail::handle_exception<ExPolicy, Iter>::call(
                            std::current_exception()));
                }
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Iter>
        inline void get_void_result(Iter)
        {
        }

        template <typename Iter>
        inline hpx::future<void> get_void_result(hpx::future<Iter>&& f)
        {
            return hpx::future<void>(std::move(f));
        }
    }    // namespace detail

    // TODO: Support bidirectional iterator. (#2826)
    // For now, only support random access iterator.

    // clang-format off
    template <typename ExPolicy, typename RandIter,
        typename Comp = detail::less, typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_iterator<RandIter>::value &&
            traits::is_projected<Proj, RandIter>::value &&
            traits::is_indirect_callable<ExPolicy, Comp,
                traits::projected<Proj, RandIter>,
                traits::projected<Proj, RandIter>
            >::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::inplace_merge is deprecated, use hpx::inplace_merge "
        "instead")
        typename util::detail::algorithm_result<ExPolicy, RandIter>::type
        inplace_merge(ExPolicy&& policy, RandIter first, RandIter middle,
            RandIter last, Comp&& comp = Comp(), Proj&& proj = Proj())
    {
        static_assert((hpx::traits::is_random_access_iterator<RandIter>::value),
            "Required at least random access iterator.");

        using is_seq = execution::is_sequenced_execution_policy<ExPolicy>;

        return detail::inplace_merge<RandIter>().call(
            std::forward<ExPolicy>(policy), is_seq(), first, middle, last,
            std::forward<Comp>(comp), std::forward<Proj>(proj));
    }
}}}    // namespace hpx::parallel::v1

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::merge
    HPX_INLINE_CONSTEXPR_VARIABLE struct merge_t final
      : hpx::functional::tag<merge_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename RandIter1, typename RandIter2,
            typename RandIter3, typename Comp = hpx::parallel::v1::detail::less,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<RandIter1>::value &&
                hpx::traits::is_iterator<RandIter2>::value &&
                hpx::traits::is_iterator<RandIter3>::value &&
                hpx::traits::is_invocable<Comp,
                    typename std::iterator_traits<RandIter1>::value_type,
                    typename std::iterator_traits<RandIter2>::value_type
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            RandIter3>::type
        tag_invoke(merge_t, ExPolicy&& policy, RandIter1 first1,
            RandIter1 last1, RandIter2 first2, RandIter2 last2, RandIter3 dest,
            Comp&& comp = Comp())
        {
            static_assert(
                (hpx::traits::is_random_access_iterator<RandIter1>::value),
                "Required at least random access iterator.");
            static_assert(
                (hpx::traits::is_random_access_iterator<RandIter2>::value),
                "Requires at least random access iterator.");
            static_assert(
                (hpx::traits::is_random_access_iterator<RandIter3>::value),
                "Requires at least random access iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;
            using result_type = hpx::parallel::util::in_in_out_result<RandIter1,
                RandIter2, RandIter3>;

            return hpx::parallel::util::get_third_element(
                hpx::parallel::v1::detail::merge<result_type>().call(
                    std::forward<ExPolicy>(policy), is_seq(), first1, last1,
                    first2, last2, dest, std::forward<Comp>(comp),
                    hpx::parallel::util::projection_identity(),
                    hpx::parallel::util::projection_identity()));
        }

        // clang-format off
        template <typename RandIter1, typename RandIter2, typename RandIter3,
            typename Comp = hpx::parallel::v1::detail::less,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<RandIter1>::value &&
                hpx::traits::is_iterator<RandIter2>::value &&
                hpx::traits::is_iterator<RandIter3>::value &&
                hpx::traits::is_invocable<Comp,
                    typename std::iterator_traits<RandIter1>::value_type,
                    typename std::iterator_traits<RandIter2>::value_type
                >::value
            )>
        // clang-format on
        friend RandIter3 tag_invoke(merge_t, RandIter1 first1, RandIter1 last1,
            RandIter2 first2, RandIter2 last2, RandIter3 dest,
            Comp&& comp = Comp())
        {
            static_assert(
                (hpx::traits::is_random_access_iterator<RandIter1>::value),
                "Required at least random access iterator.");
            static_assert(
                (hpx::traits::is_random_access_iterator<RandIter2>::value),
                "Requires at least random access iterator.");
            static_assert(
                (hpx::traits::is_random_access_iterator<RandIter3>::value),
                "Requires at least random access iterator.");

            using result_type = hpx::parallel::util::in_in_out_result<RandIter1,
                RandIter2, RandIter3>;

            return hpx::parallel::util::get_third_element(
                hpx::parallel::v1::detail::merge<result_type>().call(
                    hpx::execution::seq, std::true_type(), first1, last1,
                    first2, last2, dest, std::forward<Comp>(comp),
                    hpx::parallel::util::projection_identity(),
                    hpx::parallel::util::projection_identity()));
        }
    } merge{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::inplace_merge
    HPX_INLINE_CONSTEXPR_VARIABLE struct inplace_merge_t final
      : hpx::functional::tag<inplace_merge_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename RandIter,
            typename Comp = hpx::parallel::v1::detail::less,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<RandIter>::value &&
                hpx::traits::is_invocable<Comp,
                    typename std::iterator_traits<RandIter>::value_type,
                    typename std::iterator_traits<RandIter>::value_type
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<
            ExPolicy>::type
        tag_invoke(inplace_merge_t, ExPolicy&& policy, RandIter first,
            RandIter middle, RandIter last, Comp&& comp = Comp())
        {
            static_assert(
                (hpx::traits::is_random_access_iterator<RandIter>::value),
                "Required at least random access iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            return hpx::parallel::v1::detail::get_void_result(
                hpx::parallel::v1::detail::inplace_merge<RandIter>().call(
                    std::forward<ExPolicy>(policy), is_seq(), first, middle,
                    last, std::forward<Comp>(comp),
                    hpx::parallel::util::projection_identity()));
        }

        // clang-format off
        template <typename RandIter,
            typename Comp = hpx::parallel::v1::detail::less,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<RandIter>::value &&
                hpx::traits::is_invocable<Comp,
                    typename std::iterator_traits<RandIter>::value_type,
                    typename std::iterator_traits<RandIter>::value_type
                >::value
            )>
        // clang-format on
        friend void tag_invoke(inplace_merge_t, RandIter first, RandIter middle,
            RandIter last, Comp&& comp = Comp())
        {
            static_assert(
                (hpx::traits::is_random_access_iterator<RandIter>::value),
                "Required at least random access iterator.");

            return hpx::parallel::v1::detail::get_void_result(
                hpx::parallel::v1::detail::inplace_merge<RandIter>().call(
                    hpx::execution::seq, std::true_type(), first, middle, last,
                    std::forward<Comp>(comp),
                    hpx::parallel::util::projection_identity()));
        }
    } inplace_merge{};
}    // namespace hpx

#endif    // DOXYGEN
