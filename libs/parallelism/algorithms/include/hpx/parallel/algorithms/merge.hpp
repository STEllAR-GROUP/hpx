//  Copyright (c) 2017 Taeguk Kwon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/util/tagged_tuple.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/execution/algorithms/detail/is_negative.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/transfer.hpp>
#include <hpx/parallel/tagspec.hpp>
#include <hpx/parallel/util/compare_projected.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/util/loop.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/transfer.hpp>

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
        template <typename InIter1, typename InIter2, typename OutIter,
            typename Comp, typename Proj1, typename Proj2>
        hpx::util::tuple<InIter1, InIter2, OutIter> sequential_merge(
            InIter1 first1, InIter1 last1, InIter2 first2, InIter2 last2,
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
                            break;
                    }
                    else
                    {
                        *dest++ = *first1++;
                        if (first1 == last1)
                            break;
                    }
                }
            }
            dest = std::copy(first1, last1, dest);
            dest = std::copy(first2, last2, dest);

            return hpx::util::make_tuple(last1, last2, dest);
        }

        struct upper_bound_helper
        {
            // upper_bound with projection function.
            template <typename RandIter, typename Type, typename Comp,
                typename Proj>
            static RandIter call(RandIter first, RandIter last,
                const Type& value, Comp comp, Proj proj)
            {
                typedef typename std::iterator_traits<RandIter>::difference_type
                    difference_type;

                difference_type count = std::distance(first, last);

                while (count > 0)
                {
                    difference_type step = count / 2;
                    RandIter mid = std::next(first, step);

                    if (!hpx::util::invoke(
                            comp, value, hpx::util::invoke(proj, *mid)))
                    {
                        first = ++mid;
                        count -= step + 1;
                    }
                    else
                    {
                        count = step;
                    }
                }

                return first;
            }

            typedef struct lower_bound_helper another_type;
        };

        struct lower_bound_helper
        {
            // lower_bound with projection function.
            template <typename RandIter, typename Type, typename Comp,
                typename Proj>
            static RandIter call(RandIter first, RandIter last,
                const Type& value, Comp comp, Proj proj)
            {
                typedef typename std::iterator_traits<RandIter>::difference_type
                    difference_type;

                difference_type count = std::distance(first, last);

                while (count > 0)
                {
                    difference_type step = count / 2;
                    RandIter mid = std::next(first, step);

                    if (hpx::util::invoke(
                            comp, hpx::util::invoke(proj, *mid), value))
                    {
                        first = ++mid;
                        count -= step + 1;
                    }
                    else
                    {
                        count = step;
                    }
                }

                return first;
            }

            typedef struct upper_bound_helper another_type;
        };

        template <typename ExPolicy, typename RandIter1, typename RandIter2,
            typename RandIter3, typename Comp, typename Proj1, typename Proj2,
            typename BinarySearchHelper>
        void parallel_merge_helper(ExPolicy policy, RandIter1 first1,
            RandIter1 last1, RandIter2 first2, RandIter2 last2, RandIter3 dest,
            Comp comp, Proj1 proj1, Proj2 proj2, bool range_reversal,
            BinarySearchHelper)
        {
            const std::size_t threshold = 65536ul;
            HPX_ASSERT(threshold >= 1ul);

            std::size_t size1 = last1 - first1;
            std::size_t size2 = last2 - first2;

            // Perform sequential merge if data size is smaller than threshold.
            if (size1 + size2 <= threshold)
            {
                if (range_reversal)
                {
                    sequential_merge(first2, first2 + size2, first1,
                        first1 + size1, dest, comp, proj2, proj1);
                }
                else
                {
                    sequential_merge(first1, first1 + size1, first2,
                        first2 + size2, dest, comp, proj1, proj2);
                }
                return;
            }

            // Let size1 is bigger than size2 always.
            if (size1 < size2)
            {
                // For stability of algorithm, must switch binary search methods
                //   when swapping size1 and size2.
                parallel_merge_helper(policy, first2, last2, first1, last1,
                    dest, comp, proj2, proj1, !range_reversal,
                    typename BinarySearchHelper::another_type());
                return;
            }

            HPX_ASSERT(size1 >= size2);
            HPX_ASSERT(size1 >= 1ul);

            RandIter1 mid1 = first1 + size1 / 2;
            RandIter2 boundary2 = BinarySearchHelper::call(
                first2, last2, hpx::util::invoke(proj1, *mid1), comp, proj2);
            RandIter3 target = dest + (mid1 - first1) + (boundary2 - first2);

            *target = *mid1;

            hpx::future<void> fut =
                execution::async_execute(policy.executor(), [&]() -> void {
                    // Process leftside ranges.
                    parallel_merge_helper(policy, first1, mid1, first2,
                        boundary2, dest, comp, proj1, proj2, range_reversal,
                        BinarySearchHelper());
                });

            try
            {
                // Process rightside ranges.
                parallel_merge_helper(policy, mid1 + 1, last1, boundary2, last2,
                    target + 1, comp, proj1, proj2, range_reversal,
                    BinarySearchHelper());
            }
            catch (...)
            {
                fut.wait();

                std::vector<hpx::future<void>> futures(2);
                futures[0] = std::move(fut);
                futures[1] = hpx::make_exceptional_future<void>(
                    std::current_exception());

                std::list<std::exception_ptr> errors;
                util::detail::handle_local_exceptions<ExPolicy>::call(
                    futures, errors);

                // Not reachable.
                HPX_ASSERT(false);
                return;
            }

            fut.get();
        }

        template <typename ExPolicy, typename RandIter1, typename RandIter2,
            typename RandIter3, typename Comp, typename Proj1, typename Proj2>
        hpx::future<hpx::util::tuple<RandIter1, RandIter2, RandIter3>>
        parallel_merge(ExPolicy&& policy, RandIter1 first1, RandIter1 last1,
            RandIter2 first2, RandIter2 last2, RandIter3 dest, Comp&& comp,
            Proj1&& proj1, Proj2&& proj2)
        {
            typedef hpx::util::tuple<RandIter1, RandIter2, RandIter3>
                result_type;

            typedef typename std::remove_reference<ExPolicy>::type ExPolicy_;
            typedef typename std::remove_reference<Comp>::type Comp_;
            typedef typename std::remove_reference<Proj1>::type Proj1_;
            typedef typename std::remove_reference<Proj2>::type Proj2_;

            hpx::future<result_type> f = execution::async_execute(
                policy.executor(),
                std::bind(
                    [first1, last1, first2, last2, dest](ExPolicy_& policy,
                        Comp_& comp, Proj1_& proj1,
                        Proj2_& proj2) -> result_type {
                        try
                        {
                            parallel_merge_helper(std::move(policy), first1,
                                last1, first2, last2, dest, std::move(comp),
                                std::move(proj1), std::move(proj2), false,
                                lower_bound_helper());

                            return hpx::util::make_tuple(last1, last2,
                                dest + (last1 - first1) + (last2 - first2));
                        }
                        catch (...)
                        {
                            util::detail::handle_local_exceptions<
                                ExPolicy>::call(std::current_exception());
                        }

                        // Not reachable.
                        HPX_ASSERT(false);
                    },
                    std::forward<ExPolicy>(policy), std::forward<Comp>(comp),
                    std::forward<Proj1>(proj1), std::forward<Proj2>(proj2)));

            return f;
        }

        template <typename IterTuple>
        struct merge : public detail::algorithm<merge<IterTuple>, IterTuple>
        {
            merge()
              : merge::algorithm("merge")
            {
            }

            template <typename ExPolicy, typename InIter1, typename InIter2,
                typename OutIter, typename Comp, typename Proj1, typename Proj2>
            static hpx::util::tuple<InIter1, InIter2, OutIter> sequential(
                ExPolicy, InIter1 first1, InIter1 last1, InIter2 first2,
                InIter2 last2, OutIter dest, Comp&& comp, Proj1&& proj1,
                Proj2&& proj2)
            {
                return sequential_merge(first1, last1, first2, last2, dest,
                    std::forward<Comp>(comp), std::forward<Proj1>(proj1),
                    std::forward<Proj2>(proj2));
            }

            template <typename ExPolicy, typename RandIter1, typename RandIter2,
                typename RandIter3, typename Comp, typename Proj1,
                typename Proj2>
            static typename util::detail::algorithm_result<ExPolicy,
                hpx::util::tuple<RandIter1, RandIter2, RandIter3>>::type
            parallel(ExPolicy&& policy, RandIter1 first1, RandIter1 last1,
                RandIter2 first2, RandIter2 last2, RandIter3 dest, Comp&& comp,
                Proj1&& proj1, Proj2&& proj2)
            {
                typedef hpx::util::tuple<RandIter1, RandIter2, RandIter3>
                    result_type;
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
    /// \tparam Proj1       The type of an optional projection function to be
    ///                     used for elements of the first range. This defaults
    ///                     to \a util::projection_identity
    /// \tparam Proj2       The type of an optional projection function to be
    ///                     used for elements of the second range. This defaults
    ///                     to \a util::projection_identity
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
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     first range as a projection operation before the
    ///                     actual comparison \a comp is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     second range as a projection operation before the
    ///                     actual comparison \a comp is invoked.
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
    /// \a hpx::future<tagged_tuple<tag::in1(RandIter1), tag::in2(RandIter2), tag::out(RandIter3)> >
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns
    /// \a tagged_tuple<tag::in1(RandIter1), tag::in2(RandIter2), tag::out(RandIter3)>
    ///           otherwise.
    ///           The \a merge algorithm returns the tuple of
    ///           the source iterator \a last1,
    ///           the source iterator \a last2,
    ///           the destination iterator to the end of the \a dest range.
    ///
    template <typename ExPolicy, typename RandIter1, typename RandIter2,
        typename RandIter3, typename Comp = detail::less,
        typename Proj1 = util::projection_identity,
        typename Proj2 = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(execution::is_execution_policy<ExPolicy>::value&&
                hpx::traits::is_iterator<RandIter1>::value&& hpx::traits::
                    is_iterator<RandIter2>::value&& hpx::traits::is_iterator<
                        RandIter3>::value&& traits::is_projected<Proj1,
                        RandIter1>::value&&
                        traits::is_projected<Proj2, RandIter2>::value&&
                            traits::is_indirect_callable<ExPolicy, Comp,
                                traits::projected<Proj1, RandIter1>,
                                traits::projected<Proj2, RandIter2>>::value)>
    typename util::detail::algorithm_result<ExPolicy,
        hpx::util::tagged_tuple<tag::in1(RandIter1), tag::in2(RandIter2),
            tag::out(RandIter3)>>::type
    merge(ExPolicy&& policy, RandIter1 first1, RandIter1 last1,
        RandIter2 first2, RandIter2 last2, RandIter3 dest, Comp&& comp = Comp(),
        Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2())
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

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;
        typedef hpx::util::tuple<RandIter1, RandIter2, RandIter3> result_type;

        return hpx::util::make_tagged_tuple<tag::in1, tag::in2, tag::out>(
            detail::merge<result_type>().call(std::forward<ExPolicy>(policy),
                is_seq(), first1, last1, first2, last2, dest,
                std::forward<Comp>(comp), std::forward<Proj1>(proj1),
                std::forward<Proj2>(proj2)));
    }

    /////////////////////////////////////////////////////////////////////////////
    // inplace_merge
    namespace detail {
        /// \cond NOINTERNAL

        // sequential inplace_merge with projection function.
        template <typename BidirIter, typename Comp, typename Proj>
        inline BidirIter sequential_inplace_merge(BidirIter first,
            BidirIter middle, BidirIter last, Comp&& comp, Proj&& proj)
        {
            std::inplace_merge(first, middle, last,
                util::compare_projected<Comp, Proj>(
                    std::forward<Comp>(comp), std::forward<Proj>(proj)));
            return last;
        }

        template <typename ExPolicy, typename RandIter, typename Comp,
            typename Proj>
        void parallel_inplace_merge_helper(ExPolicy&& policy, RandIter first,
            RandIter middle, RandIter last, Comp&& comp, Proj&& proj)
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
                RandIter pivot = first + left_size / 2;
                RandIter boundary = lower_bound_helper::call(
                    middle, last, hpx::util::invoke(proj, *pivot), comp, proj);
                RandIter target = pivot + (boundary - middle);

                // Swap two blocks, [pivot, middle) and [middle, boundary).
                // After this, [first, last) will be divided into three blocks,
                //   [first, target), target, and [target+1, last).
                // And all elements of [first, target) are less than
                //   the thing of target.
                // And all elements of [target+1, last) are greater or equal than
                //   the thing of target.
                std::rotate(pivot, middle, boundary);

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

                    std::vector<hpx::future<void>> futures(2);
                    futures[0] = std::move(fut);
                    futures[1] = hpx::make_exceptional_future<void>(
                        std::current_exception());

                    std::list<std::exception_ptr> errors;
                    util::detail::handle_local_exceptions<ExPolicy>::call(
                        futures, errors);

                    // Not reachable.
                    HPX_ASSERT(false);
                }
                if (fut.valid())    // NOLINT
                    fut.get();
            }
            else /* left_size < right_size */
            {
                // Means that always 'pivot' < 'last'.
                HPX_ASSERT(right_size >= 3ul);

                // Select pivot in right-side range.
                RandIter pivot = middle + right_size / 2;
                RandIter boundary = upper_bound_helper::call(
                    first, middle, hpx::util::invoke(proj, *pivot), comp, proj);
                RandIter target = boundary + (pivot - middle);

                // Swap two blocks, [boundary, middle) and [middle, pivot+1).
                // After this, [first, last) will be divided into three blocks,
                //   [first, target), target, and [target+1, last).
                // And all elements of [first, target) are less than
                //   the thing of target.
                // And all elements of [target+1, last) are greater or equal than
                //   the thing of target.
                std::rotate(boundary, middle, pivot + 1);

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

                    std::vector<hpx::future<void>> futures(2);
                    futures[0] = std::move(fut);
                    futures[1] = hpx::make_exceptional_future<void>(
                        std::current_exception());

                    std::list<std::exception_ptr> errors;
                    util::detail::handle_local_exceptions<ExPolicy>::call(
                        futures, errors);

                    // Not reachable.
                    HPX_ASSERT(false);
                }

                if (fut.valid())    // NOLINT
                    fut.get();
            }
        }

        template <typename ExPolicy, typename RandIter, typename Comp,
            typename Proj>
        inline hpx::future<RandIter> parallel_inplace_merge(ExPolicy&& policy,
            RandIter first, RandIter middle, RandIter last, Comp&& comp,
            Proj&& proj)
        {
            return execution::async_execute(policy.executor(),
                [policy, first, middle, last, comp = std::forward<Comp>(comp),
                    proj = std::forward<Proj>(proj)]() mutable -> RandIter {
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

        template <typename Iter>
        struct inplace_merge
          : public detail::algorithm<inplace_merge<Iter>, Iter>
        {
            inplace_merge()
              : inplace_merge::algorithm("inplace_merge")
            {
            }

            template <typename ExPolicy, typename RandIter, typename Comp,
                typename Proj>
            static RandIter sequential(ExPolicy, RandIter first,
                RandIter middle, RandIter last, Comp&& comp, Proj&& proj)
            {
                return sequential_inplace_merge(first, middle, last,
                    std::forward<Comp>(comp), std::forward<Proj>(proj));
            }

            template <typename ExPolicy, typename RandIter, typename Comp,
                typename Proj>
            static typename util::detail::algorithm_result<ExPolicy,
                RandIter>::type
            parallel(ExPolicy&& policy, RandIter first, RandIter middle,
                RandIter last, Comp&& comp, Proj&& proj)
            {
                typedef util::detail::algorithm_result<ExPolicy, RandIter>
                    algorithm_result;

                try
                {
                    return algorithm_result::get(parallel_inplace_merge(
                        std::forward<ExPolicy>(policy), first, middle, last,
                        std::forward<Comp>(comp), std::forward<Proj>(proj)));
                }
                catch (...)
                {
                    return algorithm_result::get(
                        detail::handle_exception<ExPolicy, RandIter>::call(
                            std::current_exception()));
                }
            }
        };
        /// \endcond
    }    // namespace detail

    // TODO: Support bidirectional iterator. (#2826)
    // For now, only support random access iterator.
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
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
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
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each of the elements as a
    ///                     projection operation before the actual predicate
    ///                     \a is invoked.
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
    ///           \a hpx::future<RandIter> if the execution policy is of type
    ///           \a sequenced_task_policy or \a parallel_task_policy
    ///           and returns \a RandIter otherwise.
    ///           The \a inplace_merge algorithm returns
    ///           the source iterator \a last
    ///
    template <typename ExPolicy, typename RandIter,
        typename Comp = detail::less, typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(execution::is_execution_policy<ExPolicy>::value&&
                hpx::traits::is_iterator<RandIter>::value&&
                    traits::is_projected<Proj, RandIter>::value&&
                        traits::is_indirect_callable<ExPolicy, Comp,
                            traits::projected<Proj, RandIter>,
                            traits::projected<Proj, RandIter>>::value)>
    typename util::detail::algorithm_result<ExPolicy, RandIter>::type
    inplace_merge(ExPolicy&& policy, RandIter first, RandIter middle,
        RandIter last, Comp&& comp = Comp(), Proj&& proj = Proj())
    {
        static_assert((hpx::traits::is_random_access_iterator<RandIter>::value),
            "Required at least random access iterator.");

        typedef execution::is_sequenced_execution_policy<ExPolicy> is_seq;

        return detail::inplace_merge<RandIter>().call(
            std::forward<ExPolicy>(policy), is_seq(), first, middle, last,
            std::forward<Comp>(comp), std::forward<Proj>(proj));
    }
}}}    // namespace hpx::parallel::v1
