//  Copyright (c) 2017 Taeguk Kwon
//  Copyright (c) 2017-2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \page hpx::merge, hpx::inplace_merge
/// \headerfile hpx/algorithm.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    // clang-format off

    /// Merges two sorted ranges [first1, last1) and [first2, last2)
    /// into one sorted range beginning at \a dest. The order of
    /// equivalent elements in each of the original two ranges is preserved.
    /// For equivalent elements in the original two ranges, the elements from
    /// the first range precede the elements from the second range.
    /// The destination range cannot overlap with either of the input ranges.
    /// Executed according to the policy.
    ///
    /// \note   Complexity: Performs
    ///         O(std::distance(first1, last1) + std::distance(first2, last2))
    ///         applications of the comparison \a comp and each projection.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam RandIter1   The type of the source iterators used (deduced)
    ///                     representing the first sorted range.
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    /// \tparam RandIter2   The type of the source iterators used (deduced)
    ///                     representing the second sorted range.
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    /// \tparam RandIter3   The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of a
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
    ///           \a hpx::future<RandIter3> >
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns
    ///           \a RandIter3 otherwise.
    ///           The \a merge algorithm returns the destination iterator to
    ///           the end of the \a dest range.
    ///
    template <typename ExPolicy, typename RandIter1, typename RandIter2,
        typename RandIter3, typename Comp = hpx::parallel::detail::less>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, RandIter3>
    merge(ExPolicy&& policy, RandIter1 first1, RandIter1 last1,
        RandIter2 first2, RandIter2 last2, RandIter3 dest, Comp&& comp = Comp());

    /// Merges two sorted ranges [first1, last1) and [first2, last2)
    /// into one sorted range beginning at \a dest. The order of
    /// equivalent elements in each of the original two ranges is preserved.
    /// For equivalent elements in the original two ranges, the elements from
    /// the first range precede the elements from the second range.
    /// The destination range cannot overlap with either of the input ranges.
    ///
    /// \note   Complexity: Performs
    ///         O(std::distance(first1, last1) + std::distance(first2, last2))
    ///         applications of the comparison \a comp and each projection.
    ///
    /// \tparam RandIter1   The type of the source iterators used (deduced)
    ///                     representing the first sorted range.
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    /// \tparam RandIter2   The type of the source iterators used (deduced)
    ///                     representing the second sorted range.
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    /// \tparam RandIter3   The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a merge requires \a Comp to meet the
    ///                     requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    ///
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
    /// \returns  The \a merge algorithm returns a \a RandIter3.
    ///           The \a merge algorithm returns the destination iterator to
    ///           the end of the \a dest range.
    ///
    template <typename RandIter1, typename RandIter2,
        typename RandIter3, typename Comp = hpx::parallel::detail::less>
    RandIter3 merge(RandIter1 first1, RandIter1 last1,
        RandIter2 first2, RandIter2 last2, RandIter3 dest, Comp&& comp = Comp());

    /// Merges two consecutive sorted ranges [first, middle) and
    /// [middle, last) into one sorted range [first, last). The order of
    /// equivalent elements in each of the original two ranges is preserved.
    /// For equivalent elements in the original two ranges, the elements from
    /// the first range precede the elements from the second range. Executed
    /// according to the policy.
    ///
    ///
    /// \note   Complexity: Performs O(std::distance(first, last))
    ///         applications of the comparison \a comp and each projection.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam RandIter    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
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
    ///           the source iterator \a last.
    ///
    template <typename ExPolicy, typename RandIter,
        typename Comp = hpx::parallel::detail::less>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy>
    inplace_merge(ExPolicy&& policy, RandIter first, RandIter middle,
        RandIter last, Comp&& comp = Comp());

    /// Merges two consecutive sorted ranges [first, middle) and
    /// [middle, last) into one sorted range [first, last). The order of
    /// equivalent elements in each of the original two ranges is preserved.
    /// For equivalent elements in the original two ranges, the elements from
    /// the first range precede the elements from the second range.
    ///
    /// \note   Complexity: Performs O(std::distance(first, last))
    ///         applications of the comparison \a comp and each projection.
    ///
    /// \tparam RandIter    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     random access iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced). Unlike its sequential form, the parallel
    ///                     overload of \a inplace_merge requires \a Comp
    ///                     to meet the requirements of \a CopyConstructible.
    ///                     This defaults to std::less<>
    ///
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
    /// \returns  The \a inplace_merge algorithm returns a \a void.
    ///           The \a inplace_merge algorithm returns
    ///           the source iterator \a last.
    ///
    template <typename RandIter, typename Comp = hpx::parallel::detail::less>
    void inplace_merge(RandIter first, RandIter middle,
        RandIter last, Comp&& comp = Comp());
    // clang-format on
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/datastructures/tuple.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/execution/executors/execution_parameters.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/functional/invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/unwrap_iterator.hpp>
#include <hpx/modules/itt_notify.hpp>
#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/parallel/algorithms/detail/advance_and_get_distance.hpp>
#include <hpx/parallel/algorithms/detail/advance_to_sentinel.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/rotate.hpp>
#include <hpx/parallel/algorithms/detail/upper_lower_bound.hpp>
#include <hpx/parallel/util/compare_projected.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/foreach_partitioner.hpp>
#include <hpx/parallel/util/result_types.hpp>
#include <hpx/type_support/identity.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iterator>
#include <list>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx::parallel {

    /////////////////////////////////////////////////////////////////////////////
    // merge
    namespace detail {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        struct lower_bound_helper;

        struct upper_bound_helper
        {
            template <typename Iter, typename Sent, typename T, typename Comp,
                typename Proj>
            static constexpr Iter call(
                Iter first, Sent last, T const& value, Comp&& comp, Proj&& proj)
            {
                return detail::upper_bound(first, last, value,
                    HPX_FORWARD(Comp, comp), HPX_FORWARD(Proj, proj));
            }

            template <typename Iter, typename T, typename Comp, typename Proj>
            static constexpr Iter call_n(Iter first,
                typename std::iterator_traits<Iter>::difference_type count,
                T const& value, Comp&& comp, Proj&& proj)
            {
                return detail::upper_bound_n(first, count, value,
                    HPX_FORWARD(Comp, comp), HPX_FORWARD(Proj, proj));
            }

            using another_type = lower_bound_helper;
        };

        struct lower_bound_helper
        {
            template <typename Iter, typename Sent, typename T, typename Comp,
                typename Proj>
            static constexpr Iter call(
                Iter first, Sent last, T const& value, Comp&& comp, Proj&& proj)
            {
                return detail::lower_bound(first, last, value,
                    HPX_FORWARD(Comp, comp), HPX_FORWARD(Proj, proj));
            }

            template <typename Iter, typename T, typename Comp, typename Proj>
            static constexpr Iter call_n(Iter first,
                typename std::iterator_traits<Iter>::difference_type count,
                T const& value, Comp&& comp, Proj&& proj)
            {
                return detail::lower_bound_n(first, count, value,
                    HPX_FORWARD(Comp, comp), HPX_FORWARD(Proj, proj));
            }

            using another_type = upper_bound_helper;
        };

        template <typename T>
        HPX_FORCEINLINE decltype(auto) init_value([[maybe_unused]] T&& val)
        {
            if constexpr (std::is_default_constructible_v<std::decay_t<T>>)
            {
                return std::decay_t<T>{};
            }
            else
            {
                return HPX_FORWARD(T, val);
            }
        }

        // sequential merge helper with projection function.
        template <typename Iter1, typename Sent1, typename Iter2,
            typename Sent2, typename OutIter, typename Comp, typename Proj1,
            typename Proj2>
        constexpr util::in_in_out_result<Iter1, Iter2, OutIter>
        sequential_merge_helper(Iter1 first1, Sent1 last1, Iter2 first2,
            Sent2 last2, OutIter dest, Comp&& comp, Proj1&& proj1,
            Proj2&& proj2)
        {
            using value_type1 =
                typename std::iterator_traits<Iter1>::value_type;
            using value_type2 =
                typename std::iterator_traits<Iter2>::value_type;

            if (first1 != last1 && first2 != last2)
            {
                auto val1 = HPX_INVOKE(proj1, *first1);
                auto val2 = init_value(HPX_INVOKE(proj2, *first2));
                while (true)
                {
                    while (first2 != last2)
                    {
                        value_type2 elem2 = *first2;
                        val2 = HPX_INVOKE(proj2, elem2);
                        if (static_cast<bool>(HPX_INVOKE(comp, val2, val1)))
                        {
                            *dest = HPX_MOVE(elem2);
                            ++dest;
                            ++first2;
                        }
                        else
                        {
                            break;
                        }
                    }
                    if (first2 == last2)
                    {
                        break;
                    }

                    while (first1 != last1)
                    {
                        value_type1 elem1 = *first1;
                        val1 = HPX_INVOKE(proj1, elem1);
                        if (!static_cast<bool>(HPX_INVOKE(comp, val2, val1)))
                        {
                            *dest = HPX_MOVE(elem1);
                            ++dest;
                            ++first1;
                        }
                        else
                        {
                            break;
                        }
                    }
                    if (first1 == last1)
                    {
                        break;
                    }
                }
            }

            return {first1, first2, dest};
        }

        // sequential merge helper without projection function.
        template <typename Iter1, typename Sent1, typename Iter2,
            typename Sent2, typename OutIter, typename Comp>
        constexpr util::in_in_out_result<Iter1, Iter2, OutIter>
        sequential_merge_helper(Iter1 first1, Sent1 last1, Iter2 first2,
            Sent2 last2, OutIter dest, Comp&& comp)
        {
            using value_type1 =
                typename std::iterator_traits<Iter1>::value_type;
            using value_type2 =
                typename std::iterator_traits<Iter2>::value_type;

#if !defined(HPX_MSVC)
            if constexpr (std::is_same_v<value_type1, value_type2>)
            {
                while (HPX_LIKELY(first1 != last1 && first2 != last2))
                {
                    // This code enables the compiler to generate conditional
                    // mov operations instead of branches
                    Iter1 const next1 = std::next(first1);
                    Iter2 const next2 = std::next(first2);

                    value_type1 val1 = *first1;
                    value_type2 const val2 = *first2;

                    if (static_cast<bool>(HPX_INVOKE(comp, val2, val1)))
                    {
                        val1 = HPX_MOVE(val2);
                        first2 = next2;
                    }
                    else
                    {
                        first1 = next1;
                    }

                    *dest = HPX_MOVE(val1);
                    ++dest;
                }
            }
            else
#endif
            {
                if (HPX_LIKELY(first1 != last1 && first2 != last2))
                {
                    value_type1 val1 = *first1;
                    value_type2 val2 = init_value(*first2);
                    while (true)
                    {
                        while (first2 != last2)
                        {
                            val2 = *first2;
                            if (static_cast<bool>(HPX_INVOKE(comp, val2, val1)))
                            {
                                *dest = HPX_MOVE(val2);
                                ++dest;
                                ++first2;
                            }
                            else
                            {
                                break;
                            }
                        }
                        if (HPX_UNLIKELY(first2 == last2))
                        {
                            break;
                        }

                        while (first1 != last1)
                        {
                            val1 = *first1;
                            if (!static_cast<bool>(
                                    HPX_INVOKE(comp, val2, val1)))
                            {
                                *dest = HPX_MOVE(val1);
                                ++dest;
                                ++first1;
                            }
                            else
                            {
                                break;
                            }
                        }
                        if (HPX_UNLIKELY(first1 == last1))
                        {
                            break;
                        }
                    }
                }
            }

            return {first1, first2, dest};
        }

        // sequential merge with projection function.
        template <typename Iter1, typename Sent1, typename Iter2,
            typename Sent2, typename OutIter, typename Comp, typename Proj1,
            typename Proj2>
        constexpr util::in_in_out_result<Iter1, Iter2, OutIter>
        sequential_merge(Iter1 start1, Sent1 sent1, Iter2 start2, Sent2 sent2,
            OutIter out, Comp&& comp, Proj1&& proj1, Proj2&& proj2)
        {
            if constexpr (hpx::traits::is_random_access_iterator_v<Iter1> &&
                hpx::traits::is_random_access_iterator_v<Iter2>)
            {
                auto first1 = hpx::util::get_unwrapped(start1);
                auto first2 = hpx::util::get_unwrapped(start2);
                auto dest = hpx::util::get_unwrapped(out);

                auto end1 = start1;
                auto const len1 = detail::advance_and_get_distance(end1, sent1);
                auto const last1 = hpx::util::get_unwrapped(end1);

                auto end2 = start2;
                auto const len2 = detail::advance_and_get_distance(end2, sent2);
                auto const last2 = hpx::util::get_unwrapped(end2);

                auto merge_result = sequential_merge_helper(first1, last1,
                    first2, last2, dest, HPX_FORWARD(Comp, comp),
                    HPX_FORWARD(Proj1, proj1), HPX_FORWARD(Proj2, proj2));

                auto copy_result1 =
                    util::copy(merge_result.in1, last1, merge_result.out);
                [[maybe_unused]] auto copy_result2 =
                    util::copy(merge_result.in2, last2, copy_result1.out);

                if constexpr (!hpx::traits::is_input_iterator_v<OutIter>)
                {
                    return {end1, end2, copy_result2.out};
                }
                else
                {
                    return {end1, end2, std::next(out, len1 + len2)};
                }
            }
            else
            {
                auto merge_result = sequential_merge_helper(start1, sent1,
                    start2, sent2, out, HPX_FORWARD(Comp, comp),
                    HPX_FORWARD(Proj1, proj1), HPX_FORWARD(Proj2, proj2));

                auto copy_result1 =
                    util::copy(merge_result.in1, sent1, merge_result.out);
                [[maybe_unused]] auto copy_result2 =
                    util::copy(merge_result.in2, sent2, copy_result1.out);

                return {copy_result1.in, copy_result2.in, copy_result2.out};
            }
        }

        // sequential merge without projection function.
        template <typename Iter1, typename Sent1, typename Iter2,
            typename Sent2, typename OutIter, typename Comp>
        constexpr util::in_in_out_result<Iter1, Iter2, OutIter>
        sequential_merge(Iter1 start1, Sent1 sent1, Iter2 start2, Sent2 sent2,
            OutIter out, Comp&& comp, hpx::identity, hpx::identity)
        {
            if constexpr (hpx::traits::is_random_access_iterator_v<Iter1> &&
                hpx::traits::is_random_access_iterator_v<Iter2>)
            {
                auto first1 = hpx::util::get_unwrapped(start1);
                auto first2 = hpx::util::get_unwrapped(start2);
                auto dest = hpx::util::get_unwrapped(out);

                auto end1 = start1;
                auto const len1 = detail::advance_and_get_distance(end1, sent1);
                auto const last1 = hpx::util::get_unwrapped(end1);

                auto end2 = start2;
                auto const len2 = detail::advance_and_get_distance(end2, sent2);
                auto const last2 = hpx::util::get_unwrapped(end2);

                auto merge_result = sequential_merge_helper(first1, last1,
                    first2, last2, dest, HPX_FORWARD(Comp, comp));

                auto copy_result1 =
                    util::copy(merge_result.in1, last1, merge_result.out);
                [[maybe_unused]] auto copy_result2 =
                    util::copy(merge_result.in2, last2, copy_result1.out);

                if constexpr (!hpx::traits::is_input_iterator_v<OutIter>)
                {
                    return {end1, end2, copy_result2.out};
                }
                else
                {
                    return {end1, end2, std::next(out, len1 + len2)};
                }
            }
            else
            {
                auto merge_result = sequential_merge_helper(
                    start1, sent1, start2, sent2, out, HPX_FORWARD(Comp, comp));

                auto copy_result1 =
                    util::copy(merge_result.in1, sent1, merge_result.out);
                [[maybe_unused]] auto copy_result2 =
                    util::copy(merge_result.in2, sent2, copy_result1.out);

                return {copy_result1.in, copy_result2.in, copy_result2.out};
            }
        }

        template <typename Iter1, typename Iter2, typename Comp, typename Proj1,
            typename Proj2, typename BinarySearchHelper>
        auto get_reshape_chunks(std::size_t len1, Iter2 first2, Iter2 last2,
            Comp&& comp, Proj1&& proj1, Proj2&& proj2, BinarySearchHelper)
        {
            using merge_region =
                hpx::tuple<Iter1, std::size_t, Iter2, std::size_t, std::size_t>;

            auto reshape = [len1, first2, last2, comp, proj1, proj2](
                               auto&& shape) {
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
                static hpx::util::itt::event notify_event("reshape");
                hpx::util::itt::mark_event e(notify_event);
#endif
                std::vector<merge_region> reshaped;
                reshaped.reserve(2 * std::size(shape));

                Iter2 it2 = first2;
                std::size_t dest_start = 0;
                for (auto [it1, size1, base] : shape)
                {
                    Iter2 l2 = last2;
                    if (base + size1 != len1)
                    {
                        // gcc complains about using HPX_INVOKE here
                        l2 = BinarySearchHelper::call(it2, last2,
                            std::invoke(proj1, *std::next(it1, size1)), comp,
                            proj2);
                    }

                    std::size_t size2 = detail::distance(it2, l2);

                    // prevent chunks from growing too large
                    if (size2 <= static_cast<std::size_t>(
                                     1.2 * static_cast<double>(size1)))
                    {
                        reshaped.emplace_back(
                            it1, size1, it2, size2, dest_start);

                        it2 = l2;
                        dest_start += size1 + size2;
                    }
                    else
                    {
                        // split first sequence into smaller pieces based
                        // on chunks of the second (sub-)sequence
                        Iter1 begin1 = it1;
                        std::size_t remainder1 = size1;

                        std::size_t chunk_size2 = size1;
                        std::size_t remainder2 = size2 - chunk_size2;

                        Iter2 begin2 = it2;
                        Iter2 end2 = std::next(begin2, chunk_size2);

                        while (chunk_size2 != 0)
                        {
                            Iter1 end_chunk1;
                            if (end2 != l2)
                            {
                                using binary_search =
                                    typename BinarySearchHelper::another_type;

                                // gcc complains about using HPX_INVOKE here
                                end_chunk1 =
                                    binary_search::call_n(begin1, remainder1,
                                        std::invoke(proj2, *end2), comp, proj1);
                            }
                            else
                            {
                                end_chunk1 = std::next(it1, size1);
                            }

                            auto chunk_size1 =
                                detail::distance(begin1, end_chunk1);

                            reshaped.emplace_back(begin1, chunk_size1, begin2,
                                chunk_size2, dest_start);

                            dest_start += chunk_size1 + chunk_size2;

                            chunk_size2 = (std::min) (size1, remainder2);
                            remainder2 -= chunk_size2;

                            begin1 = end_chunk1;
                            remainder1 -= chunk_size1;

                            begin2 = end2;
                            end2 = std::next(begin2, chunk_size2);
                        }

                        it2 = l2;
                    }
                }
                return reshaped;
            };

            return reshape;
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename Iter1, typename Sent1,
            typename Iter2, typename Sent2, typename Iter3, typename Comp,
            typename Proj1, typename Proj2>
        decltype(auto) parallel_merge(ExPolicy&& policy, Iter1 first1,
            Sent1 last1, Iter2 first2, Sent2 last2, Iter3 dest, Comp&& comp,
            Proj1&& proj1, Proj2&& proj2)
        {
            auto end1 = first1;
            auto const len1 = detail::advance_and_get_distance(end1, last1);

            auto end2 = first2;
            auto const len2 = detail::advance_and_get_distance(end2, last2);

            using result_type = util::in_in_out_result<Iter1, Iter2, Iter3>;

            auto f1 = [dest, comp, proj1, proj2](Iter1 it1, std::size_t size1,
                          Iter2 it2, std::size_t size2, std::size_t dest_base) {
                if (size1 != 0 || size2 != 0)
                {
                    sequential_merge(it1, std::next(it1, size1), it2,
                        std::next(it2, size2), std::next(dest, dest_base), comp,
                        proj1, proj2);
                }
            };

            if (len1 > len2)
            {
                auto f2 = [first2, len1, len2, dest](Iter1 l1) {
                    return result_type{l1, std::next(first2, len2),
                        std::next(dest, len1 + len2)};
                };

                auto reshape = get_reshape_chunks<Iter1>(len1, first2, end2,
                    comp, proj1, proj2, lower_bound_helper{});

                return util::foreach_partitioner<ExPolicy>::call(
                    HPX_FORWARD(ExPolicy, policy), first1, len1, HPX_MOVE(f1),
                    HPX_MOVE(f2), HPX_MOVE(reshape));
            }

            auto f2 = [first1, len1, len2, dest](Iter2 l2) {
                return result_type{
                    std::next(first1, len1), l2, std::next(dest, len1 + len2)};
            };

            auto reshape = get_reshape_chunks<Iter2>(
                len2, first1, end1, comp, proj2, proj1, upper_bound_helper{});

            return util::foreach_partitioner<ExPolicy>::call(
                HPX_FORWARD(ExPolicy, policy), first2, len2, HPX_MOVE(f1),
                HPX_MOVE(f2), HPX_MOVE(reshape));
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename IterTuple>
        struct merge : public algorithm<merge<IterTuple>, IterTuple>
        {
            constexpr merge() noexcept
              : algorithm<merge, IterTuple>("merge")
            {
            }

            template <typename ExPolicy, typename Iter1, typename Sent1,
                typename Iter2, typename Sent2, typename Iter3, typename Comp,
                typename Proj1, typename Proj2>
            static constexpr util::in_in_out_result<Iter1, Iter2, Iter3>
            sequential(ExPolicy, Iter1 first1, Sent1 last1, Iter2 first2,
                Sent2 last2, Iter3 dest, Comp&& comp, Proj1&& proj1,
                Proj2&& proj2)
            {
                return sequential_merge(first1, last1, first2, last2, dest,
                    HPX_FORWARD(Comp, comp), HPX_FORWARD(Proj1, proj1),
                    HPX_FORWARD(Proj2, proj2));
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
                using result_type = util::in_in_out_result<Iter1, Iter2, Iter3>;
                using algorithm_result =
                    util::detail::algorithm_result<ExPolicy, result_type>;

                try
                {
                    return algorithm_result::get(parallel_merge(
                        HPX_FORWARD(ExPolicy, policy), first1, last1, first2,
                        last2, dest, HPX_FORWARD(Comp, comp),
                        HPX_FORWARD(Proj1, proj1), HPX_FORWARD(Proj2, proj2)));
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

    /////////////////////////////////////////////////////////////////////////////
    // inplace_merge
    namespace detail {

        // sequential inplace_merge with projection function.
        template <typename Iter, typename Sent, typename Comp, typename Proj>
        constexpr Iter sequential_inplace_merge(
            Iter first, Iter middle, Sent last, Comp&& comp, Proj&& proj)
        {
            std::inplace_merge(first, middle,
                detail::advance_to_sentinel(middle, last),
                util::compare_projected<Comp, Proj>(
                    HPX_FORWARD(Comp, comp), HPX_FORWARD(Proj, proj)));
            return last;
        }

        template <typename ExPolicy, typename Iter, typename Sent,
            typename Comp, typename Proj>
        void parallel_inplace_merge_helper(ExPolicy&& policy, Iter first,
            Iter middle, Sent last, Comp&& comp, Proj&& proj)
        {
            constexpr std::size_t threshold = 65536ul;
            static_assert(threshold >= 5ul);

            std::size_t const left_size = middle - first;
            std::size_t const right_size = last - middle;

            // Perform sequential inplace_merge
            //   if data size is smaller than threshold.
            if (left_size + right_size <= threshold)
            {
                sequential_inplace_merge(first, middle, last,
                    HPX_FORWARD(Comp, comp), HPX_FORWARD(Proj, proj));
                return;
            }

            if (left_size >= right_size)
            {
                // Means that always 'pivot' < 'middle'.
                HPX_ASSERT(left_size >= 3ul);

                // Select pivot in left-side range.
                Iter pivot = first + left_size / 2;
                Iter boundary = lower_bound_helper::call(
                    middle, last, HPX_INVOKE(proj, *pivot), comp, proj);
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
                    futures.emplace_back(HPX_MOVE(fut));
                    futures.emplace_back(hpx::make_exceptional_future<void>(
                        std::current_exception()));

                    std::list<std::exception_ptr> errors;
                    util::detail::handle_local_exceptions<ExPolicy>::call(
                        futures, errors);

                    HPX_UNREACHABLE;
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
                    first, middle, HPX_INVOKE(proj, *pivot), comp, proj);
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
                    futures.emplace_back(HPX_MOVE(fut));
                    futures.emplace_back(hpx::make_exceptional_future<void>(
                        std::current_exception()));

                    std::list<std::exception_ptr> errors;
                    util::detail::handle_local_exceptions<ExPolicy>::call(
                        futures, errors);

                    HPX_UNREACHABLE;
                }

                if (fut.valid())    // NOLINT
                {
                    fut.get();
                }
            }
        }

        template <typename ExPolicy, typename Iter, typename Sent,
            typename Comp, typename Proj>
        hpx::future<Iter> parallel_inplace_merge(ExPolicy&& policy, Iter first,
            Iter middle, Sent last, Comp&& comp, Proj&& proj)
        {
            return execution::async_execute(policy.executor(),
                [policy, first, middle, last, comp = HPX_FORWARD(Comp, comp),
                    proj = HPX_FORWARD(Proj, proj)]() mutable -> Iter {
                    try
                    {
                        parallel_inplace_merge_helper(policy, first, middle,
                            last, HPX_MOVE(comp), HPX_MOVE(proj));
                        return last;
                    }
                    catch (...)
                    {
                        util::detail::handle_local_exceptions<ExPolicy>::call(
                            std::current_exception());

                        HPX_UNREACHABLE;
                    }
                });
        }

        template <typename Result>
        struct inplace_merge : public algorithm<inplace_merge<Result>, Result>
        {
            constexpr inplace_merge() noexcept
              : algorithm<inplace_merge, Result>("inplace_merge")
            {
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename Comp, typename Proj>
            static constexpr Iter sequential(ExPolicy, Iter first, Iter middle,
                Sent last, Comp&& comp, Proj&& proj)
            {
                return sequential_inplace_merge(first, middle, last,
                    HPX_FORWARD(Comp, comp), HPX_FORWARD(Proj, proj));
            }

            template <typename ExPolicy, typename Iter, typename Sent,
                typename Comp, typename Proj>
            static util::detail::algorithm_result_t<ExPolicy, Iter> parallel(
                ExPolicy&& policy, Iter first, Iter middle, Sent last,
                Comp&& comp, Proj&& proj)
            {
                using result = util::detail::algorithm_result<ExPolicy, Iter>;

                try
                {
                    return result::get(parallel_inplace_merge(
                        HPX_FORWARD(ExPolicy, policy), first, middle, last,
                        HPX_FORWARD(Comp, comp), HPX_FORWARD(Proj, proj)));
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
        constexpr void get_void_result(Iter) noexcept
        {
        }

        template <typename Iter>
        hpx::future<void> get_void_result(hpx::future<Iter>&& f) noexcept
        {
            return hpx::future<void>(HPX_MOVE(f));
        }
    }    // namespace detail
}    // namespace hpx::parallel

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::merge
    inline constexpr struct merge_t final
      : hpx::detail::tag_parallel_algorithm<merge_t>
    {
    private:
        template <typename ExPolicy, typename RandIter1, typename RandIter2,
            typename RandIter3, typename Comp = hpx::parallel::detail::less>
        // clang-format off
            requires (
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<RandIter1> &&
                hpx::traits::is_iterator_v<RandIter2> &&
                hpx::traits::is_iterator_v<RandIter3> &&
                hpx::is_invocable_v<Comp,
                    typename std::iterator_traits<RandIter1>::value_type,
                    typename std::iterator_traits<RandIter2>::value_type
                >
            )
        // clang-format on
        friend hpx::parallel::util::detail::algorithm_result_t<ExPolicy,
            RandIter3>
        tag_fallback_invoke(merge_t, ExPolicy&& policy, RandIter1 first1,
            RandIter1 last1, RandIter2 first2, RandIter2 last2, RandIter3 dest,
            Comp comp = Comp())
        {
            static_assert(hpx::traits::is_random_access_iterator_v<RandIter1>,
                "Required at least random access iterator.");
            static_assert(hpx::traits::is_random_access_iterator_v<RandIter2>,
                "Requires at least random access iterator.");
            static_assert(hpx::traits::is_random_access_iterator_v<RandIter3>,
                "Requires at least random access iterator.");

            using result_type = hpx::parallel::util::in_in_out_result<RandIter1,
                RandIter2, RandIter3>;

            return hpx::parallel::util::get_third_element(
                hpx::parallel::detail::merge<result_type>().call(
                    HPX_FORWARD(ExPolicy, policy), first1, last1, first2, last2,
                    dest, HPX_MOVE(comp), hpx::identity_v, hpx::identity_v));
        }

        template <typename RandIter1, typename RandIter2, typename RandIter3,
            typename Comp = hpx::parallel::detail::less>
        // clang-format off
            requires (
                hpx::traits::is_iterator_v<RandIter1> &&
                hpx::traits::is_iterator_v<RandIter2> &&
                hpx::traits::is_iterator_v<RandIter3> &&
                hpx::is_invocable_v<Comp,
                    typename std::iterator_traits<RandIter1>::value_type,
                    typename std::iterator_traits<RandIter2>::value_type
                >
            )
        // clang-format on
        friend RandIter3 tag_fallback_invoke(merge_t, RandIter1 first1,
            RandIter1 last1, RandIter2 first2, RandIter2 last2, RandIter3 dest,
            Comp comp = Comp())
        {
            static_assert(hpx::traits::is_random_access_iterator_v<RandIter1>,
                "Required at least random access iterator.");
            static_assert(hpx::traits::is_random_access_iterator_v<RandIter2>,
                "Requires at least random access iterator.");
            static_assert(hpx::traits::is_random_access_iterator_v<RandIter3>,
                "Requires at least random access iterator.");

            using result_type = hpx::parallel::util::in_in_out_result<RandIter1,
                RandIter2, RandIter3>;

            return hpx::parallel::util::get_third_element(
                hpx::parallel::detail::merge<result_type>().call(
                    hpx::execution::seq, first1, last1, first2, last2, dest,
                    HPX_MOVE(comp), hpx::identity_v, hpx::identity_v));
        }
    } merge{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::inplace_merge
    inline constexpr struct inplace_merge_t final
      : hpx::detail::tag_parallel_algorithm<inplace_merge_t>
    {
    private:
        template <typename ExPolicy, typename RandIter,
            typename Comp = hpx::parallel::detail::less>
        // clang-format off
            requires (
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<RandIter> &&
                hpx::is_invocable_v<Comp,
                    typename std::iterator_traits<RandIter>::value_type,
                    typename std::iterator_traits<RandIter>::value_type
                >
            )
        // clang-format on
        friend hpx::parallel::util::detail::algorithm_result_t<ExPolicy>
        tag_fallback_invoke(inplace_merge_t, ExPolicy&& policy, RandIter first,
            RandIter middle, RandIter last, Comp comp = Comp())
        {
            static_assert(hpx::traits::is_random_access_iterator_v<RandIter>,
                "Required at least random access iterator.");

            return hpx::parallel::detail::get_void_result(
                hpx::parallel::detail::inplace_merge<RandIter>().call(
                    HPX_FORWARD(ExPolicy, policy), first, middle, last,
                    HPX_MOVE(comp), hpx::identity_v));
        }

        template <typename RandIter,
            typename Comp = hpx::parallel::detail::less>
        // clang-format off
            requires (
                hpx::traits::is_iterator_v<RandIter> &&
                hpx::is_invocable_v<Comp,
                    typename std::iterator_traits<RandIter>::value_type,
                    typename std::iterator_traits<RandIter>::value_type
                >
            )
        // clang-format on
        friend void tag_fallback_invoke(inplace_merge_t, RandIter first,
            RandIter middle, RandIter last, Comp comp = Comp())
        {
            static_assert(hpx::traits::is_random_access_iterator_v<RandIter>,
                "Required at least random access iterator.");

            return hpx::parallel::detail::get_void_result(
                hpx::parallel::detail::inplace_merge<RandIter>().call(
                    hpx::execution::seq, first, middle, last, HPX_MOVE(comp),
                    hpx::identity_v));
        }
    } inplace_merge{};
}    // namespace hpx

#endif    // DOXYGEN
