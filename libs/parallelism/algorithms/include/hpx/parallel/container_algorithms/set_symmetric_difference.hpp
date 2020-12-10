//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/set_symmetric_difference.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx { namespace ranges {
    // clang-format off

    /// Constructs a sorted range beginning at dest consisting of all elements
    /// present in either of the sorted ranges [first1, last1) and
    /// [first2, last2), but not in both of them are copied to the range
    /// beginning at \a dest. The resulting range is also sorted. This
    /// algorithm expects both input ranges to be sorted with the given binary
    /// predicate \a f.
    ///
    /// \note   Complexity: At most 2*(N1 + N2 - 1) comparisons, where \a N1 is
    ///         the length of the first sequence and \a N2 is the length of the
    ///         second sequence.
    ///
    /// If some element is found \a m times in [first1, last1) and \a n times
    /// in [first2, last2), it will be copied to \a dest exactly std::abs(m-n)
    /// times. If m>n, then the last m-n of those elements are copied from
    /// [first1,last1), otherwise the last n-m elements are copied from
    /// [first2,last2). The resulting range cannot overlap with either of the
    /// input ranges.
    ///
    /// The resulting range cannot overlap with either of the input ranges.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam Iter1       The type of the source iterators used (deduced)
    ///                     representing the first sequence.
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent1       The type of the end source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter1.
    /// \tparam Iter2       The type of the source iterators used (deduced)
    ///                     representing the second sequence.
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Sent2       The type of the end source iterators used (deduced)
    ///                     representing the second sequence.
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter2.
    /// \tparam Iter3       The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a set_symmetric_difference requires \a Pred to meet
    ///                     the requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    /// \tparam Proj1       The type of an optional projection function applied
    ///                     to the first sequence. This
    ///                     defaults to \a util::projection_identity
    /// \tparam Proj2       The type of an optional projection function applied
    ///                     to the second sequence. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     of the first range the algorithm will be applied to.
    /// \param last1        Refers to the end of the sequence of elements of
    ///                     the first range the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the sequence of elements
    ///                     of the second range the algorithm will be applied to.
    /// \param last2        Refers to the end of the sequence of elements of
    ///                     the second range the algorithm will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param op           The binary predicate which returns true if the
    ///                     elements should be treated as equal. The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such
    ///                     that objects of type \a InIter can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type1
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     first sequence as a projection operation before the
    ///                     actual predicate \a op is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     second sequence as a projection operation before the
    ///                     actual predicate \a op is invoked.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with a sequential execution policy object execute in sequential
    /// order in the calling thread (\a sequenced_policy) or in a
    /// single new thread spawned from the current thread
    /// (for \a sequenced_task_policy).
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a set_symmetric_difference algorithm returns a
    ///           \a hpx::future<ranges::set_symmetric_difference_result<Iter1, Iter2, Iter3>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a ranges::set_symmetric_difference_result<Iter1, Iter2, Iter3> otherwise.
    ///           The \a set_symmetric_difference algorithm returns the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename Iter1, typename Sent1, typename Iter2,
        typename Sent2, typename Iter3, typename Pred = detail::less,
        typename Proj1 = util::projection_identity,
        typename Proj2 = util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy,
        ranges::set_symmetric_difference_result<Iter1, Iter2, Iter3>>::type
    set_symmetric_difference(ExPolicy&& policy, Iter1 first1, Sent1 last1,
        Iter2 first2, Sent2 last2, Iter3 dest, Pred&& op = Pred(),
        Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2());

    /// Constructs a sorted range beginning at dest consisting of all elements
    /// present in either of the sorted ranges [first1, last1) and
    /// [first2, last2), but not in both of them are copied to the range
    /// beginning at \a dest. The resulting range is also sorted. This
    /// algorithm expects both input ranges to be sorted with the given binary
    /// predicate \a f.
    ///
    /// \note   Complexity: At most 2*(N1 + N2 - 1) comparisons, where \a N1 is
    ///         the length of the first sequence and \a N2 is the length of the
    ///         second sequence.
    ///
    /// If some element is found \a m times in [first1, last1) and \a n times
    /// in [first2, last2), it will be copied to \a dest exactly std::abs(m-n)
    /// times. If m>n, then the last m-n of those elements are copied from
    /// [first1,last1), otherwise the last n-m elements are copied from
    /// [first2,last2). The resulting range cannot overlap with either of the
    /// input ranges.
    ///
    /// The resulting range cannot overlap with either of the input ranges.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam Rng1        The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Rng2        The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Iter3       The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     output iterator.
    /// \tparam Pred        The type of an optional function/function object to use.
    ///                     Unlike its sequential form, the parallel
    ///                     overload of \a set_symmetric_difference requires \a Pred to meet
    ///                     the requirements of \a CopyConstructible. This defaults
    ///                     to std::less<>
    /// \tparam Proj1       The type of an optional projection function applied
    ///                     to the first sequence. This
    ///                     defaults to \a util::projection_identity
    /// \tparam Proj2       The type of an optional projection function applied
    ///                     to the second sequence. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng1         Refers to the first sequence of elements the algorithm
    ///                     will be applied to.
    /// \param rng2         Refers to the second sequence of elements the algorithm
    ///                     will be applied to.
    /// \param dest         Refers to the beginning of the destination range.
    /// \param op           The binary predicate which returns true if the
    ///                     elements should be treated as equal. The signature
    ///                     of the predicate function should be equivalent to
    ///                     the following:
    ///                     \code
    ///                     bool pred(const Type1 &a, const Type1 &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type1 must be such
    ///                     that objects of type \a InIter can
    ///                     be dereferenced and then implicitly converted to
    ///                     \a Type1
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     first sequence as a projection operation before the
    ///                     actual predicate \a op is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements of the
    ///                     second sequence as a projection operation before the
    ///                     actual predicate \a op is invoked.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with a sequential execution policy object execute in sequential
    /// order in the calling thread (\a sequenced_policy) or in a
    /// single new thread spawned from the current thread
    /// (for \a sequenced_task_policy).
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a set_symmetric_difference algorithm returns a
    ///           \a hpx::future<ranges::set_symmetric_difference_result<Iter1, Iter2, Iter3>>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a ranges::set_symmetric_difference_result<Iter1, Iter2, Iter3> otherwise.
    ///           where Iter1 is range_iterator_t<Rng1> and Iter2 is range_iterator_t<Rng2>
    ///           The \a set_symmetric_difference algorithm returns the output iterator to the
    ///           element in the destination range, one past the last element
    ///           copied.
    ///
    template <typename ExPolicy, typename Rng1, typename Rng2, typename Iter3,
        typename Pred = detail::less,
        typename Proj1 = util::projection_identity,
        typename Proj2 = util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy,
        ranges::set_symmetric_difference_result<
            typename traits::range_iterator<Rng1>::type,
            typename traits::range_iterator<Rng2>::type, Iter3>>::type
    set_symmetric_difference(ExPolicy&& policy, Rng1&& rng1, Rng2&& rng2,
        Iter3 dest, Pred&& op = Pred(), Proj1&& proj1 = Proj1(),
        Proj2&& proj2 = Proj2());

    // clang-format on
}}    // namespace hpx::ranges

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_sentinel_for.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/set_symmetric_difference.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace ranges {

    template <typename I1, typename I2, typename O>
    using set_symmetric_difference_result =
        parallel::util::in_in_out_result<I1, I2, O>;

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::set_symmetric_difference
    HPX_INLINE_CONSTEXPR_VARIABLE struct set_symmetric_difference_t final
      : hpx::functional::tag<set_symmetric_difference_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename Iter1, typename Sent1,
            typename Iter2, typename Sent2, typename Iter3,
            typename Pred = hpx::parallel::v1::detail::less,
            typename Proj1 = hpx::parallel::util::projection_identity,
            typename Proj2 = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_sentinel_for<Sent1, Iter1>::value &&
                hpx::parallel::traits::is_projected<Proj1, Iter1>::value &&
                hpx::traits::is_sentinel_for<Sent2, Iter2>::value &&
                hpx::parallel::traits::is_projected<Proj2, Iter2>::value &&
                hpx::traits::is_iterator<Iter3>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy, Pred,
                    hpx::parallel::traits::projected<Proj1, Iter1>,
                    hpx::parallel::traits::projected<Proj2, Iter2>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            set_symmetric_difference_result<Iter1, Iter2, Iter3>>::type
        tag_invoke(set_symmetric_difference_t, ExPolicy&& policy, Iter1 first1,
            Sent1 last1, Iter2 first2, Sent2 last2, Iter3 dest,
            Pred&& op = Pred(), Proj1&& proj1 = Proj1(),
            Proj2&& proj2 = Proj2())
        {
            static_assert((hpx::traits::is_forward_iterator<Iter1>::value),
                "Requires at least forward iterator.");
            static_assert((hpx::traits::is_forward_iterator<Iter2>::value),
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator<Iter3>::value ||
                    (hpx::is_sequenced_execution_policy<ExPolicy>::value &&
                        hpx::traits::is_output_iterator<Iter3>::value),
                "Requires at least forward iterator or sequential execution.");

            using is_seq = std::integral_constant<bool,
                hpx::is_sequenced_execution_policy<ExPolicy>::value ||
                    !hpx::traits::is_random_access_iterator<Iter1>::value ||
                    !hpx::traits::is_random_access_iterator<Iter2>::value>;

            using result_type =
                set_symmetric_difference_result<Iter1, Iter2, Iter3>;

            return hpx::parallel::v1::detail::set_symmetric_difference<
                result_type>()
                .call(std::forward<ExPolicy>(policy), is_seq(), first1, last1,
                    first2, last2, dest, std::forward<Pred>(op),
                    std::forward<Proj1>(proj1), std::forward<Proj2>(proj2));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng1, typename Rng2, typename Iter3,
            typename Pred = hpx::parallel::v1::detail::less,
            typename Proj1 = hpx::parallel::util::projection_identity,
            typename Proj2 = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng1>::value &&
                hpx::parallel::traits::is_projected_range<Proj1, Rng1>::value &&
                hpx::traits::is_range<Rng2>::value &&
                hpx::parallel::traits::is_projected_range<Proj2, Rng2>::value &&
                hpx::traits::is_iterator<Iter3>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy, Pred,
                    hpx::parallel::traits::projected_range<Proj1, Rng1>,
                    hpx::parallel::traits::projected_range<Proj2, Rng2>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            set_symmetric_difference_result<
                typename hpx::traits::range_iterator<Rng1>::type,
                typename hpx::traits::range_iterator<Rng2>::type, Iter3>>::type
        tag_invoke(set_symmetric_difference_t, ExPolicy&& policy, Rng1&& rng1,
            Rng2&& rng2, Iter3 dest, Pred&& op = Pred(),
            Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2())
        {
            using iterator_type1 =
                typename hpx::traits::range_iterator<Rng1>::type;
            using iterator_type2 =
                typename hpx::traits::range_iterator<Rng2>::type;

            static_assert(
                (hpx::traits::is_forward_iterator<iterator_type1>::value),
                "Requires at least forward iterator.");
            static_assert(
                (hpx::traits::is_forward_iterator<iterator_type2>::value),
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator<Iter3>::value ||
                    (hpx::is_sequenced_execution_policy<ExPolicy>::value &&
                        hpx::traits::is_output_iterator<Iter3>::value),
                "Requires at least forward iterator or sequential execution.");

            using is_seq = std::integral_constant<bool,
                hpx::is_sequenced_execution_policy<ExPolicy>::value ||
                    !hpx::traits::is_random_access_iterator<
                        iterator_type1>::value ||
                    !hpx::traits::is_random_access_iterator<
                        iterator_type2>::value>;

            using result_type = set_symmetric_difference_result<iterator_type1,
                iterator_type2, Iter3>;

            return hpx::parallel::v1::detail::set_symmetric_difference<
                result_type>()
                .call(std::forward<ExPolicy>(policy), is_seq(),
                    hpx::util::begin(rng1), hpx::util::end(rng1),
                    hpx::util::begin(rng2), hpx::util::end(rng2), dest,
                    std::forward<Pred>(op), std::forward<Proj1>(proj1),
                    std::forward<Proj2>(proj2));
        }

        // clang-format off
        template <typename Iter1, typename Sent1, typename Iter2, typename Sent2,
            typename Iter3, typename Pred = hpx::parallel::v1::detail::less,
            typename Proj1 = hpx::parallel::util::projection_identity,
            typename Proj2 = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_sentinel_for<Sent1, Iter1>::value &&
                hpx::parallel::traits::is_projected<Proj1, Iter1>::value &&
                hpx::traits::is_sentinel_for<Sent2, Iter2>::value &&
                hpx::parallel::traits::is_projected<Proj2, Iter2>::value &&
                hpx::traits::is_iterator<Iter3>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected<Proj1, Iter1>,
                    hpx::parallel::traits::projected<Proj2, Iter2>
                >::value
            )>
        // clang-format on
        friend set_symmetric_difference_result<Iter1, Iter2, Iter3> tag_invoke(
            set_symmetric_difference_t, Iter1 first1, Sent1 last1, Iter2 first2,
            Sent2 last2, Iter3 dest, Pred&& op = Pred(),
            Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2())
        {
            static_assert((hpx::traits::is_input_iterator<Iter1>::value),
                "Requires at least input iterator.");
            static_assert((hpx::traits::is_input_iterator<Iter2>::value),
                "Requires at least input iterator.");
            static_assert((hpx::traits::is_output_iterator<Iter3>::value),
                "Requires at least output iterator.");

            using result_type =
                set_symmetric_difference_result<Iter1, Iter2, Iter3>;

            return hpx::parallel::v1::detail::set_symmetric_difference<
                result_type>()
                .call(hpx::execution::seq, std::true_type(), first1, last1,
                    first2, last2, dest, std::forward<Pred>(op),
                    std::forward<Proj1>(proj1), std::forward<Proj2>(proj2));
        }

        // clang-format off
        template <typename Rng1, typename Rng2, typename Iter3,
            typename Pred = hpx::parallel::v1::detail::less,
            typename Proj1 = hpx::parallel::util::projection_identity,
            typename Proj2 = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng1>::value &&
                hpx::parallel::traits::is_projected_range<Proj1, Rng1>::value &&
                hpx::traits::is_range<Rng2>::value &&
                hpx::parallel::traits::is_projected_range<Proj2, Rng2>::value &&
                hpx::traits::is_iterator<Iter3>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected_range<Proj1, Rng1>,
                    hpx::parallel::traits::projected_range<Proj2, Rng2>
                >::value
            )>
        // clang-format on
        friend set_symmetric_difference_result<
            typename hpx::traits::range_iterator<Rng1>::type,
            typename hpx::traits::range_iterator<Rng2>::type, Iter3>
        tag_invoke(set_symmetric_difference_t, Rng1&& rng1, Rng2&& rng2,
            Iter3 dest, Pred&& op = Pred(), Proj1&& proj1 = Proj1(),
            Proj2&& proj2 = Proj2())
        {
            using iterator_type1 =
                typename hpx::traits::range_iterator<Rng1>::type;
            using iterator_type2 =
                typename hpx::traits::range_iterator<Rng2>::type;

            static_assert(
                (hpx::traits::is_input_iterator<iterator_type1>::value),
                "Requires at least input iterator.");
            static_assert(
                (hpx::traits::is_input_iterator<iterator_type2>::value),
                "Requires at least input iterator.");
            static_assert((hpx::traits::is_output_iterator<Iter3>::value),
                "Requires at least output iterator.");

            using result_type = set_symmetric_difference_result<iterator_type1,
                iterator_type2, Iter3>;

            return hpx::parallel::v1::detail::set_symmetric_difference<
                result_type>()
                .call(hpx::execution::seq, std::true_type(),
                    hpx::util::begin(rng1), hpx::util::end(rng1),
                    hpx::util::begin(rng2), hpx::util::end(rng2), dest,
                    std::forward<Pred>(op), std::forward<Proj1>(proj1),
                    std::forward<Proj2>(proj2));
        }
    } set_symmetric_difference{};

}}    // namespace hpx::ranges

#endif    // DOXYGEN
