//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/uninitialized_copy.hpp

#pragma once

#if defined(DOXYGEN)

namespace hpx { namespace ranges {

    /// Copies the elements in the range, defined by [first, last), to an
    /// uninitialized memory area beginning at \a dest. If an exception is
    /// thrown during the copy operation, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Sent1       The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter.
    /// \tparam FwdIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Sent2       The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter2.
    ///
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     that will be copied from
    /// \param last1        Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied
    /// \param first2       Refers to the beginning of the destination range.
    /// \param last2        Refers to sentinel value denoting the end of the
    ///                     second range the algorithm will be applied to.
    ///
    /// The assignments in the parallel \a uninitialized_copy algorithm invoked
    /// without an execution policy object will execute in sequential order in
    /// the calling thread.
    ///
    /// \returns  The \a uninitialized_copy algorithm returns an
    ///           \a in_out_result<InIter, FwdIter>.
    ///           The \a uninitialized_copy algorithm returns an input iterator
    ///           to one past the last element copied from and the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    template <typename InIter, typename Sent1, typename FwdIter, typename Sent2>
    hpx::parallel::util::in_out_result<InIter, FwdIter> uninitialized_copy(
        InIter first1, Sent1 last1, FwdIter first2, Sent2 last2);

    /// Copies the elements in the range, defined by [first, last), to an
    /// uninitialized memory area beginning at \a dest. If an exception is
    /// thrown during the copy operation, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Sent1       The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Sent2       The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter2.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     that will be copied from
    /// \param last1        Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param first2       Refers to the beginning of the destination range.
    /// \param last2        Refers to sentinel value denoting the end of the
    ///                     second range the algorithm will be applied to.
    ///
    /// The assignments in the parallel \a uninitialized_copy algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_copy algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_copy algorithm returns a
    ///           \a hpx::future<in_out_result<InIter, FwdIter>>, if the
    ///           execution policy is of type \a sequenced_task_policy
    ///           or \a parallel_task_policy and
    ///           returns \a in_out_result<InIter, FwdIter> otherwise.
    ///           The \a uninitialized_copy algorithm returns an input iterator
    ///           to one past the last element copied from and the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename Sent1,
        typename FwdIter2, typename Sent2>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        parallel::util::in_out_result<FwdIter1, FwdIter2>>::type
    uninitialized_copy(ExPolicy&& policy, FwdIter1 first1, Sent1 last1,
        FwdIter2 first2, Sent2 last2);

    /// Copies the elements in the range, defined by [first, last), to an
    /// uninitialized memory area beginning at \a dest. If an exception is
    /// thrown during the copy operation, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam Rng1        The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Rng2        The type of the destination range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    ///
    /// \param rng1         Refers to the range from which the elements
    ///                     will be copied from
    /// \param rng2         Refers to the range to which the elements
    ///                     will be copied to
    ///
    /// The assignments in the parallel \a uninitialized_copy algorithm invoked
    /// without an execution policy object will execute in sequential order in
    /// the calling thread.
    ///
    /// \returns  The \a uninitialized_copy algorithm returns an \a
    ///           in_out_result<typename hpx::traits::range_traits<Rng1>::iterator_type,
    ///           typename hpx::traits::range_traits<Rng2>::iterator_type>.
    ///           The \a uninitialized_copy algorithm returns an input iterator
    ///           to one past the last element copied from and the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    template <typename Rng1, typename Rng2>
    hpx::parallel::util::in_out_result<
        typename hpx::traits::range_traits<Rng1>::iterator_type,
        typename hpx::traits::range_traits<Rng2>::iterator_type>
    uninitialized_copy(Rng1&& rng1, Rng2&& rng2);

    /// Copies the elements in the range, defined by [first, last), to an
    /// uninitialized memory area beginning at \a dest. If an exception is
    /// thrown during the copy operation, the function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng1        The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Rng2        The type of the destination range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng1         Refers to the range from which the elements
    ///                     will be copied from
    /// \param rng2         Refers to the range to which the elements
    ///                     will be copied to
    ///
    /// The assignments in the parallel \a uninitialized_copy algorithm invoked
    /// with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_copy algorithm invoked
    /// with an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_copy algorithm returns a
    ///           \a hpx::future<in_out_result<InIter, FwdIter>>, if the
    ///           execution policy is of type \a sequenced_task_policy
    ///           or \a parallel_task_policy and
    ///           returns \a in_out_result<
    ///             typename hpx::traits::range_traits<Rng1>::iterator_type
    ///           , typename hpx::traits::range_traits<Rng2>::iterator_type>
    ///           otherwise. The \a uninitialized_copy algorithm returns the
    ///           input iterator to one past the last element copied from and
    ///           the output iterator to the element in the destination range,
    ///           one past the last element copied.
    ///
    template <typename ExPolicy, typename Rng1, typename Rng2>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        hpx::parallel::util::in_out_result<
            typename hpx::traits::range_traits<Rng1>::iterator_type,
            typename hpx::traits::range_traits<Rng2>::iterator_type>>::type
    uninitialized_copy(ExPolicy&& policy, Rng1&& rng1, Rng2&& rng2);

    /// Copies the elements in the range [first, first + count), starting from
    /// first and proceeding to first + count - 1., to another range beginning
    /// at dest. If an exception is thrown during the copy operation, the
    /// function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam InIter      The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam FwdIter     The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Sent2       The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter.
    ///
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     that will be copied from
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the destination range.
    /// \param last2        Refers to sentinel value denoting the end of the
    ///                     second range the algorithm will be applied to.
    ///
    /// The assignments in the parallel \a uninitialized_copy_n algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// \returns  The \a uninitialized_copy_n algorithm returns
    ///           \a in_out_result<InIter, FwdIter>.
    ///           The \a uninitialized_copy_n algorithm returns the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    template <typename InIter, typename Size, typename FwdIter, typename Sent2>
    hpx::parallel::util::in_out_result<InIter, FwdIter> uninitialized_copy_n(
        InIter first1, Size count, FwdIter first2, Sent2 last2);

    /// Copies the elements in the range [first, first + count), starting from
    /// first and proceeding to first + count - 1., to another range beginning
    /// at dest. If an exception is thrown during the copy operation, the
    /// function has no effects.
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     input iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam FwdIter2    The type of the iterator representing the
    ///                     destination range (deduced).
    ///                     This iterator type must meet the requirements of a
    ///                     forward iterator.
    /// \tparam Sent2       The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter2.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     that will be copied from
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    /// \param first2       Refers to the beginning of the destination range.
    /// \param last2        Refers to sentinel value denoting the end of the
    ///                     second range the algorithm will be applied to.
    ///
    /// The assignments in the parallel \a uninitialized_copy_n algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The assignments in the parallel \a uninitialized_copy_n algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a uninitialized_copy_n algorithm returns a
    ///           \a hpx::future<in_out_result<FwdIter1, FwdIter2>> if the
    ///           execution policy is of type \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter2 otherwise.
    ///           The \a uninitialized_copy_n algorithm returns the output
    ///           iterator to the element in the destination range, one past
    ///           the last element copied.
    ///
    template <typename ExPolicy, typename FwdIter1, typename Size,
        typename FwdIter2, typename Sent2>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        parallel::util::in_out_result<FwdIter1, FwdIter2>>::type
    uninitialized_copy_n(ExPolicy&& policy, FwdIter1 first1, Size count,
        FwdIter2 first2, Sent2 last2);
}}    // namespace hpx::ranges
#else

#include <hpx/config.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/algorithms/uninitialized_copy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx::ranges {
    inline constexpr struct uninitialized_copy_t final
      : hpx::detail::tag_parallel_algorithm<uninitialized_copy_t>
    {
    private:
        // clang-format off
        template <typename InIter, typename Sent1, typename FwdIter, typename Sent2,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<InIter> &&
                hpx::traits::is_sentinel_for_v<Sent1, InIter> &&
                hpx::traits::is_forward_iterator_v<FwdIter> &&
                hpx::traits::is_sentinel_for_v<Sent2, FwdIter>
            )>
        // clang-format on
        friend hpx::parallel::util::in_out_result<InIter, FwdIter>
        tag_fallback_invoke(hpx::ranges::uninitialized_copy_t, InIter first1,
            Sent1 last1, FwdIter first2, Sent2 last2)
        {
            static_assert(hpx::traits::is_input_iterator_v<InIter>,
                "Requires at least input iterator.");
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::uninitialized_copy_sent<
                parallel::util::in_out_result<InIter, FwdIter>>()
                .call(hpx::execution::seq, first1, last1, first2, last2);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename Sent1,
            typename FwdIter2, typename Sent2,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_forward_iterator_v<FwdIter1> &&
                hpx::traits::is_sentinel_for_v<Sent1, FwdIter1> &&
                hpx::traits::is_forward_iterator_v<FwdIter2> &&
                hpx::traits::is_sentinel_for_v<Sent2, FwdIter2>
            )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy,
            parallel::util::in_out_result<FwdIter1, FwdIter2>>
        tag_fallback_invoke(hpx::ranges::uninitialized_copy_t,
            ExPolicy&& policy, FwdIter1 first1, Sent1 last1, FwdIter2 first2,
            Sent2 last2)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter1>,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter2>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::uninitialized_copy_sent<
                parallel::util::in_out_result<FwdIter1, FwdIter2>>()
                .call(HPX_FORWARD(ExPolicy, policy), first1, last1, first2,
                    last2);
        }

        // clang-format off
        template <typename Rng1, typename Rng2,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range_v<Rng1> &&
                hpx::traits::is_range_v<Rng2>
            )>
        // clang-format on
        friend hpx::parallel::util::in_out_result<
            typename hpx::traits::range_traits<Rng1>::iterator_type,
            typename hpx::traits::range_traits<Rng2>::iterator_type>
        tag_fallback_invoke(
            hpx::ranges::uninitialized_copy_t, Rng1&& rng1, Rng2&& rng2)
        {
            using iterator_type1 =
                typename hpx::traits::range_traits<Rng1>::iterator_type;
            using iterator_type2 =
                typename hpx::traits::range_traits<Rng2>::iterator_type;

            static_assert(hpx::traits::is_input_iterator_v<iterator_type1>,
                "Requires at least input iterator.");

            static_assert(hpx::traits::is_forward_iterator_v<iterator_type2>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::uninitialized_copy_sent<
                parallel::util::in_out_result<iterator_type1, iterator_type2>>()
                .call(hpx::execution::seq, std::begin(rng1), std::end(rng1),
                    std::begin(rng2), std::end(rng2));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng1, typename Rng2,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_range_v<Rng1> &&
                hpx::traits::is_range_v<Rng2>
            )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy,
            hpx::parallel::util::in_out_result<
                typename hpx::traits::range_traits<Rng1>::iterator_type,
                typename hpx::traits::range_traits<Rng2>::iterator_type>>
        tag_fallback_invoke(hpx::ranges::uninitialized_copy_t,
            ExPolicy&& policy, Rng1&& rng1, Rng2&& rng2)
        {
            using iterator_type1 =
                typename hpx::traits::range_traits<Rng1>::iterator_type;
            using iterator_type2 =
                typename hpx::traits::range_traits<Rng2>::iterator_type;

            static_assert(hpx::traits::is_forward_iterator_v<iterator_type1>,
                "Requires at least forward iterator.");

            static_assert(hpx::traits::is_forward_iterator_v<iterator_type2>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::uninitialized_copy_sent<
                parallel::util::in_out_result<iterator_type1, iterator_type2>>()
                .call(HPX_FORWARD(ExPolicy, policy), std::begin(rng1),
                    std::end(rng1), std::begin(rng2), std::end(rng2));
        }
    } uninitialized_copy{};

    inline constexpr struct uninitialized_copy_n_t final
      : hpx::detail::tag_parallel_algorithm<uninitialized_copy_n_t>
    {
    private:
        // clang-format off
        template <typename InIter, typename Size, typename FwdIter, typename Sent2,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<InIter> &&
                hpx::traits::is_forward_iterator_v<FwdIter> &&
                hpx::traits::is_sentinel_for_v<Sent2, FwdIter> &&
                std::is_integral_v<Size>
            )>
        // clang-format on
        friend hpx::parallel::util::in_out_result<InIter, FwdIter>
        tag_fallback_invoke(hpx::ranges::uninitialized_copy_n_t, InIter first1,
            Size count, FwdIter first2, Sent2 last2)
        {
            static_assert(hpx::traits::is_input_iterator_v<InIter>,
                "Requires at least input iterator.");
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            std::size_t d = parallel::detail::distance(first2, last2);
            return hpx::parallel::detail::uninitialized_copy_n<
                parallel::util::in_out_result<InIter, FwdIter>>()
                .call(hpx::execution::seq, first1, count <= d ? count : d,
                    first2);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename Size,
            typename FwdIter2, typename Sent2,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_forward_iterator_v<FwdIter1> &&
                hpx::traits::is_forward_iterator_v<FwdIter2> &&
                hpx::traits::is_sentinel_for_v<Sent2, FwdIter2> &&
                std::is_integral_v<Size>
            )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy,
            hpx::parallel::util::in_out_result<FwdIter1, FwdIter2>>
        tag_fallback_invoke(hpx::ranges::uninitialized_copy_n_t,
            ExPolicy&& policy, FwdIter1 first1, Size count, FwdIter2 first2,
            Sent2 last2)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter1>,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter2>,
                "Requires at least forward iterator.");

            std::size_t d = parallel::detail::distance(first2, last2);
            return hpx::parallel::detail::uninitialized_copy_n<
                parallel::util::in_out_result<FwdIter1, FwdIter2>>()
                .call(HPX_FORWARD(ExPolicy, policy), first1,
                    count <= d ? count : d, first2);
        }
    } uninitialized_copy_n{};
}    // namespace hpx::ranges

#endif
