//  Copyright (c) 2015-2023 Hartmut Kaiser
//  Copyright (c) 2021 Akhli J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/swap_ranges.hpp

#pragma once

#if defined(DOXYGEN)

namespace hpx { namespace ranges {
    // clang-format off

    ///////////////////////////////////////////////////////////////////////////
    /// Exchanges elements between range [first1, last1) and another range
    /// starting at \a first2.
    ///
    /// \note   Complexity: Linear in the distance between \a first1 and \a last1
    ///
    /// \tparam InIter1     The type of the first range of iterators to swap
    ///                     (deduced).
    /// \tparam Sent1       The type of the first sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter1.
    /// \tparam InIter2     The type of the second range of iterators to swap
    ///                     (deduced).
    /// \tparam Sent2       The type of the second sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter2.
    ///
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     for the first range.
    /// \param last1        Refers to sentinel value denoting the end of the
    ///                     sequence of elements for the first range.
    /// \param first2       Refers to the beginning of the sequence of elements
    ///                     for the second range.
    /// \param last2        Refers to sentinel value denoting the end of the
    ///                     sequence of elements for the second range.
    ///
    /// The swap operations in the parallel \a swap_ranges algorithm
    /// invoked without an execution policy object  execute in sequential
    /// order in the calling thread.
    ///
    /// \returns  The \a swap_ranges algorithm returns
    ///           \a swap_ranges_result<InIter1, InIter2>.
    ///           The \a swap_ranges algorithm returns in_in_result with the
    ///           first element as the iterator to the element past the last
    ///           element exchanged in range beginning with \a first1 and the
    ///           second element as the iterator to the element past the last
    ///           element exchanged in the range beginning with \a first2.
    ///
    template <typename InIter1, typename Sent1, typename InIter2,
        typename Sent2>
    swap_ranges_result<InIter1, InIter2>
    swap_ranges(InIter1 first1, Sent1 last1, InIter2 first2, Sent2 last2);

    ///////////////////////////////////////////////////////////////////////////
    /// Exchanges elements between range [first1, last1) and another range
    /// starting at \a first2.
    ///
    /// \note   Complexity: Linear in the distance between \a first1 and \a last1
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the first range of iterators to swap
    ///                     (deduced).
    /// \tparam Sent1       The type of the first sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter1.
    /// \tparam FwdIter2    The type of the second range of iterators to swap
    ///                     (deduced).
    /// \tparam Sent2       The type of the second sentinel (deduced). This
    ///                     sentinel type must be a sentinel for FwdIter2.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the sequence of elements
    ///                     for the first range.
    /// \param last1        Refers to sentinel value denoting the end of the
    ///                     sequence of elements for the first range.
    /// \param first2       Refers to the beginning of the sequence of elements
    ///                     for the second range.
    /// \param last2        Refers to sentinel value denoting the end of the
    ///                     sequence of elements for the second range.
    ///
    /// The swap operations in the parallel \a swap_ranges algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in
    /// the calling thread.
    ///
    /// The swap operations in the parallel \a swap_ranges algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a swap_ranges algorithm returns a
    ///           \a hpx::future<swap_ranges_result<FwdIter1, FwdIter2>>
    ///           if the execution policy is of type \a parallel_task_policy
    ///           and returns \a FwdIter2 otherwise.
    ///           The \a swap_ranges algorithm returns in_in_result with the
    ///           first element as the iterator to the element past the last
    ///           element exchanged in range beginning with \a first1 and the
    ///           second element as the iterator to the element past the last
    ///           element exchanged in the range beginning with \a first2.
    ///
    template <typename ExPolicy, typename FwdIter1, typename Sent1,
        typename FwdIter2, typename Sent2>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        swap_ranges_result<FwdIter1, FwdIter2>>::type
    swap_ranges(ExPolicy&& policy, FwdIter1 first1, Sent1 last1,
        FwdIter2 first2, Sent2 last2);

    ///////////////////////////////////////////////////////////////////////////
    /// Exchanges elements between range [first1, last1) and another range
    /// starting at \a first2.
    ///
    /// \note   Complexity: Linear in the distance between \a first1 and \a last1
    ///
    /// \tparam Rng1        The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Rng2        The type of the destination range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    ///
    /// \param rng1         Refers to the sequence of elements of the first
    ///                     range.
    /// \param rng2         Refers to the sequence of elements of the second
    ///                     range.
    ///
    /// The swap operations in the parallel \a swap_ranges algorithm
    /// invoked without an execution policy object  execute in sequential
    /// order in the calling thread.
    ///
    /// \returns  The \a swap_ranges algorithm returns
    ///           \a swap_ranges_result<
    ///           hpx::traits::range_iterator_t<Rng1>,
    ///           hpx::traits::range_iterator_t<Rng1>>.
    ///           The \a swap_ranges algorithm returns in_in_result with the
    ///           first element as the iterator to the element past the last
    ///           element exchanged in range beginning with \a first1 and the
    ///           second element as the iterator to the element past the last
    ///           element exchanged in the range beginning with \a first2.
    ///
    template <typename Rng1, typename Rng2>
    swap_ranges_result<hpx::traits::range_iterator_t<Rng1>,
        hpx::traits::range_iterator_t<Rng2>>
    swap_ranges(Rng1&& rng1, Rng2&& rng2);

    ///////////////////////////////////////////////////////////////////////////
    /// Exchanges elements between range [first1, last1) and another range
    /// starting at \a first2.
    ///
    /// \note   Complexity: Linear in the distance between \a first1 and \a last1
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
    ///                     meet the requirements of an input iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng1         Refers to the sequence of elements of the first
    ///                     range.
    /// \param rng2         Refers to the sequence of elements of the second
    ///                     range.
    ///
    /// The swap operations in the parallel \a swap_ranges algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in
    /// the calling thread.
    ///
    /// The swap operations in the parallel \a swap_ranges algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a swap_ranges algorithm returns a
    ///           \a hpx::future<swap_ranges_result<
    ///           hpx::traits::range_iterator_t<Rng1>,
    ///           hpx::traits::range_iterator_t<Rng1>>>
    ///           if the execution policy is of type \a parallel_task_policy
    ///           and returns \a swap_ranges_result<
    ///           hpx::traits::range_iterator_t<Rng1>,
    ///           hpx::traits::range_iterator_t<Rng1>>.
    ///           otherwise.
    ///           The \a swap_ranges algorithm returns in_in_result with the
    ///           first element as the iterator to the element past the last
    ///           element exchanged in range beginning with \a first1 and the
    ///           second element as the iterator to the element past the last
    ///           element exchanged in the range beginning with \a first2.
    ///
    template <typename ExPolicy, typename Rng1, typename Rng2>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        swap_ranges_result<hpx::traits::range_iterator_t<Rng1>,
            hpx::traits::range_iterator_t<Rng2>>>
    swap_ranges(ExPolicy&& policy, Rng1&& rng1, Rng2&& rng2);

    // clang-format on
}}    // namespace hpx::ranges

#else

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/parallel/algorithms/swap_ranges.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <type_traits>
#include <utility>

namespace hpx::ranges {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter1, typename Iter2>
    using swap_ranges_result = hpx::parallel::util::in_in_result<Iter1, Iter2>;

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::swap_ranges
    inline constexpr struct swap_ranges_t final
      : hpx::detail::tag_parallel_algorithm<swap_ranges_t>
    {
    private:
        // clang-format off
        template <typename InIter1, typename Sent1, typename InIter2,
            typename Sent2,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<InIter1> &&
                hpx::traits::is_sentinel_for_v<Sent1, InIter1> &&
                hpx::traits::is_iterator_v<InIter2> &&
                hpx::traits::is_sentinel_for_v<Sent2, InIter2>
            )>
        // clang-format on
        friend swap_ranges_result<InIter1, InIter2> tag_fallback_invoke(
            hpx::ranges::swap_ranges_t, InIter1 first1, Sent1 last1,
            InIter2 first2, Sent2 last2)
        {
            static_assert(hpx::traits::is_input_iterator_v<InIter1>,
                "Requires at least input iterator.");
            static_assert(hpx::traits::is_input_iterator_v<InIter2>,
                "Requires at least input iterator.");

            return hpx::parallel::detail::swap_ranges<
                swap_ranges_result<InIter1, InIter2>>()
                .call(hpx::execution::seq, first1, last1, first2, last2);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename Sent1,
            typename FwdIter2, typename Sent2,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<FwdIter1> &&
                hpx::traits::is_sentinel_for_v<Sent1, FwdIter1> &&
                hpx::traits::is_iterator_v<FwdIter2> &&
                hpx::traits::is_sentinel_for_v<Sent2, FwdIter2>
            )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy,
            swap_ranges_result<FwdIter1, FwdIter2>>
        tag_fallback_invoke(hpx::ranges::swap_ranges_t, ExPolicy&& policy,
            FwdIter1 first1, Sent1 last1, FwdIter2 first2, Sent2 last2)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter1>,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter2>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::swap_ranges<
                swap_ranges_result<FwdIter1, FwdIter2>>()
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
        friend swap_ranges_result<hpx::traits::range_iterator_t<Rng1>,
            hpx::traits::range_iterator_t<Rng2>>
        tag_fallback_invoke(
            hpx::ranges::swap_ranges_t, Rng1&& rng1, Rng2&& rng2)
        {
            using iterator_type1 = hpx::traits::range_iterator_t<Rng1>;
            using iterator_type2 = hpx::traits::range_iterator_t<Rng2>;

            static_assert(hpx::traits::is_input_iterator_v<iterator_type1>,
                "Requires at least input iterator.");
            static_assert(hpx::traits::is_input_iterator_v<iterator_type2>,
                "Requires at least input iterator.");

            return hpx::parallel::detail::swap_ranges<
                swap_ranges_result<iterator_type1, iterator_type2>>()
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
            swap_ranges_result<hpx::traits::range_iterator_t<Rng1>,
                hpx::traits::range_iterator_t<Rng2>>>
        tag_fallback_invoke(hpx::ranges::swap_ranges_t, ExPolicy&& policy,
            Rng1&& rng1, Rng2&& rng2)
        {
            using iterator_type1 = hpx::traits::range_iterator_t<Rng1>;
            using iterator_type2 = hpx::traits::range_iterator_t<Rng2>;

            static_assert(hpx::traits::is_forward_iterator_v<iterator_type1>,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_forward_iterator_v<iterator_type2>,
                "Requires at least forward iterator.");

            return hpx::parallel::detail::swap_ranges<
                swap_ranges_result<iterator_type1, iterator_type2>>()
                .call(HPX_FORWARD(ExPolicy, policy), std::begin(rng1),
                    std::end(rng1), std::begin(rng2), std::end(rng2));
        }
    } swap_ranges{};
}    // namespace hpx::ranges

#endif
