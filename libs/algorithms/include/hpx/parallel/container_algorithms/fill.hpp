//  Copyright (c) 2018 Christopher Ogle
//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/fill.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx {
    // clang-format off

    /// Assigns the given value to the elements in the range [first, last).
    ///
    /// \note   Complexity: Performs exactly \a last - \a first assignments.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam T           The type of the value to be assigned (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param value        The value to be assigned.
    ///
    /// The comparisons in the parallel \a fill algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a fill algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a fill algorithm returns a \a hpx::future<void> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a difference_type otherwise (where \a difference_type
    ///           is defined by \a void.
    ///
    template <typename ExPolicy, typename Rng, typename T>
    typename util::detail::algorithm_result<ExPolicy>::type
    fill(ExPolicy&& policy, Rng&& rng, T const& value);

    /// Assigns the given value value to the first count elements in the range
    /// beginning at first if count > 0. Does nothing otherwise.
    ///
    /// \note   Complexity: Performs exactly \a count assignments, for
    ///         count > 0.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Iterator    The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply \a f to.
    /// \tparam T           The type of the value to be assigned (deduced).
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    /// \param value        The value to be assigned.
    ///
    /// The comparisons in the parallel \a fill_n algorithm invoked with
    /// an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The comparisons in the parallel \a fill_n algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a fill_n algorithm returns a \a hpx::future<void> if the
    ///           execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a difference_type otherwise (where \a difference_type
    ///           is defined by \a void.
    ///
    template <typename ExPolicy, typename Iterator, typename Size, typename T>
    typename util::detail::algorithm_result<ExPolicy, Iterator>::type
    fill_n(ExPolicy&& policy, Iterator first, Size count, T const& value);

    // clang-format on
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>

#include <hpx/parallel/algorithms/fill.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace v1 {

    // clang-format off
    template <typename ExPolicy, typename Rng, typename T,
        HPX_CONCEPT_REQUIRES_(
            execution::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_range<Rng>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::fill is deprecated, use hpx::ranges::fill instead")
        typename util::detail::algorithm_result<ExPolicy>::type
        fill(ExPolicy&& policy, Rng&& rng, T value)
    {
        return fill(std::forward<ExPolicy>(policy), hpx::util::begin(rng),
            hpx::util::end(rng), value);
    }

    // clang-format off
    template <typename ExPolicy, typename Rng, typename Size, typename T,
        HPX_CONCEPT_REQUIRES_(
            execution::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_range<Rng>::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 6,
        "hpx::parallel::fill_n is deprecated, use hpx::ranges::fill_n instead")
        typename util::detail::algorithm_result<ExPolicy,
            typename hpx::traits::range_traits<Rng>::iterator_type>::type
        fill_n(ExPolicy&& policy, Rng& rng, Size count, T value)
    {
        return fill_n(std::forward<ExPolicy>(policy), hpx::util::begin(rng),
            count, value);
    }

}}}    // namespace hpx::parallel::v1

namespace hpx { namespace ranges {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::fill
    HPX_INLINE_CONSTEXPR_VARIABLE struct fill_t final
      : hpx::functional::tag<fill_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename Rng, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            typename hpx::traits::range_traits<Rng>::iterator_type>::type
        tag_invoke(fill_t, ExPolicy&& policy, Rng&& rng, T const& value)
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            static_assert(
                (hpx::traits::is_forward_iterator<iterator_type>::value),
                "Requires at least forward iterator.");

            using is_segmented =
                hpx::traits::is_segmented_iterator<iterator_type>;

            return hpx::parallel::v1::detail::fill_(
                std::forward<ExPolicy>(policy), hpx::util::begin(rng),
                hpx::util::end(rng), value, is_segmented{});
        }

        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_sentinel_for<Sent, Iter>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            Iter>::type
        tag_invoke(
            fill_t, ExPolicy&& policy, Iter first, Sent last, T const& value)
        {
            static_assert((hpx::traits::is_forward_iterator<Iter>::value),
                "Requires at least forward iterator.");

            using is_segmented = hpx::traits::is_segmented_iterator<Iter>;

            return hpx::parallel::v1::detail::fill_(
                std::forward<ExPolicy>(policy), first, last, value,
                is_segmented{});
        }

        // clang-format off
        template <typename Rng, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend typename hpx::traits::range_traits<Rng>::iterator_type
        tag_invoke(fill_t, Rng&& rng, T const& value)
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            static_assert(
                (hpx::traits::is_forward_iterator<iterator_type>::value),
                "Requires at least forward iterator.");

            using is_segmented =
                hpx::traits::is_segmented_iterator<iterator_type>;

            return hpx::parallel::v1::detail::fill_(
                hpx::parallel::execution::seq, hpx::util::begin(rng),
                hpx::util::end(rng), value, is_segmented{});
        }

        // clang-format off
        template <typename Iter, typename Sent, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_sentinel_for<Sent, Iter>::value
            )>
        // clang-format on
        friend Iter tag_invoke(fill_t, Iter first, Sent last, T const& value)
        {
            static_assert((hpx::traits::is_forward_iterator<Iter>::value),
                "Requires at least forward iterator.");

            using is_segmented = hpx::traits::is_segmented_iterator<Iter>;

            return hpx::parallel::v1::detail::fill_(
                hpx::parallel::execution::seq, first, last, value,
                is_segmented{});
        }
    } fill{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::fill_n
    HPX_INLINE_CONSTEXPR_VARIABLE struct fill_n_t final
      : hpx::functional::tag<fill_n_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename Rng, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            typename hpx::traits::range_traits<Rng>::iterator_type>::type
        tag_invoke(fill_n_t, ExPolicy&& policy, Rng&& rng, T const& value)
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            static_assert(
                (hpx::traits::is_forward_iterator<iterator_type>::value),
                "Requires at least forward iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::v1::detail::is_negative(hpx::util::size(rng)))
            {
                auto first = hpx::util::begin(rng);
                return hpx::parallel::util::detail::algorithm_result<ExPolicy,
                    iterator_type>::get(std::move(first));
            }

            return hpx::parallel::v1::detail::fill_n<iterator_type>().call(
                std::forward<ExPolicy>(policy), is_seq{}, hpx::util::begin(rng),
                hpx::util::size(rng), value);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Size, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_invoke(fill_n_t, ExPolicy&& policy, FwdIter first, Size count,
            T const& value)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
                "Requires at least forward iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::v1::detail::is_negative(count))
            {
                return hpx::parallel::util::detail::algorithm_result<ExPolicy,
                    FwdIter>::get(std::move(first));
            }

            return hpx::parallel::v1::detail::fill_n<FwdIter>().call(
                std::forward<ExPolicy>(policy), is_seq{}, first,
                std::size_t(count), value);
        }

        // clang-format off
        template <typename Rng, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend typename hpx::traits::range_traits<Rng>::iterator_type
        tag_invoke(fill_n_t, Rng&& rng, T const& value)
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            static_assert(
                (hpx::traits::is_forward_iterator<iterator_type>::value),
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::v1::detail::is_negative(hpx::util::size(rng)))
            {
                return hpx::util::begin(rng);
            }

            return hpx::parallel::v1::detail::fill_n<iterator_type>().call(
                hpx::parallel::execution::seq, std::true_type{},
                hpx::util::begin(rng), hpx::util::size(rng), value);
        }

        // clang-format off
        template <typename FwdIter, typename Size, typename T,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend FwdIter tag_invoke(
            fill_n_t, FwdIter first, Size count, T const& value)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::v1::detail::is_negative(count))
            {
                return first;
            }

            return hpx::parallel::v1::detail::fill_n<FwdIter>().call(
                hpx::parallel::execution::seq, std::true_type{}, first,
                std::size_t(count), value);
        }
    } fill_n{};

}}    // namespace hpx::ranges

#endif    // DOXYGEN
