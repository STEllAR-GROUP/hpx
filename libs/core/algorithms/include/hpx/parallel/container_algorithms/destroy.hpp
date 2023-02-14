//  Copyright (c) 2020-2023 Hartmut Kaiser
//  Copyright (c) 2022 Dimitra Karatza
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/destroy.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx { namespace ranges {
    // clang-format off

    /// Destroys objects of type typename iterator_traits<ForwardIt>::value_type
    /// in the range [first, last).
    ///
    /// \note   Complexity: Performs exactly \a last - \a first operations.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    ///
    /// The operations in the parallel \a destroy
    /// algorithm invoked with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The operations in the parallel \a destroy
    /// algorithm invoked with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a destroy algorithm returns a
    ///           \a hpx::future<void>, if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a void otherwise.
    ///
    template <typename ExPolicy, typename Rng>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        hpx::traits::range_iterator_t<Rng>>
    destroy(ExPolicy&& policy, Rng&& rng);

    /// Destroys objects of type typename iterator_traits<ForwardIt>::value_type
    /// in the range [first, last).
    ///
    /// \note   Complexity: Performs exactly \a last - \a first operations.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Iter        The type of the source iterators used for the
    ///                     range (deduced).
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements
    ///                     the algorithm will be applied to.
    ///
    /// The operations in the parallel \a destroy
    /// algorithm invoked with an execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The operations in the parallel \a destroy
    /// algorithm invoked with an execution policy object of type \a parallel_policy
    /// or \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a destroy algorithm returns a
    ///           \a hpx::future<void>, if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a void otherwise.
    ///
    template <typename ExPolicy, typename Iter, typename Sent>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, Iter>
    destroy(ExPolicy&& policy, Iter first, Sent last);

    /// Destroys objects of type typename iterator_traits<ForwardIt>::value_type
    /// in the range [first, last).
    ///
    /// \note   Complexity: Performs exactly \a last - \a first operations.
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    ///
    /// \returns  The \a destroy algorithm returns \a void.
    ///
    template <typename Rng> hpx::traits::range_iterator<Rng>::type destroy(Rng&& rng);

    /// Destroys objects of type typename iterator_traits<ForwardIt>::value_type
    /// in the range [first, last).
    ///
    /// \note   Complexity: Performs exactly \a last - \a first operations.
    ///
    /// \tparam Iter        The type of the source iterators used for the
    ///                     range (deduced).
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for InIter.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to the end of the sequence of elements
    ///                     the algorithm will be applied to.
    ///
    /// \returns  The \a destroy algorithm returns \a void.
    ///
    template <typename Iter, typename Sent> Iter destroy(Iter first, Sent last);

    /// Destroys objects of type typename iterator_traits<ForwardIt>::value_type
    /// in the range [first, first + count).
    ///
    /// \note   Complexity: Performs exactly \a count operations, if
    ///         count > 0, no assignments otherwise.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply this algorithm to.
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    ///
    /// The operations in the parallel \a destroy_n
    /// algorithm invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The operations in the parallel \a destroy_n
    /// algorithm invoked with an execution policy object of type
    /// \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an
    /// unordered fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a destroy_n algorithm returns a
    ///           \a hpx::future<FwdIter> if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and
    ///           returns \a FwdIter otherwise.
    ///           The \a destroy_n algorithm returns the
    ///           iterator to the element in the source range, one past
    ///           the last element constructed.
    ///
    template <typename ExPolicy, typename FwdIter, typename Size>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        FwdIter>::type
    destroy_n(ExPolicy&& policy, FwdIter first, Size count);

    /// Destroys objects of type typename iterator_traits<ForwardIt>::value_type
    /// in the range [first, first + count).
    ///
    /// \note   Complexity: Performs exactly \a count operations, if
    ///         count > 0, no assignments otherwise.
    ///
    /// \tparam FwdIter     The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     forward iterator.
    /// \tparam Size        The type of the argument specifying the number of
    ///                     elements to apply this algorithm to.
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param count        Refers to the number of elements starting at
    ///                     \a first the algorithm will be applied to.
    ///
    /// \returns  The \a destroy_n algorithm returns the
    ///           iterator to the element in the source range, one past
    ///           the last element constructed.
    ///
    template <typename FwdIter, typename Size>
    FwdIter destroy_n(FwdIter first, Size count);

    // clang-format on
}}    // namespace hpx::ranges

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution/algorithms/detail/is_negative.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/parallel/algorithms/destroy.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx::ranges {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::destroy
    inline constexpr struct destroy_t final
      : hpx::detail::tag_parallel_algorithm<destroy_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename Rng,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_range_v<Rng>
            )>
        // clang-format on
        friend hpx::parallel::util::detail::algorithm_result_t<ExPolicy,
            hpx::traits::range_iterator_t<Rng>>
        tag_fallback_invoke(destroy_t, ExPolicy&& policy, Rng&& rng)
        {
            using iterator_type = hpx::traits::range_iterator_t<Rng>;

            static_assert(hpx::traits::is_forward_iterator_v<iterator_type>,
                "Required at least forward iterator.");

            return hpx::parallel::detail::destroy<iterator_type>().call(
                HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
                hpx::util::end(rng));
        }

        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_sentinel_for_v<Sent, Iter>
            )>
        // clang-format on
        friend hpx::parallel::util::detail::algorithm_result_t<ExPolicy, Iter>
        tag_fallback_invoke(destroy_t, ExPolicy&& policy, Iter first, Sent last)
        {
            static_assert(hpx::traits::is_forward_iterator_v<Iter>,
                "Required at least forward iterator.");

            return hpx::parallel::detail::destroy<Iter>().call(
                HPX_FORWARD(ExPolicy, policy), first, last);
        }

        // clang-format off
        template <typename Rng,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range_v<Rng>
            )>
        // clang-format on
        friend hpx::traits::range_iterator_t<Rng> tag_fallback_invoke(
            destroy_t, Rng&& rng)
        {
            using iterator_type = hpx::traits::range_iterator_t<Rng>;

            static_assert(hpx::traits::is_forward_iterator_v<iterator_type>,
                "Required at least forward iterator.");

            return hpx::parallel::detail::destroy<iterator_type>().call(
                hpx::execution::seq, hpx::util::begin(rng),
                hpx::util::end(rng));
        }

        // clang-format off
        template <typename Iter, typename Sent,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<Iter>
            )>
        // clang-format on
        friend Iter tag_fallback_invoke(destroy_t, Iter first, Sent last)
        {
            static_assert(hpx::traits::is_forward_iterator_v<Iter>,
                "Required at least forward iterator.");

            return hpx::parallel::detail::destroy<Iter>().call(
                hpx::execution::seq, first, last);
        }
    } destroy{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::destroy_n
    inline constexpr struct destroy_n_t final
      : hpx::detail::tag_parallel_algorithm<destroy_n_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Size,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<FwdIter> &&
                std::is_integral_v<Size>
            )>
        // clang-format on
        friend hpx::parallel::util::detail::algorithm_result_t<ExPolicy,
            FwdIter>
        tag_fallback_invoke(
            destroy_n_t, ExPolicy&& policy, FwdIter first, Size count)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::detail::is_negative(count))
            {
                return hpx::parallel::util::detail::algorithm_result<ExPolicy,
                    FwdIter>::get(HPX_MOVE(first));
            }

            return hpx::parallel::detail::destroy_n<FwdIter>().call(
                HPX_FORWARD(ExPolicy, policy), first,
                static_cast<std::size_t>(count));
        }

        // clang-format off
        template <typename FwdIter, typename Size,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<FwdIter> &&
                std::is_integral_v<Size>
            )>
        // clang-format on
        friend FwdIter tag_fallback_invoke(
            destroy_n_t, FwdIter first, Size count)
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter>,
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::detail::is_negative(count))
            {
                return first;
            }

            return hpx::parallel::detail::destroy_n<FwdIter>().call(
                hpx::execution::seq, first, static_cast<std::size_t>(count));
        }
    } destroy_n{};
}    // namespace hpx::ranges

#endif    // DOXYGEN
