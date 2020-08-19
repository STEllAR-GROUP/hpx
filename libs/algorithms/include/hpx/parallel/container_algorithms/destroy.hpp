//  Copyright (c) 2020 Hartmut Kaiser
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
    template <typename ExPolicy>
    typename util::detail::algorithm_result<ExPolicy,
        typename traits::range_iterator<Rng>::type>::type
    destroy(ExPolicy&& policy, Rng&& rng);

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
    typename util::detail::algorithm_result<ExPolicy, FwdIter>::type
    destroy_n(ExPolicy&& policy, FwdIter first, Size count);

    // clang-format on
}}    // namespace hpx::ranges

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/destroy.hpp>
#include <hpx/execution/algorithms/detail/is_negative.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace ranges {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::destroy
    HPX_INLINE_CONSTEXPR_VARIABLE struct destroy_t final
      : hpx::functional::tag<destroy_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename Rng,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            typename hpx::traits::range_iterator<Rng>::type>::type
        tag_invoke(destroy_t, ExPolicy&& policy, Rng&& rng)
        {
            using iterator_type =
                typename hpx::traits::range_iterator<Rng>::type;

            static_assert(
                (hpx::traits::is_forward_iterator<iterator_type>::value),
                "Required at least forward iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            return hpx::parallel::v1::detail::destroy<iterator_type>().call(
                std::forward<ExPolicy>(policy), is_seq{}, hpx::util::begin(rng),
                hpx::util::end(rng));
        }

        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_sentinel_for<Sent, Iter>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            Iter>::type
        tag_invoke(destroy_t, ExPolicy&& policy, Iter first, Sent last)
        {
            static_assert((hpx::traits::is_forward_iterator<Iter>::value),
                "Required at least forward iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            return hpx::parallel::v1::detail::destroy<Iter>().call(
                std::forward<ExPolicy>(policy), is_seq{}, first, last);
        }

        // clang-format off
        template <typename Rng,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend typename hpx::traits::range_iterator<Rng>::type tag_invoke(
            destroy_t, Rng&& rng)
        {
            using iterator_type =
                typename hpx::traits::range_iterator<Rng>::type;

            static_assert(
                (hpx::traits::is_forward_iterator<iterator_type>::value),
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::destroy<iterator_type>().call(
                hpx::parallel::execution::seq, std::false_type{},
                hpx::util::begin(rng), hpx::util::end(rng));
        }

        // clang-format off
        template <typename Iter, typename Sent,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<Iter>::value
            )>
        // clang-format on
        friend Iter tag_invoke(destroy_t, Iter first, Sent last)
        {
            static_assert((hpx::traits::is_forward_iterator<Iter>::value),
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::destroy<Iter>().call(
                hpx::parallel::execution::seq, std::false_type{}, first, last);
        }
    } destroy{};

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::destroy_n
    HPX_INLINE_CONSTEXPR_VARIABLE struct destroy_n_t final
      : hpx::functional::tag<destroy_n_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Size,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_invoke(destroy_n_t, ExPolicy&& policy, FwdIter first, Size count)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::v1::detail::is_negative(count))
            {
                return hpx::parallel::util::detail::algorithm_result<ExPolicy,
                    FwdIter>::get(std::move(first));
            }

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            return hpx::parallel::v1::detail::destroy_n<FwdIter>().call(
                std::forward<ExPolicy>(policy), is_seq{}, first,
                std::size_t(count));
        }

        // clang-format off
        template <typename FwdIter, typename Size,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter>::value
            )>
        // clang-format on
        friend FwdIter tag_invoke(destroy_n_t, FwdIter first, Size count)
        {
            static_assert((hpx::traits::is_forward_iterator<FwdIter>::value),
                "Requires at least forward iterator.");

            // if count is representing a negative value, we do nothing
            if (hpx::parallel::v1::detail::is_negative(count))
            {
                return first;
            }

            return hpx::parallel::v1::detail::destroy_n<FwdIter>().call(
                hpx::parallel::execution::seq, std::false_type{}, first,
                std::size_t(count));
        }
    } destroy_n{};
}}    // namespace hpx::ranges

#endif    // DOXYGEN
