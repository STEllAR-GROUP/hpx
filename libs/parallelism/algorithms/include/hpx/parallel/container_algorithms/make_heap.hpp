//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/make_heap.hpp

#pragma once

#if defined(DOXYGEN)
namespace hpx { namespace ranges {
    // clang-format off

    /// Constructs a \a max \a heap in the range [first, last).
    ///
    /// \note Complexity: at most (3*N) comparisons where
    ///       \a N = distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution of
    ///                     the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param comp         Refers to the binary predicate which returns true
    ///                     if the first argument should be treated as less than
    ///                     the second. The signature of the function should be
    ///                     equivalent to
    ///                     \code
    ///                     bool comp(const Type &a, const Type &b);
    ///                     \endcode \n
    ///                     The signature does not need to have const &, but
    ///                     the function must not modify the objects passed to
    ///                     it. The type \a Type must be such that objects of
    ///                     types \a RndIter can be dereferenced and then
    ///                     implicitly converted to Type.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each pair of elements as a
    ///                     projection operation before the actual predicate
    ///                     \a comp is invoked.
    ///
    /// The predicate operations in the parallel \a make_heap algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// executes in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a make_heap algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a make_heap algorithm returns a
    ///           \a hpx::future<Iter> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a Iter
    ///           otherwise.
    ///           It returns \a last.
    ///
    template <typename ExPolicy, typename Rng, typename Comp,
        typename Proj = util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy,
        typename hpx::traits::range_iterator<Rng>::type>::type
    make_heap(ExPolicy&& policy, Rng&& rng, Comp&& comp, Proj&& proj = Proj{});

    /// Constructs a \a max \a heap in the range [first, last). Uses the
    /// operator \a < for comparisons.
    ///
    /// \note Complexity: at most (3*N) comparisons where
    ///       \a N = distance(first, last).
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution of
    ///                     the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each pair of elements as a
    ///                     projection operation before the actual predicate
    ///                     \a comp is invoked.
    ///
    /// The predicate operations in the parallel \a make_heap algorithm invoked
    /// with an execution policy object of type \a sequential_execution_policy
    /// executes in sequential order in the calling thread.
    ///
    /// The comparison operations in the parallel \a make_heap algorithm invoked
    /// with an execution policy object of type \a parallel_execution_policy
    /// or \a parallel_task_execution_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a make_heap algorithm returns a \a hpx::future<void>
    ///           if the execution policy is of type \a task_execution_policy
    ///           and returns \a void otherwise.
    ///
    template <typename ExPolicy, typename Rng,
        typename Proj = util::projection_identity>
    typename util::detail::algorithm_result<ExPolicy,
        typename hpx::traits::range_iterator<Rng>::type>::type
    make_heap(ExPolicy&& policy, Rng&& rng, Proj&& proj = Proj{});

    // clang-format on
}}    // namespace hpx::ranges

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/iterator_support/traits/is_sentinel_for.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/make_heap.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace ranges {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::make_heap
    HPX_INLINE_CONSTEXPR_VARIABLE struct make_heap_t final
      : hpx::functional::tag<make_heap_t>
    {
    private:
        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent,
            typename Comp,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_sentinel_for<Sent, Iter>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy, Comp,
                    hpx::parallel::traits::projected<Proj, Iter>,
                    hpx::parallel::traits::projected<Proj, Iter>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            Iter>::type
        tag_invoke(make_heap_t, ExPolicy&& policy, Iter first, Sent last,
            Comp&& comp, Proj&& proj = Proj{})
        {
            static_assert(hpx::traits::is_random_access_iterator<Iter>::value,
                "Requires random access iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            return hpx::parallel::v1::detail::make_heap<Iter>().call(
                std::forward<ExPolicy>(policy), is_seq{}, first, last,
                std::forward<Comp>(comp), std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng, typename Comp,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy, Comp,
                    hpx::parallel::traits::projected_range<Proj, Rng>,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            typename hpx::traits::range_iterator<Rng>::type>::type
        tag_invoke(make_heap_t, ExPolicy&& policy, Rng& rng, Comp&& comp,
            Proj&& proj = Proj{})
        {
            using iterator_type =
                typename hpx::traits::range_iterator<Rng>::type;

            static_assert(
                hpx::traits::is_random_access_iterator<iterator_type>::value,
                "Requires random access iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;

            return hpx::parallel::v1::detail::make_heap<iterator_type>().call(
                std::forward<ExPolicy>(policy), is_seq{}, hpx::util::begin(rng),
                hpx::util::end(rng), std::forward<Comp>(comp),
                std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Iter, typename Sent,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_sentinel_for<Sent, Iter>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy,
                    std::less<typename std::iterator_traits<Iter>::value_type>,
                    hpx::parallel::traits::projected<Proj, Iter>,
                    hpx::parallel::traits::projected<Proj, Iter>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            Iter>::type
        tag_invoke(make_heap_t, ExPolicy&& policy, Iter first, Sent last,
            Proj&& proj = Proj{})
        {
            static_assert(hpx::traits::is_random_access_iterator<Iter>::value,
                "Requires random access iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;
            using value_type = typename std::iterator_traits<Iter>::value_type;

            return hpx::parallel::v1::detail::make_heap<Iter>().call(
                std::forward<ExPolicy>(policy), is_seq{}, first, last,
                std::less<value_type>(), std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::parallel::execution::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_indirect_callable<ExPolicy,
                    std::less<typename std::iterator_traits<
                        typename hpx::traits::range_iterator<Rng>::type
                    >::value_type>,
                    hpx::parallel::traits::projected_range<Proj, Rng>,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            typename hpx::traits::range_iterator<Rng>::type>::type
        tag_invoke(
            make_heap_t, ExPolicy&& policy, Rng&& rng, Proj&& proj = Proj{})
        {
            using iterator_type =
                typename hpx::traits::range_iterator<Rng>::type;

            static_assert(
                hpx::traits::is_random_access_iterator<iterator_type>::value,
                "Requires random access iterator.");

            using is_seq =
                hpx::parallel::execution::is_sequenced_execution_policy<
                    ExPolicy>;
            using value_type =
                typename std::iterator_traits<iterator_type>::value_type;

            return hpx::parallel::v1::detail::make_heap<iterator_type>().call(
                std::forward<ExPolicy>(policy), is_seq{}, hpx::util::begin(rng),
                hpx::util::end(rng), std::less<value_type>(),
                std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename Iter, typename Sent, typename Comp,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_sentinel_for<Sent, Iter>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::parallel::execution::sequenced_policy, Comp,
                    hpx::parallel::traits::projected<Proj, Iter>,
                    hpx::parallel::traits::projected<Proj, Iter>
                >::value
            )>
        // clang-format on
        friend Iter tag_invoke(make_heap_t, Iter first, Sent last, Comp&& comp,
            Proj&& proj = Proj{})
        {
            static_assert(hpx::traits::is_random_access_iterator<Iter>::value,
                "Requires random access iterator.");

            return hpx::parallel::v1::detail::make_heap<Iter>().call(
                hpx::parallel::execution::seq, std::true_type{}, first, last,
                std::forward<Comp>(comp), std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename Rng, typename Comp,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::parallel::execution::sequenced_policy, Comp,
                    hpx::parallel::traits::projected_range<Proj, Rng>,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend typename hpx::traits::range_iterator<Rng>::type tag_invoke(
            make_heap_t, Rng& rng, Comp&& comp, Proj&& proj = Proj{})
        {
            using iterator_type =
                typename hpx::traits::range_iterator<Rng>::type;

            static_assert(
                hpx::traits::is_random_access_iterator<iterator_type>::value,
                "Requires random access iterator.");

            return hpx::parallel::v1::detail::make_heap<iterator_type>().call(
                hpx::parallel::execution::seq, std::true_type{},
                hpx::util::begin(rng), hpx::util::end(rng),
                std::forward<Comp>(comp), std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename Iter, typename Sent,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_sentinel_for<Sent, Iter>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::parallel::execution::sequenced_policy,
                    std::less<typename std::iterator_traits<Iter>::value_type>,
                    hpx::parallel::traits::projected<Proj, Iter>,
                    hpx::parallel::traits::projected<Proj, Iter>
                >::value
            )>
        // clang-format on
        friend Iter tag_invoke(
            make_heap_t, Iter first, Sent last, Proj&& proj = Proj{})
        {
            static_assert(hpx::traits::is_random_access_iterator<Iter>::value,
                "Requires random access iterator.");

            using value_type = typename std::iterator_traits<Iter>::value_type;

            return hpx::parallel::v1::detail::make_heap<Iter>().call(
                hpx::parallel::execution::seq, std::true_type{}, first, last,
                std::less<value_type>(), std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename Rng,
            typename Proj = hpx::parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::parallel::execution::sequenced_policy,
                    std::less<typename std::iterator_traits<
                        typename hpx::traits::range_iterator<Rng>::type
                    >::value_type>,
                    hpx::parallel::traits::projected_range<Proj, Rng>,
                    hpx::parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend typename hpx::traits::range_iterator<Rng>::type tag_invoke(
            make_heap_t, Rng&& rng, Proj&& proj = Proj{})
        {
            using iterator_type =
                typename hpx::traits::range_iterator<Rng>::type;

            static_assert(
                hpx::traits::is_random_access_iterator<iterator_type>::value,
                "Requires random access iterator.");

            using value_type =
                typename std::iterator_traits<iterator_type>::value_type;

            return hpx::parallel::v1::detail::make_heap<iterator_type>().call(
                hpx::parallel::execution::seq, std::true_type{},
                hpx::util::begin(rng), hpx::util::end(rng),
                std::less<value_type>(), std::forward<Proj>(proj));
        }
    } make_heap{};
}}    // namespace hpx::ranges

#endif    // DOXYGEN
