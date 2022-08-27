//  Copyright (c) 2020 Hartmut Kaiser
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/partial_sort.hpp

#pragma once

#if defined(DOXYGEN)

namespace hpx { namespace ranges {
    // clang-format off

    ///////////////////////////////////////////////////////////////////////////
    /// Places the first middle - first elements from the range [first, last)
    /// as sorted with respect to comp into the range [first, middle). The rest
    /// of the elements in the range [middle, last) are placed in an unspecified
    /// order.
    ///
    /// \note   Complexity: Approximately (last - first) * log(middle - first)
    ///         comparisons.
    ///
    /// \tparam RandomIt    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     random iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for RandomIt.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced). Comp defaults to detail::less.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param middle       Refers to the middle of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param comp         comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that
    ///                     comp will not apply any non-constant function
    ///                     through the dereferenced iterator. Comp defaults
    ///                     to detail::less.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each pair of elements as a
    ///                     projection operation before the actual predicate
    ///                     \a comp is invoked.
    ///
    /// The assignments in the parallel \a partial_sort algorithm invoked without
    /// an execution policy object execute in sequential order in the
    /// calling thread.
    ///
    /// \returns  The \a partial_sort algorithm returns \a RandomIt.
    ///           The algorithm returns an iterator pointing to the first
    ///           element after the last element in the input sequence.
    ///
    template <typename RandomIt, typename Sent, typename Comp, typename Proj>
    RandomIt partial_sort(RandomIt first, Sent last, Comp&& comp = Comp(),
        Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Places the first middle - first elements from the range [first, last)
    /// as sorted with respect to comp into the range [first, middle). The rest
    /// of the elements in the range [middle, last) are placed in an unspecified
    /// order.
    ///
    /// \note   Complexity: Approximately (last - first) * log(middle - first)
    ///         comparisons.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam RandomIt    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     random iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for RandomIt.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced). Comp defaults to detail::less.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param middle       Refers to the middle of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param comp         comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that
    ///                     comp will not apply any non-constant function
    ///                     through the dereferenced iterator. Comp defaults
    ///                     to detail::less.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each pair of elements as a
    ///                     projection operation before the actual predicate
    ///                     \a comp is invoked.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a partial_sort algorithm returns a
    ///           \a hpx::future<RandomIt> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a RandomIt
    ///           otherwise.
    ///           The algorithm returns an iterator pointing to the first
    ///           element after the last element in the input sequence.
    ///
    template <typename ExPolicy, typename RandomIt, typename Sent,
        typename Comp, typename Proj>
    parallel::util::detail::algorithm_result_t<ExPolicy,
        RandomIt>
    partial_sort(ExPolicy&& policy, RandomIt first, RandomIt middle,
        Sent last, Comp&& comp = Comp(), Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Places the first middle - first elements from the range [first, last)
    /// as sorted with respect to comp into the range [first, middle). The rest
    /// of the elements in the range [middle, last) are placed in an unspecified
    /// order.
    ///
    /// \note   Complexity: Approximately (last - first) * log(middle - first)
    ///         comparisons.
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced). Comp defaults to detail::less.
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param middle       Refers to the middle of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param comp         comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that
    ///                     comp will not apply any non-constant function
    ///                     through the dereferenced iterator. Comp defaults
    ///                     to detail::less.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each pair of elements as a
    ///                     projection operation before the actual predicate
    ///                     \a comp is invoked.
    ///
    /// The assignments in the parallel \a partial_sort algorithm invoked without
    /// an execution policy object execute in sequential order in the
    /// calling thread.
    ///
    /// \returns  The \a partial_sort algorithm returns \a
    ///           typename hpx::traits::range_iterator<Rng>::type.
    ///           It returns \a last.
    template <typename Rng, typename Comp, typename Proj>
    hpx::traits::range_iterator<Rng>_t
    partial_sort(Rng&& rng,, hpx::traits::range_iterator<Rng>_t middle,
        Comp&& comp = Comp(), Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Sorts the elements in the range [first, last) in ascending order. The
    /// relative order of equal elements is preserved. The function
    /// uses the given comparison function object comp (defaults to using
    /// operator<()).
    ///
    /// \note   Complexity: O(Nlog(N)), where N = std::distance(first, last)
    ///                     comparisons.
    ///
    /// A sequence is sorted with respect to a comparator \a comp and a
    /// projection \a proj if for every iterator i pointing to the sequence and
    /// every non-negative integer n such that i + n is a valid iterator
    /// pointing to an element of the sequence, and
    /// INVOKE(comp, INVOKE(proj, *(i + n)), INVOKE(proj, *i)) == false.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it applies user-provided function objects.
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced). Comp defaults to detail::less;
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param middle       Refers to the middle of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param comp         comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that
    ///                     comp will not apply any non-constant function
    ///                     through the dereferenced iterator. Comp defaults
    ///                     to detail::less.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each pair of elements as a
    ///                     projection operation before the actual predicate
    ///                     \a comp is invoked.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a sequenced_policy execute in sequential order in the
    /// calling thread.
    ///
    /// The application of function objects in parallel algorithm
    /// invoked with an execution policy object of type
    /// \a parallel_policy or \a parallel_task_policy are
    /// permitted to execute in an unordered fashion in unspecified
    /// threads, and indeterminately sequenced within each thread.
    ///
    /// \returns  The \a partial_sort algorithm returns a
    ///           \a hpx::future<typename hpx::traits::range_iterator<Rng>
    ///           ::type> if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a
    ///           typename hpx::traits::range_iterator<Rng>::type
    ///           otherwise.
    ///           It returns \a last.
    ///
    template <typename ExPolicy, typename Rng, typename Comp, typename Proj>
    util::detail::algorithm_result_t<ExPolicy,
        hpx::traits::range_iterator<Rng>_t>
    partial_sort(ExPolicy&& policy, Rng&& rng,
        hpx::traits::range_iterator<Rng>_t middle,
        Comp&& comp = Comp(), Proj&& proj = Proj());

    // clang-format on
}}    // namespace hpx::ranges

#else

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>

#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/parallel/algorithms/partial_sort.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace ranges {
    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::ranges::partial_sort
    inline constexpr struct partial_sort_t final
      : hpx::detail::tag_parallel_algorithm<partial_sort_t>
    {
    private:
        // clang-format off
        template <typename RandomIt, typename Sent,
            typename Comp = ranges::less,
            typename Proj = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<RandomIt> &&
                hpx::traits::is_sentinel_for_v<Sent, RandomIt> &&
                parallel::traits::is_projected_v<Proj, RandomIt> &&
                parallel::traits::is_indirect_callable_v<
                    hpx::execution::sequenced_policy, Comp,
                    parallel::traits::projected<Proj, RandomIt>,
                    parallel::traits::projected<Proj, RandomIt>
                >
            )>
        // clang-format on
        friend RandomIt tag_fallback_invoke(hpx::ranges::partial_sort_t,
            RandomIt first, RandomIt middle, Sent last, Comp&& comp = Comp(),
            Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_random_access_iterator_v<RandomIt>,
                "Requires a random access iterator.");

            return hpx::parallel::v1::partial_sort<RandomIt>().call(
                hpx::execution::seq, first, middle, last,
                HPX_FORWARD(Comp, comp), HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename RandomIt, typename Sent,
            typename Comp = ranges::less,
            typename Proj = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<RandomIt> &&
                hpx::traits::is_sentinel_for_v<Sent, RandomIt> &&
                parallel::traits::is_projected_v<Proj, RandomIt> &&
                parallel::traits::is_indirect_callable_v<ExPolicy, Comp,
                    parallel::traits::projected<Proj, RandomIt>,
                    parallel::traits::projected<Proj, RandomIt>
                >
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            RandomIt>::type
        tag_fallback_invoke(hpx::ranges::partial_sort_t, ExPolicy&& policy,
            RandomIt first, RandomIt middle, Sent last, Comp&& comp = Comp(),
            Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_random_access_iterator_v<RandomIt>,
                "Requires a random access iterator.");

            return hpx::parallel::v1::partial_sort<RandomIt>().call(
                HPX_FORWARD(ExPolicy, policy), first, middle, last,
                HPX_FORWARD(Comp, comp), HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename Rng,
            typename Compare = ranges::less,
            typename Proj = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range_v<Rng> &&
                parallel::traits::is_projected_range_v<Proj, Rng> &&
                parallel::traits::is_indirect_callable_v<
                    hpx::execution::sequenced_policy, Compare,
                    parallel::traits::projected_range<Proj, Rng>,
                    parallel::traits::projected_range<Proj, Rng>
                >
            )>
        // clang-format on
        friend hpx::traits::range_iterator_t<Rng> tag_fallback_invoke(
            hpx::ranges::partial_sort_t, Rng&& rng,
            hpx::traits::range_iterator_t<Rng> middle,
            Compare&& comp = Compare(), Proj&& proj = Proj())
        {
            using iterator_type = hpx::traits::range_iterator_t<Rng>;

            static_assert(
                hpx::traits::is_random_access_iterator_v<iterator_type>,
                "Requires a random access iterator.");

            return hpx::parallel::v1::partial_sort<iterator_type>().call(
                hpx::execution::seq, hpx::util::begin(rng), middle,
                hpx::util::end(rng), HPX_FORWARD(Compare, comp),
                HPX_FORWARD(Proj, proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,
            typename Compare = ranges::less,
            typename Proj = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_range_v<Rng> &&
                parallel::traits::is_projected_range_v<Proj, Rng> &&
                parallel::traits::is_indirect_callable_v<ExPolicy, Compare,
                    parallel::traits::projected_range<Proj, Rng>,
                    parallel::traits::projected_range<Proj, Rng>
                >
            )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy,
            hpx::traits::range_iterator_t<Rng>>
        tag_fallback_invoke(hpx::ranges::partial_sort_t, ExPolicy&& policy,
            Rng&& rng, hpx::traits::range_iterator_t<Rng> middle,
            Compare&& comp = Compare(), Proj&& proj = Proj())
        {
            using iterator_type = hpx::traits::range_iterator_t<Rng>;

            static_assert(
                hpx::traits::is_random_access_iterator_v<iterator_type>,
                "Requires a random access iterator.");

            return hpx::parallel::v1::partial_sort<iterator_type>().call(
                HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng), middle,
                hpx::util::end(rng), HPX_FORWARD(Compare, comp),
                HPX_FORWARD(Proj, proj));
        }
    } partial_sort{};
}}    // namespace hpx::ranges

#endif
