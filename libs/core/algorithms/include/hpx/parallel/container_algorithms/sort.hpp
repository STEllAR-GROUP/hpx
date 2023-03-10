//  Copyright (c) 2015-2023 Hartmut Kaiser
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/sort.hpp

#pragma once

#if defined(DOXYGEN)

namespace hpx { namespace ranges {
    // clang-format off

    ///////////////////////////////////////////////////////////////////////////
    /// Sorts the elements in the range [first, last) in ascending order. The
    /// order of equal elements is not guaranteed to be preserved. The function
    /// uses the given comparison function object comp (defaults to using
    /// operator<()).
    ///
    /// \note   Complexity: O(N log(N)), where N = detail::distance(first, last)
    ///                     comparisons.
    ///
    /// A sequence is sorted with respect to a comparator \a comp and a
    /// projection \a proj if for every iterator i pointing to the sequence and
    /// every non-negative integer n such that i + n is a valid iterator
    /// pointing to an element of the sequence, and
    /// INVOKE(comp, INVOKE(proj, *(i + n)), INVOKE(proj, *i)) == false.
    ///
    /// \tparam RandomIt    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     random iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for RandomIt.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param comp         comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that comp
    ///                     will not apply any non-constant function through the
    ///                     dereferenced iterator.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each pair of elements as a
    ///                     projection operation before the actual predicate
    ///                     \a comp is invoked.
    ///
    /// \a comp has to induce a strict weak ordering on the values.
    ///
    /// The assignments in the parallel \a sort algorithm invoked without
    /// an execution policy object execute in sequential order in the
    /// calling thread.
    ///
    /// \returns  The \a sort algorithm returns \a RandomIt.
    ///           The algorithm returns an iterator pointing to the first
    ///           element after the last element in the input sequence.
    ///
    template <typename RandomIt, typename Sent,
        typename Comp = ranges::less,
        typename Proj = hpx::identity>
    RandomIt sort(RandomIt first, Sent last,  Comp&& comp = Comp(),
        Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Sorts the elements in the range [first, last) in ascending order. The
    /// order of equal elements is not guaranteed to be preserved. The function
    /// uses the given comparison function object comp (defaults to using
    /// operator<()).
    ///
    /// \note   Complexity: O(N log(N)), where N = detail::distance(first, last)
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
    ///                     in which it executes the assignments.
    /// \tparam RandomIt    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     random iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for RandomIt.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first        Refers to the beginning of the sequence of elements
    ///                     the algorithm will be applied to.
    /// \param last         Refers to sentinel value denoting the end of the
    ///                     sequence of elements the algorithm will be applied.
    /// \param comp         comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that comp
    ///                     will not apply any non-constant function through the
    ///                     dereferenced iterator.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each pair of elements as a
    ///                     projection operation before the actual predicate
    ///                     \a comp is invoked.
    ///
    /// \a comp has to induce a strict weak ordering on the values.
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
    /// \returns  The \a sort algorithm returns a
    ///           \a hpx::future<RandomIt> if the execution policy is of
    ///           type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a RandomIt
    ///           otherwise.
    ///           The algorithm returns an iterator pointing to the first
    ///           element after the last element in the input sequence.
    ///
    template <typename ExPolicy, typename RandomIt, typename Sent,
        typename Comp = ranges::less,
        typename Proj = hpx::identity>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        RandomIt>::type
    sort(ExPolicy&& policy, RandomIt first, Sent last, Comp&& comp = Comp(),
        Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Sorts the elements in the range \a rng  in ascending order. The
    /// order of equal elements is not guaranteed to be preserved. The function
    /// uses the given comparison function object comp (defaults to using
    /// operator<()).
    ///
    /// \note   Complexity: O(N log(N)),
    ///             where N = std::distance(begin(rng), end(rng)) comparisons.
    ///
    /// A sequence is sorted with respect to a comparator \a comp and a
    /// projection \a proj if for every iterator i pointing to the sequence and
    /// every non-negative integer n such that i + n is a valid iterator
    /// pointing to an element of the sequence, and
    /// INVOKE(comp, INVOKE(proj, *(i + n)), INVOKE(proj, *i)) == false.
    ///
    /// \tparam Rng         The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an input iterator.
    /// \tparam Comp     The type of the function/function object to use
    ///                     (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param comp         comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that comp
    ///                     will not apply any non-constant function through the
    ///                     dereferenced iterator.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each pair of elements as a
    ///                     projection operation before the actual predicate
    ///                     \a comp is invoked.
    ///
    /// \a comp has to induce a strict weak ordering on the values.
    ///
    /// The assignments in the parallel \a sort algorithm invoked without
    /// an execution policy object execute in sequential order in the
    /// calling thread.
    ///
    /// \returns  The \a sort algorithm returns \a
    ///           hpx::traits::range_iterator_t<Rng>.
    ///           It returns \a last.
    template <typename Rng, typename Comp, typename Proj>
    hpx::traits::range_iterator_t<Rng>
    sort(Rng&& rng, Comp&& comp = Comp(), Proj&& proj = Proj());

    ///////////////////////////////////////////////////////////////////////////
    /// Sorts the elements in the range \a rng  in ascending order. The
    /// order of equal elements is not guaranteed to be preserved. The function
    /// uses the given comparison function object comp (defaults to using
    /// operator<()).
    ///
    /// \note   Complexity: O(N log(N)),
    ///             where N = std::distance(begin(rng), end(rng)) comparisons.
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
    /// \tparam Comp     The type of the function/function object to use
    ///                     (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a hpx::identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng          Refers to the sequence of elements the algorithm
    ///                     will be applied to.
    /// \param comp         comp is a callable object. The return value of the
    ///                     INVOKE operation applied to an object of type Comp,
    ///                     when contextually converted to bool, yields true if
    ///                     the first argument of the call is less than the
    ///                     second, and false otherwise. It is assumed that comp
    ///                     will not apply any non-constant function through the
    ///                     dereferenced iterator.
    /// \param proj         Specifies the function (or function object) which
    ///                     will be invoked for each pair of elements as a
    ///                     projection operation before the actual predicate
    ///                     \a comp is invoked.
    ///
    /// \a comp has to induce a strict weak ordering on the values.
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
    /// \returns  The \a sort algorithm returns a
    ///           \a hpx::future<hpx::traits::range_iterator_t<Rng>
    ///           if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a
    ///           hpx::traits::range_iterator_t<Rng>
    ///           otherwise.
    ///           It returns \a last.
    ///
    template <typename ExPolicy, typename Rng,
        typename Comp = ranges::less,
        typename Proj = hpx::identity>
    typename parallel::util::detail::algorithm_result<ExPolicy,
        hpx::traits::range_iterator_t<Rng>>
    sort(ExPolicy&& policy, Rng&& rng, Comp&& comp = Comp(),
        Proj&& proj = Proj());

    // clang-format on
}}    // namespace hpx::ranges

#else

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>
#include <hpx/parallel/algorithms/sort.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>
#include <hpx/type_support/identity.hpp>

#include <type_traits>
#include <utility>

namespace hpx::parallel {

    // clang-format off
    template <typename ExPolicy, typename Rng,
        typename Comp = detail::less,
        typename Proj = hpx::identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy_v<ExPolicy> &&
            hpx::traits::is_range_v<Rng> &&
            traits::is_projected_range_v<Proj, Rng> &&
            traits::is_indirect_callable<ExPolicy, Comp,
                traits::projected_range<Proj, Rng>,
                traits::projected_range<Proj, Rng>
            >::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(
        1, 8, "hpx::parallel::sort is deprecated, use hpx::sort instead")
        util::detail::algorithm_result_t<ExPolicy,
            hpx::traits::range_iterator_t<Rng>> sort(ExPolicy&& policy,
            Rng&& rng, Comp&& comp = Comp(), Proj&& proj = Proj())
    {
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        return sort(HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
            hpx::util::end(rng), HPX_FORWARD(Comp, comp),
            HPX_FORWARD(Proj, proj));
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }
}    // namespace hpx::parallel

namespace hpx::ranges {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::ranges::sort
    inline constexpr struct sort_t final
      : hpx::detail::tag_parallel_algorithm<sort_t>
    {
    private:
        // clang-format off
        template <typename RandomIt, typename Sent,
            typename Comp = ranges::less,
            typename Proj = hpx::identity,
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
        friend RandomIt tag_fallback_invoke(hpx::ranges::sort_t, RandomIt first,
            Sent last, Comp comp = Comp(), Proj proj = Proj())
        {
            static_assert(hpx::traits::is_random_access_iterator_v<RandomIt>,
                "Requires a random access iterator.");

            return hpx::parallel::detail::sort<RandomIt>().call(
                hpx::execution::seq, first, last, HPX_MOVE(comp),
                HPX_MOVE(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename RandomIt, typename Sent,
            typename Comp = ranges::less,
            typename Proj = hpx::identity,
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
        friend parallel::util::detail::algorithm_result_t<ExPolicy, RandomIt>
        tag_fallback_invoke(hpx::ranges::sort_t, ExPolicy&& policy,
            RandomIt first, Sent last, Comp comp = Comp(), Proj proj = Proj())
        {
            static_assert(hpx::traits::is_random_access_iterator_v<RandomIt>,
                "Requires a random access iterator.");

            return hpx::parallel::detail::sort<RandomIt>().call(
                HPX_FORWARD(ExPolicy, policy), first, last, HPX_MOVE(comp),
                HPX_MOVE(proj));
        }

        // clang-format off
        template <typename Rng,
            typename Comp = ranges::less,
            typename Proj = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range_v<Rng> &&
                parallel::traits::is_projected_range_v<Proj, Rng> &&
                parallel::traits::is_indirect_callable_v<
                    hpx::execution::sequenced_policy, Comp,
                    parallel::traits::projected_range<Proj, Rng>,
                    parallel::traits::projected_range<Proj, Rng>
                >
            )>
        // clang-format on
        friend hpx::traits::range_iterator_t<Rng> tag_fallback_invoke(
            hpx::ranges::sort_t, Rng&& rng, Comp comp = Comp(),
            Proj proj = Proj())
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            static_assert(
                hpx::traits::is_random_access_iterator_v<iterator_type>,
                "Requires a random access iterator.");

            return hpx::parallel::detail::sort<iterator_type>().call(
                hpx::execution::seq, hpx::util::begin(rng), hpx::util::end(rng),
                HPX_MOVE(comp), HPX_MOVE(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,
            typename Comp = ranges::less,
            typename Proj = hpx::identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_range_v<Rng> &&
                parallel::traits::is_projected_range_v<Proj, Rng> &&
                parallel::traits::is_indirect_callable_v<ExPolicy, Comp,
                    parallel::traits::projected_range<Proj, Rng>,
                    parallel::traits::projected_range<Proj, Rng>
                >
            )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy,
            hpx::traits::range_iterator_t<Rng>>
        tag_fallback_invoke(hpx::ranges::sort_t, ExPolicy&& policy, Rng&& rng,
            Comp comp = Comp(), Proj proj = Proj())
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            static_assert(
                hpx::traits::is_random_access_iterator_v<iterator_type>,
                "Requires a random access iterator.");

            return hpx::parallel::detail::sort<iterator_type>().call(
                HPX_FORWARD(ExPolicy, policy), hpx::util::begin(rng),
                hpx::util::end(rng), HPX_MOVE(comp), HPX_MOVE(proj));
        }
    } sort{};
}    // namespace hpx::ranges

#endif
