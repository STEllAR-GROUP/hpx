//  Copyright (c) 2020 Hartmut Kaiser
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/stable_sort.hpp

#pragma once

#if defined(DOXYGEN)

namespace hpx { namespace ranges {
    // clang-format off

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
    /// \tparam RandomIt    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     random iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for RandomIt.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
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
    /// The assignments in the parallel \a stable_sort algorithm invoked without
    /// an execution policy object execute in sequential order in the
    /// calling thread.
    ///
    /// \returns  The \a stable_sort algorithm returns \a RandomIt.
    ///           The algorithm returns an iterator pointing to the first
    ///           element after the last element in the input sequence.
    ///
    template <typename RandomIt, typename Sent, typename Comp, typename Proj>
    RandomIt stable_sort(RandomIt first, Sent last, Comp&& comp, Proj&& proj);

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
    ///                     in which it executes the assignments.
    /// \tparam RandomIt    The type of the source iterators used (deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     random iterator.
    /// \tparam Sent        The type of the source sentinel (deduced). This
    ///                     sentinel type must be a sentinel for RandomIt.
    /// \tparam Comp        The type of the function/function object to use
    ///                     (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
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
    /// \returns  The \a stable_sort algorithm returns a
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
    typename parallel::util::detail::algorithm_result<ExPolicy,
        RandomIt>::type
    stable_sort(ExPolicy&& policy, RandomIt first, Sent last, Comp&& comp,
        Proj&& proj);

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
    /// The assignments in the parallel \a stable_sort algorithm invoked without
    /// an execution policy object execute in sequential order in the
    /// calling thread.
    ///
    /// \returns  The \a stable_sort algorithm returns \a
    ///           typename hpx::traits::range_iterator<Rng>::type.
    ///           It returns \a last.
    template <typename Rng, typename Comp, typename Proj>
    typename hpx::traits::range_iterator<Rng>::type
    stable_sort(Rng&& rng, Compare&& comp, Proj&& proj);

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
    ///                     (deduced).
    /// \tparam Proj        The type of an optional projection function. This
    ///                     defaults to \a util::projection_identity
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
    /// \returns  The \a stable_sort algorithm returns a
    ///           \a hpx::future<typename hpx::traits::range_iterator<Rng>
    ///           ::type> if the execution policy is of type
    ///           \a sequenced_task_policy or
    ///           \a parallel_task_policy and returns \a
    ///           typename hpx::traits::range_iterator<Rng>::type
    ///           otherwise.
    ///           It returns \a last.
    ///
    template <typename ExPolicy, typename Rng, typename Pred, typename Proj>
    typename util::detail::algorithm_result<ExPolicy,
        typename hpx::traits::range_iterator<Rng>::type>::type
    stable_sort(ExPolicy&& policy, Rng&& rng, Comp&& comp, Proj&&);

    // clang-format on
}}    // namespace hpx::ranges

#else

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>

#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/parallel/algorithms/stable_sort.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/detail/sender_util.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { inline namespace rangev1 {

    // clang-format off
    template <typename ExPolicy, typename Rng,
        typename Compare = v1::detail::less,
        typename Proj = util::projection_identity,
        HPX_CONCEPT_REQUIRES_(
            hpx::is_execution_policy<ExPolicy>::value &&
            hpx::traits::is_range<Rng>::value &&
            traits::is_projected_range<Proj, Rng>::value &&
            traits::is_indirect_callable<ExPolicy, Compare,
                traits::projected_range<Proj, Rng>,
                traits::projected_range<Proj, Rng>
            >::value
        )>
    // clang-format on
    HPX_DEPRECATED_V(1, 8,
        "hpx::parallel::stable_sort is deprecated, use hpx::stable_sort "
        "instead") typename util::detail::algorithm_result<ExPolicy,
        typename hpx::traits::range_iterator<Rng>::type>::type
        stable_sort(ExPolicy&& policy, Rng&& rng, Compare&& comp = Compare(),
            Proj&& proj = Proj())
    {
        using iterator_type = typename hpx::traits::range_iterator<Rng>::type;

#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        static_assert(hpx::traits::is_random_access_iterator_v<iterator_type>,
            "Requires a random access iterator.");

        return parallel::v1::detail::stable_sort<iterator_type>().call(
            std::forward<ExPolicy>(policy), hpx::util::begin(rng),
            hpx::util::end(rng), std::forward<Compare>(comp),
            std::forward<Proj>(proj));
#if defined(HPX_GCC_VERSION) && HPX_GCC_VERSION >= 100000
#pragma GCC diagnostic pop
#endif
    }
}}}    // namespace hpx::parallel::rangev1

namespace hpx { namespace ranges {
    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::ranges::stable_sort
    HPX_INLINE_CONSTEXPR_VARIABLE struct stable_sort_t final
      : hpx::detail::tag_parallel_algorithm<stable_sort_t>
    {
    private:
        // clang-format off
        template <typename RandomIt, typename Sent,
            typename Comp = ranges::less,
            typename Proj = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<RandomIt> &&
                hpx::traits::is_sentinel_for<Sent, RandomIt>::value &&
                parallel::traits::is_projected<Proj, RandomIt>::value &&
                parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Comp,
                    parallel::traits::projected<Proj, RandomIt>,
                    parallel::traits::projected<Proj, RandomIt>
                >::value
            )>
        // clang-format on
        friend RandomIt tag_fallback_dispatch(hpx::ranges::stable_sort_t,
            RandomIt first, Sent last, Comp&& comp = Comp(),
            Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_random_access_iterator_v<RandomIt>,
                "Requires a random access iterator.");

            return hpx::parallel::v1::detail::stable_sort<RandomIt>().call(
                hpx::execution::seq, first, last, std::forward<Comp>(comp),
                std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename RandomIt, typename Sent,
            typename Comp = ranges::less,
            typename Proj = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator_v<RandomIt> &&
                hpx::traits::is_sentinel_for<Sent, RandomIt>::value &&
                parallel::traits::is_projected<Proj, RandomIt>::value &&
                parallel::traits::is_indirect_callable<ExPolicy, Comp,
                    parallel::traits::projected<Proj, RandomIt>,
                    parallel::traits::projected<Proj, RandomIt>
                >::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            RandomIt>::type
        tag_fallback_dispatch(hpx::ranges::stable_sort_t, ExPolicy&& policy,
            RandomIt first, Sent last, Comp&& comp = Comp(),
            Proj&& proj = Proj())
        {
            static_assert(hpx::traits::is_random_access_iterator_v<RandomIt>,
                "Requires a random access iterator.");

            return hpx::parallel::v1::detail::stable_sort<RandomIt>().call(
                std::forward<ExPolicy>(policy), first, last,
                std::forward<Comp>(comp), std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename Rng,
            typename Compare = ranges::less,
            typename Proj = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value &&
                parallel::traits::is_projected_range<Proj, Rng>::value &&
                parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Compare,
                    parallel::traits::projected_range<Proj, Rng>,
                    parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend typename hpx::traits::range_iterator<Rng>::type
        tag_fallback_dispatch(hpx::ranges::stable_sort_t, Rng&& rng,
            Compare&& comp = Compare(), Proj&& proj = Proj())
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            static_assert(
                hpx::traits::is_random_access_iterator_v<iterator_type>,
                "Requires a random access iterator.");

            return hpx::parallel::v1::detail::stable_sort<iterator_type>().call(
                hpx::execution::seq, hpx::util::begin(rng), hpx::util::end(rng),
                std::forward<Compare>(comp), std::forward<Proj>(proj));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,
            typename Compare = ranges::less,
            typename Proj = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value &&
                parallel::traits::is_projected_range<Proj, Rng>::value &&
                parallel::traits::is_indirect_callable<ExPolicy, Compare,
                    parallel::traits::projected_range<Proj, Rng>,
                    parallel::traits::projected_range<Proj, Rng>
                >::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            typename hpx::traits::range_iterator<Rng>::type>::type
        tag_fallback_dispatch(hpx::ranges::stable_sort_t, ExPolicy&& policy,
            Rng&& rng, Compare&& comp = Compare(), Proj&& proj = Proj())
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            static_assert(
                hpx::traits::is_random_access_iterator_v<iterator_type>,
                "Requires a random access iterator.");

            return hpx::parallel::v1::detail::stable_sort<iterator_type>().call(
                std::forward<ExPolicy>(policy), hpx::util::begin(rng),
                hpx::util::end(rng), std::forward<Compare>(comp),
                std::forward<Proj>(proj));
        }
    } stable_sort{};
}}    // namespace hpx::ranges

#endif
