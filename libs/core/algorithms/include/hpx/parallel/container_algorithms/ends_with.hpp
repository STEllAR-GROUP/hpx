//  Copyright (c) 2015-2020 Hartmut Kaiser
//  Copyright (c) 2021 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/ends_with.hpp

#pragma once

#if defined(DOXYGEN)

namespace hpx { namespace ranges {
    // clang-format off

    /// Checks whether the second range defined by [first1, last1) matches the
    /// suffix of the first range defined by [first2, last2)
    ///
    /// \note   Complexity: Linear: at most min(N1, N2) applications of the
    ///                     predicate and both projections.
    ///
    /// \tparam Iter1       The type of the begin source iterators used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an input iterator.
    /// \tparam Sent1       The type of the end source iterators used(deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter1.
    /// \tparam Iter2       The type of the begin destination iterators used
    ///                     deduced). This iterator type must meet the
    ///                     requirements of a input iterator.
    /// \tparam Sent2       The type of the end destination iterators used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an sentinel for Iter2.
    /// \tparam Pred        The binary predicate that compares the projected
    ///                     elements.
    /// \tparam Proj1       The type of an optional projection function for
    ///                     the source range. This defaults to
    ///                     \a util::projection_identity
    /// \tparam Proj1       The type of an optional projection function for
    ///                     the destination range. This defaults to
    ///                     \a util::projection_identity
    ///
    /// \param first1       Refers to the beginning of the source range.
    /// \param last1        Sentinel value referring to the end of the source
    ///                     range.
    /// \param first2       Refers to the beginning of the destination range.
    /// \param last2        Sentinel value referring to the end of the
    ///                     destination range.
    /// \param pred         Specifies the binary predicate function
    ///                     (or function object) which will be invoked for
    ///                     comparison of the elements in the in two ranges
    ///                     projected by proj1 and proj2 respectively.
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     source range as a projection operation before the
    ///                     actual predicate \a is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     destination range as a projection operation before
    ///                     the actual predicate \a is invoked.
    ///
    /// The assignments in the parallel \a ends_with algorithm invoked
    /// without an execution policy object execute in sequential order
    /// in the calling thread.
    ///
    /// \returns  The \a ends_with algorithm returns \a bool.
    ///           The \a ends_with algorithm returns a boolean with the
    ///           value true if the second range matches the suffix of the
    ///           first range, false otherwise.
    template <typename FwdIter1, typename Sent1, typename FwdIter2,
        typename Sent2, typename Pred, typename Proj1, typename Proj2>
    bool ends_with(FwdIter1 first1, Sent1 last1, FwdIter2 first2,
        Sent2 last2, Pred&& pred, Proj1&& proj1, Proj2&& proj2);

    /// Checks whether the second range defined by [first1, last1) matches the
    /// suffix of the first range defined by [first2, last2)
    ///
    /// \note   Complexity: Linear: at most min(N1, N2) applications of the
    ///                     predicate and both projections.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam FwdIter1    The type of the begin source iterators used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an forward iterator.
    /// \tparam Sent1       The type of the end source iterators used(deduced).
    ///                     This iterator type must meet the requirements of an
    ///                     sentinel for Iter1.
    /// \tparam FwdIter2    The type of the begin destination iterators used
    ///                     deduced). This iterator type must meet the
    ///                     requirements of a forward iterator.
    /// \tparam Sent2       The type of the end destination iterators used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an sentinel for Iter2.
    /// \tparam Pred        The binary predicate that compares the projected
    ///                     elements.
    /// \tparam Proj1       The type of an optional projection function for
    ///                     the source range. This defaults to
    ///                     \a util::projection_identity
    /// \tparam Proj1       The type of an optional projection function for
    ///                     the destination range. This defaults to
    ///                     \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param first1       Refers to the beginning of the source range.
    /// \param last1        Sentinel value referring to the end of the source
    ///                     range.
    /// \param first2       Refers to the beginning of the destination range.
    /// \param last2        Sentinel value referring to the end of the
    ///                     destination range.
    /// \param pred         Specifies the binary predicate function
    ///                     (or function object) which will be invoked for
    ///                     comparison of the elements in the in two ranges
    ///                     projected by proj1 and proj2 respectively.
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     source range as a projection operation before the
    ///                     actual predicate \a is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     destination range as a projection operation before
    ///                     the actual predicate \a is invoked.
    ///
    /// The assignments in the parallel \a ends_with algorithm invoked with an
    /// execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a ends_with algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a ends_with algorithm returns a
    ///           \a hpx::future<bool> if the execution policy is of type
    ///           \a sequenced_task_policy or \a parallel_task_policy and
    ///           returns \a bool otherwise.
    ///           The \a ends_with algorithm returns a boolean with the
    ///           value true if the second range matches the suffix of the
    ///           first range, false otherwise.
    template <typename ExPolicy, typename FwdIter1, typename Sent1,
        typename FwdIter2, typename Sent2, typename Pred, typename Proj1,
        typename Proj2>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        bool>::type
    ends_with(ExPolicy&& policy, FwdIter1 first1, Sent1 last1,
        FwdIter2 first2, Sent2 last2, Pred&& pred, Proj1&& proj1,
        Proj2&& proj2);

    /// Checks whether the second range \a rng2 matches the
    /// suffix of the first range \a rng1.
    ///
    /// \note   Complexity: Linear: at most min(N1, N2) applications of the
    ///                     predicate and both projections.
    ///
    /// \tparam Rng1        The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Rng2        The type of the destination range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Pred        The binary predicate that compares the projected
    ///                     elements.
    /// \tparam Proj1       The type of an optional projection function for
    ///                     the source range. This defaults to
    ///                     \a util::projection_identity
    /// \tparam Proj1       The type of an optional projection function for
    ///                     the destination range. This defaults to
    ///                     \a util::projection_identity
    ///
    /// \param rng1         Refers to the source range.
    /// \param rng2         Refers to the destination range.
    /// \param pred         Specifies the binary predicate function
    ///                     (or function object) which will be invoked for
    ///                     comparison of the elements in the in two ranges
    ///                     projected by proj1 and proj2 respectively.
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     source range as a projection operation before the
    ///                     actual predicate \a is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     destination range as a projection operation before
    ///                     the actual predicate \a is invoked.
    ///
    /// The assignments in the parallel \a ends_with algorithm invoked
    /// without an execution policy object execute in sequential order
    /// in the calling thread.
    ///
    /// \returns  The \a ends_with algorithm returns \a bool.
    ///           The \a ends_with algorithm returns a boolean with the
    ///           value true if the second range matches the suffix of the
    ///           first range, false otherwise.
    template <typename Rng1, typename Rng2, typename Pred,
        typename Proj1, typename Proj2>
    bool ends_with(Rng1&& rng1, Rng2&& rng2, Pred&& pred,
        Proj1&& proj1, Proj2&& proj2);

    /// Checks whether the second range \a rng2 matches the
    /// suffix of the first range \a rng1.
    ///
    /// \note   Complexity: Linear: at most min(N1, N2) applications of the
    ///                     predicate and both projections.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam Rng1        The type of the source range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Rng2        The type of the destination range used (deduced).
    ///                     The iterators extracted from this range type must
    ///                     meet the requirements of an forward iterator.
    /// \tparam Pred        The binary predicate that compares the projected
    ///                     elements.
    /// \tparam Proj1       The type of an optional projection function for
    ///                     the source range. This defaults to
    ///                     \a util::projection_identity
    /// \tparam Proj1       The type of an optional projection function for
    ///                     the destination range. This defaults to
    ///                     \a util::projection_identity
    ///
    /// \param policy       The execution policy to use for the scheduling of
    ///                     the iterations.
    /// \param rng1         Refers to the source range.
    /// \param rng2         Refers to the destination range.
    /// \param pred         Specifies the binary predicate function
    ///                     (or function object) which will be invoked for
    ///                     comparison of the elements in the in two ranges
    ///                     projected by proj1 and proj2 respectively.
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     source range as a projection operation before the
    ///                     actual predicate \a is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     destination range as a projection operation before
    ///                     the actual predicate \a is invoked.
    ///
    /// The assignments in the parallel \a ends_with algorithm invoked with an
    /// execution policy object of type \a sequenced_policy
    /// execute in sequential order in the calling thread.
    ///
    /// The assignments in the parallel \a ends_with algorithm invoked with
    /// an execution policy object of type \a parallel_policy or
    /// \a parallel_task_policy are permitted to execute in an unordered
    /// fashion in unspecified threads, and indeterminately sequenced
    /// within each thread.
    ///
    /// \returns  The \a ends_with algorithm returns a
    ///           \a hpx::future<bool> if the execution policy is of type
    ///           \a sequenced_task_policy or \a parallel_task_policy and
    ///           returns \a bool otherwise.
    ///           The \a ends_with algorithm returns a boolean with the
    ///           value true if the second range matches the suffix of the
    ///           first range, false otherwise.
    template <typename ExPolicy, typename Rng1, typename Rng2,
        typename Pred, typename Proj1, typename Proj2>
    typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
        bool>::type
    ends_with(ExPolicy&& policy, Rng1&& rng1, Rng2&& rng2,
        Pred&& pred, Proj1&& proj1, Proj2&& proj2);

    // clang-format on
}}    // namespace hpx::ranges

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>

#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/parallel/algorithms/ends_with.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace ranges {

    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::ranges::copy
    HPX_INLINE_CONSTEXPR_VARIABLE struct ends_with_t final
      : hpx::functional::tag_fallback<ends_with_t>
    {
    private:
        // clang-format off
        template <typename Iter1, typename Sent1, typename Iter2, typename Sent2,
            typename Pred = ranges::equal_to,
            typename Proj1 = parallel::util::projection_identity,
            typename Proj2 = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<Iter1> &&
                hpx::traits::is_sentinel_for<Sent1, Iter1>::value &&
                hpx::traits::is_iterator_v<Iter2> &&
                hpx::traits::is_sentinel_for<Sent2, Iter2>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected<Proj1, Iter1>,
                    hpx::parallel::traits::projected<Proj2, Iter2>
                >::value
            )>
        // clang-format on
        friend bool tag_fallback_dispatch(hpx::ranges::ends_with_t,
            Iter1 first1, Sent1 last1, Iter2 first2, Sent2 last2,
            Pred&& pred = Pred(), Proj1&& proj1 = Proj1(),
            Proj2&& proj2 = Proj2())
        {
            static_assert(hpx::traits::is_input_iterator_v<Iter1>,
                "Required at least input iterator.");

            static_assert(hpx::traits::is_input_iterator_v<Iter2>,
                "Required at least input iterator.");

            return hpx::parallel::v1::detail::ends_with().call(
                hpx::execution::seq, first1, last1, first2, last2,
                std::forward<Pred>(pred), std::forward<Proj1>(proj1),
                std::forward<Proj2>(proj2));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename Sent1, typename FwdIter2,
            typename Sent2, typename Pred = ranges::equal_to,
            typename Proj1 = parallel::util::projection_identity,
            typename Proj2 = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator_v<FwdIter1> &&
                hpx::traits::is_sentinel_for<Sent1, FwdIter1>::value &&
                hpx::traits::is_iterator_v<FwdIter2> &&
                hpx::traits::is_sentinel_for<Sent2, FwdIter2>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    ExPolicy, Pred,
                    hpx::parallel::traits::projected<Proj1, FwdIter1>,
                    hpx::parallel::traits::projected<Proj2, FwdIter2>
                >::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_dispatch(hpx::ranges::ends_with_t, ExPolicy&& policy,
            FwdIter1 first1, Sent1 last1, FwdIter2 first2, Sent2 last2,
            Pred&& pred = Pred(), Proj1&& proj1 = Proj1(),
            Proj2&& proj2 = Proj2())
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter1>,
                "Required at least forward iterator.");

            static_assert(hpx::traits::is_forward_iterator_v<FwdIter2>,
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::ends_with().call(
                std::forward<ExPolicy>(policy), first1, last1, first2, last2,
                std::forward<Pred>(pred), std::forward<Proj1>(proj1),
                std::forward<Proj2>(proj2));
        }

        // clang-format off
        template <typename Rng1, typename Rng2,
            typename Pred = ranges::equal_to,
            typename Proj1 = parallel::util::projection_identity,
            typename Proj2 = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng1>::value &&
                hpx::parallel::traits::is_projected_range<Proj1, Rng1>::value &&
                hpx::traits::is_range<Rng2>::value &&
                hpx::parallel::traits::is_projected_range<Proj2, Rng2>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected<Proj1,
                        typename hpx::traits::range_traits<Rng1>::iterator_type>,
                    hpx::parallel::traits::projected<Proj2,
                        typename hpx::traits::range_traits<Rng2>::iterator_type>
                >::value
            )>
        // clang-format on
        friend bool tag_fallback_dispatch(hpx::ranges::ends_with_t, Rng1&& rng1,
            Rng2&& rng2, Pred&& pred = Pred(), Proj1&& proj1 = Proj1(),
            Proj2&& proj2 = Proj2())
        {
            using iterator_type1 =
                typename hpx::traits::range_iterator<Rng1>::type;

            using iterator_type2 =
                typename hpx::traits::range_iterator<Rng2>::type;

            static_assert(hpx::traits::is_input_iterator_v<iterator_type1>,
                "Required at least input iterator.");

            static_assert(hpx::traits::is_input_iterator_v<iterator_type2>,
                "Required at least input iterator.");

            return hpx::parallel::v1::detail::ends_with().call(
                hpx::execution::seq, hpx::util::begin(rng1),
                hpx::util::end(rng1), hpx::util::begin(rng2),
                hpx::util::end(rng2), std::forward<Pred>(pred),
                std::forward<Proj1>(proj1), std::forward<Proj2>(proj2));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng1, typename Rng2,
            typename Pred = ranges::equal_to,
            typename Proj1 = parallel::util::projection_identity,
            typename Proj2 = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng1>::value &&
                hpx::parallel::traits::is_projected_range<Proj1, Rng1>::value &&
                hpx::traits::is_range<Rng2>::value &&
                hpx::parallel::traits::is_projected_range<Proj2, Rng2>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    ExPolicy, Pred,
                    hpx::parallel::traits::projected<Proj1,
                        typename hpx::traits::range_traits<Rng1>::iterator_type>,
                    hpx::parallel::traits::projected<Proj2,
                        typename hpx::traits::range_traits<Rng2>::iterator_type>
                >::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_dispatch(hpx::ranges::ends_with_t, ExPolicy&& policy,
            Rng1&& rng1, Rng2&& rng2, Pred&& pred = Pred(),
            Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2())
        {
            using iterator_type1 =
                typename hpx::traits::range_iterator<Rng1>::type;

            using iterator_type2 =
                typename hpx::traits::range_iterator<Rng2>::type;

            static_assert(hpx::traits::is_forward_iterator_v<iterator_type1>,
                "Required at least forward iterator.");

            static_assert(hpx::traits::is_forward_iterator_v<iterator_type2>,
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::ends_with().call(
                std::forward<ExPolicy>(policy), hpx::util::begin(rng1),
                hpx::util::end(rng1), hpx::util::begin(rng2),
                hpx::util::end(rng2), std::forward<Pred>(pred),
                std::forward<Proj1>(proj1), std::forward<Proj2>(proj2));
        }
    } ends_with{};

}}    // namespace hpx::ranges

#endif    // DOXYGEN
