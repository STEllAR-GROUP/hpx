//  Copyright (c) 2022 Bhumit Attarde
//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/starts_with.hpp

#pragma once

#if defined(DOXYGEN)

namespace hpx {
    // clang-format off

    /// Checks whether the second range defined by [first1, last1) matches the
    /// prefix of the first range defined by [first2, last2)
    ///
    /// \note   Complexity: Linear: at most min(N1, N2) applications of the
    ///                     predicate and both projections.
    ///
    /// \tparam InIter1     The type of the source iterators used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an input iterator.
    /// \tparam InIter2     The type of the destination iterators used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an input iterator.
    /// \tparam Pred        The binary predicate that compares the projected
    ///                     elements. This defaults to
    ///                     \a hpx::parallel::detail::equal_to.
    /// \tparam Proj1       The type of an optional projection function for
    ///                     the source range. This defaults to
    ///                     \a hpx::identity.
    /// \tparam Proj2       The type of an optional projection function for
    ///                     the destination range. This defaults to
    ///                     \a hpx::identity.
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
    ///                     projected by \a proj1 and \a proj2 respectively.
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     source range as a projection operation before the
    ///                     actual predicate \a pred is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     destination range as a projection operation before
    ///                     the actual predicate \a pred is invoked.
    ///
    /// The assignments in the parallel \a starts_with algorithm invoked
    /// without an execution policy object execute in sequential order
    /// in the calling thread.
    ///
    /// \returns  The \a starts_with algorithm returns \a bool.
    ///           The \a starts_with algorithm returns a boolean with the
    ///           value true if the second range matches the prefix of the
    ///           first range, false otherwise.
    template <typename InIter1, typename InIter2,
        typename Pred = hpx::parallel::detail::equal_to,
        typename Proj1 = hpx::identity,
        typename Proj2 = hpx::identity>
    bool starts_with( InIter1 first1, InIter1 last1, InIter2 first2,
        InIter2 last2, Pred&& pred = Pred(), Proj1&& proj1 = Proj1(),
        Proj2&& proj2 = Proj2());

    /// Checks whether the second range defined by [first1, last1) matches the
    /// prefix of the first range defined by [first2, last2).
    /// Executed according to the policy.
    ///
    /// \note   Complexity: Linear: at most min(N1, N2) applications of the
    ///                     predicate and both projections.
    ///
    /// \tparam ExPolicy    The type of the execution policy to use (deduced).
    ///                     It describes the manner in which the execution
    ///                     of the algorithm may be parallelized and the manner
    ///                     in which it executes the assignments.
    /// \tparam InIter1     The type of the source iterators used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an input iterator.
    /// \tparam InIter2     The type of the destination iterators used
    ///                     (deduced). This iterator type must meet the
    ///                     requirements of an input iterator.
    /// \tparam Pred        The binary predicate that compares the projected
    ///                     elements. This defaults to
    ///                     \a hpx::parallel::detail::equal_to.
    /// \tparam Proj1       The type of an optional projection function for
    ///                     the source range. This defaults to
    ///                     \a hpx::identity.
    /// \tparam Proj2       The type of an optional projection function for
    ///                     the destination range. This defaults to
    ///                     \a hpx::identity.
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
    ///                     projected by \a proj1 and \a proj2 respectively.
    /// \param proj1        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     source range as a projection operation before the
    ///                     actual predicate \a pred is invoked.
    /// \param proj2        Specifies the function (or function object) which
    ///                     will be invoked for each of the elements in the
    ///                     destination range as a projection operation before
    ///                     the actual predicate \a pred is invoked.
    ///
    /// The assignments in the parallel \a starts_with algorithm invoked
    /// without an execution policy object execute in sequential order
    /// in the calling thread.
    ///
    /// \returns  The \a starts_with algorithm returns a
    ///           \a hpx::future<bool> if the execution policy is of type
    ///           \a sequenced_task_policy or \a parallel_task_policy and
    ///           returns \a bool otherwise.
    ///           The \a starts_with algorithm returns a boolean with the
    ///           value true if the second range matches the prefix of the
    ///           first range, false otherwise.
    template <typename ExPolicy, typename InIter1, typename InIter2,
        typename Pred = hpx::parallel::detail::equal_to,
        typename Proj1 = hpx::identity,
        typename Proj2 = hpx::identity>
    hpx::parallel::util::detail::algorithm_result_t<ExPolicy, bool>
    starts_with(ExPolicy&& policy, InIter1 first1, InIter1 last1,
        InIter2 first2,InIter2 last2, Pred&& pred = Pred(),
        Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2());

    // clang-format on
}    // namespace hpx

#else    // DOXYGEN

#include <hpx/config.hpp>
#include <hpx/algorithms/traits/projected.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/algorithms/mismatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/result_types.hpp>
#include <hpx/type_support/identity.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx::parallel {

    ///////////////////////////////////////////////////////////////////////////
    // starts_with
    namespace detail {
        template <typename FwdIter1, typename FwdIter2, typename Sent2>
        constexpr bool get_starts_with_result(
            util::in_in_result<FwdIter1, FwdIter2>&& p, Sent2 last2)
        {
            return p.in2 == last2;
        }

        template <typename FwdIter1, typename FwdIter2, typename Sent2>
        hpx::future<bool> get_starts_with_result(
            hpx::future<util::in_in_result<FwdIter1, FwdIter2>>&& f,
            Sent2 last2)
        {
            return hpx::make_future<bool>(HPX_MOVE(f),
                [last2 = HPX_MOVE(last2)](
                    util::in_in_result<FwdIter1, FwdIter2>&& p) -> bool {
                    return p.in2 == last2;
                });
        }

        struct starts_with : public algorithm<starts_with, bool>
        {
            constexpr starts_with() noexcept
              : algorithm("starts_with")
            {
            }

            template <typename ExPolicy, typename Iter1, typename Sent1,
                typename Iter2, typename Sent2, typename Pred, typename Proj1,
                typename Proj2>
            static constexpr bool sequential(ExPolicy, Iter1 first1,
                Sent1 last1, Iter2 first2, Sent2 last2, Pred&& pred,
                Proj1&& proj1, Proj2&& proj2)
            {
                auto dist1 = detail::distance(first1, last1);
                auto dist2 = detail::distance(first2, last2);

                if (dist1 < dist2)
                {
                    return false;
                }

                auto end_first = std::next(first1, dist2);
                return detail::get_starts_with_result<Iter1, Iter2, Sent2>(
                    hpx::parallel::detail::mismatch_binary<
                        util::in_in_result<Iter1, Iter2>>()
                        .call(hpx::execution::seq, HPX_MOVE(first1),
                            HPX_MOVE(end_first), HPX_MOVE(first2), last2,
                            HPX_FORWARD(Pred, pred), HPX_FORWARD(Proj1, proj1),
                            HPX_FORWARD(Proj2, proj2)),
                    last2);
            }

            template <typename ExPolicy, typename FwdIter1, typename Sent1,
                typename FwdIter2, typename Sent2, typename Pred,
                typename Proj1, typename Proj2>
            static util::detail::algorithm_result_t<ExPolicy, bool> parallel(
                ExPolicy&& policy, FwdIter1 first1, Sent1 last1,
                FwdIter2 first2, Sent2 last2, Pred&& pred, Proj1&& proj1,
                Proj2&& proj2)
            {
                auto dist1 = detail::distance(first1, last1);
                auto dist2 = detail::distance(first2, last2);

                if (dist1 < dist2)
                {
                    return util::detail::algorithm_result<ExPolicy, bool>::get(
                        false);
                }

                auto end_first = std::next(first1, dist2);
                return detail::get_starts_with_result<FwdIter1, FwdIter2,
                    Sent2>(
                    detail::mismatch_binary<
                        util::in_in_result<FwdIter1, FwdIter2>>()
                        .call(HPX_FORWARD(ExPolicy, policy), first1, end_first,
                            first2, last2, HPX_FORWARD(Pred, pred),
                            HPX_FORWARD(Proj1, proj1),
                            HPX_FORWARD(Proj2, proj2)),
                    last2);
            }
        };
        /// \endcond
    }    // namespace detail
}    // namespace hpx::parallel

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // CPO for hpx::starts_with
    inline constexpr struct starts_with_t final
      : hpx::functional::detail::tag_fallback<starts_with_t>
    {
    private:
        // clang-format off
        template <typename InIter1, typename InIter2,
            typename Pred = hpx::parallel::detail::equal_to,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator_v<InIter1> &&
                hpx::traits::is_iterator_v<InIter2> &&
                hpx::is_invocable_v<Pred,
                    hpx::traits::iter_value_t<InIter1>,
                    hpx::traits::iter_value_t<InIter2>
                >
            )>
        // clang-format on
        friend bool tag_fallback_invoke(hpx::starts_with_t, InIter1 first1,
            InIter1 last1, InIter2 first2, InIter2 last2, Pred pred = Pred())
        {
            static_assert(hpx::traits::is_input_iterator_v<InIter1>,
                "Required at least input iterator.");

            static_assert(hpx::traits::is_input_iterator_v<InIter2>,
                "Required at least input iterator.");

            return hpx::parallel::detail::starts_with().call(
                hpx::execution::seq, first1, last1, first2, last2,
                HPX_MOVE(pred), hpx::identity_v, hpx::identity_v);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Pred = ranges::equal_to,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy_v<ExPolicy> &&
                hpx::traits::is_iterator_v<FwdIter1> &&
                hpx::traits::is_iterator_v<FwdIter2> &&
                hpx::is_invocable_v<Pred,
                    hpx::traits::iter_value_t<FwdIter1>,
                    hpx::traits::iter_value_t<FwdIter2>
                >
            )>
        // clang-format on
        friend parallel::util::detail::algorithm_result_t<ExPolicy, bool>
        tag_fallback_invoke(hpx::starts_with_t, ExPolicy&& policy,
            FwdIter1 first1, FwdIter1 last1, FwdIter2 first2, FwdIter2 last2,
            Pred pred = Pred())
        {
            static_assert(hpx::traits::is_forward_iterator_v<FwdIter1>,
                "Required at least forward iterator.");

            static_assert(hpx::traits::is_forward_iterator_v<FwdIter2>,
                "Required at least forward iterator.");

            return hpx::parallel::detail::starts_with().call(
                HPX_FORWARD(ExPolicy, policy), first1, last1, first2, last2,
                HPX_MOVE(pred), hpx::identity_v, hpx::identity_v);
        }
    } starts_with{};
}    // namespace hpx

#endif
