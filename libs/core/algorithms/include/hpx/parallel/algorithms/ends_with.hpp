//  Copyright (c) 2020 ETH Zurich
//  Copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/algorithms/starts_with.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/executors/execution_policy.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/parallel/algorithms/detail/dispatch.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>
#include <hpx/parallel/algorithms/equal.hpp>
#include <hpx/parallel/algorithms/mismatch.hpp>
#include <hpx/parallel/util/detail/algorithm_result.hpp>
#include <hpx/parallel/util/invoke_projected.hpp>
#include <hpx/parallel/util/projection_identity.hpp>
#include <hpx/parallel/util/result_types.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx { namespace parallel { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    // ends_with
    namespace detail {
        /// \cond NOINTERNAL
        struct ends_with : public detail::algorithm<ends_with, bool>
        {
            ends_with()
              : ends_with::algorithm("ends_with")
            {
            }

            template <typename ExPolicy, typename Iter1, typename Sent1,
                typename Iter2, typename Sent2, typename Pred, typename Proj1,
                typename Proj2>
            static bool sequential(ExPolicy, Iter1 first1, Sent1 last1,
                Iter2 first2, Sent2 last2, Pred&& pred, Proj1&& proj1,
                Proj2&& proj2)
            {
                const auto drop = detail::distance(first1, last1) -
                    detail::distance(first2, last2);

                if (drop < 0)
                    return false;

                return hpx::parallel::v1::detail::equal_binary().call(
                    hpx::execution::seq, std::next(std::move(first1), drop),
                    std::move(last1), std::move(first2), std::move(last2),
                    std::forward<Pred>(pred), std::forward<Proj1>(proj1),
                    std::forward<Proj2>(proj2));
            }

            template <typename ExPolicy, typename FwdIter1, typename Sent1,
                typename FwdIter2, typename Sent2, typename Pred,
                typename Proj1, typename Proj2>
            static typename util::detail::algorithm_result<ExPolicy, bool>::type
            parallel(ExPolicy&& policy, FwdIter1 first1, Sent1 last1,
                FwdIter2 first2, Sent2 last2, Pred&& pred, Proj1&& proj1,
                Proj2&& proj2)
            {
                const auto drop = detail::distance(first1, last1) -
                    detail::distance(first2, last2);

                if (drop < 0)
                {
                    return util::detail::algorithm_result<ExPolicy, bool>::get(
                        false);
                }

                return hpx::parallel::v1::detail::equal_binary().call(
                    std::forward<ExPolicy>(policy),
                    std::next(std::move(first1), drop), std::move(last1),
                    std::move(first2), std::move(last2),
                    std::forward<Pred>(pred), std::forward<Proj1>(proj1),
                    std::forward<Proj2>(proj2));
            }
        };
        /// \endcond
    }    // namespace detail
}}}      // namespace hpx::parallel::v1

namespace hpx {

    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::ends_with
    HPX_INLINE_CONSTEXPR_VARIABLE struct ends_with_t final
      : hpx::functional::tag_fallback<ends_with_t>
    {
    private:
        // clang-format off
        template <typename InIter1, typename InIter2,
            typename Pred = hpx::parallel::v1::detail::equal_to,
            typename Proj1 = parallel::util::projection_identity,
            typename Proj2 = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<InIter1>::value &&
                hpx::traits::is_iterator<InIter2>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    hpx::execution::sequenced_policy, Pred,
                    hpx::parallel::traits::projected<Proj1, InIter1>,
                    hpx::parallel::traits::projected<Proj2, InIter2>
                >::value
            )>
        // clang-format on
        friend bool tag_fallback_dispatch(hpx::ends_with_t, InIter1 first1,
            InIter1 last1, InIter2 first2, InIter2 last2, Pred&& pred = Pred(),
            Proj1&& proj1 = Proj1(), Proj2&& proj2 = Proj2())
        {
            static_assert(hpx::traits::is_input_iterator<InIter1>::value,
                "Required at least input iterator.");

            static_assert(hpx::traits::is_input_iterator<InIter2>::value,
                "Required at least input iterator.");

            return hpx::parallel::v1::detail::ends_with().call(
                hpx::execution::seq, first1, last1, first2, last2,
                std::forward<Pred>(pred), std::forward<Proj1>(proj1),
                std::forward<Proj2>(proj2));
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename FwdIter2,
            typename Pred = ranges::equal_to,
            typename Proj1 = parallel::util::projection_identity,
            typename Proj2 = parallel::util::projection_identity,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::parallel::traits::is_indirect_callable<
                    ExPolicy, Pred,
                    hpx::parallel::traits::projected<Proj1, FwdIter1>,
                    hpx::parallel::traits::projected<Proj2, FwdIter2>
                >::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            bool>::type
        tag_fallback_dispatch(hpx::ends_with_t, ExPolicy&& policy,
            FwdIter1 first1, FwdIter1 last1, FwdIter2 first2, FwdIter2 last2,
            Pred&& pred = Pred(), Proj1&& proj1 = Proj1(),
            Proj2&& proj2 = Proj2())
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter1>::value,
                "Required at least forward iterator.");

            static_assert(hpx::traits::is_forward_iterator<FwdIter2>::value,
                "Required at least forward iterator.");

            return hpx::parallel::v1::detail::ends_with().call(
                std::forward<ExPolicy>(policy), first1, last1, first2, last2,
                std::forward<Pred>(pred), std::forward<Proj1>(proj1),
                std::forward<Proj2>(proj2));
        }
    } ends_with{};

}    // namespace hpx
