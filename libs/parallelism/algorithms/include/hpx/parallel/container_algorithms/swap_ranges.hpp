//  Copyright (c) 2015-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/sort.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>

#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/parallel/algorithms/swap_ranges.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace ranges {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iter1, typename Iter2>
    using swap_ranges_result = hpx::parallel::util::in_in_result<Iter1, Iter2>;

    ///////////////////////////////////////////////////////////////////////////
    // DPO for hpx::ranges::swap_ranges
    HPX_INLINE_CONSTEXPR_VARIABLE struct swap_ranges_t final
      : hpx::functional::tag_fallback<swap_ranges_t>
    {
    private:
        // clang-format off
        template <typename FwdIter1, typename Sent1, typename FwdIter2,
            typename Sent2,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_sentinel_for<Sent1, FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::traits::is_sentinel_for<Sent2, FwdIter2>::value
            )>
        // clang-format on
        friend swap_ranges_result<FwdIter1, FwdIter2> tag_fallback_dispatch(
            hpx::ranges::swap_ranges_t, FwdIter1 first1, Sent1 last1,
            FwdIter2 first2, Sent2 last2)
        {
            static_assert(hpx::traits::is_input_iterator<FwdIter1>::value,
                "Requires at least input iterator.");
            static_assert(hpx::traits::is_input_iterator<FwdIter2>::value,
                "Requires at least input iterator.");

            return hpx::parallel::v1::detail::swap_ranges<
                swap_ranges_result<FwdIter1, FwdIter2>>()
                .call(hpx::execution::seq, first1, last1, first2, last2);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename Sent1,
            typename FwdIter2, typename Sent2,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter1>::value &&
                hpx::traits::is_sentinel_for<Sent1, FwdIter1>::value &&
                hpx::traits::is_iterator<FwdIter2>::value &&
                hpx::traits::is_sentinel_for<Sent2, FwdIter2>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            swap_ranges_result<FwdIter1, FwdIter2>>::type
        tag_fallback_dispatch(hpx::ranges::swap_ranges_t, ExPolicy&& policy,
            FwdIter1 first1, Sent1 last1, FwdIter2 first2, Sent2 last2)
        {
            static_assert(hpx::traits::is_input_iterator<FwdIter1>::value,
                "Requires at least input iterator.");
            static_assert(hpx::traits::is_input_iterator<FwdIter2>::value,
                "Requires at least input iterator.");

            return hpx::parallel::v1::detail::swap_ranges<
                swap_ranges_result<FwdIter1, FwdIter2>>()
                .call(std::forward<ExPolicy>(policy), first1, last1, first2,
                    last2);
        }

        // clang-format off
        template <typename Rng1, typename Rng2,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng1>::value &&
                hpx::traits::is_range<Rng2>::value
            )>
        // clang-format on
        friend swap_ranges_result<
            typename hpx::traits::range_traits<Rng1>::iterator_type,
            typename hpx::traits::range_traits<Rng2>::iterator_type>
        tag_fallback_dispatch(
            hpx::ranges::swap_ranges_t, Rng1&& rng1, Rng2&& rng2)
        {
            using iterator_type1 =
                typename hpx::traits::range_traits<Rng1>::iterator_type;
            using iterator_type2 =
                typename hpx::traits::range_traits<Rng2>::iterator_type;

            static_assert(hpx::traits::is_input_iterator<iterator_type1>::value,
                "Requires at least input iterator.");
            static_assert(hpx::traits::is_input_iterator<iterator_type2>::value,
                "Requires at least input iterator.");

            return hpx::parallel::v1::detail::swap_ranges<
                swap_ranges_result<iterator_type1, iterator_type2>>()
                .call(hpx::execution::seq, std::begin(rng1), std::end(rng1),
                    std::begin(rng2), std::end(rng2));
        }

        // clang-format off
        template <typename ExPolicy, typename Rng1, typename Rng2,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng1>::value &&
                hpx::traits::is_range<Rng2>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            swap_ranges_result<
                typename hpx::traits::range_traits<Rng1>::iterator_type,
                typename hpx::traits::range_traits<Rng2>::iterator_type>>::type
        tag_fallback_dispatch(hpx::ranges::swap_ranges_t, ExPolicy&& policy,
            Rng1&& rng1, Rng2&& rng2)
        {
            using iterator_type1 =
                typename hpx::traits::range_traits<Rng1>::iterator_type;
            using iterator_type2 =
                typename hpx::traits::range_traits<Rng2>::iterator_type;

            static_assert(hpx::traits::is_input_iterator<iterator_type1>::value,
                "Requires at least input iterator.");
            static_assert(hpx::traits::is_input_iterator<iterator_type2>::value,
                "Requires at least input iterator.");

            return hpx::parallel::v1::detail::swap_ranges<
                swap_ranges_result<iterator_type1, iterator_type2>>()
                .call(std::forward<ExPolicy>(policy), std::begin(rng1),
                    std::end(rng1), std::begin(rng2), std::end(rng2));
        }
    } swap_ranges{};
}}    // namespace hpx::ranges
