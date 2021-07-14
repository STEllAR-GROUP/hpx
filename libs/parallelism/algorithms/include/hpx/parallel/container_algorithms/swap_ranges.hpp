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
    HPX_INLINE_CONSTEXPR_VARIABLE struct swap_ranges_t final
      : hpx::functional::tag_fallback<swap_ranges_t>
    {
    private:
        // clang-format off
        template <typename FwdIter1, typename Sent1, typename FwdIter2,
            typename Sent2,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_forward_iterator<FwdIter1>::value &&
                hpx::traits::is_sentinel_for<Sent1, FwdIter1>::value &&
                hpx::traits::is_forward_iterator<FwdIter2>::value &&
                hpx::traits::is_sentinel_for<Sent2, FwdIter2>::value
            )>
        // clang-format on
        friend FwdIter2 tag_fallback_dispatch(hpx::ranges::swap_ranges_t,
            FwdIter1 first1, Sent1 last1, FwdIter2 first2, Sent2 last2)
        {
            static_assert(hpx::traits::is_input_iterator<FwdIter1>::value,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_output_iterator<FwdIter2>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::swap_ranges<FwdIter2>().call(
                hpx::execution::seq, first1, last1, first2, last2);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter1, typename Sent1,
            typename FwdIter2, typename Sent2,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_forward_iterator<FwdIter1>::value &&
                hpx::traits::is_sentinel_for<Sent1, FwdIter1>::value &&
                hpx::traits::is_forward_iterator<FwdIter2>::value &&
                hpx::traits::is_sentinel_for<Sent2, FwdIter2>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter2>::type
        tag_fallback_dispatch(hpx::ranges::swap_ranges_t, ExPolicy&& policy,
            FwdIter1 first1, Sent1 last1, FwdIter2 first2, Sent2 last2)
        {
            static_assert(hpx::traits::is_input_iterator<FwdIter1>::value,
                "Requires at least forward iterator.");
            static_assert(hpx::traits::is_output_iterator<FwdIter2>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::swap_ranges<FwdIter2>().call(
                std::forward<ExPolicy>(policy), first1, last1, first2, last2);
        }

        // clang-format off
        template <typename Rng1, typename Rng2,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng1>::value &&
                hpx::traits::is_range<Rng2>::value
            )>
        // clang-format on
        friend typename hpx::traits::range_traits<Rng2>::iterator_type
        tag_fallback_dispatch(
            hpx::ranges::swap_ranges_t, Rng1&& rng1, Rng2&& rng2)
        {
            using iterator_type1 =
                typename hpx::traits::range_traits<Rng1>::iterator_type;
            using iterator_type2 =
                typename hpx::traits::range_traits<Rng2>::iterator_type;

            static_assert(hpx::traits::is_input_iterator<iterator_type1>::value,
                "Requires at least forward iterator.");
            static_assert(
                hpx::traits::is_forward_iterator<iterator_type2>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::swap_ranges<iterator_type2>()
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
            typename hpx::traits::range_traits<Rng2>::iterator_type>::type
        tag_fallback_dispatch(hpx::ranges::swap_ranges_t, ExPolicy&& policy,
            Rng1&& rng1, Rng2&& rng2)
        {
            using iterator_type1 =
                typename hpx::traits::range_traits<Rng1>::iterator_type;
            using iterator_type2 =
                typename hpx::traits::range_traits<Rng2>::iterator_type;

            static_assert(hpx::traits::is_input_iterator<iterator_type1>::value,
                "Requires at least forward iterator.");
            static_assert(
                hpx::traits::is_forward_iterator<iterator_type2>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::swap_ranges<iterator_type2>()
                .call(std::forward<ExPolicy>(policy), std::begin(rng1),
                    std::end(rng1), std::begin(rng2), std::end(rng2));
        }
    } swap_ranges{};
}}    // namespace hpx::ranges
