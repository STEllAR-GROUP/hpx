//  Copyright (c) 2015-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/container_algorithms/shift_left.hpp

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>

#include <hpx/algorithms/traits/projected_range.hpp>
#include <hpx/parallel/algorithms/shift_left.hpp>
#include <hpx/parallel/util/projection_identity.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace ranges {
    HPX_INLINE_CONSTEXPR_VARIABLE struct shift_left_t final
      : hpx::functional::tag_fallback<shift_left_t>
    {
    private:
        // clang-format off
        template <typename FwdIter, typename Sent, typename Size,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_iterator<FwdIter>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value
            )>
        // clang-format on
        friend FwdIter tag_fallback_dispatch(
            hpx::ranges::shift_left_t, FwdIter first, Sent last, Size n)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::shift_left<FwdIter>().call(
                hpx::execution::seq, first, last, n);
        }

        // clang-format off
        template <typename ExPolicy, typename FwdIter, typename Sent,
            typename Size,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_iterator<FwdIter>::value &&
                hpx::traits::is_sentinel_for<Sent, FwdIter>::value
            )>
        // clang-format on
        friend typename hpx::parallel::util::detail::algorithm_result<ExPolicy,
            FwdIter>::type
        tag_fallback_dispatch(hpx::ranges::shift_left_t, ExPolicy&& policy,
            FwdIter first, Sent last, Size n)
        {
            static_assert(hpx::traits::is_forward_iterator<FwdIter>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::shift_left<FwdIter>().call(
                std::forward<ExPolicy>(policy), first, last, n);
        }

        // clang-format off
        template <typename Rng, typename Size,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend typename hpx::traits::range_traits<Rng>::iterator_type
        tag_fallback_dispatch(hpx::ranges::shift_left_t, Rng&& rng, Size n)
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            static_assert(
                hpx::traits::is_forward_iterator<iterator_type>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::shift_left<iterator_type>().call(
                hpx::execution::seq, std::begin(rng), std::end(rng), n);
        }

        // clang-format off
        template <typename ExPolicy, typename Rng,  typename Size,
            HPX_CONCEPT_REQUIRES_(
                hpx::is_execution_policy<ExPolicy>::value &&
                hpx::traits::is_range<Rng>::value
            )>
        // clang-format on
        friend typename parallel::util::detail::algorithm_result<ExPolicy,
            typename hpx::traits::range_traits<Rng>::iterator_type>::type
        tag_fallback_dispatch(
            hpx::ranges::shift_left_t, ExPolicy&& policy, Rng&& rng, Size n)
        {
            using iterator_type =
                typename hpx::traits::range_traits<Rng>::iterator_type;

            static_assert(
                hpx::traits::is_forward_iterator<iterator_type>::value,
                "Requires at least forward iterator.");

            return hpx::parallel::v1::detail::shift_left<iterator_type>().call(
                std::forward<ExPolicy>(policy), std::begin(rng), std::end(rng),
                n);
        }
    } shift_left{};
}}    // namespace hpx::ranges
