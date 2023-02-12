//  Copyright (c) 2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/concepts/concepts.hpp>
#include <hpx/execution/algorithms/detail/predicates.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_sentinel_for.hpp>
#include <hpx/parallel/algorithms/detail/advance_to_sentinel.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>

namespace hpx::ranges {

    ///////////////////////////////////////////////////////////////////////////
    // clang-format off
    template <typename Iter,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_input_iterator_v<Iter> ||
            hpx::traits::is_output_iterator_v<Iter>
        )>
    // clang-format on
    constexpr Iter next(
        Iter first, hpx::traits::iter_difference_t<Iter> dist = 1)
    {
        std::advance(first, dist);
        return first;
    }

    // clang-format off
    template <typename Iter, typename Sent,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_sentinel_for_v<Sent, Iter> &&
            (hpx::traits::is_input_iterator_v<Iter> ||
            hpx::traits::is_output_iterator_v<Iter>)
        )>
    // clang-format on
    constexpr Iter next(Iter first, Sent bound)
    {
        return hpx::parallel::detail::advance_to_sentinel(first, bound);
    }

    // clang-format off
    template <typename Iter, typename Sent,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_sentinel_for_v<Sent, Iter> &&
            (hpx::traits::is_input_iterator_v<Iter> ||
            hpx::traits::is_output_iterator_v<Iter>)
        )>
    // clang-format on
    constexpr Iter next(
        Iter first, hpx::traits::iter_difference_t<Iter> n, Sent bound)
    {
        if constexpr (hpx::traits::is_sized_sentinel_for_v<Sent, Iter> &&
            hpx::traits::is_random_access_iterator_v<Iter>)
        {
            if (hpx::parallel::detail::distance(first, bound) <
                static_cast<std::size_t>(n))
            {
                return hpx::parallel::detail::advance_to_sentinel(first, bound);
            }

            hpx::parallel::detail::advance(first, n);
            return first;
        }
        else
        {
            while (n > 0 || first != bound)
            {
                --n;
                ++first;
            }
            return first;
        }
    }
}    // namespace hpx::ranges
