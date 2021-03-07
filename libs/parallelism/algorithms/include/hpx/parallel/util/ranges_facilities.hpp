//  Copyright (c) 2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/concepts/concepts.hpp>

#include <hpx/iterator_support/counting_iterator.hpp>
#include <hpx/iterator_support/traits/is_sentinel_for.hpp>
#include <hpx/parallel/algorithms/detail/advance_to_sentinel.hpp>
#include <hpx/parallel/algorithms/detail/distance.hpp>

#include <iterator>

namespace hpx { namespace ranges {

    // clang-format off
    template <typename Iter, 
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_input_iterator<Iter>::value ||
            hpx::traits::is_output_iterator<Iter>::value
        )>
    // clang-format on
    constexpr inline Iter next(Iter first,
        typename std::iterator_traits<Iter>::difference_type dist = 1)
    {
        std::advance(first, dist);
        return first;
    }

    // clang-format off
    template <typename Iter, typename Sent,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_sentinel_for<Sent, Iter>::value &&
            (hpx::traits::is_input_iterator<Iter>::value ||
            hpx::traits::is_output_iterator<Iter>::value)
        )>
    // clang-format on
    constexpr inline Iter next(Iter first, Sent bound)
    {
        return hpx::parallel::v1::detail::advance_to_sentinel(first, bound);
    }

    // clang-format off
    template <typename Iter, typename Sent,
        HPX_CONCEPT_REQUIRES_(
            hpx::traits::is_sentinel_for<Sent, Iter>::value &&
            (hpx::traits::is_input_iterator<Iter>::value ||
            hpx::traits::is_output_iterator<Iter>::value)
        )>
    // clang-format on
    constexpr inline Iter next(Iter first,
        typename std::iterator_traits<Iter>::difference_type n, Sent bound)
    {
        if (hpx::parallel::v1::detail::distance(first, bound) < n)
        {
            return hpx::parallel::v1::detail::advance_to_sentinel(first, bound);
        }
        else
        {
            std::advance(first, n);
            return first;
        }
    }

}}    // namespace hpx::ranges
