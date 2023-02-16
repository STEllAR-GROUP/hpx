//  Copyright (c) 2007-2022 Hartmut Kaiser
//  Copyright (c) 2016 Agustin Berge
//  Copyright (c) 2017 Anton Bikineev
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/has_member_xxx.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>

namespace hpx::traits::detail {

    ///////////////////////////////////////////////////////////////////////
    // not every random access sequence is reservable
    // so we need an explicit trait to determine this
    HPX_HAS_MEMBER_XXX_TRAIT_DEF(reserve)

    template <typename Range>
    using is_reservable = std::integral_constant<bool,
        is_range_v<std::decay_t<Range>> && has_reserve_v<std::decay_t<Range>>>;

    template <typename Range>
    inline constexpr bool is_reservable_v = is_reservable<Range>::value;

    ///////////////////////////////////////////////////////////////////////
    template <typename Container>
    HPX_FORCEINLINE void reserve_if_reservable(Container& v, std::size_t n)
    {
        if constexpr (is_reservable_v<Container>)
        {
            v.reserve(n);
        }
    }

    ///////////////////////////////////////////////////////////////////////
    // Reserve sufficient space in the given vector if the underlying
    // iterator type of the given range allow calculating the size in O(1).
    template <typename Container, typename Range>
    HPX_FORCEINLINE void reserve_if_random_access_by_range(
        Container& v, Range const& r)
    {
        using iterator_type = typename range_traits<Range>::iterator_type;
        if constexpr (is_random_access_iterator_v<iterator_type> &&
            is_reservable_v<Container>)
        {
            v.reserve(hpx::util::size(r));
        }
    }

    template <typename Container, typename Iterator>
    HPX_FORCEINLINE void reserve_if_random_access_by_range(
        Container& v, Iterator begin, Iterator end)
    {
        if constexpr (is_random_access_iterator_v<Iterator> &&
            is_reservable_v<Container>)
        {
            v.reserve(std::distance(begin, end));
        }
    }
}    // namespace hpx::traits::detail
