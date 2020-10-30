//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/iterator_support/traits/is_sentinel_for.hpp>

#include <iterator>
#include <type_traits>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {

    // Generic implementation for advancing a given iterator to its sentinel
    template <typename Iter, typename Sent>
    constexpr inline Iter advance_to_sentinel(
        Iter first, Sent last, std::false_type)
    {
        for (/**/; first != last; ++first)
        {
            /**/;
        }
        return first;
    }

    template <typename Iter, typename Sent>
    constexpr inline Iter advance_to_sentinel(
        Iter first, Sent last, std::true_type)
    {
        return first + (last - first);
    }

    template <typename Iter, typename Sent>
    constexpr inline Iter advance_to_sentinel(Iter first, Sent last)
    {
        return advance_to_sentinel(first, last,
            typename hpx::traits::is_sized_sentinel_for<Iter, Sent>::type{});
    }

    template <typename Iter>
    constexpr inline Iter advance_to_sentinel(Iter, Iter last)
    {
        return last;
    }

}}}}    // namespace hpx::parallel::v1::detail
