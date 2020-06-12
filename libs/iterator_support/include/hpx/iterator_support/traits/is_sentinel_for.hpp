//  Copyright (c) 2020 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/type_support/always_void.hpp>
#include <hpx/iterator_support/traits/is_sentinel_for.hpp>

#include <utility>


namespace hpx { namespace traits {

    template <typename Sent, typename Iter, typename Enable = void>
    struct is_sentinel_for : std::false_type
    {
    };

    template <typename Sent, typename Iter>
    struct is_sentinel_for<Sent, Iter,
        typename util::always_void<
            typename is_iterator<Iter>::type,
            typename detail::equality_result<Iter, Sent>::type,
            typename detail::equality_result<Sent, Iter>::type,
            typename detail::inequality_result<Iter, Sent>::type,
            typename detail::inequality_result<Sent, Iter>::type>::type>
      : std::true_type
    {
    };
}}    // namespace hpx::traits
