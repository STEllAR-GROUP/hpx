//  Copyright (c) 2017 Denis Blank
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/traits/is_tuple_like.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>

namespace hpx::util::detail {

    /// A tag for dispatching based on the tuple like or container properties of
    /// a type.
    template <bool IsContainer, bool IsTupleLike>
    struct container_category_tag
    {
    };

    /// Deduces to the container_category_tag of the given type T.
    template <typename T>
    using container_category_of_t =
        container_category_tag<traits::is_range_v<T>,
            traits::is_tuple_like_v<T>>;
}    // namespace hpx::util::detail
