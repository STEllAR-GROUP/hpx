//  Copyright (c) 2017 Denis Blank
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_CONTAINER_CATEGORY_HPP
#define HPX_UTIL_DETAIL_CONTAINER_CATEGORY_HPP

#include <hpx/config.hpp>
#include <hpx/traits/is_range.hpp>
#include <hpx/traits/is_tuple_like.hpp>

namespace hpx {
namespace util {
    namespace detail {
        /// A tag for dispatching based on the tuple like
        /// or container properties of a type.
        template <bool IsContainer, bool IsTupleLike>
        struct container_category_tag
        {
        };

        /// Deduces to the container_category_tag of the given type T.
        template <typename T>
        using container_category_of_t =
            container_category_tag<traits::is_range<T>::value,
                traits::is_tuple_like<T>::value>;
    }    // end namespace detail
}    // end namespace util
}    // end namespace hpx

#endif    // HPX_UTIL_DETAIL_CONTAINER_CATEGORY_HPP
