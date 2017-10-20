//  Copyright (c) 2017 Denis Blank
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_TUPLE_LIKE_HPP)
#define HPX_TRAITS_IS_TUPLE_LIKE_HPP

#include <type_traits>

#include <hpx/util/always_void.hpp>
#include <hpx/util/tuple.hpp>

namespace hpx {
namespace traits {
    /// Deduces to a true type if the given parameter T
    /// has a specific tuple like size.
    template <typename T, typename = void>
    struct is_tuple_like : std::false_type
    {
    };
    template <typename T>
    struct is_tuple_like<T,
        typename util::always_void<decltype(util::tuple_size<T>::value)>::type>
      : std::true_type
    {
    };
}    // namespace traits
}    // namespace hpx

#endif
