//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_IS_RANGE_HPP
#define HPX_TRAITS_IS_RANGE_HPP

#include <hpx/traits/has_member_xxx.hpp>
#include <hpx/util/always_void.hpp>

#include <type_traits>

namespace hpx { namespace traits
{
    namespace detail
    {
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(begin);
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(end);
    }

    template <typename T, typename Enable = void>
    struct is_range
      : std::integral_constant<bool,
            detail::has_begin<T>::value && detail::has_end<T>::value>
    {};
}}

#endif
