//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_IS_RANGE_HPP
#define HPX_TRAITS_IS_RANGE_HPP

#include <hpx/util/always_void.hpp>
#include <hpx/traits/has_member_xxx.hpp>

#include <boost/mpl/and.hpp>

namespace hpx { namespace traits
{
    namespace detail
    {
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(begin);
        HPX_HAS_MEMBER_XXX_TRAIT_DEF(end);
    }

    template <typename T, typename Enable = void>
    struct is_range
      : boost::mpl::and_<detail::has_begin<T>, detail::has_end<T> >
    {};
}}

#endif
