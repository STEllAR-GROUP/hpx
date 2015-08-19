//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_TRAITS_IS_RANGE_HPP
#define HPX_TRAITS_IS_RANGE_HPP

#include <hpx/util/always_void.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/utility/declval.hpp>

namespace hpx { namespace traits
{
    template <class T, class>
    struct is_range: boost::mpl::false_ {};

    template <class T>
    struct is_range<T, typename util::always_void<decltype(
            boost::declval<T>().begin(),
            boost::declval<T>().end())>::type >:
                boost::mpl::true_ {};

}}

#endif
