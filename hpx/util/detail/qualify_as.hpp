//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_QUALIFY_AS_HPP
#define HPX_UTIL_DETAIL_QUALIFY_AS_HPP

#include <hpx/util/detail/pp_strip_parens.hpp>

#include <boost/move/move.hpp>

namespace hpx { namespace util { namespace detail
{
    // creates a type `T` with the (cv-ref)qualifiers of `U`
    template <typename T, typename U>
    struct qualify_as
    {
        typedef T type;
    };

    template <typename T, typename U>
    struct qualify_as<T, U&>
    {
        typedef typename qualify_as<T, U>::type& type;
    };
    template <typename T, typename U>
    struct qualify_as<T, BOOST_FWD_REF(U)>
    {
        typedef BOOST_FWD_REF(HPX_UTIL_STRIP((typename qualify_as<T, U>::type))) type;
    };
    
    template <typename T, typename U>
    struct qualify_as<T, U const>
    {
        typedef typename qualify_as<T, U>::type const type;
    };
    template <typename T, typename U>
    struct qualify_as<T, U volatile>
    {
        typedef typename qualify_as<T, U>::type volatile type;
    };
    template <typename T, typename U>
    struct qualify_as<T, U const volatile>
    {
        typedef typename qualify_as<T, U>::type const volatile type;
    };
}}}

#endif
