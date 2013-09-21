//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_ADD_RVALUE_REFERENCE_HPP
#define HPX_UTIL_DETAIL_ADD_RVALUE_REFERENCE_HPP

#include <boost/move/move.hpp>

namespace hpx { namespace util { namespace detail
{
    template <typename T>
    struct add_rvalue_reference
    {
        typedef BOOST_RV_REF(T) type;
    };

    template <typename T>
    struct add_rvalue_reference<T&>
    {
        typedef T& type;
    };

    template <typename T>
    struct add_rvalue_reference<BOOST_RV_REF(T)>
    {
        typedef BOOST_RV_REF(T) type;
    };
}}}

#endif
