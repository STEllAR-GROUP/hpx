//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_ADD_LVALUE_REFERENCE_HPP
#define HPX_UTIL_ADD_LVALUE_REFERENCE_HPP

#include <boost/move/move.hpp>

namespace hpx { namespace util
{
    template <typename T>
    struct add_lvalue_reference
    {
        typedef T& type;
    };

    template <typename T>
    struct add_lvalue_reference<T&>
    {
        typedef T& type;
    };

    template <typename T>
    struct add_lvalue_reference<BOOST_RV_REF(T)>
    {
        typedef T& type;
    };
}}

#endif
