//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_REMOVE_REFERENCE_HPP
#define HPX_UTIL_DETAIL_REMOVE_REFERENCE_HPP

#include <boost/move/move.hpp>
#include <boost/type_traits/remove_reference.hpp>

namespace hpx { namespace util {
    namespace detail
    {
#if defined(BOOST_NO_RVALUE_REFERENCES)
        template <typename T>
        struct remove_reference
        {
            typedef typename boost::remove_reference<T>::type type;
        };

        template <typename T>
        struct remove_reference<boost::rv<T> >
        {
            typedef T type;
        };
#else
        using std::remove_reference;
#endif
    }
}}

#endif
