//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_LAZY_ENABLE_IF_HPP
#define HPX_UTIL_LAZY_ENABLE_IF_HPP

namespace hpx { namespace util
{
    template <bool Enable, typename T>
    struct lazy_enable_if
    {
    };

    template <typename T>
    struct lazy_enable_if<true, T>
    {
        typedef typename T::type type;
    };
}}

#endif
