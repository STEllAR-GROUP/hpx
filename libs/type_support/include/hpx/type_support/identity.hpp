//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_IDENTITY_HPP)
#define HPX_UTIL_IDENTITY_HPP

namespace hpx { namespace util {
    template <typename T>
    struct identity
    {
        typedef T type;
    };
}}

#endif
