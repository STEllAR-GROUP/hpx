//  Copyright (c) 2019 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_MAX_HPP
#define HPX_UTIL_MAX_HPP

#include <hpx/config.hpp>

namespace hpx { namespace util {

    template <typename T>
    HPX_HOST_DEVICE HPX_CONSTEXPR inline T const& (max)(T const& a, T const& b)
    {
        return a < b ? b : a;
    }

}}

#endif
