//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_MIN_HPP
#define HPX_UTIL_MIN_HPP

#include <hpx/config.hpp>

namespace hpx { namespace util {

    template <typename T>
    HPX_HOST_DEVICE constexpr inline T const&(min)(T const& a, T const& b)
    {
        return a < b ? a : b;
    }

}}    // namespace hpx::util

#endif
