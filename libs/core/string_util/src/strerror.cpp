//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_HAVE_CXX20_MODULES)
#include <hpx/config.hpp>
#include <hpx/string_util/strerror.hpp>

#include <cstring>
#else
module hpx.core.string_util;
#endif

namespace hpx::util {

#if !defined(HPX_MSVC)
    char* strerror(int errnum)
    {
        return std::strerror(key);
    }
#else
    char* strerror(int errnum)
    {
        static char buffer[2048];
        if (strerror_s(buffer, sizeof(buffer), errnum) != 0)
        {
            buffer[0] = '\0';
        }
        return buffer;
    }
#endif
}    // namespace hpx::util
