//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_HAVE_CXX20_MODULES)
#include <hpx/config.hpp>
#include <hpx/string_util/getenv.hpp>

#include <cstddef>
#include <cstdlib>
#else
module hpx.core.string_util;
#endif

namespace hpx::util {

#if !defined(HPX_MSVC)
    char* getenv(char const* key)
    {
        return std::getenv(key);
    }
#else
    char* getenv(char const* key)
    {
        static char buffer[2048];
        std::size_t value_len = 0;

        if (getenv_s(&value_len, buffer, sizeof(buffer), key) != 0)
        {
            return nullptr;
        }
        return buffer;
    }
#endif
}    // namespace hpx::util
