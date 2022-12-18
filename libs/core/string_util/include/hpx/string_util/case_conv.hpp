//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <algorithm>
#include <cctype>
#include <string>

namespace hpx::string_util {

    template <typename CharT, class Traits, class Alloc>
    void to_lower(std::basic_string<CharT, Traits, Alloc>& s)
    {
        std::transform(std::begin(s), std::end(s), std::begin(s),
            [](int c) { return std::tolower(c); });
    }
}    // namespace hpx::string_util
