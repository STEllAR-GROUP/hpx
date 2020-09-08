//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <algorithm>
#include <cstddef>
#include <string>

namespace hpx { namespace string_util {
    template <typename CharT, class Traits, class Alloc>
    void trim(std::basic_string<CharT, Traits, Alloc>& s)
    {
        auto first = std::find_if_not(std::cbegin(s), std::cend(s),
            [](int c) { return std::isspace(c); });
        s.erase(std::begin(s), first);

        auto last = std::find_if_not(std::crbegin(s), std::crend(s),
            [](int c) { return std::isspace(c); });
        s.erase(last.base(), std::end(s));
    }

    template <typename CharT, class Traits, class Alloc>
    std::basic_string<CharT, Traits, Alloc> trim_copy(
        std::basic_string<CharT, Traits, Alloc> const& s)
    {
        auto t = s;
        trim(t);

        return t;
    }
}}    // namespace hpx::string_util
