//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// (c) Copyright John R. Bandela 2001.

// See http://www.boost.org/libs/tokenizer for documentation

#include <hpx/modules/string_util.hpp>
#include <hpx/modules/testing.hpp>

#include <string>
#include <vector>

int main()
{
    std::vector<std::string> tokens;

    std::string s = "12252001";
    hpx::string_util::offset_separator f({2, 2, 4});

    auto beg = hpx::string_util::make_token_iterator<std::string>(
        s.begin(), s.end(), f);
    auto end =
        hpx::string_util::make_token_iterator<std::string>(s.end(), s.end(), f);

    for (; beg != end; ++beg)
    {
        tokens.push_back(*beg);
    }

    std::vector<std::string> expected = {"12", "25", "2001"};
    HPX_TEST(tokens == expected);

    return hpx::util::report_errors();
}
