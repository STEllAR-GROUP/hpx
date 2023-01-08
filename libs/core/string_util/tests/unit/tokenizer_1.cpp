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

    std::string s = "This is,  a test";
    hpx::string_util::tokenizer<> tok(s);
    for (auto const& t : tok)
    {
        tokens.push_back(t);
    }

    std::vector<std::string> expected = {"This", "is", ",", "a", "test"};
    HPX_TEST(tokens == expected);

    return hpx::util::report_errors();
}
