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

    std::string s =
        "Field 1,\"putting quotes around fields, allows commas\",Field 3";

    hpx::string_util::tokenizer<hpx::string_util::escaped_list_separator<char>>
        tok(s);
    for (auto const& t : tok)
    {
        tokens.push_back(t);
    }

    std::vector<std::string> expected = {
        "Field 1", "putting quotes around fields, allows commas", "Field 3"};
    HPX_TEST(tokens == expected);

    return hpx::util::report_errors();
}
