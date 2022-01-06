//  Copyright (c) 2020      ETH Zurich
//  Copyright (c) 2002-2003 Pavol Droba
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/testing.hpp>
#include <hpx/string_util/classification.hpp>
#include <hpx/string_util/split.hpp>

#include <cstddef>
#include <iostream>
#include <list>
#include <string>
#include <vector>

int main()
{
    std::string str2("Xx-abc--xX-abb-xx");
    std::string str3("xx");
    std::string strempty("");
    const char* pch1 = "xx-abc--xx-abb";
    std::vector<std::string> tokens;

    // split tests
    hpx::string_util::split(tokens, str2, hpx::string_util::is_any_of("xX"),
        hpx::string_util::token_compress_mode::on);

    HPX_TEST_EQ(tokens.size(), std::size_t(4));
    HPX_TEST_EQ(tokens[0], std::string(""));
    HPX_TEST_EQ(tokens[1], std::string("-abc--"));
    HPX_TEST_EQ(tokens[2], std::string("-abb-"));
    HPX_TEST_EQ(tokens[3], std::string(""));

    hpx::string_util::split(tokens, str2, hpx::string_util::is_any_of("xX"),
        hpx::string_util::token_compress_mode::off);

    HPX_TEST_EQ(tokens.size(), std::size_t(7));
    HPX_TEST_EQ(tokens[0], std::string(""));
    HPX_TEST_EQ(tokens[1], std::string(""));
    HPX_TEST_EQ(tokens[2], std::string("-abc--"));
    HPX_TEST_EQ(tokens[3], std::string(""));
    HPX_TEST_EQ(tokens[4], std::string("-abb-"));
    HPX_TEST_EQ(tokens[5], std::string(""));
    HPX_TEST_EQ(tokens[6], std::string(""));

    hpx::string_util::split(tokens, pch1, hpx::string_util::is_any_of("x"),
        hpx::string_util::token_compress_mode::on);

    HPX_TEST_EQ(tokens.size(), std::size_t(3));
    HPX_TEST_EQ(tokens[0], std::string(""));
    HPX_TEST_EQ(tokens[1], std::string("-abc--"));
    HPX_TEST_EQ(tokens[2], std::string("-abb"));

    hpx::string_util::split(tokens, pch1, hpx::string_util::is_any_of("x"),
        hpx::string_util::token_compress_mode::off);

    HPX_TEST_EQ(tokens.size(), std::size_t(5));
    HPX_TEST_EQ(tokens[0], std::string(""));
    HPX_TEST_EQ(tokens[1], std::string(""));
    HPX_TEST_EQ(tokens[2], std::string("-abc--"));
    HPX_TEST_EQ(tokens[3], std::string(""));
    HPX_TEST_EQ(tokens[4], std::string("-abb"));

    hpx::string_util::split(tokens, str3, hpx::string_util::is_any_of(","),
        hpx::string_util::token_compress_mode::on);

    HPX_TEST_EQ(tokens.size(), std::size_t(1));
    HPX_TEST_EQ(tokens[0], std::string("xx"));

    hpx::string_util::split(tokens, str3, hpx::string_util::is_any_of(","),
        hpx::string_util::token_compress_mode::off);

    hpx::string_util::split(tokens, str3, hpx::string_util::is_any_of("xX"),
        hpx::string_util::token_compress_mode::on);

    HPX_TEST_EQ(tokens.size(), std::size_t(2));
    HPX_TEST_EQ(tokens[0], std::string(""));
    HPX_TEST_EQ(tokens[1], std::string(""));

    hpx::string_util::split(tokens, str3, hpx::string_util::is_any_of("xX"),
        hpx::string_util::token_compress_mode::off);

    HPX_TEST_EQ(tokens.size(), std::size_t(3));
    HPX_TEST_EQ(tokens[0], std::string(""));
    HPX_TEST_EQ(tokens[1], std::string(""));
    HPX_TEST_EQ(tokens[2], std::string(""));

    split(tokens, strempty, hpx::string_util::is_any_of(".:,;"),
        hpx::string_util::token_compress_mode::on);

    HPX_TEST(tokens.size() == 1);
    HPX_TEST(tokens[0] == std::string(""));

    split(tokens, strempty, hpx::string_util::is_any_of(".:,;"),
        hpx::string_util::token_compress_mode::off);

    HPX_TEST(tokens.size() == 1);
    HPX_TEST(tokens[0] == std::string(""));

    // If using a compiler that supports forwarding references, we should be
    // able to use rvalues, too
    hpx::string_util::split(tokens, std::string("Xx-abc--xX-abb-xx"),
        hpx::string_util::is_any_of("xX"),
        hpx::string_util::token_compress_mode::on);

    HPX_TEST_EQ(tokens.size(), std::size_t(4));
    HPX_TEST_EQ(tokens[0], std::string(""));
    HPX_TEST_EQ(tokens[1], std::string("-abc--"));
    HPX_TEST_EQ(tokens[2], std::string("-abb-"));
    HPX_TEST_EQ(tokens[3], std::string(""));

    hpx::string_util::split(tokens, std::string("Xx-abc--xX-abb-xx"),
        hpx::string_util::is_any_of("xX"),
        hpx::string_util::token_compress_mode::off);

    HPX_TEST_EQ(tokens.size(), std::size_t(7));
    HPX_TEST_EQ(tokens[0], std::string(""));
    HPX_TEST_EQ(tokens[1], std::string(""));
    HPX_TEST_EQ(tokens[2], std::string("-abc--"));
    HPX_TEST_EQ(tokens[3], std::string(""));
    HPX_TEST_EQ(tokens[4], std::string("-abb-"));
    HPX_TEST_EQ(tokens[5], std::string(""));
    HPX_TEST_EQ(tokens[6], std::string(""));

    return hpx::util::report_errors();
}
