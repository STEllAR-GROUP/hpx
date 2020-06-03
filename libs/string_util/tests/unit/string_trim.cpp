//  Copyright (c) 2020      ETH Zurich
//  Copyright (c) 2002-2003 Pavol Droba
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/testing.hpp>
#include <hpx/string_util/trim.hpp>

#include <string>

int main()
{
    std::string str1("     1x x x x1     ");
    std::string str2("     2x x x x2     ");
    std::string str3("x     ");
    std::string str4("     x");
    std::string str5("    ");

    // general string test
    HPX_TEST_EQ(hpx::string_util::trim_copy(str1), "1x x x x1");
    HPX_TEST_EQ(hpx::string_util::trim_copy(str3), "x");
    HPX_TEST_EQ(hpx::string_util::trim_copy(str4), "x");

    // spaces-only string test
    HPX_TEST_EQ(hpx::string_util::trim_copy(str5), "");

    // empty string check
    HPX_TEST_EQ(hpx::string_util::trim_copy(std::string("")), "");

    // general string test
    hpx::string_util::trim(str2);
    HPX_TEST_EQ(str2, "2x x x x2");

    hpx::string_util::trim(str3);
    HPX_TEST_EQ(str3, "x");

    hpx::string_util::trim(str4);
    HPX_TEST_EQ(str4, "x");

    // spaces-only string test
    hpx::string_util::trim(str5);
    HPX_TEST_EQ(str5, "");

    // empty string check
    str5 = "";
    hpx::string_util::trim(str5);
    HPX_TEST_EQ(str5, "");

    return hpx::util::report_errors();
}
