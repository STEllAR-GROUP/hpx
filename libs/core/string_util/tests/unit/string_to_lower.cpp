//  Copyright (c) 2020      ETH Zurich
//  Copyright (c) 2002-2003 Pavol Droba
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/testing.hpp>
#include <hpx/string_util/case_conv.hpp>

#include <string>

int main()
{
    std::string str1("AbCdEfG 123 xxxYYYzZzZ");
    std::string str2("");

    hpx::string_util::to_lower(str1);
    HPX_TEST(str1 == "abcdefg 123 xxxyyyzzzz");

    // to_lower is idempotent
    hpx::string_util::to_lower(str1);
    HPX_TEST(str1 == "abcdefg 123 xxxyyyzzzz");

    hpx::string_util::to_lower(str2);
    HPX_TEST(str2 == "");

    return hpx::util::report_errors();
}
