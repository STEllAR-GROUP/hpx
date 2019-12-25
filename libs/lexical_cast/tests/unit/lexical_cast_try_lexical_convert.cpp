//  Unit test for hpx::util::lexical_cast.
//
//  See http://www.boost.org for most recent version, including documentation.
//
//  Copyright Antony Polukhin, 2014-2019.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost
//  Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt).

#include <hpx/config.hpp>

#include <hpx/lexical_cast/try_lexical_convert.hpp>
#include <hpx/testing.hpp>

#include <string>

using namespace hpx::util;

void try_uncommon_cases()
{
    std::string sres;
    const bool res1 = try_lexical_convert(std::string("Test string"), sres);
    HPX_TEST(res1);
    HPX_TEST_EQ(sres, "Test string");

    volatile int vires;
    const bool res2 = try_lexical_convert(100, vires);
    HPX_TEST(res2);
    HPX_TEST_EQ(vires, 100);

    const bool res3 = try_lexical_convert("Test string", sres);
    HPX_TEST(res3);
    HPX_TEST_EQ(sres, "Test string");

    const bool res4 =
        try_lexical_convert("Test string", sizeof("Test string") - 1, sres);
    HPX_TEST(res4);
    HPX_TEST_EQ(sres, "Test string");

    int ires;
    HPX_TEST(!try_lexical_convert("Test string", ires));
    HPX_TEST(!try_lexical_convert(1.1, ires));
    HPX_TEST(!try_lexical_convert(-1.9, ires));
    HPX_TEST(!try_lexical_convert("1.1", ires));
    HPX_TEST(
        !try_lexical_convert("1000000000000000000000000000000000000000", ires));
}

void try_common_cases()
{
    int ires = 0;
    const bool res1 = try_lexical_convert(std::string("100"), ires);
    HPX_TEST(res1);
    HPX_TEST_EQ(ires, 100);

    ires = 0;
    const bool res2 = try_lexical_convert("-100", ires);
    HPX_TEST(res2);
    HPX_TEST_EQ(ires, -100);

    float fres = 1.0f;
    const bool res3 = try_lexical_convert("0.0", fres);
    HPX_TEST(res3);
    HPX_TEST_EQ(fres, 0.0f);

    fres = 1.0f;
    const bool res4 = try_lexical_convert("0.0", sizeof("0.0") - 1, fres);
    HPX_TEST(res4);
    HPX_TEST_EQ(fres, 0.0f);
}

int main()
{
    try_uncommon_cases();
    try_common_cases();

    return hpx::util::report_errors();
}
