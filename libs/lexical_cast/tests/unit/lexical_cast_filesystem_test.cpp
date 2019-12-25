//  Unit test for hpx::util::lexical_cast.
//
//  See http://www.boost.org for most recent version, including documentation.
//
//  Copyright Antony Polukhin, 2013-2019.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost
//  Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt).
//
// Test lexical_cast usage with long filesystem::path. Bug 7704.

#include <hpx/config.hpp>

#include <hpx/filesystem.hpp>
#include <hpx/lexical_cast.hpp>
#include <hpx/testing.hpp>

#include <string>

using namespace hpx::util;

void test_filesystem()
{
    hpx::filesystem::path p;
    std::string s1 = "aaaaaaaaaaaaaaaaaaaaaaa";
    p = hpx::util::lexical_cast<hpx::filesystem::path>(s1);
    HPX_TEST(!p.empty());
    HPX_TEST_EQ(p, s1);
    p.clear();

    const char ab[] =
        "aaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    p = hpx::util::lexical_cast<hpx::filesystem::path>(ab);
    HPX_TEST(!p.empty());
    HPX_TEST_EQ(p, ab);

    // Tests for
    // https://github.com/boostorg/lexical_cast/issues/25

    const char quoted_path[] = "\"/home/my user\"";
    p = hpx::util::lexical_cast<hpx::filesystem::path>(quoted_path);
    HPX_TEST(!p.empty());
    const char unquoted_path[] = "/home/my user";
    HPX_TEST_EQ(p, hpx::filesystem::path(unquoted_path));

    // Converting back to std::string gives the initial string
    HPX_TEST_EQ(hpx::util::lexical_cast<std::string>(p), quoted_path);

    try
    {
        // Without quotes the path will have only `/home/my` in it.
        // `user` remains in the stream, so an exception must be thrown.
        p = hpx::util::lexical_cast<hpx::filesystem::path>(unquoted_path);
        HPX_TEST(false);
    }
    catch (const hpx::util::bad_lexical_cast&)
    {
        // Exception is expected
    }
}

int main()
{
    test_filesystem();

    return hpx::util::report_errors();
}
