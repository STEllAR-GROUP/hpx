// Copyright Vladimir Prus 2002-2004.
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>

#if defined(HPX_WINDOWS)
#include <hpx/modules/testing.hpp>
#include <hpx/preprocessor/cat.hpp>
#include <hpx/program_options/parsers.hpp>

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace hpx::program_options;
using namespace std;

void check_equal(
    const std::vector<string>& actual, char const** expected, int n)
{
    if (actual.size() != n)
    {
        std::cerr << "Size mismatch between expected and actual data\n";
        abort();
    }
    for (int i = 0; i < n; ++i)
    {
        if (actual[i] != expected[i])
        {
            std::cerr << "Unexpected content\n";
            abort();
        }
    }
}

#define COMMA ,
#define TEST(input, expected)                                                  \
    char const* HPX_PP_CAT(e, __LINE__)[] = expected;                          \
    vector<string> HPX_PP_CAT(v, __LINE__) = split_winmain(input);             \
    check_equal(HPX_PP_CAT(v, __LINE__), HPX_PP_CAT(e, __LINE__),              \
        sizeof(HPX_PP_CAT(e, __LINE__)) / sizeof(char*));                      \
    /**/

void test_winmain()
{
    // The following expectations were obtained in Win2000 shell:
    TEST("1 ", {"1"});
    TEST("1\"2\" ", {"12"});
    TEST("1\"2  ", {"12  "});
    TEST("1\"\\\"2\" ", {"1\"2"});
    TEST("\"1\" \"2\" ", {"1" COMMA "2"});
    TEST("1\\\" ", {"1\""});
    TEST("1\\\\\" ", {"1\\ "});
    TEST("1\\\\\\\" ", {"1\\\""});
    TEST("1\\\\\\\\\" ", {"1\\\\ "});

    TEST("1\" 1 ", {"1 1 "});
    TEST("1\\\" 1 ", {"1\"" COMMA "1"});
    TEST("1\\1 ", {"1\\1"});
    TEST("1\\\\1 ", {"1\\\\1"});
}

int main(int, char*[])
{
    test_winmain();
    return hpx::util::report_errors();
}
#else
int main(int, char*[])
{
    return 0;
}
#endif
