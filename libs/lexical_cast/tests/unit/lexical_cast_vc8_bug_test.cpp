//  Unit test for hpx::util::lexical_cast.
//
//  See http://www.boost.org for most recent version, including documentation.
//
//  Copyright Alexander Nasonov, 2007.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost
//  Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt).
//
// This tests now must pass on vc8, because lexical_cast
// implementation has changed and it does not use stringstream for casts
// to integral types

#include <hpx/config.hpp>
#include <hpx/lexical_cast.hpp>
#include <hpx/testing.hpp>

#include <cstdint>
#include <string>

#ifdef HPX_MSVC_WARNING_PRAGMA
#pragma warning(disable : 4127)    // conditional expression is constant
#endif

using namespace hpx::util;

// See also test_conversion_from_string_to_integral(char)
// in libs/conversion/lexical_cast_test.cpp
template <class T>
void test_too_long_number(char zero)
{
    typedef std::numeric_limits<T> limits;

    std::basic_string<char> s;

    std::basic_ostringstream<char> o;
    o << (limits::max)() << zero;
    s = o.str();
    HPX_TEST_THROW(lexical_cast<T>(s), bad_lexical_cast);
    s[s.size() - 1] += static_cast<char>(9);    // '0' -> '9'
    HPX_TEST_THROW(lexical_cast<T>(s), bad_lexical_cast);

    if (limits::is_signed)
    {
        std::basic_ostringstream<char> o2;
        o2 << (limits::min)() << zero;
        s = o2.str();
        HPX_TEST_THROW(lexical_cast<T>(s), bad_lexical_cast);
        s[s.size() - 1] += static_cast<char>(9);    // '0' -> '9'
        HPX_TEST_THROW(lexical_cast<T>(s), bad_lexical_cast);
    }
}

void test_vc8_bug()
{
    test_too_long_number<boost::intmax_t>('0');
    test_too_long_number<boost::uintmax_t>('0');
}

int main()
{
    test_vc8_bug();

    return hpx::util::report_errors();
}
