//  Unit test for hpx::util::lexical_cast.
//
//  See http://www.boost.org for most recent version, including documentation.
//
//  Copyright Terje Sletteb and Kevlin Henney, 2005.
//  Copyright Alexander Nasonov, 2006.
//  Copyright Antony Polukhin, 2011-2019.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost
//  Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt).
//
// Note: The unit test no longer compile on MSVC 6, but lexical_cast itself works for it.

//
// We need this #define before any #includes: otherwise msvc will emit warnings
// deep within std::string, resulting from our (perfectly legal) use of basic_string
// with a custom traits class:
//
#define _SCL_SECURE_NO_WARNINGS

#include <hpx/config.hpp>

#if defined(__INTEL_COMPILER)
#pragma warning(disable : 193 383 488 981 1418 1419)
#elif defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(                                                               \
    disable : 4097 4100 4121 4127 4146 4244 4245 4511 4512 4701 4800)
#endif

#include <hpx/lexical_cast.hpp>
#include <hpx/testing.hpp>

#define HPX_TEST_CLOSE_FRACTION(expected, actual)                              \
    HPX_TEST_LTE(std::abs(actual - expected),                                  \
        std::abs(expected) * std::numeric_limits<decltype(actual)>::epsilon())

#include <algorithm>    // std::transform
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

template <class CharT>
struct my_traits : std::char_traits<CharT>
{
};

template <class CharT>
struct my_allocator : std::allocator<CharT>
{
    typedef std::allocator<CharT> base_t;

    my_allocator() {}
    template <class U>
    my_allocator(const my_allocator<U>& v)
      : base_t(v)
    {
    }

    template <class U>
    struct rebind
    {
        typedef my_allocator<U> other;
    };
};

using namespace hpx::util;

void test_conversion_to_char()
{
    HPX_TEST_EQ('A', lexical_cast<char>('A'));
    HPX_TEST_EQ(' ', lexical_cast<char>(' '));
    HPX_TEST_EQ('1', lexical_cast<char>(1));
    HPX_TEST_EQ('0', lexical_cast<char>(0));
    HPX_TEST_THROW(lexical_cast<char>(123), bad_lexical_cast);
    HPX_TEST_EQ('1', lexical_cast<char>(1.0));
    HPX_TEST_EQ('1', lexical_cast<char>(true));
    HPX_TEST_EQ('0', lexical_cast<char>(false));
    HPX_TEST_EQ('A', lexical_cast<char>("A"));
    HPX_TEST_EQ(' ', lexical_cast<char>(" "));
    HPX_TEST_THROW(lexical_cast<char>(""), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<char>("Test"), bad_lexical_cast);
    HPX_TEST_EQ('A', lexical_cast<char>(std::string("A")));
    HPX_TEST_EQ(' ', lexical_cast<char>(std::string(" ")));
    HPX_TEST_THROW(lexical_cast<char>(std::string("")), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<char>(std::string("Test")), bad_lexical_cast);
}

void test_conversion_to_int()
{
    HPX_TEST_EQ(1, lexical_cast<int>('1'));
    HPX_TEST_EQ(0, lexical_cast<int>('0'));
    HPX_TEST_THROW(lexical_cast<int>('A'), bad_lexical_cast);
    HPX_TEST_EQ(1, lexical_cast<int>(1));
    HPX_TEST_EQ(1, lexical_cast<int>(1.0));

    HPX_TEST_EQ((std::numeric_limits<int>::max)(),
        lexical_cast<int>((std::numeric_limits<int>::max)()));

    HPX_TEST_EQ((std::numeric_limits<int>::min)(),
        lexical_cast<int>((std::numeric_limits<int>::min)()));

    HPX_TEST_THROW(lexical_cast<int>(1.23), bad_lexical_cast);

    HPX_TEST_THROW(lexical_cast<int>(1e20), bad_lexical_cast);
    HPX_TEST_EQ(1, lexical_cast<int>(true));
    HPX_TEST_EQ(0, lexical_cast<int>(false));
    HPX_TEST_EQ(123, lexical_cast<int>("123"));
    HPX_TEST_THROW(lexical_cast<int>(" 123"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<int>(""), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<int>("Test"), bad_lexical_cast);
    HPX_TEST_EQ(123, lexical_cast<int>("123"));
    HPX_TEST_EQ(123, lexical_cast<int>(std::string("123")));
    HPX_TEST_THROW(lexical_cast<int>(std::string(" 123")), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<int>(std::string("")), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<int>(std::string("Test")), bad_lexical_cast);
}

void test_conversion_with_nonconst_char()
{
    std::vector<char> buffer;
    buffer.push_back('1');
    buffer.push_back('\0');
    HPX_TEST_EQ(hpx::util::lexical_cast<int>(&buffer[0]), 1);

    std::vector<unsigned char> buffer2;
    buffer2.push_back('1');
    buffer2.push_back('\0');
    HPX_TEST_EQ(hpx::util::lexical_cast<int>(&buffer2[0]), 1);

    std::vector<unsigned char> buffer3;
    buffer3.push_back('1');
    buffer3.push_back('\0');
    HPX_TEST_EQ(hpx::util::lexical_cast<int>(&buffer3[0]), 1);
}

void test_conversion_to_double()
{
    HPX_TEST_CLOSE_FRACTION(1.0, lexical_cast<double>('1'));
    HPX_TEST_THROW(lexical_cast<double>('A'), bad_lexical_cast);
    HPX_TEST_CLOSE_FRACTION(1.0, lexical_cast<double>(1));
    HPX_TEST_CLOSE_FRACTION(1.23, lexical_cast<double>(1.23));
    HPX_TEST_CLOSE_FRACTION(1.234567890, lexical_cast<double>(1.234567890));
    HPX_TEST_CLOSE_FRACTION(1.234567890, lexical_cast<double>("1.234567890"));
    HPX_TEST_CLOSE_FRACTION(1.0, lexical_cast<double>(true));
    HPX_TEST_CLOSE_FRACTION(0.0, lexical_cast<double>(false));
    HPX_TEST_CLOSE_FRACTION(1.23, lexical_cast<double>("1.23"));
    HPX_TEST_THROW(lexical_cast<double>(""), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<double>("Test"), bad_lexical_cast);
    HPX_TEST_CLOSE_FRACTION(1.23, lexical_cast<double>(std::string("1.23")));
    HPX_TEST_THROW(lexical_cast<double>(std::string("")), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<double>(std::string("Test")), bad_lexical_cast);
}

void test_conversion_to_bool()
{
    HPX_TEST_EQ(true, lexical_cast<bool>('1'));
    HPX_TEST_EQ(false, lexical_cast<bool>('0'));
    HPX_TEST_THROW(lexical_cast<bool>('A'), bad_lexical_cast);
    HPX_TEST_EQ(true, lexical_cast<bool>(1));
    HPX_TEST_EQ(false, lexical_cast<bool>(0));
    HPX_TEST_THROW(lexical_cast<bool>(123), bad_lexical_cast);
    HPX_TEST_EQ(true, lexical_cast<bool>(1.0));
    HPX_TEST_THROW(lexical_cast<bool>(-123), bad_lexical_cast);
    HPX_TEST_EQ(false, lexical_cast<bool>(0.0));
    HPX_TEST_THROW(lexical_cast<bool>(1234), bad_lexical_cast);
#if !defined(_CRAYC)
    // Looks like a bug in CRAY compiler (throws bad_lexical_cast)
    // TODO: localize the bug and report it to developers.
    HPX_TEST_EQ(true, lexical_cast<bool>(true));
    HPX_TEST_EQ(false, lexical_cast<bool>(false));
#endif
    HPX_TEST_EQ(true, lexical_cast<bool>("1"));
    HPX_TEST_EQ(false, lexical_cast<bool>("0"));
    HPX_TEST_THROW(lexical_cast<bool>(""), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<bool>("Test"), bad_lexical_cast);
    HPX_TEST_EQ(true, lexical_cast<bool>("1"));
    HPX_TEST_EQ(false, lexical_cast<bool>("0"));
    HPX_TEST_EQ(true, lexical_cast<bool>(std::string("1")));
    HPX_TEST_EQ(false, lexical_cast<bool>(std::string("0")));

    HPX_TEST_THROW(lexical_cast<bool>(1.0001L), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<bool>(2), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<bool>(2u), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<bool>(-1), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<bool>(-2), bad_lexical_cast);

    HPX_TEST_THROW(lexical_cast<bool>(std::string("")), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<bool>(std::string("Test")), bad_lexical_cast);

    HPX_TEST(lexical_cast<bool>("+1") == true);
    HPX_TEST(lexical_cast<bool>("+0") == false);
    HPX_TEST(lexical_cast<bool>("-0") == false);
    HPX_TEST_THROW(lexical_cast<bool>("--0"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<bool>("-+-0"), bad_lexical_cast);

    HPX_TEST(lexical_cast<bool>("0") == false);
    HPX_TEST(lexical_cast<bool>("1") == true);
    HPX_TEST(lexical_cast<bool>("00") == false);
    HPX_TEST(lexical_cast<bool>("00000000000") == false);
    HPX_TEST(lexical_cast<bool>("000000000001") == true);
    HPX_TEST(lexical_cast<bool>("+00") == false);
    HPX_TEST(lexical_cast<bool>("-00") == false);
    HPX_TEST(lexical_cast<bool>("+00000000001") == true);

    HPX_TEST_THROW(lexical_cast<bool>("020"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<bool>("00200"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<bool>("-00200"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<bool>("+00200"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<bool>("000000000002"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<bool>("-1"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<bool>("-0000000001"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<bool>("00000000011"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<bool>("001001"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<bool>("-00000000010"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<bool>("-000000000100"), bad_lexical_cast);
}

void test_conversion_to_string()
{
    char buf[] = "hello";
    char* str = buf;
    HPX_TEST_EQ(str, lexical_cast<std::string>(str));
    HPX_TEST_EQ("A", lexical_cast<std::string>('A'));
    HPX_TEST_EQ(" ", lexical_cast<std::string>(' '));
    HPX_TEST_EQ("123", lexical_cast<std::string>(123));
    HPX_TEST_EQ("1.23", lexical_cast<std::string>(1.23));
    HPX_TEST_EQ("1.111111111", lexical_cast<std::string>(1.111111111));
    HPX_TEST_EQ("1", lexical_cast<std::string>(true));
    HPX_TEST_EQ("0", lexical_cast<std::string>(false));
    HPX_TEST_EQ("Test", lexical_cast<std::string>("Test"));
    HPX_TEST_EQ(" ", lexical_cast<std::string>(" "));
    HPX_TEST_EQ("", lexical_cast<std::string>(""));
    HPX_TEST_EQ("Test", lexical_cast<std::string>(std::string("Test")));
    HPX_TEST_EQ(" ", lexical_cast<std::string>(std::string(" ")));
    HPX_TEST_EQ("", lexical_cast<std::string>(std::string("")));
}

void test_bad_lexical_cast()
{
    try
    {
        lexical_cast<int>(std::string("Test"));

        HPX_TEST(false);    // Exception expected
    }
    catch (const bad_lexical_cast& e)
    {
        HPX_TEST(e.source_type() == typeid(std::string));
        HPX_TEST(e.target_type() == typeid(int));
    }
}

void test_no_whitespace_stripping()
{
    HPX_TEST_THROW(lexical_cast<int>(" 123"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<int>("123 "), bad_lexical_cast);
}

void test_volatile_types_conversions()
{
    volatile int i1 = 100000;
    HPX_TEST_EQ("100000", hpx::util::lexical_cast<std::string>(i1));

    volatile const int i2 = 100000;
    HPX_TEST_EQ("100000", hpx::util::lexical_cast<std::string>(i2));

    volatile const long int i3 = 1000000;
    HPX_TEST_EQ("1000000", hpx::util::lexical_cast<std::string>(i3));
}

void test_traits()
{
    typedef std::basic_string<char, my_traits<char>> my_string;

    my_string const s("s");
    HPX_TEST(hpx::util::lexical_cast<char>(s) == s[0]);
    HPX_TEST(hpx::util::lexical_cast<my_string>(s) == s);
    HPX_TEST(hpx::util::lexical_cast<my_string>(-1) == "-1");
}

void test_allocator()
{
    typedef std::basic_string<char, std::char_traits<char>, my_allocator<char>>
        my_string;

    my_string s("s");
    HPX_TEST(hpx::util::lexical_cast<char>(s) == s[0]);
    HPX_TEST(hpx::util::lexical_cast<std::string>(s) == "s");
    HPX_TEST(hpx::util::lexical_cast<my_string>(s) == s);
    HPX_TEST(hpx::util::lexical_cast<my_string>(1) == "1");
    HPX_TEST(hpx::util::lexical_cast<my_string>("s") == s);
    HPX_TEST(hpx::util::lexical_cast<my_string>(std::string("s")) == s);
}

void test_char_types_conversions()
{
    const char c_arr[] = "Test array of chars";
    HPX_TEST(hpx::util::lexical_cast<std::string>(c_arr) == std::string(c_arr));
    HPX_TEST(hpx::util::lexical_cast<char>(c_arr[0]) == c_arr[0]);
}

struct foo_operators_test
{
    foo_operators_test()
      : f(2)
    {
    }
    int f;
};

template <typename OStream>
OStream& operator<<(OStream& ostr, const foo_operators_test& foo)
{
    ostr << foo.f;
    return ostr;
}

template <typename IStream>
IStream& operator>>(IStream& istr, foo_operators_test& foo)
{
    istr >> foo.f;
    return istr;
}

void operators_overload_test()
{
    foo_operators_test foo;
    HPX_TEST_EQ(hpx::util::lexical_cast<std::string>(foo), "2");
    HPX_TEST_EQ((hpx::util::lexical_cast<foo_operators_test>("2")).f, 2);

    // Must compile
    (void) hpx::util::lexical_cast<foo_operators_test>(foo);
}

void test_getting_pointer_to_function()
{
    // Just checking that &lexical_cast<To, From> is not ambiguous
    typedef char char_arr[4];
    typedef int (*f1)(const char_arr&);
    f1 p1 = &hpx::util::lexical_cast<int, char_arr>;
    HPX_TEST(p1);

    typedef int (*f2)(const std::string&);
    f2 p2 = &hpx::util::lexical_cast<int, std::string>;
    HPX_TEST(p2);

    typedef std::string (*f3)(const int&);
    f3 p3 = &hpx::util::lexical_cast<std::string, int>;
    HPX_TEST(p3);

    std::vector<int> values;
    std::vector<std::string> ret;
    std::transform(values.begin(), values.end(), ret.begin(),
        hpx::util::lexical_cast<std::string, int>);
}

int main()
{
    test_conversion_to_char();
    test_conversion_to_int();
    test_conversion_to_double();
    test_conversion_to_bool();
    test_conversion_to_string();
    test_conversion_with_nonconst_char();
    test_bad_lexical_cast();
    test_no_whitespace_stripping();
    test_volatile_types_conversions();
    test_traits();
    test_allocator();

    test_char_types_conversions();
    operators_overload_test();
    test_getting_pointer_to_function();

    return hpx::util::report_errors();
}
