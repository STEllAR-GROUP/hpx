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
// We need this #define before any #includes: otherwise msvc will emit warnings
// deep within std::string, resulting from our (perfectly legal) use of
// basic_string with a custom traits class:
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

#include <boost/type_traits/integral_promotion.hpp>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

// Test all 65536 values if true:
bool const lcast_test_small_integral_types_completely = false;

// lcast_integral_test_counter: use when testing all values of an integral
// types is not possible. Max. portable value is 32767.
int const lcast_integral_test_counter = 500;

using namespace hpx::util;

template <class T, class CharT>
void test_conversion_from_integral_to_char(CharT zero)
{
    HPX_TEST(lexical_cast<CharT>(static_cast<T>(0)) == zero + 0);
    HPX_TEST(lexical_cast<CharT>(static_cast<T>(1)) == zero + 1);
    HPX_TEST(lexical_cast<CharT>(static_cast<T>(2)) == zero + 2);
    HPX_TEST(lexical_cast<CharT>(static_cast<T>(3)) == zero + 3);
    HPX_TEST(lexical_cast<CharT>(static_cast<T>(4)) == zero + 4);
    HPX_TEST(lexical_cast<CharT>(static_cast<T>(5)) == zero + 5);
    HPX_TEST(lexical_cast<CharT>(static_cast<T>(6)) == zero + 6);
    HPX_TEST(lexical_cast<CharT>(static_cast<T>(7)) == zero + 7);
    HPX_TEST(lexical_cast<CharT>(static_cast<T>(8)) == zero + 8);
    HPX_TEST(lexical_cast<CharT>(static_cast<T>(9)) == zero + 9);

    HPX_TEST_THROW(lexical_cast<CharT>(static_cast<T>(10)), bad_lexical_cast);

    T t = (std::numeric_limits<T>::max)();
    HPX_TEST_THROW(lexical_cast<CharT>(t), bad_lexical_cast);
}

template <class T, class CharT>
void test_conversion_from_char_to_integral(CharT zero)
{
    HPX_TEST(
        lexical_cast<T>(static_cast<CharT>(zero + 0)) == static_cast<T>(0));
    HPX_TEST(
        lexical_cast<T>(static_cast<CharT>(zero + 1)) == static_cast<T>(1));
    HPX_TEST(
        lexical_cast<T>(static_cast<CharT>(zero + 2)) == static_cast<T>(2));
    HPX_TEST(
        lexical_cast<T>(static_cast<CharT>(zero + 3)) == static_cast<T>(3));
    HPX_TEST(
        lexical_cast<T>(static_cast<CharT>(zero + 4)) == static_cast<T>(4));
    HPX_TEST(
        lexical_cast<T>(static_cast<CharT>(zero + 5)) == static_cast<T>(5));
    HPX_TEST(
        lexical_cast<T>(static_cast<CharT>(zero + 6)) == static_cast<T>(6));
    HPX_TEST(
        lexical_cast<T>(static_cast<CharT>(zero + 7)) == static_cast<T>(7));
    HPX_TEST(
        lexical_cast<T>(static_cast<CharT>(zero + 8)) == static_cast<T>(8));
    HPX_TEST(
        lexical_cast<T>(static_cast<CharT>(zero + 9)) == static_cast<T>(9));

    HPX_TEST_THROW(
        lexical_cast<T>(static_cast<CharT>(zero + 10)), bad_lexical_cast);
    HPX_TEST_THROW(
        lexical_cast<T>(static_cast<CharT>(zero - 1)), bad_lexical_cast);
}

template <class T>
void test_conversion_from_integral_to_integral()
{
    T t = 0;
    HPX_TEST(lexical_cast<T>(t) == t);

    // Next two variables are used to suppress warnings.
    int st = 32767;
    unsigned int ut = st;
    t = st;
    HPX_TEST(lexical_cast<short>(t) == st);
    HPX_TEST(lexical_cast<unsigned short>(t) == ut);
    HPX_TEST(lexical_cast<int>(t) == st);
    HPX_TEST(lexical_cast<unsigned int>(t) == ut);
    HPX_TEST(lexical_cast<long>(t) == st);
    HPX_TEST(lexical_cast<unsigned long>(t) == ut);

    t = (std::numeric_limits<T>::max)();
    HPX_TEST(lexical_cast<T>(t) == t);

    t = (std::numeric_limits<T>::min)();
    HPX_TEST(lexical_cast<T>(t) == t);
}

template <class CharT, class T>
std::basic_string<CharT> to_str(T t)
{
    std::basic_ostringstream<CharT> o;
    o << t;
    return o.str();
}

template <class T, class CharT>
void test_conversion_from_integral_to_string(CharT)
{
    typedef std::numeric_limits<T> limits;
    typedef std::basic_string<CharT> string_type;

    T t;

    t = (limits::min)();
    HPX_TEST(lexical_cast<string_type>(t) == to_str<CharT>(t));

    t = (limits::max)();
    HPX_TEST(lexical_cast<string_type>(t) == to_str<CharT>(t));

    if (limits::digits <= 16 && lcast_test_small_integral_types_completely)
        // min and max have already been tested.
        for (t = 1 + (limits::min)(); t != (limits::max)(); ++t)
            HPX_TEST(lexical_cast<string_type>(t) == to_str<CharT>(t));
    else
    {
        T const min_val = (limits::min)();
        T const max_val = (limits::max)();
        T const half_max_val = max_val / 2;
        T const cnt = lcast_integral_test_counter;    // to suppress warnings
        unsigned int const counter = cnt < half_max_val ? cnt : half_max_val;

        unsigned int i;

        // Test values around min:
        t = min_val;
        for (i = 0; i < counter; ++i, ++t)
            HPX_TEST(lexical_cast<string_type>(t) == to_str<CharT>(t));

        // Test values around max:
        t = max_val;
        for (i = 0; i < counter; ++i, --t)
            HPX_TEST(lexical_cast<string_type>(t) == to_str<CharT>(t));

        // Test values around zero:
        if (limits::is_signed)
            for (t = static_cast<T>(-counter); t < static_cast<T>(counter); ++t)
                HPX_TEST(lexical_cast<string_type>(t) == to_str<CharT>(t));

        // Test values around 100, 1000, 10000, ...
        T ten_power = 100;
        for (int e = 2; e < limits::digits10; ++e, ten_power *= 10)
        {
            // ten_power + 100 probably never overflows
            for (t = ten_power - 100; t != ten_power + 100; ++t)
                HPX_TEST(lexical_cast<string_type>(t) == to_str<CharT>(t));
        }
    }
}

template <class T, class CharT>
void test_conversion_from_string_to_integral(CharT)
{
    typedef std::numeric_limits<T> limits;
    typedef std::basic_string<CharT> string_type;

    string_type s;
    string_type const zero = to_str<CharT>(0);
    string_type const nine = to_str<CharT>(9);
    T const min_val = (limits::min)();
    T const max_val = (limits::max)();

    s = to_str<CharT>(min_val);
    HPX_TEST_EQ(lexical_cast<T>(s), min_val);
    if (limits::is_signed)
    {
        HPX_TEST_THROW(lexical_cast<T>(s + zero), bad_lexical_cast);
        HPX_TEST_THROW(lexical_cast<T>(s + nine), bad_lexical_cast);
    }

    s = to_str<CharT>(max_val);
    HPX_TEST_EQ(lexical_cast<T>(s), max_val);
    {
        HPX_TEST_THROW(lexical_cast<T>(s + zero), bad_lexical_cast);
        HPX_TEST_THROW(lexical_cast<T>(s + nine), bad_lexical_cast);

        s = to_str<CharT>(max_val);
        for (int i = 1; i <= 10; ++i)
        {
            s[s.size() - 1] += 1;
            HPX_TEST_THROW(lexical_cast<T>(s), bad_lexical_cast);
        }

        for (int i = 1; i <= 256; ++i)
        {
            HPX_TEST_THROW(
                lexical_cast<T>(to_str<CharT>(i) + s), bad_lexical_cast);
        }

        typedef typename boost::integral_promotion<T>::type promoted;
        if (!(std::is_same<T, promoted>::value))
        {
            promoted prom = max_val;
            s = to_str<CharT>(max_val);
            for (int i = 1; i <= 256; ++i)
            {
                HPX_TEST_THROW(
                    lexical_cast<T>(to_str<CharT>(prom + i)), bad_lexical_cast);
                HPX_TEST_THROW(
                    lexical_cast<T>(to_str<CharT>(i) + s), bad_lexical_cast);
            }
        }
    }

    if (limits::digits <= 16 && lcast_test_small_integral_types_completely)
        // min and max have already been tested.
        for (T t = 1 + min_val; t != max_val; ++t)
            HPX_TEST(lexical_cast<T>(to_str<CharT>(t)) == t);
    else
    {
        T const half_max_val = max_val / 2;
        T const cnt = lcast_integral_test_counter;    // to suppress warnings
        unsigned int const counter = cnt < half_max_val ? cnt : half_max_val;

        T t;
        unsigned int i;

        // Test values around min:
        t = min_val;
        for (i = 0; i < counter; ++i, ++t)
            HPX_TEST(lexical_cast<T>(to_str<CharT>(t)) == t);

        // Test values around max:
        t = max_val;
        for (i = 0; i < counter; ++i, --t)
            HPX_TEST(lexical_cast<T>(to_str<CharT>(t)) == t);

        // Test values around zero:
        if (limits::is_signed)
            for (t = static_cast<T>(-counter); t < static_cast<T>(counter); ++t)
                HPX_TEST(lexical_cast<T>(to_str<CharT>(t)) == t);

        // Test values around 100, 1000, 10000, ...
        T ten_power = 100;
        for (int e = 2; e < limits::digits10; ++e, ten_power *= 10)
        {
            // ten_power + 100 probably never overflows
            for (t = ten_power - 100; t != ten_power + 100; ++t)
                HPX_TEST(lexical_cast<T>(to_str<CharT>(t)) == t);
        }
    }
}

template <class T>
void test_conversion_from_to_integral()
{
    char const zero = '0';
    test_conversion_from_integral_to_char<T>(zero);
    test_conversion_from_char_to_integral<T>(zero);

    HPX_TEST(lexical_cast<T>("-1") == static_cast<T>(-1));
    HPX_TEST(lexical_cast<T>("-9") == static_cast<T>(-9));
    HPX_TEST(lexical_cast<T>(-1) == static_cast<T>(-1));
    HPX_TEST(lexical_cast<T>(-9) == static_cast<T>(-9));

    HPX_TEST_THROW(lexical_cast<T>("-1.0"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<T>("-9.0"), bad_lexical_cast);
    HPX_TEST(lexical_cast<T>(-1.0) == static_cast<T>(-1));
    HPX_TEST(lexical_cast<T>(-9.0) == static_cast<T>(-9));

    HPX_TEST(lexical_cast<T>(static_cast<T>(1)) == static_cast<T>(1));
    HPX_TEST(lexical_cast<T>(static_cast<T>(9)) == static_cast<T>(9));
    HPX_TEST_THROW(lexical_cast<T>(1.1f), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<T>(1.1), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<T>(1.1L), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<T>(1.0001f), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<T>(1.0001), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<T>(1.0001L), bad_lexical_cast);

    HPX_TEST(lexical_cast<T>("+1") == static_cast<T>(1));
    HPX_TEST(lexical_cast<T>("+9") == static_cast<T>(9));
    HPX_TEST(lexical_cast<T>("+10") == static_cast<T>(10));
    HPX_TEST(lexical_cast<T>("+90") == static_cast<T>(90));
    HPX_TEST_THROW(lexical_cast<T>("++1"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<T>("-+9"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<T>("--1"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<T>("+-9"), bad_lexical_cast);

    // Overflow test case from David W. Birdsall
    std::string must_owerflow_str =
        (sizeof(T) < 16 ? "160000000000000000000" :
                          "1600000000000000000000000000000000000000");
    std::string must_owerflow_negative_str =
        (sizeof(T) < 16 ? "-160000000000000000000" :
                          "-1600000000000000000000000000000000000000");
    for (int i = 0; i < 15; ++i)
    {
        HPX_TEST_THROW(lexical_cast<T>(must_owerflow_str), bad_lexical_cast);
        HPX_TEST_THROW(
            lexical_cast<T>(must_owerflow_negative_str), bad_lexical_cast);

        must_owerflow_str += '0';
        must_owerflow_negative_str += '0';
    }
}

void test_conversion_from_to_short()
{
    test_conversion_from_to_integral<short>();
}

void test_conversion_from_to_ushort()
{
    test_conversion_from_to_integral<unsigned short>();
}

void test_conversion_from_to_int()
{
    test_conversion_from_to_integral<int>();
}

void test_conversion_from_to_uint()
{
    test_conversion_from_to_integral<unsigned int>();
}

void test_conversion_from_to_long()
{
    test_conversion_from_to_integral<long>();
}

void test_conversion_from_to_ulong()
{
    test_conversion_from_to_integral<unsigned long>();
}

void test_conversion_from_to_intmax_t()
{
    test_conversion_from_to_integral<boost::intmax_t>();
}

void test_conversion_from_to_uintmax_t()
{
    test_conversion_from_to_integral<boost::uintmax_t>();
}

void test_conversion_from_to_longlong()
{
    test_conversion_from_to_integral<long long>();
}

void test_conversion_from_to_ulonglong()
{
    test_conversion_from_to_integral<unsigned long long>();
}

template <bool Specialized, class T>
struct test_if_specialized
{
    static void test() {}
};

template <class T>
struct test_if_specialized<true, T>
{
    static void test()
    {
        test_conversion_from_to_integral_minimal<T>();
    }
};

template <typename SignedT>
void test_integral_conversions_on_min_max_impl()
{
    typedef SignedT signed_t;
    typedef typename std::make_unsigned<signed_t>::type unsigned_t;

    typedef std::numeric_limits<signed_t> s_limits;
    typedef std::numeric_limits<unsigned_t> uns_limits;

    HPX_TEST_EQ(
        lexical_cast<unsigned_t>((uns_limits::max)()), (uns_limits::max)());
    HPX_TEST_EQ(
        lexical_cast<unsigned_t>((uns_limits::min)()), (uns_limits::min)());

    HPX_TEST_EQ(lexical_cast<signed_t>((s_limits::max)()), (s_limits::max)());
    HPX_TEST_EQ(lexical_cast<signed_t>((uns_limits::min)()),
        static_cast<signed_t>((uns_limits::min)()));

    HPX_TEST_EQ(lexical_cast<unsigned_t>((s_limits::max)()),
        static_cast<unsigned_t>((s_limits::max)()));
    HPX_TEST_EQ(lexical_cast<unsigned_t>((s_limits::min)()),
        static_cast<unsigned_t>((s_limits::min)()));
}

void test_integral_conversions_on_min_max()
{
    test_integral_conversions_on_min_max_impl<int>();
    test_integral_conversions_on_min_max_impl<short>();
    test_integral_conversions_on_min_max_impl<long int>();
    test_integral_conversions_on_min_max_impl<long long>();
}

int main()
{
    test_conversion_from_to_short();
    test_conversion_from_to_ushort();
    test_conversion_from_to_int();
    test_conversion_from_to_uint();
    test_conversion_from_to_long();
    test_conversion_from_to_ulong();
    test_conversion_from_to_intmax_t();
    test_conversion_from_to_uintmax_t();
    test_conversion_from_to_longlong();
    test_conversion_from_to_ulonglong();
    test_integral_conversions_on_min_max();

    return hpx::util::report_errors();
}
