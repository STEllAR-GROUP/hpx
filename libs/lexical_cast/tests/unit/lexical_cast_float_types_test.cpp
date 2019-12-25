//  Unit test for hpx::util::lexical_cast.
//
//  See http://www.boost.org for most recent version, including documentation.
//
//  Copyright Antony Polukhin, 2011-2019.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost
//  Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt).

#include <hpx/config.hpp>

#if defined(__INTEL_COMPILER)
#pragma warning(disable : 193 383 488 981 1418 1419)
#elif defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(                                                               \
    disable : 4097 4100 4121 4127 4146 4244 4245 4511 4512 4701 4800)
#endif

#include <hpx/lexical_cast.hpp>
#include <hpx/testing.hpp>

#include <cstdint>
#include <sstream>
#include <string>
#include <type_traits>

#define HPX_TEST_CLOSE_FRACTION(expected, actual)                              \
    HPX_TEST_LTE(std::abs(actual - expected),                                  \
        std::abs(expected) * std::numeric_limits<decltype(actual)>::epsilon())

using namespace hpx::util;

template <class T>
std::string to_str(T t)
{
    std::ostringstream o;
    o << t;
    return o.str();
}

/*
 * Converts char* to float number type and checks, that generated
 * number does not exceeds allowed epsilon.
 */
#define CHECK_CLOSE_ABS_DIFF(VAL, TYPE)                                        \
    converted_val = lexical_cast<test_t>(#VAL);                                \
    HPX_TEST_CLOSE_FRACTION(                                                   \
        (static_cast<bool>(VAL##L) ? VAL##L :                                  \
                                     std::numeric_limits<test_t>::epsilon()),  \
        (converted_val ? converted_val :                                       \
                         std::numeric_limits<test_t>::epsilon()));

template <class TestType>
void test_converion_to_float_types()
{
    typedef TestType test_t;
    test_t converted_val;

    HPX_TEST_CLOSE_FRACTION(1.0, lexical_cast<test_t>('1'));
    HPX_TEST_EQ(0.0, lexical_cast<test_t>('0'));

    HPX_TEST_CLOSE_FRACTION(
        1e34L, lexical_cast<test_t>("10000000000000000000000000000000000"));

    HPX_TEST_CLOSE_FRACTION(
        1e-35L, lexical_cast<test_t>("0.00000000000000000000000000000000001"));
    HPX_TEST_CLOSE_FRACTION(
        0.1111111111111111111111111111111111111111111111111111111111111111111111111L,
        lexical_cast<test_t>("0."
                             "1111111111111111111111111111111111111111111111111"
                             "111111111111111111111111"));

    CHECK_CLOSE_ABS_DIFF(1, test_t);
    HPX_TEST_EQ(0, lexical_cast<test_t>("0"));
    CHECK_CLOSE_ABS_DIFF(-1, test_t);

    CHECK_CLOSE_ABS_DIFF(1.0, test_t);
    CHECK_CLOSE_ABS_DIFF(0.0, test_t);
    CHECK_CLOSE_ABS_DIFF(-1.0, test_t);

    CHECK_CLOSE_ABS_DIFF(1e1, test_t);
    CHECK_CLOSE_ABS_DIFF(0e1, test_t);
    CHECK_CLOSE_ABS_DIFF(-1e1, test_t);

    CHECK_CLOSE_ABS_DIFF(1.0e1, test_t);
    CHECK_CLOSE_ABS_DIFF(0.0e1, test_t);
    CHECK_CLOSE_ABS_DIFF(-1.0e1, test_t);

    CHECK_CLOSE_ABS_DIFF(1e-1, test_t);
    CHECK_CLOSE_ABS_DIFF(0e-1, test_t);
    CHECK_CLOSE_ABS_DIFF(-1e-1, test_t);

    CHECK_CLOSE_ABS_DIFF(1.0e-1, test_t);
    CHECK_CLOSE_ABS_DIFF(0.0e-1, test_t);
    CHECK_CLOSE_ABS_DIFF(-1.0e-1, test_t);

    CHECK_CLOSE_ABS_DIFF(1E1, test_t);
    CHECK_CLOSE_ABS_DIFF(0E1, test_t);
    CHECK_CLOSE_ABS_DIFF(-1E1, test_t);

    CHECK_CLOSE_ABS_DIFF(1.0E1, test_t);
    CHECK_CLOSE_ABS_DIFF(0.0E1, test_t);
    CHECK_CLOSE_ABS_DIFF(-1.0E1, test_t);

    CHECK_CLOSE_ABS_DIFF(1E-1, test_t);
    CHECK_CLOSE_ABS_DIFF(0E-1, test_t);
    CHECK_CLOSE_ABS_DIFF(-1E-1, test_t);

    CHECK_CLOSE_ABS_DIFF(1.0E-1, test_t);
    CHECK_CLOSE_ABS_DIFF(0.0E-1, test_t);
    CHECK_CLOSE_ABS_DIFF(-1.0E-1, test_t);

    CHECK_CLOSE_ABS_DIFF(.0E-1, test_t);
    CHECK_CLOSE_ABS_DIFF(.0E-1, test_t);
    CHECK_CLOSE_ABS_DIFF(-.0E-1, test_t);

    CHECK_CLOSE_ABS_DIFF(10.0, test_t);
    CHECK_CLOSE_ABS_DIFF(00.0, test_t);
    CHECK_CLOSE_ABS_DIFF(-10.0, test_t);

    CHECK_CLOSE_ABS_DIFF(10e1, test_t);
    CHECK_CLOSE_ABS_DIFF(00e1, test_t);
    CHECK_CLOSE_ABS_DIFF(-10e1, test_t);

    CHECK_CLOSE_ABS_DIFF(10.0e1, test_t);
    CHECK_CLOSE_ABS_DIFF(00.0e1, test_t);
    CHECK_CLOSE_ABS_DIFF(-10.0e1, test_t);

    CHECK_CLOSE_ABS_DIFF(10e-1, test_t);
    CHECK_CLOSE_ABS_DIFF(00e-1, test_t);
    CHECK_CLOSE_ABS_DIFF(-10e-1, test_t);

    CHECK_CLOSE_ABS_DIFF(10.0e-1, test_t);
    CHECK_CLOSE_ABS_DIFF(00.0e-1, test_t);
    CHECK_CLOSE_ABS_DIFF(-10.0e-1, test_t);

    CHECK_CLOSE_ABS_DIFF(10E1, test_t);
    CHECK_CLOSE_ABS_DIFF(00E1, test_t);
    CHECK_CLOSE_ABS_DIFF(-10E1, test_t);

    CHECK_CLOSE_ABS_DIFF(10.0E1, test_t);
    CHECK_CLOSE_ABS_DIFF(00.0E1, test_t);
    CHECK_CLOSE_ABS_DIFF(-10.0E1, test_t);

    CHECK_CLOSE_ABS_DIFF(10E-1, test_t);
    CHECK_CLOSE_ABS_DIFF(00E-1, test_t);
    CHECK_CLOSE_ABS_DIFF(-10E-1, test_t);

    CHECK_CLOSE_ABS_DIFF(10.0E-1, test_t);
    CHECK_CLOSE_ABS_DIFF(00.0E-1, test_t);
    CHECK_CLOSE_ABS_DIFF(-10.0E-1, test_t);

    CHECK_CLOSE_ABS_DIFF(-10101.0E-011, test_t);
    CHECK_CLOSE_ABS_DIFF(-10101093, test_t);
    CHECK_CLOSE_ABS_DIFF(10101093, test_t);

    CHECK_CLOSE_ABS_DIFF(-.34, test_t);
    CHECK_CLOSE_ABS_DIFF(.34, test_t);
    CHECK_CLOSE_ABS_DIFF(.34e10, test_t);

    HPX_TEST_THROW(lexical_cast<test_t>("-1.e"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>("-1.E"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>("1.e"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>("1.E"), bad_lexical_cast);

    HPX_TEST_THROW(lexical_cast<test_t>("1.0e"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>("1.0E"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>("10E"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>("10e"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>("1.0e-"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>("1.0E-"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>("10E-"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>("10e-"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>("e1"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>("e-1"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>("e-"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>(".e"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>(".1111111111111111111111111111111111111"
                                        "1111111111111111111111111111111ee"),
        bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>(".1111111111111111111111111111111111111"
                                        "1111111111111111111111111111111e-"),
        bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>("."), bad_lexical_cast);

    HPX_TEST_THROW(lexical_cast<test_t>("-B"), bad_lexical_cast);

    // Following two tests are not valid for C++11 compilers
    //HPX_TEST_THROW(lexical_cast<test_t>("0xB"), bad_lexical_cast);
    //HPX_TEST_THROW(lexical_cast<test_t>("0x0"), bad_lexical_cast);

    HPX_TEST_THROW(lexical_cast<test_t>("--1.0"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>("1.0e--1"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>("1.0.0"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>("1e1e1"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>("1.0e-1e-1"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>(" 1.0"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>("1.0 "), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>(""), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>("-"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>('\0'), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>('-'), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>('.'), bad_lexical_cast);
}

template <class T>
void test_float_typess_for_overflows()
{
    typedef T test_t;
    test_t minvalue = (std::numeric_limits<test_t>::min)();
    std::string s_min_value = lexical_cast<std::string>(minvalue);
    HPX_TEST_CLOSE_FRACTION(minvalue, lexical_cast<test_t>(minvalue));
    HPX_TEST_CLOSE_FRACTION(minvalue, lexical_cast<test_t>(s_min_value));

    test_t maxvalue = (std::numeric_limits<test_t>::max)();
    std::string s_max_value = lexical_cast<std::string>(maxvalue);
    HPX_TEST_CLOSE_FRACTION(maxvalue, lexical_cast<test_t>(maxvalue));
    HPX_TEST_CLOSE_FRACTION(maxvalue, lexical_cast<test_t>(s_max_value));

#ifndef _LIBCPP_VERSION
    // libc++ had a bug in implementation of stream conversions for values that
    // must be represented as infinity.
    // http://llvm.org/bugs/show_bug.cgi?id=15723#c4
    HPX_TEST_THROW(lexical_cast<test_t>(s_max_value + "1"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>(s_max_value + "9"), bad_lexical_cast);

    // VC9 can fail the following tests on floats and doubles when using stingstream
    HPX_TEST_THROW(lexical_cast<test_t>("1" + s_max_value), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<test_t>("9" + s_max_value), bad_lexical_cast);
#endif

    if (std::is_same<test_t, float>::value)
    {
        HPX_TEST_THROW(
            lexical_cast<test_t>((std::numeric_limits<double>::max)()),
            bad_lexical_cast);
        HPX_TEST((std::numeric_limits<double>::min)() -
                    std::numeric_limits<test_t>::epsilon() <=
                lexical_cast<test_t>((std::numeric_limits<double>::min)()) &&
            lexical_cast<test_t>((std::numeric_limits<double>::min)()) <=
                (std::numeric_limits<double>::min)() +
                    std::numeric_limits<test_t>::epsilon());
    }

    if (sizeof(test_t) < sizeof(long double))
    {
        HPX_TEST_THROW(
            lexical_cast<test_t>((std::numeric_limits<long double>::max)()),
            bad_lexical_cast);
        HPX_TEST((std::numeric_limits<long double>::min)() -
                    std::numeric_limits<test_t>::epsilon() <=
                lexical_cast<test_t>(
                    (std::numeric_limits<long double>::min)()) &&
            lexical_cast<test_t>((std::numeric_limits<long double>::min)()) <=
                (std::numeric_limits<long double>::min)() +
                    std::numeric_limits<test_t>::epsilon());
    }
}

#undef CHECK_CLOSE_ABS_DIFF

// Epsilon is multiplied by 2 because of two lexical conversions
#define TEST_TO_FROM_CAST_AROUND_TYPED(VAL, STRING_TYPE)                       \
    test_value = VAL + std::numeric_limits<test_t>::epsilon() * i;             \
    converted_val =                                                            \
        lexical_cast<test_t>(lexical_cast<STRING_TYPE>(test_value));           \
    HPX_TEST_CLOSE_FRACTION(test_value, converted_val);

/*
 * For interval [ from_mult*epsilon+VAL, to_mult*epsilon+VAL ], converts float
 * type numbers to string and then back to float type, then compares initial
 * values and generated.
 * Step is epsilon
 */
#define TEST_TO_FROM_CAST_AROUND(VAL)                                          \
    for (i = from_mult; i <= to_mult; ++i)                                     \
    {                                                                          \
        TEST_TO_FROM_CAST_AROUND_TYPED(VAL, std::string)                       \
    }

template <class TestType>
void test_converion_from_to_float_types()
{
    typedef TestType test_t;
    test_t test_value;
    test_t converted_val;

    int i;
    int from_mult = -50;
    int to_mult = 50;

    TEST_TO_FROM_CAST_AROUND(0.0);

    long double val1;
    for (val1 = 1.0e-10L; val1 < 1e11; val1 *= 10)
        TEST_TO_FROM_CAST_AROUND(val1);

    long double val2;
    for (val2 = -1.0e-10L; val2 > -1e11; val2 *= 10)
        TEST_TO_FROM_CAST_AROUND(val2);

    from_mult = -100;
    to_mult = 0;
    TEST_TO_FROM_CAST_AROUND((std::numeric_limits<test_t>::max)());

    from_mult = 0;
    to_mult = 100;
    TEST_TO_FROM_CAST_AROUND((std::numeric_limits<test_t>::min)());
}

#undef TEST_TO_FROM_CAST_AROUND
#undef TEST_TO_FROM_CAST_AROUND_TYPED

template <class T>
void test_conversion_from_float_to_char(char zero)
{
    HPX_TEST(lexical_cast<char>(static_cast<T>(0)) == zero + 0);
    HPX_TEST(lexical_cast<char>(static_cast<T>(1)) == zero + 1);
    HPX_TEST(lexical_cast<char>(static_cast<T>(2)) == zero + 2);
    HPX_TEST(lexical_cast<char>(static_cast<T>(3)) == zero + 3);
    HPX_TEST(lexical_cast<char>(static_cast<T>(4)) == zero + 4);
    HPX_TEST(lexical_cast<char>(static_cast<T>(5)) == zero + 5);
    HPX_TEST(lexical_cast<char>(static_cast<T>(6)) == zero + 6);
    HPX_TEST(lexical_cast<char>(static_cast<T>(7)) == zero + 7);
    HPX_TEST(lexical_cast<char>(static_cast<T>(8)) == zero + 8);
    HPX_TEST(lexical_cast<char>(static_cast<T>(9)) == zero + 9);

    HPX_TEST_THROW(lexical_cast<char>(static_cast<T>(10)), bad_lexical_cast);

    T t = (std::numeric_limits<T>::max)();
    HPX_TEST_THROW(lexical_cast<char>(t), bad_lexical_cast);
}

template <class T>
void test_conversion_from_char_to_float(char zero)
{
    HPX_TEST_CLOSE_FRACTION(
        lexical_cast<T>(static_cast<char>(zero + 0)), static_cast<T>(0));
    HPX_TEST_CLOSE_FRACTION(
        lexical_cast<T>(static_cast<char>(zero + 1)), static_cast<T>(1));
    HPX_TEST_CLOSE_FRACTION(
        lexical_cast<T>(static_cast<char>(zero + 2)), static_cast<T>(2));
    HPX_TEST_CLOSE_FRACTION(
        lexical_cast<T>(static_cast<char>(zero + 3)), static_cast<T>(3));
    HPX_TEST_CLOSE_FRACTION(
        lexical_cast<T>(static_cast<char>(zero + 4)), static_cast<T>(4));
    HPX_TEST_CLOSE_FRACTION(
        lexical_cast<T>(static_cast<char>(zero + 5)), static_cast<T>(5));
    HPX_TEST_CLOSE_FRACTION(
        lexical_cast<T>(static_cast<char>(zero + 6)), static_cast<T>(6));
    HPX_TEST_CLOSE_FRACTION(
        lexical_cast<T>(static_cast<char>(zero + 7)), static_cast<T>(7));
    HPX_TEST_CLOSE_FRACTION(
        lexical_cast<T>(static_cast<char>(zero + 8)), static_cast<T>(8));
    HPX_TEST_CLOSE_FRACTION(
        lexical_cast<T>(static_cast<char>(zero + 9)), static_cast<T>(9));

    HPX_TEST_THROW(
        lexical_cast<T>(static_cast<char>(zero + 10)), bad_lexical_cast);
    HPX_TEST_THROW(
        lexical_cast<T>(static_cast<char>(zero - 1)), bad_lexical_cast);
}

template <class T>
void test_conversion_from_to_float()
{
    char const zero = '0';
    test_conversion_from_float_to_char<T>(zero);
    test_conversion_from_char_to_float<T>(zero);

    HPX_TEST_CLOSE_FRACTION(lexical_cast<T>("+1"), 1);
    HPX_TEST_CLOSE_FRACTION(lexical_cast<T>("+9"), 9);

    HPX_TEST_THROW(lexical_cast<T>("++1"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<T>("-+9"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<T>("--1"), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<T>("+-9"), bad_lexical_cast);

    test_converion_to_float_types<T>();
    test_float_typess_for_overflows<T>();
    test_converion_from_to_float_types<T>();
}

void test_conversion_from_to_float()
{
    test_conversion_from_to_float<float>();
}
void test_conversion_from_to_double()
{
    test_conversion_from_to_float<double>();
}
void test_conversion_from_to_long_double()
{
    test_conversion_from_to_float<long double>();
}

int main()
{
    test_conversion_from_to_float();
    test_conversion_from_to_double();
    test_conversion_from_to_long_double();

    return hpx::util::report_errors();
}
