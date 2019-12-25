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
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/math/special_functions/sign.hpp>

using namespace hpx::util;

template <class T>
bool is_pos_inf(T value)
{
    return (boost::math::isinf)(value) && !(boost::math::signbit)(value);
}

template <class T>
bool is_neg_inf(T value)
{
    return (boost::math::isinf)(value) && (boost::math::signbit)(value);
}

template <class T>
bool is_pos_nan(T value)
{
    return (boost::math::isnan)(value) && !(boost::math::signbit)(value);
}

template <class T>
bool is_neg_nan(T value)
{
    /* There is some strange behaviour on Itanium platform with -nan nuber for
    * long double. It is a IA64 feature, not a lexical_cast bug */
#if defined(__ia64__) || defined(_M_IA64)
    return (boost::math::isnan)(value) &&
        (std::is_same<T, long double>::value || (boost::math::signbit)(value));
#else
    return (boost::math::isnan)(value) && (boost::math::signbit)(value);
#endif
}

template <class T>
void test_inf_nan_templated()
{
    typedef T test_t;

    HPX_TEST(is_pos_inf(lexical_cast<test_t>("inf")));
    HPX_TEST(is_pos_inf(lexical_cast<test_t>("INF")));

    HPX_TEST(is_neg_inf(lexical_cast<test_t>("-inf")));
    HPX_TEST(is_neg_inf(lexical_cast<test_t>("-INF")));

    HPX_TEST(is_pos_inf(lexical_cast<test_t>("+inf")));
    HPX_TEST(is_pos_inf(lexical_cast<test_t>("+INF")));

    HPX_TEST(is_pos_inf(lexical_cast<test_t>("infinity")));
    HPX_TEST(is_pos_inf(lexical_cast<test_t>("INFINITY")));

    HPX_TEST(is_neg_inf(lexical_cast<test_t>("-infinity")));
    HPX_TEST(is_neg_inf(lexical_cast<test_t>("-INFINITY")));

    HPX_TEST(is_pos_inf(lexical_cast<test_t>("+infinity")));
    HPX_TEST(is_pos_inf(lexical_cast<test_t>("+INFINITY")));

    HPX_TEST(is_pos_inf(lexical_cast<test_t>("iNfiNity")));
    HPX_TEST(is_pos_inf(lexical_cast<test_t>("INfinity")));

    HPX_TEST(is_neg_inf(lexical_cast<test_t>("-inFINITY")));
    HPX_TEST(is_neg_inf(lexical_cast<test_t>("-INFINITY")));

    HPX_TEST(is_pos_nan(lexical_cast<test_t>("nan")));
    HPX_TEST(is_pos_nan(lexical_cast<test_t>("NAN")));

    HPX_TEST(is_neg_nan(lexical_cast<test_t>("-nan")));
    HPX_TEST(is_neg_nan(lexical_cast<test_t>("-NAN")));

    HPX_TEST(is_pos_nan(lexical_cast<test_t>("+nan")));
    HPX_TEST(is_pos_nan(lexical_cast<test_t>("+NAN")));

    HPX_TEST(is_pos_nan(lexical_cast<test_t>("nAn")));
    HPX_TEST(is_pos_nan(lexical_cast<test_t>("NaN")));

    HPX_TEST(is_neg_nan(lexical_cast<test_t>("-nAn")));
    HPX_TEST(is_neg_nan(lexical_cast<test_t>("-NaN")));

    HPX_TEST(is_pos_nan(lexical_cast<test_t>("+Nan")));
    HPX_TEST(is_pos_nan(lexical_cast<test_t>("+nAN")));

    HPX_TEST(is_pos_nan(lexical_cast<test_t>("nan()")));
    HPX_TEST(is_pos_nan(lexical_cast<test_t>("NAN(some string)")));
    HPX_TEST_THROW(lexical_cast<test_t>("NAN(some string"), bad_lexical_cast);

    HPX_TEST(lexical_cast<std::string>((boost::math::changesign)(
                 std::numeric_limits<test_t>::infinity())) == "-inf");
    HPX_TEST(lexical_cast<std::string>(
                 std::numeric_limits<test_t>::infinity()) == "inf");
    HPX_TEST(lexical_cast<std::string>(
                 std::numeric_limits<test_t>::quiet_NaN()) == "nan");
#if !defined(__ia64__) && !defined(_M_IA64)
    HPX_TEST(lexical_cast<std::string>((boost::math::changesign)(
                 std::numeric_limits<test_t>::quiet_NaN())) == "-nan");
#endif
}

void test_inf_nan_float()
{
    test_inf_nan_templated<float>();
}

void test_inf_nan_double()
{
    test_inf_nan_templated<double>();
}

void test_inf_nan_long_double()
{
    test_inf_nan_templated<long double>();
}

int main()
{
    test_inf_nan_float();
    test_inf_nan_double();
    test_inf_nan_long_double();

    return hpx::util::report_errors();
}
