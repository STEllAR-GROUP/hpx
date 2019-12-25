//  Unit test for hpx::util::lexical_cast.
//
//  See http://www.boost.org for most recent version, including documentation.
//
//  Copyright Alexander Nasonov, 2006.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost
//  Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt).
//
//  Test round-tripping conversion FPT -> string -> FPT,
//  where FPT is Floating Point Type.

#include <hpx/config.hpp>

#if defined(__INTEL_COMPILER)
#pragma warning(disable : 193 383 488 981 1418 1419)
#elif defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(                                                               \
    disable : 4097 4100 4121 4127 4146 4244 4245 4511 4512 4701 4800)
#endif

#include <hpx/lexical_cast.hpp>
#include <hpx/testing.hpp>

#include <string>

using namespace hpx::util;

template <class T>
void test_round_conversion()
{
    T epsilon = std::numeric_limits<T>::epsilon();
    std::string const epsilon_s = hpx::util::lexical_cast<std::string>(epsilon);
    HPX_TEST(epsilon == hpx::util::lexical_cast<T>(epsilon_s));

    T max_ = (std::numeric_limits<T>::max)();
    std::string const max_s = hpx::util::lexical_cast<std::string>(max_);
    HPX_TEST(max_ == hpx::util::lexical_cast<T>(max_s));

    T min_ = (std::numeric_limits<T>::min)();
    std::string const min_s = hpx::util::lexical_cast<std::string>(min_);
    HPX_TEST(min_ == hpx::util::lexical_cast<T>(min_s));

    T max_div137 = max_ / 137;
    std::string max_div137_s = hpx::util::lexical_cast<std::string>(max_div137);
    HPX_TEST(max_div137 == hpx::util::lexical_cast<T>(max_div137_s));

    T epsilon_mult137 = epsilon * 137;
    std::string epsilon_mult137_s(
        hpx::util::lexical_cast<std::string>(epsilon_mult137));
    HPX_TEST(epsilon_mult137 == hpx::util::lexical_cast<T>(epsilon_mult137_s));
}

// See bug http://tinyurl.com/vhpvo
template <class T>
void test_msvc_magic_values()
{
    T magic_msvc = 0.00010000433948393407;
    std::string magic_msvc_s = hpx::util::lexical_cast<std::string>(magic_msvc);
    HPX_TEST(magic_msvc == hpx::util::lexical_cast<T>(magic_msvc_s));
}

void test_round_conversion_float()
{
    test_round_conversion<float>();
}

void test_round_conversion_double()
{
    test_round_conversion<double>();
    test_msvc_magic_values<double>();
}

void test_round_conversion_long_double()
{
    test_round_conversion<long double>();
    test_msvc_magic_values<long double>();
}

int main()
{
    test_round_conversion_float();
    test_round_conversion_double();
    test_round_conversion_long_double();

    return hpx::util::report_errors();
}
