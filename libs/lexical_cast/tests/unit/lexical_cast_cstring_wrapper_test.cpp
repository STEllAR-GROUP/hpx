//  Unit test for hpx::util::lexical_cast.
//
//  See http://www.boost.org for most recent version, including documentation.
//
//  Copyright Antony Polukhin, 2012-2019.
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
#include <hpx/lexical_cast/detail/cstring_wrapper.hpp>
#include <hpx/testing.hpp>

using namespace hpx::util;
using detail::cstring_wrapper;

struct class_with_user_defined_sream_operators
{
    int i;

    operator int() const
    {
        return i;
    }
};

inline std::istream& operator>>(
    std::istream& istr, class_with_user_defined_sream_operators& rhs)
{
    return istr >> rhs.i;
}

template <class RngT>
void do_test_cstring_wrapper_impl(const RngT& rng)
{
    HPX_TEST_EQ((lexical_cast<int>(rng)), 1);
    HPX_TEST_EQ((lexical_cast<int>(rng.data, rng.length)), 1);
    HPX_TEST_EQ((lexical_cast<unsigned int>(rng)), 1u);
    HPX_TEST_EQ((lexical_cast<unsigned int>(rng.data, rng.length)), 1u);
    HPX_TEST_EQ((lexical_cast<short>(rng)), 1);
    HPX_TEST_EQ((lexical_cast<short>(rng.data, rng.length)), 1);
    HPX_TEST_EQ((lexical_cast<unsigned short>(rng)), 1u);
    HPX_TEST_EQ((lexical_cast<unsigned short>(rng.data, rng.length)), 1u);
    HPX_TEST_EQ((lexical_cast<long int>(rng)), 1);
    HPX_TEST_EQ((lexical_cast<long int>(rng.data, rng.length)), 1);
    HPX_TEST_EQ((lexical_cast<unsigned long int>(rng)), 1u);
    HPX_TEST_EQ((lexical_cast<unsigned long int>(rng.data, rng.length)), 1u);
    HPX_TEST_EQ((lexical_cast<long long>(rng)), 1);
    HPX_TEST_EQ((lexical_cast<long long>(rng.data, rng.length)), 1);
    HPX_TEST_EQ((lexical_cast<unsigned long long>(rng)), 1u);
    HPX_TEST_EQ((lexical_cast<unsigned long long>(rng.data, rng.length)), 1u);

    HPX_TEST_EQ((lexical_cast<float>(rng)), 1.0f);
    HPX_TEST_EQ((lexical_cast<float>(rng.data, rng.length)), 1.0f);
    HPX_TEST_EQ((lexical_cast<double>(rng)), 1.0);
    HPX_TEST_EQ((lexical_cast<double>(rng.data, rng.length)), 1.0);
    HPX_TEST_EQ((lexical_cast<long double>(rng)), 1.0L);
    HPX_TEST_EQ((lexical_cast<long double>(rng.data, rng.length)), 1.0L);
    HPX_TEST_EQ(
        (lexical_cast<class_with_user_defined_sream_operators>(rng)), 1);
}

void test_it_range_using_any_chars(char* one, char* eleven)
{
    // Zero terminated
    cstring_wrapper rng1(one, 1);
    do_test_cstring_wrapper_impl(rng1);

    // Non zero terminated
    cstring_wrapper rng2(eleven, 1);
    do_test_cstring_wrapper_impl(rng2);
}

void test_it_range_using_char(char* one, char* eleven)
{
    cstring_wrapper rng1(one, 1);
    HPX_TEST_EQ(lexical_cast<std::string>(rng1), "1");

    cstring_wrapper rng2(eleven, 1);
    HPX_TEST_EQ(lexical_cast<std::string>(rng2), "1");

    HPX_TEST_EQ(lexical_cast<float>(rng2), 1.0f);
    HPX_TEST_EQ(lexical_cast<double>(rng2), 1.0);
    HPX_TEST_EQ(lexical_cast<long double>(rng2), 1.0L);
    HPX_TEST_EQ(lexical_cast<class_with_user_defined_sream_operators>(rng2), 1);
}

void test_char_cstring_wrappers()
{
    char data1[] = "1";
    char data2[] = "11";
    test_it_range_using_any_chars(data1, data2);
    test_it_range_using_char(data1, data2);
}
int main()
{
    test_char_cstring_wrappers();

    return hpx::util::report_errors();
}
