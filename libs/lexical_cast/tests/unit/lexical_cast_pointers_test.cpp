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
#include <hpx/testing.hpp>

#include <sstream>
#include <string>

using namespace hpx::util;

typedef std::stringstream ss_t;

void test_void_pointers_conversions()
{
    void* p_to_null = nullptr;
    const void* cp_to_data = "Some data";
    char nonconst_data[5];
    void* p_to_data = nonconst_data;
    ss_t ss;

    ss << p_to_null;
    HPX_TEST_EQ(hpx::util::lexical_cast<std::string>(p_to_null), ss.str());
    ss.str(std::string());

    ss << cp_to_data;
    HPX_TEST_EQ(hpx::util::lexical_cast<std::string>(cp_to_data), ss.str());
    ss.str(std::string());

    ss << p_to_data;
    HPX_TEST_EQ(hpx::util::lexical_cast<std::string>(p_to_data), ss.str());
    ss.str(std::string());
}

struct incomplete_type;

void test_incomplete_type_pointers_conversions()
{
    incomplete_type* p_to_null = nullptr;
    const incomplete_type* cp_to_data = nullptr;
    char nonconst_data[5];
    incomplete_type* p_to_data =
        reinterpret_cast<incomplete_type*>(nonconst_data);
    ss_t ss;

    ss << p_to_null;
    HPX_TEST_EQ(hpx::util::lexical_cast<std::string>(p_to_null), ss.str());
    ss.str(std::string());

    ss << cp_to_data;
    HPX_TEST_EQ(hpx::util::lexical_cast<std::string>(cp_to_data), ss.str());
    ss.str(std::string());

    ss << p_to_data;
    HPX_TEST_EQ(hpx::util::lexical_cast<std::string>(p_to_data), ss.str());
    ss.str(std::string());
}

struct ble;
typedef struct ble* meh;
std::ostream& operator<<(std::ostream& o, meh)
{
    o << "yay";
    return o;
}

void test_inomplete_type_with_overloaded_ostream_op()
{
    meh heh = nullptr;
    ss_t ss;
    ss << heh;
    HPX_TEST_EQ(hpx::util::lexical_cast<std::string>(heh), ss.str());
}

int main()
{
    test_void_pointers_conversions();
    test_incomplete_type_pointers_conversions();
    test_inomplete_type_with_overloaded_ostream_op();

    return hpx::util::report_errors();
}
