//  Testing hpx::util::lexical_cast with arrays.
//
//  See http://www.boost.org for most recent version, including documentation.
//
//  Copyright Antony Polukhin, 2012-2019.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost
//  Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt).

#include <hpx/lexical_cast.hpp>
#include <hpx/testing.hpp>

#include <array>
#include <string>

using namespace hpx::util;

template <class T>
static void testing_template_array_output_on_spec_value(T val)
{
    typedef std::array<char, 300> arr_type;
    typedef std::array<char, 1> short_arr_type;

    std::string ethalon("100");
    using namespace std;

    {
        arr_type res1 = lexical_cast<arr_type>(val);
        HPX_TEST_EQ(&res1[0], ethalon);
        const arr_type res2 = lexical_cast<arr_type>(val);
        HPX_TEST_EQ(&res2[0], ethalon);
        HPX_TEST_THROW(
            lexical_cast<short_arr_type>(val), hpx::util::bad_lexical_cast);
    }
}

static void testing_template_array_output_on_char_value()
{
    typedef std::array<char, 300> arr_type;
    typedef std::array<char, 1> short_arr_type;

    const char val[] = "100";
    std::string ethalon("100");
    using namespace std;

    {
        arr_type res1 = lexical_cast<arr_type>(val);
        HPX_TEST_EQ(&res1[0], ethalon);
        const arr_type res2 = lexical_cast<arr_type>(val);
        HPX_TEST_EQ(&res2[0], ethalon);
        HPX_TEST_THROW(
            lexical_cast<short_arr_type>(val), hpx::util::bad_lexical_cast);
    }
}

void testing_std_array_output_conversion()
{
    testing_template_array_output_on_char_value();
    testing_template_array_output_on_spec_value(100);
    testing_template_array_output_on_spec_value(static_cast<short>(100));
    testing_template_array_output_on_spec_value(
        static_cast<unsigned short>(100));
    testing_template_array_output_on_spec_value(static_cast<unsigned int>(100));

    HPX_TEST(true);
}

static void testing_array_input_conversion()
{
    {
        std::array<char, 4> var_zero_terminated = {{'1', '0', '0', '\0'}};
        HPX_TEST_EQ(lexical_cast<std::string>(var_zero_terminated), "100");
        HPX_TEST_EQ(lexical_cast<int>(var_zero_terminated), 100);

        std::array<char, 3> var_none_terminated = {{'1', '0', '0'}};
        HPX_TEST_EQ(lexical_cast<std::string>(var_none_terminated), "100");
        HPX_TEST_EQ(
            lexical_cast<short>(var_none_terminated), static_cast<short>(100));

        std::array<const char, 4> var_zero_terminated_const_char = {
            {'1', '0', '0', '\0'}};
        HPX_TEST_EQ(
            lexical_cast<std::string>(var_zero_terminated_const_char), "100");

        std::array<const char, 3> var_none_terminated_const_char = {
            {'1', '0', '0'}};
        HPX_TEST_EQ(
            lexical_cast<std::string>(var_none_terminated_const_char), "100");

        const std::array<char, 4> var_zero_terminated_const_var = {
            {'1', '0', '0', '\0'}};
        HPX_TEST_EQ(
            lexical_cast<std::string>(var_zero_terminated_const_var), "100");

        const std::array<char, 3> var_none_terminated_const_var = {
            {'1', '0', '0'}};
        HPX_TEST_EQ(
            lexical_cast<std::string>(var_none_terminated_const_var), "100");

        const std::array<const char, 4>
            var_zero_terminated_const_var_const_char = {{'1', '0', '0', '\0'}};
        HPX_TEST_EQ(
            lexical_cast<std::string>(var_zero_terminated_const_var_const_char),
            "100");

        const std::array<const char, 3>
            var_none_terminated_const_var_const_char = {{'1', '0', '0'}};
        HPX_TEST_EQ(
            lexical_cast<std::string>(var_none_terminated_const_var_const_char),
            "100");
        HPX_TEST_EQ(
            lexical_cast<int>(var_none_terminated_const_var_const_char), 100);
    }
}

void testing_std_array_input_conversion()
{
    testing_array_input_conversion();

    HPX_TEST(true);
}

int main()
{
    testing_std_array_output_conversion();
    testing_std_array_input_conversion();

    return hpx::util::report_errors();
}
