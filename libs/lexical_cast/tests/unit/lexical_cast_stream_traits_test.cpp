//  Unit test for hpx::util::lexical_cast.
//
//  See http://www.boost.org for most recent version, including documentation.
//
//  Copyright Antony Polukhin, 2012-2014.
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost
//  Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt).

#include <hpx/config.hpp>

#include <hpx/lexical_cast/detail/converter_lexical.hpp>
#include <hpx/lexical_cast/detail/cstring_wrapper.hpp>

using namespace hpx::util;

#include <hpx/testing.hpp>

template <class T>
static void test_optimized_types_to_string_const()
{
    typedef detail::lexical_cast_stream_traits<T, std::string> trait_1;
    HPX_TEST(!trait_1::is_source_input_not_optimized_t::value);
    HPX_TEST((std::is_same<typename trait_1::src_char_t, char>::value));
    HPX_TEST((std::is_same<typename trait_1::target_char_t, char>::value));
    HPX_TEST((std::is_same<typename trait_1::char_type, char>::value));
    HPX_TEST(!trait_1::is_source_input_not_optimized_t::value);

    typedef detail::lexical_cast_stream_traits<const T, std::string> trait_2;
    HPX_TEST(!trait_2::is_source_input_not_optimized_t::value);
    HPX_TEST((std::is_same<typename trait_2::src_char_t, char>::value));
    HPX_TEST((std::is_same<typename trait_2::target_char_t, char>::value));
    HPX_TEST((std::is_same<typename trait_2::char_type, char>::value));
    HPX_TEST(!trait_2::is_source_input_not_optimized_t::value);
}

template <class T>
static void test_optimized_types_to_string()
{
    test_optimized_types_to_string_const<T>();

    typedef detail::lexical_cast_stream_traits<std::string, T> trait_4;
    HPX_TEST(!trait_4::is_source_input_not_optimized_t::value);
    HPX_TEST((std::is_same<typename trait_4::src_char_t, char>::value));
    HPX_TEST((std::is_same<typename trait_4::target_char_t, char>::value));
    HPX_TEST((std::is_same<typename trait_4::char_type, char>::value));
    HPX_TEST(!trait_4::is_source_input_not_optimized_t::value);

    typedef detail::lexical_cast_stream_traits<const std::string, T> trait_5;
    HPX_TEST(!trait_5::is_source_input_not_optimized_t::value);
    HPX_TEST((std::is_same<typename trait_5::src_char_t, char>::value));
    HPX_TEST((std::is_same<typename trait_5::target_char_t, char>::value));
    HPX_TEST((std::is_same<typename trait_5::char_type, char>::value));
    HPX_TEST(!trait_5::is_source_input_not_optimized_t::value);
}

void test_metafunctions()
{
    test_optimized_types_to_string<bool>();
    test_optimized_types_to_string<char>();
    test_optimized_types_to_string<short>();
    test_optimized_types_to_string<unsigned short>();
    test_optimized_types_to_string<int>();
    test_optimized_types_to_string<unsigned int>();
    test_optimized_types_to_string<long>();
    test_optimized_types_to_string<unsigned long>();
    test_optimized_types_to_string<long long>();
    test_optimized_types_to_string<unsigned long long>();

    test_optimized_types_to_string<std::string>();
    test_optimized_types_to_string<char*>();
    //test_optimized_types_to_string<char[5]>();
    //test_optimized_types_to_string<char[1]>();
    test_optimized_types_to_string_const<detail::cstring_wrapper<char>>();

    test_optimized_types_to_string<std::array<char, 1>>();
    test_optimized_types_to_string<std::array<char, 5>>();

    test_optimized_types_to_string_const<std::array<const char, 1>>();
    test_optimized_types_to_string_const<std::array<const char, 5>>();
}

int main()
{
    test_metafunctions();

    return hpx::util::report_errors();
}
