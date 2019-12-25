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

#include <hpx/lexical_cast.hpp>
#include <hpx/testing.hpp>

#include <istream>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>

///////////////////////// char streamable classes ///////////////////////////////////////////

struct streamable_easy
{
    enum ENU
    {
        value = 0
    };
};
std::ostream& operator<<(std::ostream& ostr, const streamable_easy&)
{
    return ostr << streamable_easy::value;
}
std::istream& operator>>(std::istream& istr, const streamable_easy&)
{
    int i;
    istr >> i;
    HPX_TEST_EQ(i, streamable_easy::value);
    return istr;
}

struct streamable_medium
{
    enum ENU
    {
        value = 1
    };
};
template <class CharT>
typename std::enable_if<std::is_same<CharT, char>::value,
    std::basic_ostream<CharT>&>::type
operator<<(std::basic_ostream<CharT>& ostr, const streamable_medium&)
{
    return ostr << streamable_medium::value;
}
template <class CharT>
typename std::enable_if<std::is_same<CharT, char>::value,
    std::basic_istream<CharT>&>::type
operator>>(std::basic_istream<CharT>& istr, const streamable_medium&)
{
    int i;
    istr >> i;
    HPX_TEST_EQ(i, streamable_medium::value);
    return istr;
}

struct streamable_hard
{
    enum ENU
    {
        value = 2
    };
};
template <class CharT, class TraitsT>
typename std::enable_if<std::is_same<CharT, char>::value,
    std::basic_ostream<CharT, TraitsT>&>::type
operator<<(std::basic_ostream<CharT, TraitsT>& ostr, const streamable_hard&)
{
    return ostr << streamable_hard::value;
}
template <class CharT, class TraitsT>
typename std::enable_if<std::is_same<CharT, char>::value,
    std::basic_istream<CharT, TraitsT>&>::type
operator>>(std::basic_istream<CharT, TraitsT>& istr, const streamable_hard&)
{
    int i;
    istr >> i;
    HPX_TEST_EQ(i, streamable_hard::value);
    return istr;
}

struct streamable_hard2
{
    enum ENU
    {
        value = 3
    };
};
template <class TraitsT>
std::basic_ostream<char, TraitsT>& operator<<(
    std::basic_ostream<char, TraitsT>& ostr, const streamable_hard2&)
{
    return ostr << streamable_hard2::value;
}
template <class TraitsT>
std::basic_istream<char, TraitsT>& operator>>(
    std::basic_istream<char, TraitsT>& istr, const streamable_hard2&)
{
    int i;
    istr >> i;
    HPX_TEST_EQ(i, streamable_hard2::value);
    return istr;
}

template <class T>
static void test_ostr_impl()
{
    T streamable;
    HPX_TEST_EQ(T::value, hpx::util::lexical_cast<int>(streamable));
    HPX_TEST_EQ(hpx::util::lexical_cast<std::string>(T::value),
        hpx::util::lexical_cast<std::string>(streamable));
}

void test_ostream_character_detection()
{
    test_ostr_impl<streamable_easy>();
    test_ostr_impl<streamable_medium>();
    test_ostr_impl<streamable_hard>();
    test_ostr_impl<streamable_hard2>();
}

template <class T>
static void test_istr_impl()
{
    hpx::util::lexical_cast<T>(T::value);
    hpx::util::lexical_cast<T>(hpx::util::lexical_cast<std::string>(T::value));
}

template <class T>
static void test_wistr_impl()
{
    hpx::util::lexical_cast<T>(T::value);
    //hpx::util::lexical_cast<T>(hpx::util::lexical_cast<std::string>(T::value)); // Shall not compile???
    hpx::util::lexical_cast<T>(hpx::util::lexical_cast<std::wstring>(T::value));
}

void test_istream_character_detection()
{
    test_istr_impl<streamable_easy>();
    test_istr_impl<streamable_medium>();
    test_istr_impl<streamable_hard>();
    test_istr_impl<streamable_hard2>();
}

int main()
{
    test_ostream_character_detection();
    test_istream_character_detection();

    return hpx::util::report_errors();
}
