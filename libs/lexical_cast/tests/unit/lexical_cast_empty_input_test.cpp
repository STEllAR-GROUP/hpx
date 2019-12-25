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
#include <hpx/lexical_cast/detail/cstring_wrapper.hpp>
#include <hpx/testing.hpp>

#include <string>
#include <vector>

using namespace hpx::util;

template <class T>
void do_test_on_empty_input(T& v)
{
    HPX_TEST_THROW(lexical_cast<int>(v), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<float>(v), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<double>(v), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<long double>(v), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<unsigned int>(v), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<unsigned short>(v), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<unsigned long long>(v), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<long long>(v), bad_lexical_cast);
}

void test_empty_cstring_wrapper()
{
    detail::cstring_wrapper<char> v;
    do_test_on_empty_input(v);
    HPX_TEST_EQ(lexical_cast<std::string>(v), std::string());
    HPX_TEST_THROW(lexical_cast<char>(v), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<unsigned char>(v), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<signed char>(v), bad_lexical_cast);

    const detail::cstring_wrapper<char> cv;
    do_test_on_empty_input(cv);
    HPX_TEST_EQ(lexical_cast<std::string>(cv), std::string());
    HPX_TEST_THROW(lexical_cast<char>(cv), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<unsigned char>(cv), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<signed char>(cv), bad_lexical_cast);
}

void test_empty_string()
{
    std::string v;
    do_test_on_empty_input(v);
    HPX_TEST_THROW(lexical_cast<char>(v), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<unsigned char>(v), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<signed char>(v), bad_lexical_cast);
}

struct Escaped
{
    Escaped(const std::string& s)
      : str_(s)
    {
    }

    std::string str_;
};

inline std::ostream& operator<<(std::ostream& o, const Escaped& rhs)
{
    return o << rhs.str_;
}

void test_empty_user_class()
{
    Escaped v("");
    do_test_on_empty_input(v);
    HPX_TEST_THROW(lexical_cast<char>(v), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<unsigned char>(v), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<signed char>(v), bad_lexical_cast);
}

namespace std {
    inline std::ostream& operator<<(
        std::ostream& out, const std::vector<long>& v)
    {
        std::ostream_iterator<long> it(out);
        std::copy(v.begin(), v.end(), it);
        assert(out);
        return out;
    }
}    // namespace std

void test_empty_vector()
{
    std::vector<long> v;
    do_test_on_empty_input(v);
    HPX_TEST_THROW(lexical_cast<char>(v), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<unsigned char>(v), bad_lexical_cast);
    HPX_TEST_THROW(lexical_cast<signed char>(v), bad_lexical_cast);
}

struct my_string
{
    friend std::ostream& operator<<(
        std::ostream& sout, my_string const& /* st*/)
    {
        return sout << "";
    }
};

void test_empty_zero_terminated_string()
{
    my_string st;
    HPX_TEST_EQ(hpx::util::lexical_cast<std::string>(st), std::string());
    ;
}

int main()
{
    test_empty_cstring_wrapper();
    test_empty_string();
    test_empty_user_class();
    test_empty_vector();
    test_empty_zero_terminated_string();

    return hpx::util::report_errors();
}
