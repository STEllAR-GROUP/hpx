//  Copyright (c) 2016 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/modules/iterator_support.hpp>
#include <hpx/modules/testing.hpp>

#include <array>
#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
void array_range()
{
    typedef int range[3];

    HPX_TEST_MSG((hpx::traits::is_range<range>::value == true), "array");
    HPX_TEST_MSG(
        (hpx::traits::is_range<range const>::value == true), "array-const");
}

///////////////////////////////////////////////////////////////////////////////
struct member
{
    int x;

    int* begin()
    {
        return &x;
    }

    int const* begin() const
    {
        return &x;
    }

    int* end()
    {
        return &x + 1;
    }

    int const* end() const
    {
        return &x + 1;
    }
};

void member_range()
{
    typedef member range;

    HPX_TEST_MSG((hpx::traits::is_range<range>::value == true), "member-const");
    HPX_TEST_MSG(
        (hpx::traits::is_range<range const>::value == true), "member-const");
}

///////////////////////////////////////////////////////////////////////////////
namespace adl {
    struct free
    {
        int x;
    };

    int* begin(free& r)
    {
        return &r.x;
    }

    int const* begin(free const& r)
    {
        return &r.x;
    }

    int* end(free& r)
    {
        return &r.x + 1;
    }

    int const* end(free const& r)
    {
        return &r.x + 1;
    }
}    // namespace adl

void adl_range()
{
    typedef adl::free range;

    HPX_TEST_MSG((hpx::traits::is_range<range>::value == true), "adl-const");
    HPX_TEST_MSG(
        (hpx::traits::is_range<range const>::value == true), "adl-const");
}

///////////////////////////////////////////////////////////////////////////////
void vector_range()
{
    typedef std::vector<int> range;

    HPX_TEST_MSG((hpx::traits::is_range<range>::value == true), "vector");
    HPX_TEST_MSG(
        (hpx::traits::is_range<range const>::value == true), "vector-const");
}

///////////////////////////////////////////////////////////////////////////////
void test_is_input_range()
{
    HPX_TEST_MSG((hpx::traits::is_input_range_v<std::vector<int>> == true),
        "vector is input_range");
    HPX_TEST_MSG(
        (hpx::traits::is_input_range_v<std::vector<int> const> == true),
        "vector const is input_range");

    HPX_TEST_MSG((hpx::traits::is_input_range_v<std::deque<int>> == true),
        "deque is input_range");
    HPX_TEST_MSG((hpx::traits::is_input_range_v<std::deque<int> const> == true),
        "deque const is input_range");

    HPX_TEST_MSG((hpx::traits::is_input_range_v<std::list<int>> == true),
        "list is input_range");
    HPX_TEST_MSG((hpx::traits::is_input_range_v<std::list<int> const> == true),
        "list const is input_range");

    HPX_TEST_MSG(
        (hpx::traits::is_input_range_v<std::forward_list<int>> == true),
        "forward_list is input_range");
    HPX_TEST_MSG(
        (hpx::traits::is_input_range_v<std::forward_list<int> const> == true),
        "forward_list const is input_range");

    HPX_TEST_MSG((hpx::traits::is_input_range_v<std::set<int>> == true),
        "set is input_range");
    HPX_TEST_MSG((hpx::traits::is_input_range_v<std::set<int> const> == true),
        "set const is input_range");

    HPX_TEST_MSG((hpx::traits::is_input_range_v<std::map<int, int>> == true),
        "map is input_range");
    HPX_TEST_MSG(
        (hpx::traits::is_input_range_v<std::map<int, int> const> == true),
        "map const is input_range");

    HPX_TEST_MSG(
        (hpx::traits::is_input_range_v<std::unordered_set<int>> == true),
        "unordered_set is input_range");
    HPX_TEST_MSG(
        (hpx::traits::is_input_range_v<std::unordered_set<int> const> == true),
        "unordered_set const is input_range");

    HPX_TEST_MSG(
        (hpx::traits::is_input_range_v<std::unordered_map<int, int>> == true),
        "unordered_map is input_range");
    HPX_TEST_MSG(
        (hpx::traits::is_input_range_v<std::unordered_map<int, int> const> ==
            true),
        "unordered_map const is input_range");

    HPX_TEST_MSG((hpx::traits::is_input_range_v<std::string> == true),
        "string is input_range");
    HPX_TEST_MSG((hpx::traits::is_input_range_v<std::string const> == true),
        "string const is input_range");

    HPX_TEST_MSG((hpx::traits::is_input_range_v<std::array<int, 3>> == true),
        "std::array is input_range");
    HPX_TEST_MSG(
        (hpx::traits::is_input_range_v<std::array<int, 3> const> == true),
        "std::array const is input_range");

    HPX_TEST_MSG((hpx::traits::is_input_range_v<int[3]> == true),
        "C-array is input_range");
    HPX_TEST_MSG((hpx::traits::is_input_range_v<int const[3]> == true),
        "C-array const is input_range");

    HPX_TEST_MSG((hpx::traits::is_input_range_v<int> == false),
        "int is not input_range");
}

///////////////////////////////////////////////////////////////////////////////
void test_is_forward_range()
{
    HPX_TEST_MSG((hpx::traits::is_forward_range_v<std::vector<int>> == true),
        "vector is forward_range");

    HPX_TEST_MSG((hpx::traits::is_forward_range_v<std::deque<int>> == true),
        "deque is forward_range");

    HPX_TEST_MSG((hpx::traits::is_forward_range_v<std::list<int>> == true),
        "list is forward_range");

    HPX_TEST_MSG(
        (hpx::traits::is_forward_range_v<std::forward_list<int>> == true),
        "forward_list is forward_range");

    HPX_TEST_MSG((hpx::traits::is_forward_range_v<std::set<int>> == true),
        "set is forward_range");

    HPX_TEST_MSG((hpx::traits::is_forward_range_v<std::map<int, int>> == true),
        "map is forward_range");

    HPX_TEST_MSG(
        (hpx::traits::is_forward_range_v<std::unordered_set<int>> == true),
        "unordered_set is forward_range");

    HPX_TEST_MSG(
        (hpx::traits::is_forward_range_v<std::unordered_map<int, int>> == true),
        "unordered_map is forward_range");

    HPX_TEST_MSG((hpx::traits::is_forward_range_v<std::string> == true),
        "string is forward_range");

    HPX_TEST_MSG((hpx::traits::is_forward_range_v<std::array<int, 3>> == true),
        "std::array is forward_range");

    HPX_TEST_MSG((hpx::traits::is_forward_range_v<int[3]> == true),
        "C-array is forward_range");

    HPX_TEST_MSG((hpx::traits::is_forward_range_v<int> == false),
        "int is not forward_range");
}

///////////////////////////////////////////////////////////////////////////////
void test_is_bidirectional_range()
{
    HPX_TEST_MSG(
        (hpx::traits::is_bidirectional_range_v<std::vector<int>> == true),
        "vector is bidirectional_range");

    HPX_TEST_MSG(
        (hpx::traits::is_bidirectional_range_v<std::deque<int>> == true),
        "deque is bidirectional_range");

    HPX_TEST_MSG(
        (hpx::traits::is_bidirectional_range_v<std::list<int>> == true),
        "list is bidirectional_range");

    HPX_TEST_MSG(
        (hpx::traits::is_bidirectional_range_v<std::forward_list<int>> ==
            false),
        "forward_list is not bidirectional_range");

    HPX_TEST_MSG((hpx::traits::is_bidirectional_range_v<std::set<int>> == true),
        "set is bidirectional_range");

    HPX_TEST_MSG(
        (hpx::traits::is_bidirectional_range_v<std::map<int, int>> == true),
        "map is bidirectional_range");

    HPX_TEST_MSG(
        (hpx::traits::is_bidirectional_range_v<std::unordered_set<int>> ==
            false),
        "unordered_set is not bidirectional_range");

    HPX_TEST_MSG(
        (hpx::traits::is_bidirectional_range_v<std::unordered_map<int, int>> ==
            false),
        "unordered_map is not bidirectional_range");

    HPX_TEST_MSG((hpx::traits::is_bidirectional_range_v<std::string> == true),
        "string is bidirectional_range");

    HPX_TEST_MSG(
        (hpx::traits::is_bidirectional_range_v<std::array<int, 3>> == true),
        "std::array is bidirectional_range");

    HPX_TEST_MSG((hpx::traits::is_bidirectional_range_v<int[3]> == true),
        "C-array is bidirectional_range");

    HPX_TEST_MSG((hpx::traits::is_bidirectional_range_v<int> == false),
        "int is not bidirectional_range");
}

///////////////////////////////////////////////////////////////////////////////
void test_is_random_access_range()
{
    HPX_TEST_MSG(
        (hpx::traits::is_random_access_range_v<std::vector<int>> == true),
        "vector is random_access_range");

    HPX_TEST_MSG(
        (hpx::traits::is_random_access_range_v<std::deque<int>> == true),
        "deque is random_access_range");

    HPX_TEST_MSG(
        (hpx::traits::is_random_access_range_v<std::list<int>> == false),
        "list is not random_access_range");

    HPX_TEST_MSG(
        (hpx::traits::is_random_access_range_v<std::forward_list<int>> ==
            false),
        "forward_list is not random_access_range");

    HPX_TEST_MSG(
        (hpx::traits::is_random_access_range_v<std::set<int>> == false),
        "set is not random_access_range");

    HPX_TEST_MSG(
        (hpx::traits::is_random_access_range_v<std::map<int, int>> == false),
        "map is not random_access_range");

    // std::unordered_set - forward iterator (not random access)
    HPX_TEST_MSG(
        (hpx::traits::is_random_access_range_v<std::unordered_set<int>> ==
            false),
        "unordered_set is not random_access_range");

    HPX_TEST_MSG(
        (hpx::traits::is_random_access_range_v<std::unordered_map<int, int>> ==
            false),
        "unordered_map is not random_access_range");

    HPX_TEST_MSG((hpx::traits::is_random_access_range_v<std::string> == true),
        "string is random_access_range");

    HPX_TEST_MSG(
        (hpx::traits::is_random_access_range_v<std::array<int, 3>> == true),
        "std::array is random_access_range");
    HPX_TEST_MSG(
        (hpx::traits::is_random_access_range_v<std::array<int, 3> const> ==
            true),
        "std::array const is random_access_range");

    HPX_TEST_MSG((hpx::traits::is_random_access_range_v<int[3]> == true),
        "C-array is random_access_range");

    HPX_TEST_MSG((hpx::traits::is_random_access_range_v<int> == false),
        "int is not random_access_range");
}

///////////////////////////////////////////////////////////////////////////////
// Tests for is_sized_range
void test_is_sized_range()
{
    HPX_TEST_MSG((hpx::traits::is_sized_range_v<std::vector<int>> == true),
        "vector is sized_range");

    HPX_TEST_MSG((hpx::traits::is_sized_range_v<std::deque<int>> == true),
        "deque is sized_range");

    HPX_TEST_MSG((hpx::traits::is_sized_range_v<std::list<int>> == true),
        "list is sized_range");

    HPX_TEST_MSG(
        (hpx::traits::is_sized_range_v<std::forward_list<int>> == false),
        "forward_list is not sized_range");

    HPX_TEST_MSG((hpx::traits::is_sized_range_v<std::set<int>> == true),
        "set is sized_range");

    HPX_TEST_MSG((hpx::traits::is_sized_range_v<std::map<int, int>> == true),
        "map is sized_range");

    HPX_TEST_MSG(
        (hpx::traits::is_sized_range_v<std::unordered_set<int>> == true),
        "unordered_set is sized_range");

    HPX_TEST_MSG(
        (hpx::traits::is_sized_range_v<std::unordered_map<int, int>> == true),
        "unordered_map is sized_range");

    HPX_TEST_MSG((hpx::traits::is_sized_range_v<std::string> == true),
        "string is sized_range");

    HPX_TEST_MSG((hpx::traits::is_sized_range_v<std::array<int, 3>> == true),
        "std::array is sized_range");

    HPX_TEST_MSG((hpx::traits::is_sized_range_v<int[3]> == true),
        "C-array is sized_range");

    HPX_TEST_MSG((hpx::traits::is_sized_range_v<int> == false),
        "int is not sized_range");
}

///////////////////////////////////////////////////////////////////////////////
void test_member_range_concepts()
{
    HPX_TEST_MSG((hpx::traits::is_input_range_v<member> == true),
        "member is input_range");
    HPX_TEST_MSG((hpx::traits::is_forward_range_v<member> == true),
        "member is forward_range");
    HPX_TEST_MSG((hpx::traits::is_bidirectional_range_v<member> == true),
        "member is bidirectional_range");
    HPX_TEST_MSG((hpx::traits::is_random_access_range_v<member> == true),
        "member is random_access_range");
    // member uses int* as iterator/sentinel, which are sized sentinels
    HPX_TEST_MSG((hpx::traits::is_sized_range_v<member> == true),
        "member is sized_range");
}

///////////////////////////////////////////////////////////////////////////////
void test_adl_range_concepts()
{
    HPX_TEST_MSG((hpx::traits::is_input_range_v<adl::free> == true),
        "adl::free is input_range");
    HPX_TEST_MSG((hpx::traits::is_forward_range_v<adl::free> == true),
        "adl::free is forward_range");
    HPX_TEST_MSG((hpx::traits::is_bidirectional_range_v<adl::free> == true),
        "adl::free is bidirectional_range");
    HPX_TEST_MSG((hpx::traits::is_random_access_range_v<adl::free> == true),
        "adl::free is random_access_range");
    // adl::free uses int* as iterator/sentinel, which are sized sentinels
    HPX_TEST_MSG((hpx::traits::is_sized_range_v<adl::free> == true),
        "adl::free is sized_range");
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    {
        array_range();
        member_range();
        adl_range();
        vector_range();
    }

    {
        test_is_input_range();
        test_is_forward_range();
        test_is_bidirectional_range();
        test_is_random_access_range();
        test_is_sized_range();
        test_member_range_concepts();
        test_adl_range_concepts();
    }

    return hpx::util::report_errors();
}
