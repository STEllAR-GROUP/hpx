//  Copyright (c) 2016 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/traits/is_range.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
void array_range()
{
    typedef int range[3];

    HPX_TEST_MSG((hpx::traits::is_range<range>::value == true), "array");
    HPX_TEST_MSG((hpx::traits::is_range<range const>::value == true), "array-const");
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
    HPX_TEST_MSG((hpx::traits::is_range<range const>::value == true), "member-const");
}

///////////////////////////////////////////////////////////////////////////////
namespace adl
{
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
}

void adl_range()
{
    typedef adl::free range;

    HPX_TEST_MSG((hpx::traits::is_range<range>::value == true), "adl-const");
    HPX_TEST_MSG((hpx::traits::is_range<range const>::value == true), "adl-const");
}

///////////////////////////////////////////////////////////////////////////////
void vector_range()
{
    typedef std::vector<int> range;

    HPX_TEST_MSG((hpx::traits::is_range<range>::value == true), "vector");
    HPX_TEST_MSG((hpx::traits::is_range<range const>::value == true), "vector-const");
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    {
        array_range();
        member_range();
        adl_range();
        vector_range();
    }

    return hpx::util::report_errors();
}
