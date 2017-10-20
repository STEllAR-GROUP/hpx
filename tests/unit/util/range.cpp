//  Copyright (c) 2016 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/util/range.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
void array_range()
{
    int r[3] = { 0, 1, 2 };
    HPX_TEST_EQ(hpx::util::begin(r), &r[0]);
    HPX_TEST_EQ(hpx::util::end(r), &r[3]);

    int const cr[3] = { 0, 1, 2 };
    HPX_TEST_EQ(hpx::util::begin(cr), &cr[0]);
    HPX_TEST_EQ(hpx::util::end(cr), &cr[3]);
    HPX_TEST_EQ(hpx::util::size(cr), 3u);
    HPX_TEST_EQ(hpx::util::empty(cr), false);
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
    member r = member();
    HPX_TEST_EQ(hpx::util::begin(r), &r.x);
    HPX_TEST_EQ(hpx::util::end(r), &r.x + 1);

    member const cr = member();
    HPX_TEST_EQ(hpx::util::begin(cr), &cr.x);
    HPX_TEST_EQ(hpx::util::end(cr), &cr.x + 1);
    HPX_TEST_EQ(hpx::util::size(cr), 1u);
    HPX_TEST_EQ(hpx::util::empty(cr), false);
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
    adl::free r = adl::free();
    HPX_TEST_EQ(hpx::util::begin(r), &r.x);
    HPX_TEST_EQ(hpx::util::end(r), &r.x + 1);

    adl::free const cr = adl::free();
    HPX_TEST_EQ(hpx::util::begin(cr), &cr.x);
    HPX_TEST_EQ(hpx::util::end(cr), &cr.x + 1);
    HPX_TEST_EQ(hpx::util::size(cr), 1u);
    HPX_TEST_EQ(hpx::util::empty(cr), false);
}

///////////////////////////////////////////////////////////////////////////////
void vector_range()
{
    std::vector<int> r(3);
    HPX_TEST(hpx::util::begin(r) == r.begin());
    HPX_TEST(hpx::util::end(r) == r.end());

    std::vector<int> cr(3);
    HPX_TEST(hpx::util::begin(cr) == cr.begin());
    HPX_TEST(hpx::util::end(cr) == cr.end());
    HPX_TEST_EQ(hpx::util::size(cr), 3u);
    HPX_TEST_EQ(hpx::util::empty(cr), false);
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
