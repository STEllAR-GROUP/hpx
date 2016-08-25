//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/util/coordinate.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range.hpp>

#include <cstddef>

///////////////////////////////////////////////////////////////////////////////
std::ptrdiff_t const N1 = 3;
std::ptrdiff_t const N2 = 7;
std::ptrdiff_t const N3 = 11;

///////////////////////////////////////////////////////////////////////////////
void test_bounds1()
{
    hpx::util::bounds<1> bnds(N1);

    {
        hpx::util::bounds_iterator<1> it = boost::begin(bnds);
        hpx::util::bounds_iterator<1> end = boost::end(bnds);

        std::ptrdiff_t count = 0;
        for (/**/; it != end; ++it)
        {
            ++count;
        }
        HPX_TEST_EQ(count, N1);
    }

    {
        hpx::util::bounds_iterator<1> it = boost::begin(bnds);
        hpx::util::bounds_iterator<1> end = boost::end(bnds);

        std::ptrdiff_t count = 0;
        for (/**/; it != end; it += 1)
        {
            ++count;
        }
        HPX_TEST_EQ(count, N1);
    }
}

void test_bounds2()
{
    hpx::util::bounds<2> bnds({ N1, N2 });

    {
        hpx::util::bounds_iterator<2> it = boost::begin(bnds);
        hpx::util::bounds_iterator<2> end = boost::end(bnds);

        std::ptrdiff_t count = 0;
        for (/**/; it != end; ++it)
        {
            ++count;
        }
        HPX_TEST_EQ(count, N1 * N2);
    }

    {
        hpx::util::bounds_iterator<2> it = boost::begin(bnds);
        hpx::util::bounds_iterator<2> end = boost::end(bnds);

        std::ptrdiff_t count = 0;
        for (/**/; it != end; it += 1)
        {
            ++count;
        }
        HPX_TEST_EQ(count, N1 * N2);
    }
}

void test_bounds3()
{
    hpx::util::bounds<3> bnds({ N1, N2, N3 });

    {
        hpx::util::bounds_iterator<3> it = boost::begin(bnds);
        hpx::util::bounds_iterator<3> end = boost::end(bnds);

        std::ptrdiff_t count = 0;
        for (/**/; it != end; ++it)
        {
            ++count;
        }
        HPX_TEST_EQ(count, N1 * N2 * N3);
    }

    {
        hpx::util::bounds_iterator<3> it = boost::begin(bnds);
        hpx::util::bounds_iterator<3> end = boost::end(bnds);

        std::ptrdiff_t count = 0;
        for (/**/; it != end; it += 1)
        {
            ++count;
        }
        HPX_TEST_EQ(count, N1 * N2 * N3);
    }
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    test_bounds1();
    test_bounds2();
    test_bounds3();

    return 0;
}

