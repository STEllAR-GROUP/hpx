//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/components/vector/vector.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <hpx/include/parallel_for_each.hpp>
#include <hpx/parallel/segmented_algorithms/for_each.hpp>

///////////////////////////////////////////////////////////////////////////////
// Define the vector types to be used.
HPX_REGISTER_VECTOR(double);
HPX_REGISTER_VECTOR(int);

struct pfo
{
    template <typename T>
    void operator()(T& val) const
    {
        ++val;
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename T>
void for_each_tests()
{
    std::size_t const length = 12;

    {
        hpx::vector<T> v;
        hpx::parallel::for_each(hpx::parallel::seq, v.begin(), v.end(), pfo());
    }

//     {
//         hpx::vector<T> v(length);
//
//         HPX_TEST_EQ(
//             hpx::parallel::count(hpx::parallel::seq, v.begin(), v.end(), T(0)),
//             length);
//         hpx::parallel::for_each(hpx::parallel::seq, v.begin(), v.end(), pfo());
//         HPX_TEST_EQ(
//             hpx::parallel::count(hpx::parallel::seq, v.begin(), v.end(), T(1)),
//             length);
//     }
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    for_each_tests<double>();

    return 0;
}

