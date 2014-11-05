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
    void operator()(T && val) const
    {
        val = val + 1;
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename T>
void verify_values(hpx::vector<T> const& v, T const& val)
{
    typedef typename hpx::vector<T>::const_iterator const_iterator;

    std::size_t size = 0;

    const_iterator end = v.end();
    for (const_iterator it = v.begin(); it != end; ++it, ++size)
    {
        HPX_TEST_EQ(*it, val);
    }

    HPX_TEST_EQ(size, v.size());
}

template <typename ExPolicy, typename T>
void test_for_each(ExPolicy && policy, hpx::vector<T>& v, T val)
{
    hpx::parallel::for_each(hpx::parallel::seq, v.begin(), v.end(), pfo());
    verify_values(v, ++val);

    hpx::parallel::for_each(hpx::parallel::par, v.begin(), v.end(), pfo());
    verify_values(v, ++val);

    hpx::parallel::for_each(hpx::parallel::seq(hpx::parallel::task),
        v.begin(), v.end(), pfo()).get();
    verify_values(v, ++val);

    hpx::parallel::for_each(hpx::parallel::par(hpx::parallel::task),
        v.begin(), v.end(), pfo()).get();
    verify_values(v, ++val);
}

///////////////////////////////////////////////////////////////////////////////
template <typename T>
void for_each_tests()
{
    std::size_t const length = 12;

    {
        hpx::vector<T> v;
        hpx::parallel::for_each(hpx::parallel::seq, v.begin(), v.end(), pfo());
        hpx::parallel::for_each(hpx::parallel::par, v.begin(), v.end(), pfo());
        hpx::parallel::for_each(hpx::parallel::seq(hpx::parallel::task),
            v.begin(), v.end(), pfo()).get();
        hpx::parallel::for_each(hpx::parallel::par(hpx::parallel::task),
            v.begin(), v.end(), pfo()).get();
    }

    {
        hpx::vector<T> v(length, T(0));
        test_for_each(hpx::parallel::seq, v, T(0));
        test_for_each(hpx::parallel::par, v, T(4));
        test_for_each(hpx::parallel::seq(hpx::parallel::task), v, T(8));
        test_for_each(hpx::parallel::par(hpx::parallel::task), v, T(12));
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

