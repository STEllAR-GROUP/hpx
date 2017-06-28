//  Copyright (c) 2017 Ajai V George
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/partitioned_vector.hpp>
#include <hpx/include/parallel_find.hpp>
#include <hpx/include/parallel_scan.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <vector>
#include <iostream>
///////////////////////////////////////////////////////////////////////////////
// Define the vector types to be used.
HPX_REGISTER_PARTITIONED_VECTOR(double);
HPX_REGISTER_PARTITIONED_VECTOR(int);

///////////////////////////////////////////////////////////////////////////////

template<typename T>
struct cond1
{
    bool operator()(T v) const
    {
        return v > T(511);
    }
};

template<typename T>
struct cond2
{
    bool operator()(T v) const
    {
        return v < T(512);
    }
};
template <typename ExPolicy, typename T>
void test_find(ExPolicy && policy,
    hpx::partitioned_vector<T> & xvalues, T val)
{
    auto last = hpx::parallel::find(policy, xvalues.begin(),
        xvalues.end(), val);
    HPX_TEST_EQ(*last,val);
}

template <typename ExPolicy, typename T>
void test_find_async(ExPolicy && policy,
    hpx::partitioned_vector<T> & xvalues, T val)
{
    auto last = hpx::parallel::find(policy, xvalues.begin(),
        xvalues.end(), val).get();
    HPX_TEST_EQ(*last,val);
}

template <typename ExPolicy, typename T>
void test_find_if(ExPolicy && policy,
    hpx::partitioned_vector<T> & xvalues, T val)
{
    auto last = hpx::parallel::find_if(policy, xvalues.begin(),
        xvalues.end(), cond1<T>());
    HPX_TEST_EQ(*last,val);
}

template <typename ExPolicy, typename T>
void test_find_if_async(ExPolicy && policy,
    hpx::partitioned_vector<T> & xvalues, T val)
{
    auto last = hpx::parallel::find_if(policy, xvalues.begin(),
        xvalues.end(), cond1<T>()).get();
    HPX_TEST_EQ(*last,val);
}

template <typename ExPolicy, typename T>
void test_find_if_not(ExPolicy && policy,
    hpx::partitioned_vector<T> & xvalues, T val)
{
    auto last = hpx::parallel::find_if_not(policy, xvalues.begin(),
        xvalues.end(), cond2<T>());
    HPX_TEST_EQ(*last,val);
}

template <typename ExPolicy, typename T>
void test_find_if_not_async(ExPolicy && policy,
    hpx::partitioned_vector<T> & xvalues, T val)
{
    auto last = hpx::parallel::find_if_not(policy, xvalues.begin(),
        xvalues.end(), cond2<T>()).get();
    HPX_TEST_EQ(*last,val);
}


template <typename T>
void find_tests(std::vector<hpx::id_type> &localities)
{
    std::size_t const num = 1000;
    hpx::partitioned_vector<T> xvalues(num, T(1),hpx::container_layout(localities));
    hpx::parallel::inclusive_scan(hpx::parallel::execution::seq, xvalues.begin(),
        xvalues.end(), xvalues.begin(), T(0), std::plus<T>());

    test_find(hpx::parallel::execution::seq, xvalues, T(512));
    test_find(hpx::parallel::execution::par, xvalues, T(512));
    test_find_async(hpx::parallel::execution::seq(hpx::parallel::execution::task),
        xvalues, T(512));
    test_find_async(hpx::parallel::execution::par(hpx::parallel::execution::task),
        xvalues, T(512));

    test_find_if(hpx::parallel::execution::seq, xvalues, T(512));
    test_find_if(hpx::parallel::execution::par, xvalues, T(512));
    test_find_if_async(hpx::parallel::execution::seq(hpx::parallel::execution::task),
        xvalues, T(512));
    test_find_if_async(hpx::parallel::execution::par(hpx::parallel::execution::task),
        xvalues, T(512));

    test_find_if_not(hpx::parallel::execution::seq, xvalues, T(512));
    test_find_if_not(hpx::parallel::execution::par, xvalues, T(512));
    test_find_if_not_async(hpx::parallel::execution::seq(
        hpx::parallel::execution::task), xvalues, T(512));
    test_find_if_not_async(hpx::parallel::execution::par(
        hpx::parallel::execution::task), xvalues, T(512));
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    find_tests<int>(localities);
    find_tests<double>(localities);
    return 0;
}
