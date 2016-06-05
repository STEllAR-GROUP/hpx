//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/partitioned_vector.hpp>
#include <hpx/include/parallel_fill.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Define the vector types to be used.
HPX_REGISTER_PARTITIONED_VECTOR(double);
HPX_REGISTER_PARTITIONED_VECTOR(int);

///////////////////////////////////////////////////////////////////////////////
template <typename T>
void iota_vector(hpx::partitioned_vector<T>& v, T val)
{
    typename hpx::partitioned_vector<T>::iterator it = v.begin(), end = v.end();
    for(/**/; it != end; ++it)
        *it = val++;
}

///////////////////////////////////////////////////////////////////////////////
template <typename T, typename InIter>
void verify_values(InIter first, InIter last, T const& val,
    bool must_be_equal = true)
{
    for (InIter it = first; it != last; ++it)
    {
        if(must_be_equal)
        {
            HPX_TEST_EQ(*it, val);
        }
        else
        {
            HPX_TEST_NEQ(*it, val);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename T>
void verify_vector(hpx::partitioned_vector<T> const& v, T const& val,
    bool must_be_equal = true)
{
    typedef typename hpx::partitioned_vector<T>::const_iterator const_iterator;

    std::size_t size = 0;

    const_iterator end = v.end();
    for (const_iterator it = v.begin(); it != end; ++it, ++size)
    {
        HPX_TEST_EQ(*it, val);
    }

    HPX_TEST_EQ(size, v.size());
}

///////////////////////////////////////////////////////////////////////////////
template <typename T, typename DistPolicy, typename ExPolicy>
void fill_algo_tests_with_policy(std::size_t size,
    DistPolicy const& policy, ExPolicy const& fill_policy)
{
    hpx::partitioned_vector<T> c(size, policy);
    iota_vector(c, T(1234));

    const T v(42);
    hpx::parallel::fill(fill_policy, c.begin(), c.end(), v);
    verify_vector(c, v);

    const T v1(43);
    hpx::parallel::fill(fill_policy, c.begin()+1, c.end()-1, v1);
    verify_values(c.begin()+1, c.end()-1, v1);
    verify_values(c.begin(), c.begin()+1, v1, false);
    verify_values(c.end()-1, c.end(), v1, false);
}

template <typename T, typename DistPolicy, typename ExPolicy>
void fill_algo_tests_with_policy_async(std::size_t size,
    DistPolicy const& policy, ExPolicy const& fill_policy)
{
    hpx::partitioned_vector<T> c(size, policy);
    iota_vector(c, T(1234));

    const T v(42);
    hpx::future<void> f = hpx::parallel::fill(fill_policy, c.begin(), c.end(), v);
    f.wait();

    verify_vector(c, v);

    const T v1(43);
    hpx::future<void> f1 = hpx::parallel::fill(fill_policy, c.begin()+1, c.end()-1, v1);
    f1.wait();

    verify_values(c.begin()+1, c.end()-1, v1);
    verify_values(c.begin(), c.begin()+1, v1, false);
    verify_values(c.end()-1, c.end(), v1, false);
}

template <typename T, typename DistPolicy>
void fill_tests_with_policy(std::size_t size, std::size_t localities,
    DistPolicy const& policy)
{
    using namespace hpx::parallel;
    using hpx::parallel::task;

    fill_algo_tests_with_policy<T>(size, policy, seq);
    fill_algo_tests_with_policy<T>(size, policy, par);

    //async
    fill_algo_tests_with_policy_async<T>(size, policy, seq(task));
    fill_algo_tests_with_policy_async<T>(size, policy, par(task));
}

template <typename T>
void fill_tests()
{
    std::size_t const length = 12;
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    fill_tests_with_policy<T>(length, 1, hpx::container_layout);
    fill_tests_with_policy<T>(length, 3, hpx::container_layout(3));
    fill_tests_with_policy<T>(length, 3, hpx::container_layout(3, localities));
    fill_tests_with_policy<T>(length, localities.size(),
        hpx::container_layout(localities));
}


///////////////////////////////////////////////////////////////////////////////
int main()
{
    fill_tests<double>();
    fill_tests<int>();

    return 0;
}

