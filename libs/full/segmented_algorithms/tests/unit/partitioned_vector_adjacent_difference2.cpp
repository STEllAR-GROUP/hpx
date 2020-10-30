//  Copyright (c) 2017 Ajai V George
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_main.hpp>
#include <hpx/include/parallel_adjacent_difference.hpp>
#include <hpx/include/parallel_count.hpp>
#include <hpx/include/parallel_scan.hpp>
#include <hpx/include/partitioned_vector_predef.hpp>

#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename T>
void verify_values(
    ExPolicy&&, hpx::partitioned_vector<T> const& v, T const& val)
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

template <typename ExPolicy, typename T>
void verify_values(ExPolicy&&, hpx::partitioned_vector<T> const& v)
{
    typedef typename hpx::partitioned_vector<T>::const_iterator const_iterator;
    std::size_t size = 0;
    T val = 1;
    const_iterator end = v.end();
    for (const_iterator it = v.begin(); it != end; ++it, ++size)
    {
        HPX_TEST_EQ(*it, val++);
    }
    HPX_TEST_EQ(size, v.size());
}

template <typename ExPolicy, typename T>
void test_adjacent_difference(ExPolicy&& policy, hpx::partitioned_vector<T>& v,
    hpx::partitioned_vector<T>& w, T val)
{
    hpx::parallel::adjacent_difference(policy, v.begin(), v.end(), w.begin());

    verify_values(policy, w, val);
    verify_values(policy, v);
}

template <typename ExPolicy, typename T>
void test_adjacent_difference_async(ExPolicy&& policy,
    hpx::partitioned_vector<T>& v, hpx::partitioned_vector<T>& w, T val)
{
    hpx::parallel::adjacent_difference(policy, v.begin(), v.end(), w.begin())
        .get();

    verify_values(policy, w, val);
    verify_values(policy, v);
}

///////////////////////////////////////////////////////////////////////////////
template <typename T>
void adjacent_difference_tests(std::vector<hpx::id_type>& localities)
{
    std::size_t const length = 12;

    hpx::partitioned_vector<T> v(
        length, T(1), hpx::container_layout(localities));
    hpx::parallel::inclusive_scan(
        hpx::execution::seq, v.begin(), v.end(), v.begin());
    hpx::partitioned_vector<T> w(length, hpx::container_layout(localities));
    test_adjacent_difference(hpx::execution::seq, v, w, T(1));
    test_adjacent_difference(hpx::execution::par, v, w, T(1));
    test_adjacent_difference_async(
        hpx::execution::seq(hpx::execution::task), v, w, T(1));
    test_adjacent_difference_async(
        hpx::execution::par(hpx::execution::task), v, w, T(1));
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    adjacent_difference_tests<double>(localities);
    return hpx::util::report_errors();
}
#endif
