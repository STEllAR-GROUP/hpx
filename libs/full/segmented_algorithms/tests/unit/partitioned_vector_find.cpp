//  Copyright (c) 2017 Ajai V George
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_main.hpp>
#include <hpx/include/parallel_find.hpp>
#include <hpx/include/parallel_scan.hpp>
#include <hpx/include/partitioned_vector_predef.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <iostream>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
template <typename T>
struct cond1
{
    bool operator()(T v) const
    {
        return v > T(511);
    }
};

template <typename T>
struct cond2
{
    bool operator()(T v) const
    {
        return v < T(512);
    }
};

template <typename T>
void test_find(hpx::partitioned_vector<T>& xvalues, T val)
{
    auto last = hpx::find(xvalues.begin(), xvalues.end(), val);
    HPX_TEST_EQ(*last, val);
}

template <typename ExPolicy, typename T>
void test_find(ExPolicy&& policy, hpx::partitioned_vector<T>& xvalues, T val)
{
    auto last = hpx::find(policy, xvalues.begin(), xvalues.end(), val);
    HPX_TEST_EQ(*last, val);
}

template <typename ExPolicy, typename T>
void test_find_async(
    ExPolicy&& policy, hpx::partitioned_vector<T>& xvalues, T val)
{
    auto last = hpx::find(policy, xvalues.begin(), xvalues.end(), val).get();
    HPX_TEST_EQ(*last, val);
}

template <typename T>
void test_find_if(hpx::partitioned_vector<T>& xvalues, T val)
{
    auto last = hpx::find_if(xvalues.begin(), xvalues.end(), cond1<T>());
    HPX_TEST_EQ(*last, val);
}

template <typename ExPolicy, typename T>
void test_find_if(ExPolicy&& policy, hpx::partitioned_vector<T>& xvalues, T val)
{
    auto last =
        hpx::find_if(policy, xvalues.begin(), xvalues.end(), cond1<T>());
    HPX_TEST_EQ(*last, val);
}

template <typename ExPolicy, typename T>
void test_find_if_async(
    ExPolicy&& policy, hpx::partitioned_vector<T>& xvalues, T val)
{
    auto last =
        hpx::find_if(policy, xvalues.begin(), xvalues.end(), cond1<T>()).get();
    HPX_TEST_EQ(*last, val);
}

template <typename T>
void test_find_if_not(hpx::partitioned_vector<T>& xvalues, T val)
{
    auto last = hpx::find_if_not(xvalues.begin(), xvalues.end(), cond2<T>());
    HPX_TEST_EQ(*last, val);
}

template <typename ExPolicy, typename T>
void test_find_if_not(
    ExPolicy&& policy, hpx::partitioned_vector<T>& xvalues, T val)
{
    auto last =
        hpx::find_if_not(policy, xvalues.begin(), xvalues.end(), cond2<T>());
    HPX_TEST_EQ(*last, val);
}

template <typename ExPolicy, typename T>
void test_find_if_not_async(
    ExPolicy&& policy, hpx::partitioned_vector<T>& xvalues, T val)
{
    auto last =
        hpx::find_if_not(policy, xvalues.begin(), xvalues.end(), cond2<T>())
            .get();
    HPX_TEST_EQ(*last, val);
}

template <typename T>
void find_tests(std::vector<hpx::id_type>& localities)
{
    std::size_t const num = 1000;
    hpx::partitioned_vector<T> xvalues(
        num, T(1), hpx::container_layout(localities));
    hpx::inclusive_scan(hpx::execution::seq, xvalues.begin(), xvalues.end(),
        xvalues.begin(), std::plus<T>(), T(0));

    test_find(xvalues, T(512));
    test_find(hpx::execution::seq, xvalues, T(512));
    test_find(hpx::execution::par, xvalues, T(512));
    test_find_async(hpx::execution::seq(hpx::execution::task), xvalues, T(512));
    test_find_async(hpx::execution::par(hpx::execution::task), xvalues, T(512));

    test_find_if(xvalues, T(512));
    test_find_if(hpx::execution::seq, xvalues, T(512));
    test_find_if(hpx::execution::par, xvalues, T(512));
    test_find_if_async(
        hpx::execution::seq(hpx::execution::task), xvalues, T(512));
    test_find_if_async(
        hpx::execution::par(hpx::execution::task), xvalues, T(512));

    test_find_if_not(xvalues, T(512));
    test_find_if_not(hpx::execution::seq, xvalues, T(512));
    test_find_if_not(hpx::execution::par, xvalues, T(512));
    test_find_if_not_async(
        hpx::execution::seq(hpx::execution::task), xvalues, T(512));
    test_find_if_not_async(
        hpx::execution::par(hpx::execution::task), xvalues, T(512));
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    find_tests<int>(localities);
    return hpx::util::report_errors();
}
#endif
