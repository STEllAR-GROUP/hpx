//  Copyright (c) 2017 Ajai V George
//  Copyright (c) 2020 Akhil J Nair
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_main.hpp>
#include <hpx/include/parallel_minmax.hpp>
#include <hpx/include/partitioned_vector.hpp>

#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <iostream>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Define the vector types to be used.
HPX_REGISTER_PARTITIONED_VECTOR(int)

///////////////////////////////////////////////////////////////////////////////
#define SIZE 64

template <typename T>
void initialize(hpx::partitioned_vector<T>& xvalues)
{
    T init_array[SIZE] = {1, 2, 3, 4, 5, 6, 2, 3, 3, 5, 5, 3, 4, 2, 3, 2, 6, 2,
        3, 4, 5, 6, 5, 6, 6, 2, 3, 4, 6, 6, 2, 3, 4, 5, 4, 3, 2, 6, 6, 2, 3, 4,
        6, 2, 3, 6, 6, 6, 6, 6, 6, 6, 6, 7, 6, 5, 8, 5, 4, 2, 3, 4, 5, 2};
    for (int i = 0; i < SIZE; i++)
    {
        xvalues.set_value(i, init_array[i]);
    }
}

template <typename ExPolicy, typename T, typename Func>
void test_max_element(ExPolicy&& policy, hpx::partitioned_vector<T>& xvalues,
    Func&& f, T expected_result)
{
    auto result = hpx::max_element(policy, xvalues.begin(), xvalues.end(), f);
    HPX_TEST_EQ(*result, expected_result);
}

template <typename ExPolicy, typename T, typename Func>
void test_max_element_async(ExPolicy&& policy,
    hpx::partitioned_vector<T>& xvalues, Func&& f, T expected_result)
{
    auto result =
        hpx::max_element(policy, xvalues.begin(), xvalues.end(), f).get();
    HPX_TEST_EQ(*result, expected_result);
}

template <typename T>
void max_element_tests(std::vector<hpx::id_type>& localities)
{
    hpx::partitioned_vector<T> xvalues(
        SIZE, T(0), hpx::container_layout(localities));
    initialize(xvalues);

    test_max_element(hpx::execution::seq, xvalues, std::less<T>(), T(8));
    test_max_element(hpx::execution::par, xvalues, std::less<T>(), T(8));
    test_max_element_async(hpx::execution::seq(hpx::execution::task), xvalues,
        std::less<T>(), T(8));
    test_max_element_async(hpx::execution::par(hpx::execution::task), xvalues,
        std::less<T>(), T(8));

    test_max_element(hpx::execution::seq, xvalues, std::greater<T>(), T(1));
    test_max_element(hpx::execution::par, xvalues, std::greater<T>(), T(1));
    test_max_element_async(hpx::execution::seq(hpx::execution::task), xvalues,
        std::greater<T>(), T(1));
    test_max_element_async(hpx::execution::par(hpx::execution::task), xvalues,
        std::greater<T>(), T(1));
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    max_element_tests<int>(localities);
    return hpx::util::report_errors();
}
#endif
