//  Copyright (c) 2017 Ajai V George
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/parallel_all_any_none_of.hpp>
#include <hpx/include/partitioned_vector.hpp>

#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <iostream>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Define the vector types to be used.
HPX_REGISTER_PARTITIONED_VECTOR(double);

///////////////////////////////////////////////////////////////////////////////
#define SIZE 64

template <typename T>
void initialize(hpx::partitioned_vector<T>& xvalues)
{
    T init_array[SIZE] = {1, 2, 3, 4, 5, 1, 2, 3, 3, 5, 5, 3, 4, 2, 3, 2, 1, 2,
        3, 4, 5, 6, 5, 6, 1, 2, 3, 4, 1, 1, 2, 3, 4, 5, 4, 3, 2, 1, 1, 2, 3, 4,
        1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 7, 6, 5, 7, 5, 4, 2, 3, 4, 5, 2};
    for (int i = 0; i < SIZE; i++)
    {
        xvalues.set_value(i, init_array[i]);
    }
}

struct op5
{
    template <typename T>
    bool operator()(T& value)
    {
        return value > 5;
    }
};

struct op0
{
    template <typename T>
    bool operator()(T& value)
    {
        return value > 0;
    }
};

struct op8
{
    template <typename T>
    bool operator()(T& value)
    {
        return value > 8;
    }
};

template <typename ExPolicy, typename T, typename Func>
void test_none(ExPolicy&& policy, hpx::partitioned_vector<T>& xvalues, Func&& f,
    bool expected_result)
{
    bool result = hpx::none_of(policy, xvalues.begin(), xvalues.end(), f);
    HPX_TEST_EQ(result, expected_result);
}

template <typename ExPolicy, typename T, typename Func>
void test_none_async(ExPolicy&& policy, hpx::partitioned_vector<T>& xvalues,
    Func&& f, bool expected_result)
{
    bool result = hpx::none_of(policy, xvalues.begin(), xvalues.end(), f).get();
    HPX_TEST_EQ(result, expected_result);
}

template <typename T>
void none_tests(std::vector<hpx::id_type>& localities)
{
    hpx::partitioned_vector<T> xvalues(
        SIZE, T(0), hpx::container_layout(localities));
    initialize(xvalues);

    test_none(hpx::parallel::execution::seq, xvalues, op8(), true);
    test_none(hpx::parallel::execution::par, xvalues, op8(), true);
    test_none_async(
        hpx::parallel::execution::seq(hpx::parallel::execution::task), xvalues,
        op8(), true);
    test_none_async(
        hpx::parallel::execution::par(hpx::parallel::execution::task), xvalues,
        op8(), true);

    test_none(hpx::parallel::execution::seq, xvalues, op5(), false);
    test_none(hpx::parallel::execution::par, xvalues, op5(), false);
    test_none_async(
        hpx::parallel::execution::seq(hpx::parallel::execution::task), xvalues,
        op5(), false);
    test_none_async(
        hpx::parallel::execution::par(hpx::parallel::execution::task), xvalues,
        op5(), false);

    test_none(hpx::parallel::execution::seq, xvalues, op0(), false);
    test_none(hpx::parallel::execution::par, xvalues, op0(), false);
    test_none_async(
        hpx::parallel::execution::seq(hpx::parallel::execution::task), xvalues,
        op0(), false);
    test_none_async(
        hpx::parallel::execution::par(hpx::parallel::execution::task), xvalues,
        op0(), false);
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    none_tests<double>(localities);
    return hpx::util::report_errors();
}
