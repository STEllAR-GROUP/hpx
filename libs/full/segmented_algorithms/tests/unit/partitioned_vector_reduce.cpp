//  Copyright (c) 2017 Ajai V George
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/parallel_reduce.hpp>
#include <hpx/include/partitioned_vector_predef.hpp>

#include <hpx/modules/testing.hpp>

#include <boost/range/functions.hpp>

#include <cstddef>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// The vector types to be used are defined in partitioned_vector module.
// HPX_REGISTER_PARTITIONED_VECTOR(double);
// HPX_REGISTER_PARTITIONED_VECTOR(int);

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename T>
T test_reduce(ExPolicy&& policy, hpx::partitioned_vector<T> const& xvalues)
{
    return hpx::reduce(
        policy, xvalues.begin(), xvalues.end(), T(1), std::plus<T>());
}

template <typename ExPolicy, typename T>
hpx::future<T> test_reduce_async(
    ExPolicy&& policy, hpx::partitioned_vector<T> const& xvalues)
{
    return hpx::reduce(
        policy, xvalues.begin(), xvalues.end(), T(1), std::plus<T>());
}

template <typename T>
void reduce_tests(std::size_t num, hpx::partitioned_vector<T> const& xvalues)
{
    HPX_TEST_EQ(test_reduce(hpx::execution::seq, xvalues), T(num + 1));
    HPX_TEST_EQ(test_reduce(hpx::execution::par, xvalues), T(num + 1));

    HPX_TEST_EQ(
        test_reduce_async(hpx::execution::seq(hpx::execution::task), xvalues)
            .get(),
        T(num + 1));
    HPX_TEST_EQ(
        test_reduce_async(hpx::execution::par(hpx::execution::task), xvalues)
            .get(),
        T(num + 1));
}

template <typename T>
void reduce_tests(std::vector<hpx::id_type>& localities)
{
    std::size_t const num = 10007;
    hpx::partitioned_vector<T> xvalues(
        num, T(1), hpx::container_layout(localities));
    reduce_tests(num, xvalues);
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    reduce_tests<int>(localities);
    reduce_tests<double>(localities);
    return hpx::util::report_errors();
}
