//  Copyright (c) 2017 Ajai V George
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/parallel_transform_scan.hpp>
#include <hpx/include/partitioned_vector_predef.hpp>

#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
struct conv
{
    template <typename T>
    T operator()(T in) const
    {
        return (T) 2 * in;
    }
};

struct op
{
    template <typename T>
    T operator()(T in1, T in2) const
    {
        return in1 + in2;
    }
};

template <typename ExPolicy, typename T>
void test_transform_inclusive_scan(ExPolicy&& policy,
    hpx::partitioned_vector<T>& xvalues, hpx::partitioned_vector<T>& out)
{
    hpx::parallel::transform_inclusive_scan(policy, xvalues.begin(),
        xvalues.end(), out.begin(), op(), conv(), T(0));
}

template <typename ExPolicy, typename T>
void test_transform_inclusive_scan_async(ExPolicy&& policy,
    hpx::partitioned_vector<T>& xvalues, hpx::partitioned_vector<T>& out)
{
    hpx::parallel::transform_inclusive_scan(
        policy, xvalues.begin(), xvalues.end(), out.begin(), op(), conv(), T(0))
        .get();
}

template <typename ExPolicy, typename T>
void test_transform_exclusive_scan(ExPolicy&& policy,
    hpx::partitioned_vector<T>& xvalues, hpx::partitioned_vector<T>& out)
{
    hpx::parallel::transform_exclusive_scan(policy, xvalues.begin(),
        xvalues.end(), out.begin(), T(0), op(), conv());
}

template <typename ExPolicy, typename T>
void test_transform_exclusive_scan_async(ExPolicy&& policy,
    hpx::partitioned_vector<T>& xvalues, hpx::partitioned_vector<T>& out)
{
    hpx::parallel::transform_exclusive_scan(
        policy, xvalues.begin(), xvalues.end(), out.begin(), T(0), op(), conv())
        .get();
}

template <typename T>
void transform_scan_tests(std::size_t num, hpx::partitioned_vector<T>& xvalues,
    hpx::partitioned_vector<T>& out)
{
    test_transform_inclusive_scan(hpx::execution::seq, xvalues, out);
    HPX_TEST_EQ((out[num - 1]), T(2 * num));
    test_transform_inclusive_scan(hpx::execution::par, xvalues, out);
    HPX_TEST_EQ((out[num - 1]), T(2 * num));
    test_transform_inclusive_scan_async(
        hpx::execution::seq(hpx::execution::task), xvalues, out);
    HPX_TEST_EQ((out[num - 1]), T(2 * num));
    test_transform_inclusive_scan_async(
        hpx::execution::par(hpx::execution::task), xvalues, out);
    HPX_TEST_EQ((out[num - 1]), T(2 * num));

    test_transform_exclusive_scan(hpx::execution::seq, xvalues, out);
    HPX_TEST_EQ((out[num - 1]), T(2 * (num - 1)));
    test_transform_exclusive_scan(hpx::execution::par, xvalues, out);
    HPX_TEST_EQ((out[num - 1]), T(2 * (num - 1)));
    test_transform_exclusive_scan_async(
        hpx::execution::seq(hpx::execution::task), xvalues, out);
    HPX_TEST_EQ((out[num - 1]), T(2 * (num - 1)));
    test_transform_exclusive_scan_async(
        hpx::execution::par(hpx::execution::task), xvalues, out);
    HPX_TEST_EQ((out[num - 1]), T(2 * (num - 1)));
}

template <typename T>
void transform_scan_tests(std::vector<hpx::id_type>& localities)
{
    std::size_t const num = 12;
    hpx::partitioned_vector<T> xvalues(
        num, T(1), hpx::container_layout(localities));
    hpx::partitioned_vector<T> out(num, hpx::container_layout(localities));
    transform_scan_tests(num, xvalues, out);
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    transform_scan_tests<double>(localities);
    return hpx::util::report_errors();
}
