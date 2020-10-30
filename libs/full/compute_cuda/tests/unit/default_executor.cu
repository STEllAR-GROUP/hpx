//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/modules/testing.hpp>

#include <hpx/async_cuda/target.hpp>
#include <hpx/include/compute.hpp>

#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <boost/range/functions.hpp>

///////////////////////////////////////////////////////////////////////////////
auto seed = std::random_device{}();
std::mt19937 gen(seed);
struct test
{
    HPX_HOST_DEVICE void operator()() {}
};

void test_sync()
{
    typedef hpx::cuda::experimental::default_executor executor;

    hpx::cuda::experimental::target target;
    executor exec(target);
    hpx::parallel::execution::sync_execute(exec, test());
}

void test_async()
{
    typedef hpx::cuda::experimental::default_executor executor;

    hpx::cuda::experimental::target target;
    executor exec(target);
    hpx::parallel::execution::async_execute(exec, test()).get();
}

///////////////////////////////////////////////////////////////////////////////
struct bulk_test
{
    // FIXME : call operator of bulk_test is momentarily defined as
    //         HPX_HOST_DEVICE in place of HPX_DEVICE to allow the host_side
    //         result_of<> (used in traits::bulk_execute()) to get the return
    //         type

    HPX_HOST_DEVICE void operator()(int) {}
};

void test_bulk_sync()
{
    typedef hpx::cuda::experimental::default_executor executor;

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), gen());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    hpx::cuda::experimental::target target;
    executor exec(target);
    hpx::parallel::execution::bulk_sync_execute(exec, bulk_test(), v);
}

void test_bulk_async()
{
    typedef hpx::cuda::experimental::default_executor executor;

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), gen());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    hpx::cuda::experimental::target target;
    executor exec(target);
    hpx::when_all(
        hpx::parallel::execution::bulk_async_execute(exec, bulk_test(), v))
        .get();
}

int hpx_main()
{
    test_sync();
    test_async();
    test_bulk_sync();
    test_bulk_async();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(
        hpx::init(argc, argv), 0, "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
