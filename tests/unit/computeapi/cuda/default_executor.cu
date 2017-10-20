//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <hpx/include/compute.hpp>

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include <boost/range/functions.hpp>

///////////////////////////////////////////////////////////////////////////////
struct test
{
    __device__ void operator()() {}
};

void test_sync()
{
    typedef hpx::compute::cuda::default_executor executor;

    hpx::compute::cuda::target target;
    executor exec(target);
    hpx::parallel::execution::sync_execute(exec, test());
}

void test_async()
{
    typedef hpx::compute::cuda::default_executor executor;

    hpx::compute::cuda::target target;
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
    typedef hpx::compute::cuda::default_executor executor;

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    hpx::compute::cuda::target target;
    executor exec(target);
//    traits::bulk_execute(exec, hpx::util::bind(&bulk_test, _1), v);
    hpx::parallel::execution::bulk_sync_execute(exec, bulk_test(), v);
}

void test_bulk_async()
{
    typedef hpx::compute::cuda::default_executor executor;

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    hpx::compute::cuda::target target;
    executor exec(target);
//    hpx::when_all(traits::bulk_async_execute(
//        exec, hpx::util::bind(&bulk_test, _1), v)
//    ).get();
    hpx::when_all(
        hpx::parallel::execution::bulk_async_execute(exec, bulk_test(), v)
    ).get();
}

int hpx_main(int argc, char* argv[])
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
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
