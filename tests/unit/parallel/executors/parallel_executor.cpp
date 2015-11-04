//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <algorithm>
#include <cstdlib>
#include <vector>

#include <boost/range/functions.hpp>

///////////////////////////////////////////////////////////////////////////////
hpx::thread::id test() { return hpx::this_thread::get_id(); }

void test_sync()
{
    typedef hpx::parallel::parallel_executor executor;
    typedef hpx::parallel::executor_traits<executor> traits;

    executor exec;
    HPX_TEST(traits::execute(exec, &test) != hpx::this_thread::get_id());
}

void test_async()
{
    typedef hpx::parallel::parallel_executor executor;
    typedef hpx::parallel::executor_traits<executor> traits;

    executor exec;
    HPX_TEST(
        traits::async_execute(exec, &test).get() !=
        hpx::this_thread::get_id());
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test(hpx::thread::id tid, int value)
{
    HPX_TEST(tid != hpx::this_thread::get_id());
}

void test_bulk_sync()
{
    typedef hpx::parallel::parallel_executor executor;
    typedef hpx::parallel::executor_traits<executor> traits;

    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;

    executor exec;
    traits::execute(exec, hpx::util::bind(&bulk_test, tid, _1), v);
}

void test_bulk_async()
{
    typedef hpx::parallel::parallel_executor executor;
    typedef hpx::parallel::executor_traits<executor> traits;

    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;

    executor exec;
    hpx::when_all(traits::async_execute(
        exec, hpx::util::bind(&bulk_test, tid, _1), v)).get();
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
    // By default this test should run on all available cores
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
