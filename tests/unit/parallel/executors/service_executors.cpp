//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/parallel/executors/service_executors.hpp>
#include <hpx/lcos/wait_all.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <algorithm>
#include <cstdlib>
#include <vector>

#include <boost/range/functions.hpp>
#include <boost/thread.hpp>

///////////////////////////////////////////////////////////////////////////////
boost::thread::id test() { return boost::this_thread::get_id(); }

template <typename Executor>
void test_sync(Executor& exec)
{
    typedef hpx::parallel::executor_traits<Executor> traits;
    HPX_TEST(traits::execute(exec, &test) != boost::this_thread::get_id());
}

template <typename Executor>
void test_async(Executor& exec)
{
    typedef hpx::parallel::executor_traits<Executor> traits;

    HPX_TEST(
        traits::async_execute(exec, &test).get() !=
        boost::this_thread::get_id());
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test(boost::thread::id tid, int value)
{
    HPX_TEST(tid != boost::this_thread::get_id());
}

template <typename Executor>
void test_bulk_sync(Executor& exec)
{
    typedef hpx::parallel::executor_traits<Executor> traits;

    boost::thread::id tid = boost::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;
    traits::execute(exec, hpx::util::bind(&bulk_test, tid, _1), v);
}

template <typename Executor>
void test_bulk_async(Executor& exec)
{
    typedef hpx::parallel::executor_traits<Executor> traits;

    boost::thread::id tid = boost::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;
    hpx::wait_all(traits::async_execute(
        exec, hpx::util::bind(&bulk_test, tid, _1), v));
}

template <typename Executor>
void test_service_executor(Executor& exec)
{
    test_sync(exec);
    test_async(exec);
    test_bulk_sync(exec);
    test_bulk_async(exec);
}

int hpx_main(int argc, char* argv[])
{
    using hpx::parallel::service_executor_type;

    {
        hpx::parallel::service_executor exec(service_executor_type::io_thread_pool);
        test_service_executor(exec);
    }

    {
        hpx::parallel::service_executor exec(service_executor_type::parcel_thread_pool);
        test_service_executor(exec);
    }

    {
        hpx::parallel::service_executor exec(service_executor_type::timer_thread_pool);
        test_service_executor(exec);
    }

    {
        hpx::parallel::service_executor exec(service_executor_type::main_thread);
        test_service_executor(exec);
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
