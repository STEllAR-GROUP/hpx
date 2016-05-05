//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/parallel/executors/thread_pool_executors.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include <boost/range/functions.hpp>

///////////////////////////////////////////////////////////////////////////////
hpx::thread::id test(int passed_through)
{
    HPX_TEST_EQ(passed_through, 42);
    return hpx::this_thread::get_id();
}

template <typename Executor>
void test_sync(Executor& exec)
{
    typedef hpx::parallel::executor_traits<Executor> traits;

    HPX_TEST(traits::execute(exec, &test, 42) != hpx::this_thread::get_id());
}

template <typename Executor>
void test_async(Executor& exec)
{
    typedef hpx::parallel::executor_traits<Executor> traits;

    HPX_TEST(
        traits::async_execute(exec, &test, 42).get() !=
        hpx::this_thread::get_id());
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test(int value, hpx::thread::id tid, int passed_through) //-V813
{
    HPX_TEST(tid != hpx::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);
}

template <typename Executor>
void test_bulk_sync(Executor& exec)
{
    typedef hpx::parallel::executor_traits<Executor> traits;

    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    traits::bulk_execute(exec, hpx::util::bind(&bulk_test, _1, tid, _2), v, 42);
    traits::bulk_execute(exec, &bulk_test, v, tid, 42);
}

template <typename Executor>
void test_bulk_async(Executor& exec)
{
    typedef hpx::parallel::executor_traits<Executor> traits;

    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    hpx::when_all(traits::bulk_async_execute(
        exec, hpx::util::bind(&bulk_test, _1, tid, _2), v, 42)).get();
    hpx::when_all(traits::bulk_async_execute(exec, &bulk_test, v, tid, 42)).get();
}

template <typename Executor>
void test_thread_pool_executor(Executor& exec)
{
    test_sync(exec);
    test_async(exec);
    test_bulk_sync(exec);
    test_bulk_async(exec);
}

int hpx_main(int argc, char* argv[])
{
    std::size_t num_threads = hpx::get_os_thread_count();

#if defined(HPX_HAVE_STATIC_SCHEDULER)
    {
        hpx::parallel::static_queue_executor exec(num_threads);
        test_thread_pool_executor(exec);
    }
#endif

    {
        hpx::parallel::local_priority_queue_executor exec(num_threads);
        test_thread_pool_executor(exec);
    }

#if defined(HPX_HAVE_STATIC_PRIORITY_SCHEDULER)
    {
        hpx::parallel::static_priority_queue_executor exec(num_threads);
        test_thread_pool_executor(exec);
    }
#endif

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        std::to_string(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
