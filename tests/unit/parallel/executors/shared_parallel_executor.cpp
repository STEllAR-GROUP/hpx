//  Copyright (c) 2007-2016 Hartmut Kaiser
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
#include <type_traits>

#include <boost/range/functions.hpp>

///////////////////////////////////////////////////////////////////////////////
struct shared_parallel_executor
  : hpx::parallel::executor_tag
{
    template <typename T>
    struct future_type
    {
        typedef hpx::shared_future<T> type;
    };

    template <typename F>
    hpx::shared_future<typename hpx::util::result_of<F()>::type>
    async_execute(F && f)
    {
        return hpx::async(std::forward<F>(f));
    }
};

///////////////////////////////////////////////////////////////////////////////
hpx::thread::id test() { return hpx::this_thread::get_id(); }

void test_sync()
{
    typedef shared_parallel_executor executor;
    typedef hpx::parallel::executor_traits<executor> traits;

    executor exec;
    HPX_TEST(traits::execute(exec, &test) != hpx::this_thread::get_id());
}

void test_async()
{
    typedef shared_parallel_executor executor;
    typedef hpx::parallel::executor_traits<executor> traits;

    executor exec;

    hpx::shared_future<hpx::thread::id> fut =
        traits::async_execute(exec, &test);

    HPX_TEST(
        fut.get() !=
        hpx::this_thread::get_id());
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test(hpx::thread::id tid, int value)
{
    HPX_TEST(tid != hpx::this_thread::get_id());
}

void test_bulk_sync()
{
    typedef shared_parallel_executor executor;
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
    typedef shared_parallel_executor executor;
    typedef hpx::parallel::executor_traits<executor> traits;

    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;

    executor exec;
    std::vector<hpx::shared_future<void> > futs =
        traits::async_execute(exec, hpx::util::bind(&bulk_test, tid, _1), v);

    hpx::when_all(futs).get();
}

///////////////////////////////////////////////////////////////////////////////
void void_test() {}

void test_sync_void()
{
    typedef shared_parallel_executor executor;
    typedef hpx::parallel::executor_traits<executor> traits;

    executor exec;
    traits::execute(exec, &void_test);
}

void test_async_void()
{
    typedef shared_parallel_executor executor;
    typedef hpx::parallel::executor_traits<executor> traits;

    executor exec;
    hpx::shared_future<void> fut = traits::async_execute(exec, &void_test);
    fut.get();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char* argv[])
{
    test_sync();
    test_async();
    test_bulk_sync();
    test_bulk_async();

    test_sync_void();
    test_async_void();

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
