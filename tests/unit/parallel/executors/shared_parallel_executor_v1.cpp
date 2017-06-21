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
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

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

    template <typename F, typename ... Ts>
    hpx::shared_future<typename hpx::util::invoke_result<F, Ts...>::type>
    async_execute(F && f, Ts &&... ts)
    {
        return hpx::async(std::forward<F>(f), std::forward<Ts>(ts)...);
    }
};

///////////////////////////////////////////////////////////////////////////////
hpx::thread::id test(int passed_through)
{
    HPX_TEST_EQ(passed_through, 42);
    return hpx::this_thread::get_id();
}

void test_sync()
{
    typedef shared_parallel_executor executor;
    typedef hpx::parallel::executor_traits<executor> traits;

    executor exec;
    HPX_TEST(traits::execute(exec, &test, 42) != hpx::this_thread::get_id());
}

void test_async()
{
    typedef shared_parallel_executor executor;
    typedef hpx::parallel::executor_traits<executor> traits;

    executor exec;

    hpx::shared_future<hpx::thread::id> fut =
        traits::async_execute(exec, &test, 42);

    HPX_TEST(
        fut.get() !=
        hpx::this_thread::get_id());
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test(int value, hpx::thread::id tid, int passed_through) //-V813
{
    HPX_TEST(tid != hpx::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);
}

void test_bulk_sync()
{
    typedef shared_parallel_executor executor;
    typedef hpx::parallel::executor_traits<executor> traits;

    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    executor exec;
    traits::bulk_execute(exec, hpx::util::bind(&bulk_test, _1, tid, _2), v, 42);
    traits::bulk_execute(exec, &bulk_test, v, tid, 42);
}

void test_bulk_async()
{
    typedef shared_parallel_executor executor;
    typedef hpx::parallel::executor_traits<executor> traits;

    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    executor exec;
    std::vector<hpx::shared_future<void> > futs =
        traits::bulk_async_execute(exec,
            hpx::util::bind(&bulk_test, _1, tid, _2), v, 42);
    hpx::when_all(futs).get();

    futs = traits::bulk_async_execute(exec, &bulk_test, v, tid, 42);
    hpx::when_all(futs).get();
}

///////////////////////////////////////////////////////////////////////////////
void void_test(int passed_through)
{
    HPX_TEST_EQ(passed_through, 42);
}

void test_sync_void()
{
    typedef shared_parallel_executor executor;
    typedef hpx::parallel::executor_traits<executor> traits;

    executor exec;
    traits::execute(exec, &void_test, 42);
}

void test_async_void()
{
    typedef shared_parallel_executor executor;
    typedef hpx::parallel::executor_traits<executor> traits;

    executor exec;
    hpx::shared_future<void> fut = traits::async_execute(exec, &void_test, 42);
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
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
