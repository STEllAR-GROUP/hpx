//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/execution.hpp>
#include <hpx/local/future.hpp>
#include <hpx/local/init.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
struct shared_parallel_executor
{
    template <typename F, typename... Ts>
    hpx::shared_future<typename hpx::util::invoke_result<F, Ts...>::type>
    async_execute(F&& f, Ts&&... ts)
    {
        return hpx::async(std::forward<F>(f), std::forward<Ts>(ts)...);
    }
};

namespace hpx { namespace parallel { namespace execution {
    template <>
    struct is_two_way_executor<shared_parallel_executor> : std::true_type
    {
    };
}}}    // namespace hpx::parallel::execution

///////////////////////////////////////////////////////////////////////////////
hpx::thread::id test(int passed_through)
{
    HPX_TEST_EQ(passed_through, 42);
    return hpx::this_thread::get_id();
}

void test_sync()
{
    typedef shared_parallel_executor executor;

    executor exec;
    HPX_TEST(hpx::parallel::execution::sync_execute(exec, &test, 42) !=
        hpx::this_thread::get_id());
}

void test_async()
{
    typedef shared_parallel_executor executor;

    executor exec;

    hpx::shared_future<hpx::thread::id> fut =
        hpx::parallel::execution::async_execute(exec, &test, 42);

    HPX_TEST_NEQ(fut.get(), hpx::this_thread::get_id());
}

///////////////////////////////////////////////////////////////////////////////
void bulk_test(int, hpx::thread::id tid, int passed_through)    //-V813
{
    HPX_TEST_NEQ(tid, hpx::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);
}

void test_bulk_sync()
{
    typedef shared_parallel_executor executor;

    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::placeholders::_1;
    using hpx::placeholders::_2;

    executor exec;
    hpx::parallel::execution::bulk_sync_execute(
        exec, hpx::bind(&bulk_test, _1, tid, _2), v, 42);
    hpx::parallel::execution::bulk_sync_execute(exec, &bulk_test, v, tid, 42);
}

void test_bulk_async()
{
    typedef shared_parallel_executor executor;

    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::placeholders::_1;
    using hpx::placeholders::_2;

    executor exec;
    std::vector<hpx::shared_future<void>> futs =
        hpx::parallel::execution::bulk_async_execute(
            exec, hpx::bind(&bulk_test, _1, tid, _2), v, 42);
    hpx::when_all(futs).get();

    futs = hpx::parallel::execution::bulk_async_execute(
        exec, &bulk_test, v, tid, 42);
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

    executor exec;
    hpx::parallel::execution::sync_execute(exec, &void_test, 42);
}

void test_async_void()
{
    typedef shared_parallel_executor executor;

    executor exec;
    hpx::shared_future<void> fut =
        hpx::parallel::execution::async_execute(exec, &void_test, 42);
    fut.get();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_sync();
    test_async();
    test_bulk_sync();
    test_bulk_async();

    test_sync_void();
    test_async_void();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
