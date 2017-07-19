//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

using namespace std::chrono;

///////////////////////////////////////////////////////////////////////////////
hpx::thread::id test(int passed_through)
{
    HPX_TEST_EQ(passed_through, 42);
    return hpx::this_thread::get_id();
}

void test_timed_sync()
{
    typedef hpx::parallel::parallel_executor executor;
    typedef hpx::parallel::timed_executor_traits<executor> traits;

    executor exec;
    HPX_TEST(
        traits::execute_after(
            exec, milliseconds(1), &test, 42
        ) != hpx::this_thread::get_id());

    HPX_TEST(
        traits::execute_at(
            exec, steady_clock::now() + milliseconds(1), &test, 42
        ) != hpx::this_thread::get_id());
}

void test_timed_async()
{
    typedef hpx::parallel::parallel_executor executor;
    typedef hpx::parallel::timed_executor_traits<executor> traits;

    executor exec;
    HPX_TEST(
        traits::async_execute_after(
            exec, milliseconds(1), &test, 42
        ).get() != hpx::this_thread::get_id());
    HPX_TEST(
        traits::async_execute_at(
            exec, steady_clock::now() + milliseconds(1), &test, 42
        ).get() != hpx::this_thread::get_id());
}

void test_timed_apply()
{
    typedef hpx::parallel::parallel_executor executor;
    typedef hpx::parallel::timed_executor_traits<executor> traits;

    executor exec;
    traits::apply_execute_after(exec, milliseconds(1), &test, 42);
    traits::apply_execute_at(
        exec, steady_clock::now() + milliseconds(1), &test, 42);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char* argv[])
{
    test_timed_sync();
    test_timed_async();
    test_timed_apply();

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
