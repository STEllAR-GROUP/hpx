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
#include <boost/chrono.hpp>

using namespace boost::chrono;

///////////////////////////////////////////////////////////////////////////////
hpx::thread::id test() { return hpx::this_thread::get_id(); }

void test_timed_sync()
{
    typedef hpx::parallel::parallel_executor executor;
    typedef hpx::parallel::timed_executor_traits<executor> traits;

    executor exec;
    HPX_TEST(
        traits::execute_after(exec, milliseconds(1), &test) !=
            hpx::this_thread::get_id());

    HPX_TEST(
        traits::execute_at(exec, steady_clock::now()+milliseconds(1), &test) !=
            hpx::this_thread::get_id());
}

void test_timed_async()
{
    typedef hpx::parallel::parallel_executor executor;
    typedef hpx::parallel::timed_executor_traits<executor> traits;

    executor exec;
    HPX_TEST(
        traits::async_execute_after(
            exec, milliseconds(1), &test
        ).get() != hpx::this_thread::get_id());
    HPX_TEST(
        traits::async_execute_at(
            exec, steady_clock::now()+milliseconds(1), &test
        ).get() != hpx::this_thread::get_id());
}

void test_timed_apply()
{
    typedef hpx::parallel::parallel_executor executor;
    typedef hpx::parallel::timed_executor_traits<executor> traits;

    executor exec;
    traits::apply_execute_after(exec, milliseconds(1), &test);
    traits::apply_execute_at(exec, steady_clock::now()+milliseconds(1), &test);
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
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
