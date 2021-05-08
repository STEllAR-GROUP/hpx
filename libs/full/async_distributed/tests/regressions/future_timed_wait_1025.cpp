//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/threadmanager.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/modules/testing.hpp>

#include <chrono>

void wait_for(hpx::lcos::shared_future<int> f)
{
    try
    {
        f.wait_for(std::chrono::nanoseconds(1));
        hpx::this_thread::suspend(
            hpx::threads::thread_schedule_state::suspended);
    }
    catch (hpx::thread_interrupted const&)
    {
        // we should get an error reporting hpx::thread_interrupted
        HPX_TEST(true);
        return;
    }
    HPX_TEST(false);
}

void wait_until(hpx::lcos::shared_future<int> f)
{
    try
    {
        f.wait_until(
            std::chrono::system_clock::now() + std::chrono::nanoseconds(1));
        hpx::this_thread::suspend(
            hpx::threads::thread_schedule_state::suspended);
    }
    catch (hpx::thread_interrupted const&)
    {
        // we should get an error reporting hpx::thread_interrupted
        HPX_TEST(true);
        return;
    }
    HPX_TEST(false);
}

void test_wait_for()
{
    hpx::lcos::local::promise<int> promise;
    hpx::lcos::shared_future<int> future = promise.get_future();

    hpx::thread thread(&wait_for, future);

    HPX_TEST(thread.joinable());

    hpx::this_thread::sleep_for(std::chrono::seconds(5));
    promise.set_value(42);
    hpx::this_thread::sleep_for(std::chrono::seconds(5));

    hpx::threads::thread_state thread_state =
        hpx::threads::get_thread_state(thread.native_handle());
    HPX_TEST_EQ(
        thread_state.state(), hpx::threads::thread_schedule_state::suspended);

    if (thread.joinable())
    {
        thread.interrupt();
        thread.join();
    }
}

void test_wait_until()
{
    hpx::lcos::local::promise<int> promise;
    hpx::lcos::shared_future<int> future = promise.get_future();

    hpx::thread thread(&wait_until, future);

    HPX_TEST(thread.joinable());

    hpx::this_thread::sleep_for(std::chrono::seconds(5));
    promise.set_value(42);
    hpx::this_thread::sleep_for(std::chrono::seconds(5));

    hpx::threads::thread_state thread_state =
        hpx::threads::get_thread_state(thread.native_handle());
    HPX_TEST_EQ(
        thread_state.state(), hpx::threads::thread_schedule_state::suspended);

    if (thread.joinable())
    {
        thread.interrupt();
        thread.join();
    }
}

///////////////////////////////////////////////////////////////////////////////
using hpx::program_options::options_description;
using hpx::program_options::variables_map;

int hpx_main(variables_map&)
{
    {
        test_wait_for();
        test_wait_until();
    }

    hpx::finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return hpx::init(argc, argv, init_args);
}
