//  Copyright (c) 2018-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/condition_variable.hpp>
#include <hpx/exception.hpp>
#include <hpx/functional.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/mutex.hpp>
#include <hpx/runtime_local/run_as_hpx_thread.hpp>

#include <functional>
#include <mutex>
#include <thread>

std::mutex startup_mtx;
std::condition_variable startup_cond;

bool running = false;
bool stop_running = false;

int start_func(hpx::spinlock& mtx, hpx::condition_variable_any& cond)
{
    // Signal to constructor that thread has started running.
    {
        std::lock_guard<std::mutex> lk(startup_mtx);
        running = true;
    }

    {
        std::unique_lock<hpx::spinlock> lk(mtx);
        startup_cond.notify_one();
        while (!stop_running)
            cond.wait(lk);
    }

    return hpx::finalize();
}

void hpx_thread_func()
{
    HPX_THROW_EXCEPTION(hpx::error::invalid_status, "hpx_thread_func", "test");
}

int main(int argc, char** argv)
{
    hpx::spinlock mtx;
    hpx::condition_variable_any cond;

    hpx::function<int(int, char**)> start_function =
        hpx::bind(&start_func, std::ref(mtx), std::ref(cond));

    hpx::start(start_function, argc, argv);

    // wait for the main HPX thread to run
    {
        std::unique_lock<std::mutex> lk(startup_mtx);
        while (!running)
            startup_cond.wait(lk);
    }

    bool exception_caught = false;
    try
    {
        hpx::threads::run_as_hpx_thread(&hpx_thread_func);
        HPX_TEST(false);    // this should not be executed
    }
    catch (...)
    {
        exception_caught = true;
    }
    HPX_TEST(exception_caught);

    {
        std::lock_guard<hpx::spinlock> lk(mtx);
        stop_running = true;
    }

    cond.notify_one();

    return hpx::stop();
}
