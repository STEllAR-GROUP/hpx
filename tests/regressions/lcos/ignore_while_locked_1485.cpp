//  Copyright 2015 (c) Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #1485:
// `ignore_while_locked` doesn't support all Lockable types

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/include/local_lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <mutex>

struct wait_for_flag
{
    hpx::lcos::local::spinlock mutex;
    hpx::lcos::local::condition_variable_any cond_var;

    wait_for_flag()
      : flag(false), woken(0)
    {}

    void wait(hpx::lcos::local::spinlock& local_mutex,
        hpx::lcos::local::condition_variable_any& local_cond_var, bool& running)
    {
        bool first = true;
        while (!flag)
        {
            // signal successful initialization
            if (first)
            {
                {
                    std::lock_guard<hpx::lcos::local::spinlock> lk(local_mutex);
                    running = true;
                }

                first = false;
                local_cond_var.notify_all();
            }

            std::unique_lock<hpx::lcos::local::spinlock> lk(mutex);
            cond_var.wait(mutex);
        }
        ++woken;
    }

    boost::atomic<bool> flag;
    boost::atomic<unsigned> woken;
};

void test_condition_with_mutex()
{
    wait_for_flag data;

    bool running = false;
    hpx::lcos::local::spinlock local_mutex;
    hpx::lcos::local::condition_variable_any local_cond_var;

    hpx::thread thread(&wait_for_flag::wait, boost::ref(data),
        boost::ref(local_mutex), boost::ref(local_cond_var), boost::ref(running));

    // wait for the thread to run
    {
        std::unique_lock<hpx::lcos::local::spinlock> lk(local_mutex);
        while (!running)
            local_cond_var.wait(lk);
    }

    // now start actual test
    data.flag.store(true);

    {
        std::lock_guard<hpx::lcos::local::spinlock> lock(data.mutex);
        data.cond_var.notify_one();
    }

    thread.join();
    HPX_TEST_EQ(data.woken, 1u);
}

int main()
{
    test_condition_with_mutex();
    return hpx::util::report_errors();
}
