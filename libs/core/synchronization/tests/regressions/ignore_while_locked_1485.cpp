//  Copyright 2015-2022 (c) Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case tests the workaround for the issue described in #1485:
// `ignore_while_checking` doesn't support all Lockable types.
// `ignore_all_while_checking` can be used instead to ignore all locks
// (including the ones that are not supported by `ignore_while_checking`).

#include <hpx/init.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <atomic>
#include <functional>
#include <mutex>

struct wait_for_flag
{
    hpx::spinlock mutex;
    hpx::condition_variable_any cond_var;

    wait_for_flag()
      : flag(false)
      , woken(0)
    {
    }

    void wait(hpx::spinlock& local_mutex,
        hpx::condition_variable_any& local_cond_var, bool& running)
    {
        bool first = true;
        while (!flag)
        {
            // signal successful initialization
            if (first)
            {
                {
                    std::lock_guard<hpx::spinlock> lk(local_mutex);
                    running = true;
                }

                first = false;
                local_cond_var.notify_all();
            }

            std::unique_lock<hpx::spinlock> lk(mutex);
            if (!flag)
            {
                cond_var.wait(mutex);
            }
        }
        ++woken;
    }

    std::atomic<bool> flag;
    std::atomic<unsigned> woken;
};

void test_condition_with_mutex()
{
    wait_for_flag data;

    bool running = false;
    hpx::spinlock local_mutex;
    hpx::condition_variable_any local_cond_var;

    hpx::thread thread(&wait_for_flag::wait, std::ref(data),
        std::ref(local_mutex), std::ref(local_cond_var), std::ref(running));

    // wait for the thread to run
    {
        std::unique_lock<hpx::spinlock> lk(local_mutex);
        // NOLINTNEXTLINE(bugprone-infinite-loop)
        while (!running)
            local_cond_var.wait(lk);
    }

    // now start actual test
    data.flag.store(true);

    {
        std::lock_guard<hpx::spinlock> lock(data.mutex);
        hpx::util::ignore_all_while_checking il;
        HPX_UNUSED(il);

        data.cond_var.notify_one();
    }

    thread.join();
    HPX_TEST_EQ(data.woken, 1u);
}

int hpx_main()
{
    test_condition_with_mutex();
    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
