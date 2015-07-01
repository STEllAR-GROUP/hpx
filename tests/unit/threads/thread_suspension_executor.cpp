// Copyright (C) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/threadmanager.hpp>
#include <hpx/include/thread_executors.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/assign/std/vector.hpp>
#include <boost/lexical_cast.hpp>

#include <vector>

#define NUM_SUSPEND_TESTS 1000
#define NUM_YIELD_TESTS 1000

///////////////////////////////////////////////////////////////////////////////
void test_executor_association_yield()
{
    hpx::threads::thread_id_type id = hpx::threads::get_self_id();
    hpx::threads::executor exec_before = hpx::threads::get_executor(id);

    for (int i = 0; i != NUM_YIELD_TESTS; ++i)
    {
        hpx::this_thread::yield();

        hpx::threads::executor exec_after = hpx::threads::get_executor(id);
        HPX_TEST_EQ(exec_before, exec_after);
    }
}

///////////////////////////////////////////////////////////////////////////////
void wakeup_thread(hpx::threads::thread_id_type id)
{
    hpx::threads::set_thread_state(id, hpx::threads::pending);
}

void test_executor_association_suspend()
{
    hpx::threads::thread_id_type id = hpx::threads::get_self_id();
    hpx::threads::executor exec_before = hpx::threads::get_executor(id);

    for (int i = 0; i != NUM_YIELD_TESTS; ++i)
    {
        hpx::apply(exec_before, &wakeup_thread, id);

        hpx::this_thread::suspend(hpx::threads::suspended);

        hpx::threads::executor exec_after = hpx::threads::get_executor(id);
        HPX_TEST_EQ(exec_before, exec_after);
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    {
        hpx::threads::executors::local_priority_queue_executor exec(
            hpx::get_os_thread_count());

        std::vector<hpx::future<void> > result;
        for (std::size_t i = 0; i != hpx::get_os_thread_count(); ++i)
        {
            hpx::apply(exec, &test_executor_association_yield);
            hpx::apply(exec, &test_executor_association_suspend);
        }

        // destructor synchronizes will all running tasks
    }

    {
        hpx::threads::executors::local_priority_queue_executor exec(
            hpx::get_os_thread_count());

        std::vector<hpx::future<void> > result;
        for (std::size_t i = 0; i != hpx::get_os_thread_count(); ++i)
        {
            result.push_back(hpx::async(exec, &test_executor_association_yield));
            result.push_back(hpx::async(exec, &test_executor_association_suspend));
        }
        hpx::wait_all();
    }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(0, hpx::init(argc, argv), "hpx::init returned non-zero value");
    return hpx::util::report_errors();
}

