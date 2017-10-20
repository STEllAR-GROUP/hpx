//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>

void low_priority()
{
    HPX_TEST_EQ(hpx::threads::thread_priority_low,
        hpx::this_thread::get_priority());
    hpx::this_thread::yield();
    HPX_TEST_EQ(hpx::threads::thread_priority_low,
        hpx::this_thread::get_priority());
}
HPX_DECLARE_ACTION(low_priority);
HPX_ACTION_HAS_LOW_PRIORITY(low_priority_action);
HPX_PLAIN_ACTION(low_priority);

void normal_priority()
{
    HPX_TEST_EQ(hpx::threads::thread_priority_normal,
        hpx::this_thread::get_priority());
    hpx::this_thread::yield();
    HPX_TEST_EQ(hpx::threads::thread_priority_normal,
        hpx::this_thread::get_priority());
}
HPX_DECLARE_ACTION(normal_priority);
HPX_ACTION_HAS_NORMAL_PRIORITY(normal_priority_action);
HPX_PLAIN_ACTION(normal_priority);

void high_priority()
{
    HPX_TEST_EQ(hpx::threads::thread_priority_high,
        hpx::this_thread::get_priority());
    hpx::this_thread::yield();
    HPX_TEST_EQ(hpx::threads::thread_priority_high,
        hpx::this_thread::get_priority());
}
HPX_DECLARE_ACTION(high_priority);
HPX_ACTION_HAS_HIGH_PRIORITY(high_priority_action);
HPX_PLAIN_ACTION(high_priority);

void high_recursive_priority()
{
    HPX_TEST_EQ(hpx::threads::thread_priority_high_recursive,
        hpx::this_thread::get_priority());
    hpx::this_thread::yield();
    HPX_TEST_EQ(hpx::threads::thread_priority_high_recursive,
        hpx::this_thread::get_priority());
}
HPX_DECLARE_ACTION(high_recursive_priority);
HPX_ACTION_HAS_HIGH_RECURSIVE_PRIORITY(high_recursive_priority_action);
HPX_PLAIN_ACTION(high_recursive_priority);

int hpx_main(int argc, char ** argv)
{
    low_priority_action()(hpx::find_here());
    normal_priority_action()(hpx::find_here());
    high_priority_action()(hpx::find_here());
    high_recursive_priority_action()(hpx::find_here());
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(0, hpx::init(argc, argv));
    return hpx::util::report_errors();
}
