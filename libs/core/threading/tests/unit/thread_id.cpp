// Copyright (C) 2013-2022 Hartmut Kaiser
// Copyright (C) 2007 Anthony Williams
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/barrier.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <functional>
#include <memory>

using hpx::program_options::options_description;
using hpx::program_options::variables_map;

///////////////////////////////////////////////////////////////////////////////
void do_nothing(
    std::shared_ptr<hpx::barrier<>> b1, std::shared_ptr<hpx::barrier<>> b2)
{
    b1->arrive_and_wait();
    hpx::this_thread::suspend(100);    // wait for 100 ms
    b2->arrive_and_wait();
}

void test_thread_id_for_default_constructed_thread_is_default_constructed_id()
{
    hpx::thread t;
    HPX_TEST_EQ(t.get_id(), hpx::thread::id());
}

void test_thread_id_for_running_thread_is_not_default_constructed_id()
{
    std::shared_ptr<hpx::barrier<>> b1 = std::make_shared<hpx::barrier<>>(2);
    std::shared_ptr<hpx::barrier<>> b2 = std::make_shared<hpx::barrier<>>(2);

    hpx::thread t(&do_nothing, b1, b2);
    b1->arrive_and_wait();

    HPX_TEST_NEQ(t.get_id(), hpx::thread::id());

    b2->arrive_and_wait();
    t.join();
}

void test_different_threads_have_different_ids()
{
    std::shared_ptr<hpx::barrier<>> b1 = std::make_shared<hpx::barrier<>>(3);
    std::shared_ptr<hpx::barrier<>> b2 = std::make_shared<hpx::barrier<>>(3);

    hpx::thread t(&do_nothing, b1, b2);
    hpx::thread t2(&do_nothing, b1, b2);
    b1->arrive_and_wait();

    HPX_TEST_NEQ(t.get_id(), t2.get_id());

    b2->arrive_and_wait();
    t.join();
    t2.join();
}

void test_thread_ids_have_a_total_order()
{
    std::shared_ptr<hpx::barrier<>> b1 = std::make_shared<hpx::barrier<>>(4);
    std::shared_ptr<hpx::barrier<>> b2 = std::make_shared<hpx::barrier<>>(4);

    hpx::thread t1(&do_nothing, b1, b2);
    hpx::thread t2(&do_nothing, b1, b2);
    hpx::thread t3(&do_nothing, b1, b2);
    b1->arrive_and_wait();

    hpx::thread::id t1_id = t1.get_id();
    hpx::thread::id t2_id = t2.get_id();
    hpx::thread::id t3_id = t3.get_id();

    HPX_TEST_NEQ(t1_id, t2_id);
    HPX_TEST_NEQ(t1_id, t3_id);
    HPX_TEST_NEQ(t2_id, t3_id);

    HPX_TEST((t1_id < t2_id) != (t2_id < t1_id));
    HPX_TEST((t1_id < t3_id) != (t3_id < t1_id));
    HPX_TEST((t2_id < t3_id) != (t3_id < t2_id));

    HPX_TEST((t1_id > t2_id) != (t2_id > t1_id));
    HPX_TEST((t1_id > t3_id) != (t3_id > t1_id));
    HPX_TEST((t2_id > t3_id) != (t3_id > t2_id));

    HPX_TEST((t1_id < t2_id) == (t2_id > t1_id));
    HPX_TEST((t2_id < t1_id) == (t1_id > t2_id));
    HPX_TEST((t1_id < t3_id) == (t3_id > t1_id));
    HPX_TEST((t3_id < t1_id) == (t1_id > t3_id));
    HPX_TEST((t2_id < t3_id) == (t3_id > t2_id));
    HPX_TEST((t3_id < t2_id) == (t2_id > t3_id));

    HPX_TEST((t1_id < t2_id) == (t2_id >= t1_id));
    HPX_TEST((t2_id < t1_id) == (t1_id >= t2_id));
    HPX_TEST((t1_id < t3_id) == (t3_id >= t1_id));
    HPX_TEST((t3_id < t1_id) == (t1_id >= t3_id));
    HPX_TEST((t2_id < t3_id) == (t3_id >= t2_id));
    HPX_TEST((t3_id < t2_id) == (t2_id >= t3_id));

    HPX_TEST((t1_id <= t2_id) == (t2_id > t1_id));
    HPX_TEST((t2_id <= t1_id) == (t1_id > t2_id));
    HPX_TEST((t1_id <= t3_id) == (t3_id > t1_id));
    HPX_TEST((t3_id <= t1_id) == (t1_id > t3_id));
    HPX_TEST((t2_id <= t3_id) == (t3_id > t2_id));
    HPX_TEST((t3_id <= t2_id) == (t2_id > t3_id));

    if ((t1_id < t2_id) && (t2_id < t3_id))
    {
        HPX_TEST_LT(t1_id, t3_id);
    }
    else if ((t1_id < t3_id) && (t3_id < t2_id))
    {
        HPX_TEST_LT(t1_id, t2_id);
    }
    else if ((t2_id < t3_id) && (t3_id < t1_id))
    {
        HPX_TEST_LT(t2_id, t1_id);
    }
    else if ((t2_id < t1_id) && (t1_id < t3_id))
    {
        HPX_TEST_LT(t2_id, t3_id);
    }
    else if ((t3_id < t1_id) && (t1_id < t2_id))
    {
        HPX_TEST_LT(t3_id, t2_id);
    }
    else if ((t3_id < t2_id) && (t2_id < t1_id))
    {
        HPX_TEST_LT(t3_id, t1_id);
    }
    else
    {
        HPX_TEST(false);
    }

    hpx::thread::id default_id;

    HPX_TEST_LT(default_id, t1_id);
    HPX_TEST_LT(default_id, t2_id);
    HPX_TEST_LT(default_id, t3_id);

    HPX_TEST_LTE(default_id, t1_id);
    HPX_TEST_LTE(default_id, t2_id);
    HPX_TEST_LTE(default_id, t3_id);

    HPX_TEST(!(default_id > t1_id));
    HPX_TEST(!(default_id > t2_id));
    HPX_TEST(!(default_id > t3_id));

    HPX_TEST(!(default_id >= t1_id));
    HPX_TEST(!(default_id >= t2_id));
    HPX_TEST(!(default_id >= t3_id));

    b2->arrive_and_wait();

    t1.join();
    t2.join();
    t3.join();
}

void get_thread_id(hpx::thread::id* id)
{
    *id = hpx::this_thread::get_id();
}

void test_thread_id_of_running_thread_returned_by_this_thread_get_id()
{
    hpx::thread::id id;
    hpx::thread t(&get_thread_id, &id);
    hpx::thread::id t_id = t.get_id();
    t.join();
    HPX_TEST_EQ(id, t_id);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map&)
{
    {
        test_thread_id_for_default_constructed_thread_is_default_constructed_id();
        test_thread_id_for_running_thread_is_not_default_constructed_id();
        test_different_threads_have_different_ids();
        test_thread_ids_have_a_total_order();
        test_thread_id_of_running_thread_returned_by_this_thread_get_id();
    }

    hpx::local::finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return hpx::local::init(hpx_main, argc, argv, init_args);
}
