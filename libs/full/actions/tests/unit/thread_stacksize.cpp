// Copyright (C) 2012-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_main.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/threading_base/thread_data.hpp>

#include <cstring>
#include <iostream>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
void test_small_stacksize_helper()
{
    // allocate HPX_SMALL_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD memory on
    // the stack
    char array[HPX_SMALL_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD] = {};
    std::memset(array, '\0', std::size(array));
}

void test_small_stacksize()
{
    std::cout << "test_small_stacksize: "
              << hpx::this_thread::get_available_stack_space() << ", "
              << hpx::threads::get_ctx_ptr()->get_stacksize() << '\n';

    HPX_TEST(hpx::threads::get_self_ptr());

    HPX_TEST(hpx::this_thread::get_available_stack_space() >
        (HPX_SMALL_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD));
    HPX_TEST(hpx::threads::get_ctx_ptr()->get_stacksize() >
        (HPX_SMALL_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD));

    // verify that sufficient stack has been allocated
    HPX_TEST(hpx::threads::get_ctx_ptr()->get_stacksize() >=
        hpx::get_runtime().get_config().get_stack_size(
            hpx::threads::thread_stacksize::small_));

    static_assert(HPX_SMALL_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD > 0,
        "HPX_SMALL_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD > 0");

    test_small_stacksize_helper();
}
HPX_DECLARE_ACTION(test_small_stacksize, test_small_stacksize_action)
HPX_ACTION_USES_SMALL_STACK(test_small_stacksize_action)
HPX_PLAIN_ACTION(test_small_stacksize, test_small_stacksize_action)

///////////////////////////////////////////////////////////////////////////////
void test_medium_stacksize_helper()
{
    // allocate HPX_MEDIUM_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD memory on
    // the stack
    char array[HPX_MEDIUM_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD] = {};
    std::memset(array, '\0', std::size(array));
}

void test_medium_stacksize()
{
    std::cout << "test_medium_stacksize: "
              << hpx::this_thread::get_available_stack_space() << ", "
              << hpx::threads::get_ctx_ptr()->get_stacksize() << '\n';

    HPX_TEST(hpx::threads::get_self_ptr());

    HPX_TEST(hpx::this_thread::get_available_stack_space() >
        (HPX_MEDIUM_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD));
    HPX_TEST(hpx::threads::get_ctx_ptr()->get_stacksize() >
        (HPX_MEDIUM_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD));

    // verify that sufficient stack has been allocated
    HPX_TEST(hpx::threads::get_ctx_ptr()->get_stacksize() >=
        hpx::get_runtime().get_config().get_stack_size(
            hpx::threads::thread_stacksize::medium));

    static_assert(HPX_MEDIUM_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD > 0,
        "HPX_MEDIUM_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD > 0");

    test_medium_stacksize_helper();
}
HPX_DECLARE_ACTION(test_medium_stacksize, test_medium_stacksize_action)
HPX_ACTION_USES_MEDIUM_STACK(test_medium_stacksize_action)
HPX_PLAIN_ACTION(test_medium_stacksize, test_medium_stacksize_action)

///////////////////////////////////////////////////////////////////////////////
void test_large_stacksize_helper()
{
    // allocate HPX_LARGE_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD memory on
    // the stack
    char array[HPX_LARGE_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD] = {};
    std::memset(array, '\0', std::size(array));
}

void test_large_stacksize()
{
    std::cout << "test_large_stacksize: "
              << hpx::this_thread::get_available_stack_space() << ", "
              << hpx::threads::get_ctx_ptr()->get_stacksize() << '\n';

    HPX_TEST(hpx::threads::get_self_ptr());

    HPX_TEST(hpx::this_thread::get_available_stack_space() >
        (HPX_LARGE_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD));
    HPX_TEST(hpx::threads::get_ctx_ptr()->get_stacksize() >
        (HPX_LARGE_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD));

    // verify that sufficient stack has been allocated
    HPX_TEST(hpx::threads::get_ctx_ptr()->get_stacksize() >=
        hpx::get_runtime().get_config().get_stack_size(
            hpx::threads::thread_stacksize::large));

    static_assert(HPX_LARGE_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD > 0,
        "HPX_LARGE_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD > 0");

    test_large_stacksize_helper();
}
HPX_DECLARE_ACTION(test_large_stacksize, test_large_stacksize_action)
HPX_ACTION_USES_LARGE_STACK(test_large_stacksize_action)
HPX_PLAIN_ACTION(test_large_stacksize, test_large_stacksize_action)

///////////////////////////////////////////////////////////////////////////////
void test_huge_stacksize_helper()
{
    // allocate HPX_HUGE_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD memory on
    // the stack
    char array[HPX_HUGE_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD] = {};
    std::memset(array, '\0', std::size(array));
}

void test_huge_stacksize()
{
    std::cout << "test_huge_stacksize: "
              << hpx::this_thread::get_available_stack_space() << ", "
              << hpx::threads::get_ctx_ptr()->get_stacksize() << '\n';

    HPX_TEST(hpx::threads::get_self_ptr());

    HPX_TEST(hpx::this_thread::get_available_stack_space() >
        (HPX_HUGE_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD));
    HPX_TEST(hpx::threads::get_ctx_ptr()->get_stacksize() >
        (HPX_HUGE_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD));

    // verify that sufficient stack has been allocated
    HPX_TEST(hpx::threads::get_ctx_ptr()->get_stacksize() >=
        hpx::get_runtime().get_config().get_stack_size(
            hpx::threads::thread_stacksize::huge));

    static_assert(HPX_HUGE_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD > 0,
        "HPX_HUGE_STACK_SIZE - HPX_THREADS_STACK_OVERHEAD > 0");

    test_huge_stacksize_helper();
}
HPX_DECLARE_ACTION(test_huge_stacksize, test_huge_stacksize_action)
HPX_ACTION_USES_HUGE_STACK(test_huge_stacksize_action)
HPX_PLAIN_ACTION(test_huge_stacksize, test_huge_stacksize_action)

///////////////////////////////////////////////////////////////////////////////
int main()
{
    std::vector<hpx::id_type> const localities = hpx::find_all_localities();

    for (hpx::id_type const& id : localities)
    {
        {
            test_small_stacksize_action test_action;
            hpx::async(test_action, id).get();
        }

        {
            test_medium_stacksize_action test_action;
            hpx::async(test_action, id).get();
        }

        {
            test_large_stacksize_action test_action;
            hpx::async(test_action, id).get();
        }

        {
            test_huge_stacksize_action test_action;
            hpx::async(test_action, id).get();
        }
    }

    return hpx::util::report_errors();
}

#endif
