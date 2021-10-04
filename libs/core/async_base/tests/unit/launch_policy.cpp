//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/modules/async_base.hpp>
#include <hpx/modules/coroutines.hpp>
#include <hpx/modules/testing.hpp>

#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
template <typename Launch>
void test_policy(Launch policy)
{
    HPX_TEST(policy.priority() == hpx::threads::thread_priority::default_);
    HPX_TEST(policy.stacksize() == hpx::threads::thread_stacksize::default_);
    HPX_TEST(
        policy.hint().mode == hpx::threads::thread_schedule_hint_mode::none);
    HPX_TEST_EQ(policy.hint().hint, std::int16_t(-1));

    policy.set_priority(hpx::threads::thread_priority::normal);
    HPX_TEST(policy.priority() == hpx::threads::thread_priority::normal);

    auto p = hpx::execution::experimental::with_priority(
        policy, hpx::threads::thread_priority::high);
    HPX_TEST(hpx::execution::experimental::get_priority(p) ==
        hpx::threads::thread_priority::high);

    policy.set_stacksize(hpx::threads::thread_stacksize::medium);
    HPX_TEST(policy.stacksize() == hpx::threads::thread_stacksize::medium);

    auto p1 = hpx::execution::experimental::with_stacksize(
        policy, hpx::threads::thread_stacksize::small_);
    HPX_TEST(hpx::execution::experimental::get_stacksize(p1) ==
        hpx::threads::thread_stacksize::small_);

    hpx::threads::thread_schedule_hint hint(0);
    policy.set_hint(hint);
    HPX_TEST(policy.hint().mode == hint.mode);
    HPX_TEST_EQ(policy.hint().hint, hint.hint);

    hpx::threads::thread_schedule_hint hint1(1);
    auto p2 = hpx::execution::experimental::with_hint(policy, hint1);
    HPX_TEST(hpx::execution::experimental::get_hint(p2).mode == hint1.mode);
    HPX_TEST(hpx::execution::experimental::get_hint(p2).hint == hint1.hint);
}

int main()
{
    static_assert(sizeof(hpx::launch::async_policy) <= sizeof(std::int64_t));
    static_assert(sizeof(hpx::launch::sync_policy) <= sizeof(std::int64_t));
    static_assert(sizeof(hpx::launch::deferred_policy) <= sizeof(std::int64_t));
    static_assert(sizeof(hpx::launch::fork_policy) <= sizeof(std::int64_t));
    static_assert(sizeof(hpx::launch::apply_policy) <= sizeof(std::int64_t));
    static_assert(sizeof(hpx::launch) <= sizeof(std::int64_t));

    test_policy(hpx::launch::async);
    test_policy(hpx::launch::sync);
    test_policy(hpx::launch::deferred);
    test_policy(hpx::launch::fork);
    test_policy(hpx::launch::apply);

    test_policy(hpx::launch());
    test_policy(hpx::launch(hpx::launch::async));
    test_policy(hpx::launch(hpx::launch::sync));
    test_policy(hpx::launch(hpx::launch::deferred));
    test_policy(hpx::launch(hpx::launch::fork));
    test_policy(hpx::launch(hpx::launch::apply));

    return 0;
}
