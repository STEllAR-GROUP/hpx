//  Copyright (c) 2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This demonstrates the issue as reported by #1523: Remote async with deferred
// launch policy never executes

#include <hpx/hpx_main.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/testing.hpp>
#include <utility>

bool null_action_executed = false;
bool int_action_executed = false;
bool nt_executed = false;
bool it_executed = false;

void null_thread()
{
    null_action_executed = true;
}
HPX_PLAIN_ACTION(null_thread, null_action);

int int_thread()
{
    int_action_executed = true;
    return 42;
}
HPX_PLAIN_ACTION(int_thread, int_action);

hpx::id_type get_locality()
{
    return hpx::find_here();
}
HPX_PLAIN_ACTION(get_locality, get_locality_action);

int main()
{
    {
        hpx::async<null_action>(hpx::launch::deferred, hpx::find_here()).get();
        HPX_TEST(null_action_executed);

        int result = hpx::async<int_action>(
            hpx::launch::deferred, hpx::find_here()).get();
        HPX_TEST(int_action_executed);
        HPX_TEST_EQ(result, 42);

        for (hpx::id_type const& loc : hpx::find_all_localities())
        {
            hpx::id_type id = hpx::async<get_locality_action>(
                hpx::launch::deferred, loc).get();
            HPX_TEST_EQ(loc, id);
        }
    }

    // Same test with lambdas
    // action lambdas inhibit undefined behavior...
#if !defined(HPX_HAVE_SANITIZERS)
    {
        auto nt =
            hpx::actions::lambda_to_action(
            []()
            {
                nt_executed = true;
            });

        auto it =
            hpx::actions::lambda_to_action(
            []() -> int
            {
                it_executed = true;
                return 42;
            });

        auto gl =
            hpx::actions::lambda_to_action(
            []() -> hpx::id_type
            {
                return hpx::find_here();
            });

        hpx::async(
            hpx::launch::deferred, std::move(nt), hpx::find_here()).get();
        HPX_TEST(nt_executed);

        int result = hpx::async(
            hpx::launch::deferred, std::move(it), hpx::find_here()).get();
        HPX_TEST(it_executed);
        HPX_TEST_EQ(result, 42);

        for (hpx::id_type const& loc : hpx::find_all_localities())
        {
            hpx::id_type id = hpx::async(
                hpx::launch::deferred, gl, loc).get();
            HPX_TEST_EQ(loc, id);
        }
    }
#endif

    return 0;
}
