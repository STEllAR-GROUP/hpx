//  Copyright (c) 2019 National Technology & Engineering Solutions of Sandia,
//                     LLC (NTESS).
//  Copyright (c) 2018-2020 Hartmut Kaiser
//  Copyright (c) 2018-2019 Adrian Serio
//  Copyright (c) 2019 Nikunj Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/resiliency.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <stdexcept>

std::atomic<int> answer(35);
struct vogon_exception : std::exception
{
};

int universal_answer()
{
    return ++answer;
}

bool validate(int result)
{
    return result == 42;
}

int no_answer()
{
    throw hpx::resiliency::experimental::abort_replay_exception();
}

int deep_thought()
{
    static int ans = 35;
    ++ans;
    if (ans == 42)
        return ans;
    else
        throw vogon_exception();
}

int hpx_main()
{
    {
        // successful replay
        hpx::future<int> f =
            hpx::resiliency::experimental::dataflow_replay(10, &deep_thought);
        HPX_TEST(f.get() == 42);

        // successful replay validate
        f = hpx::resiliency::experimental::dataflow_replay_validate(
            10, &validate, &universal_answer);
        HPX_TEST(f.get() == 42);

        // unsuccessful replay
        f = hpx::resiliency::experimental::dataflow_replay(6, &deep_thought);

        bool exception_caught = false;
        try
        {
            f.get();
            HPX_TEST(false);
        }
        catch (vogon_exception const&)
        {
            exception_caught = true;
        }
        catch (...)
        {
            HPX_TEST(false);
        }
        HPX_TEST(exception_caught);

        // unsuccessful replay validate
        f = hpx::resiliency::experimental::dataflow_replay_validate(
            6, &validate, &universal_answer);

        exception_caught = false;
        try
        {
            f.get();
            HPX_TEST(false);
        }
        catch (hpx::resiliency::experimental::abort_replay_exception const&)
        {
            exception_caught = true;
        }
        catch (...)
        {
            HPX_TEST(false);
        }
        HPX_TEST(exception_caught);

        // aborted replay
        f = hpx::resiliency::experimental::dataflow_replay(1, &no_answer);

        exception_caught = false;
        try
        {
            f.get();
            HPX_TEST(false);
        }
        catch (hpx::resiliency::experimental::abort_replay_exception const&)
        {
            exception_caught = true;
        }
        catch (...)
        {
            HPX_TEST(false);
        }
        HPX_TEST(exception_caught);

        // aborted replay validate
        f = hpx::resiliency::experimental::dataflow_replay_validate(
            1, &validate, &no_answer);

        exception_caught = false;
        try
        {
            f.get();
        }
        catch (hpx::resiliency::experimental::abort_replay_exception const&)
        {
            exception_caught = true;
        }
        catch (...)
        {
            HPX_TEST(false);
        }
        HPX_TEST(exception_caught);
    }

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST(hpx::local::init(hpx_main, argc, argv) == 0);
    return hpx::util::report_errors();
}
