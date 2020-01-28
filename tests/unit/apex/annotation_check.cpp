//  Copyright (c) 2019 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Test that creates a set of tasks using normal priority, but every
// Nth normal task spawns a set of high priority tasks.
// The test is intended to be used with a task plotting/profiling
// tool to verify that high priority tasks run before low ones.

#include <hpx/hpx_init.hpp>

#include <hpx/async.hpp>
#include <hpx/include/parallel_execution.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/lcos/when_all.hpp>
#include <hpx/program_options.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>
#include <hpx/util/annotated_function.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/testing.hpp>
#include "apex_options.hpp"

#include <atomic>
#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <type_traits>

//
// This test generates a set of tasks with certain names, then checks
// if the names are present in the screen output from apex.
// The tasks are spawned using dataflow, or continuations with different
// launch policies and executors
// To make scanning the output possible, we prefix the names so that
// the alphabetical output from apex can be scanned with a regex to
// check that the expected names are present.
//
// See the CMakeLists
// set_tests_properties( ...  PROPERTIES PASS_REGULAR_EXPRESSION ...)

// --------------------------------------------------------------------------
// dummy function that just triggers a delay that can be seen in task plots
void dummy_task(std::size_t n)
{
    // no other work can take place on this thread whilst it sleeps
    bool sleep = true;
    auto start = std::chrono::steady_clock::now();
    do
    {
        std::this_thread::sleep_for(std::chrono::microseconds(n) / 25);
        auto now = std::chrono::steady_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::microseconds>(now - start);
        sleep = (elapsed < std::chrono::microseconds(n));
    } while (sleep);
}

// --------------------------------------------------------------------------
// string for a policy
std::string policy_string(const hpx::launch &policy)
{
    if (policy == hpx::launch::async)
    {
        return "async";
    }
    else if (policy == hpx::launch::sync)
    {
        return "sync";
    }
    else if (policy == hpx::launch::fork)
    {
        return "fork";
    }
    else if (policy == hpx::launch::apply)
    {
        return "apply";
    }
    else if (policy == hpx::launch::deferred)
    {
        return "deferred";
    }
    else
    {
        return "policy ?";
    }
}

// string for an executor
template <typename Executor>
std::string exec_string(const Executor &)
{
    bool threaded = hpx::traits::is_threads_executor<Executor>::value;
    return "Executor " + std::string(threaded ? "threaded" : "non-threaded");
}

// --------------------------------------------------------------------------
template <typename Executor>
typename std::enable_if<hpx::traits::is_threads_executor<Executor>::value,
         std::string>::type
execution_string(const Executor &exec) { return exec_string(exec); }

template <typename Executor>
typename std::enable_if<hpx::traits::is_executor_any<Executor>::value, std::string>::type
execution_string(const Executor &exec) { return exec_string(exec); }

template <typename Policy>
typename std::enable_if<hpx::traits::is_launch_policy<Policy>::value, std::string>::type
execution_string(const Policy &policy) { return policy_string(policy); }

// --------------------------------------------------------------------------
// can be called with an executor or a policy
template <typename Execution>
void test_execution(Execution &exec)
{
    static int prefix = 1;
    // these string need to have lifetimes that don't go out of scope
    std::string dfs = std::to_string(prefix++) + "-" + execution_string(exec) +
        std::string(" Dataflow");
    std::string pcs = std::to_string(prefix++) + "-" + execution_string(exec) +
        std::string(" Continuation");
    std::string pcsu = std::to_string(prefix++) + "-" + execution_string(exec) +
        std::string(" Unwrapping Continuation");

    std::vector<hpx::future<void>> results;
    {
        hpx::future<int> f1 = hpx::make_ready_future(5);
        hpx::future<int> f2 = hpx::make_ready_future(5);
        results.emplace_back(hpx::dataflow(
            exec,
            hpx::util::annotated_function(
                [](auto &&, auto &&) {
                    dummy_task(std::size_t(1000));
                },
                dfs.c_str()),
                f1, f2));
    }
    {
        hpx::future<int> f1 = hpx::make_ready_future(5);
        results.emplace_back(f1.then(
            exec,
            hpx::util::annotated_function(
                [](auto &&f1) {
                    dummy_task(std::size_t(1000));
                },
                pcs.c_str())));
    }
    {
        hpx::future<int> f1 = hpx::make_ready_future(5);
        results.emplace_back(f1.then(
            exec,
            hpx::util::unwrapping(
                hpx::util::annotated_function(
                    [](auto &&f1) {
                        dummy_task(std::size_t(1000));
                    },
                    pcsu.c_str()))));
    }
    // wait for completion
    hpx::when_all(results).get();
}

// --------------------------------------------------------------------------
// no executor or policy
void test_none()
{
    // these string need to have lifetimes that don't go out of scope
    std::string dfs = std::string("1-Dataflow");
    std::string pcs = std::string("2-Continuation");

    std::vector<hpx::future<void>> results;
    {
        hpx::future<int> f1 = hpx::make_ready_future(5);
        hpx::future<int> f2 = hpx::make_ready_future(5);
        results.emplace_back(hpx::dataflow(
            hpx::util::annotated_function(
                [](auto &&, auto &&) {
                    dummy_task(std::size_t(1000));
                },
                dfs.c_str()),
            f1, f2));
    }

    {
        hpx::future<int> f1 = hpx::make_ready_future(5);
        results.emplace_back(f1.then(
            hpx::util::annotated_function(
                [](auto &&f1) {
                    dummy_task(std::size_t(1000));
                },
                pcs.c_str())));
    }

    // wait for completion
    hpx::when_all(results).get();
}


int hpx_main()
{
    // setup executors
    hpx::threads::scheduled_executor NP_executor =
        hpx::threads::executors::pool_executor(
            "default", hpx::threads::thread_priority_default);
    hpx::parallel::execution::parallel_executor par_exec{};

    test_none();
    //
    test_execution(hpx::launch::apply);
    test_execution(hpx::launch::async);
    test_execution(hpx::launch::deferred);
    test_execution(hpx::launch::fork);
    test_execution(hpx::launch::sync);
    //
    test_execution(NP_executor);
    test_execution(par_exec);
    //
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    apex::apex_options::use_screen_output(true);
    HPX_TEST_EQ(hpx::init(), 0);
    return hpx::util::report_errors();
}
