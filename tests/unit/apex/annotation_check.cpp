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

#include <hpx/include/parallel_execution.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/modules/async_base.hpp>
#include <hpx/modules/async_combinators.hpp>
#include <hpx/modules/async_local.hpp>
#include <hpx/modules/futures.hpp>
#include <hpx/modules/program_options.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/threading_base.hpp>
#include <hpx/runtime_local/pool_executor.hpp>
#include "apex_options.hpp"

#include <atomic>
#include <cstddef>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

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
std::string policy_string(const hpx::launch& policy)
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
std::string exec_string(const Executor&)
{
#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
    bool threaded = hpx::traits::is_threads_executor<Executor>::value;
    return "Executor " + std::string(threaded ? "threaded" : "non-threaded");
#else
    return "Executor non-threaded";
#endif
}

// --------------------------------------------------------------------------
#if defined(HPX_HAVE_THREAD_EXECUTORS_COMPATIBILITY)
template <typename Executor>
typename std::enable_if<hpx::traits::is_threads_executor<Executor>::value,
    std::string>::type
execution_string(const Executor& exec)
{
    return exec_string(exec);
}
#endif

template <typename Executor>
typename std::enable_if<hpx::traits::is_executor_any<Executor>::value,
    std::string>::type
execution_string(const Executor& exec)
{
    return exec_string(exec);
}

template <typename Policy>
typename std::enable_if<hpx::traits::is_launch_policy<Policy>::value,
    std::string>::type
execution_string(const Policy& policy)
{
    return policy_string(policy);
}

// --------------------------------------------------------------------------
// can be called with an executor or a policy
template <typename Execution>
hpx::future<void> test_execution(Execution& exec)
{
    static int prefix = 1;
    std::string dfs = std::to_string(prefix++) + "-" + execution_string(exec) +
        std::string(" Dataflow");
    std::string pcs = std::to_string(prefix++) + "-" + execution_string(exec) +
        std::string(" Continuation");
    std::string pcsu = std::to_string(prefix++) + "-" + execution_string(exec) +
        std::string(" Unwrapping Continuation");

    std::vector<hpx::future<void>> results;
    {
        hpx::future<int> f1 = hpx::async([]() { return 5; });
        hpx::future<int> f2 = hpx::make_ready_future(5);
        results.emplace_back(hpx::dataflow(exec,
            hpx::util::annotated_function(
                [](auto&&, auto&&) { dummy_task(std::size_t(1000)); }, dfs),
            f1, f2));
    }
    {
        hpx::future<int> f1 = hpx::async([]() { return 5; });
        results.emplace_back(f1.then(exec,
            hpx::util::annotated_function(
                [](auto&&) { dummy_task(std::size_t(1000)); }, pcs)));
    }
    {
        hpx::future<int> f1 = hpx::async([]() { return 5; });
        results.emplace_back(f1.then(exec,
            hpx::util::unwrapping(hpx::util::annotated_function(
                [](auto&&) { dummy_task(std::size_t(1000)); }, pcsu))));
    }
    // wait for completion
    return hpx::when_all(results);
}

// --------------------------------------------------------------------------
// no executor or policy
hpx::future<void> test_none()
{
    std::string dfs = std::string("1-Dataflow");
    std::string pcs = std::string("2-Continuation");

    std::vector<hpx::future<void>> results;
    {
        hpx::future<int> f1 = hpx::async([]() { return 5; });
        hpx::future<int> f2 = hpx::make_ready_future(5);
        results.emplace_back(hpx::dataflow(
            hpx::util::annotated_function(
                [](auto&&, auto&&) { dummy_task(std::size_t(1000)); }, dfs),
            f1, f2));
    }

    {
        hpx::future<int> f1 = hpx::async([]() { return 5; });
        results.emplace_back(f1.then(hpx::util::annotated_function(
            [](auto&&) { dummy_task(std::size_t(1000)); }, pcs)));
    }

    // wait for completion
    return hpx::when_all(results);
}

int hpx_main()
{
    // setup executors
#if defined(HPX_HAVE_POOL_EXECUTOR_COMPATIBILITY)
    hpx::parallel::execution::pool_executor NP_executor =
        hpx::parallel::execution::pool_executor(
            "default", hpx::threads::thread_priority::default_);
#endif
    hpx::execution::parallel_executor par_exec{};

    test_none().get();
    //
    test_execution(hpx::launch::apply).get();
    test_execution(hpx::launch::async).get();
    test_execution(hpx::launch::deferred).get();
    test_execution(hpx::launch::fork).get();
    test_execution(hpx::launch::sync).get();
    //
#if defined(HPX_HAVE_POOL_EXECUTOR_COMPATIBILITY)
    test_execution(NP_executor).get();
#endif
    test_execution(par_exec).get();
    //
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    apex::apex_options::use_screen_output(true);
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
