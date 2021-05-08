//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/task_group.hpp>

#include <cstddef>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int fib(int n)
{
    if (n < 2)
    {
        return n;
    }

    int x = 0, y = 0;

    hpx::execution::experimental::task_group g;
    g.run([&x, n] { x = fib(n - 1); });    // spawn a task
    g.run([&y, n] { y = fib(n - 2); });    // spawn another task
    g.wait();                              // wait for both tasks to complete

    return x + y;
}

void task_group_test1()
{
    HPX_TEST_EQ(fib(22), 17711);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
int fib(Executor&& exec, int n)
{
    if (n < 2)
    {
        return n;
    }

    int x = 0, y = 0;

    hpx::execution::experimental::task_group g;
    g.run(
        exec, [&](int n) { x = fib(exec, n); }, n - 1);
    g.run(
        exec, [&](int n) { y = fib(exec, n); }, n - 2);
    g.wait();    // wait for both tasks to complete

    return x + y;
}

void task_group_test2()
{
    HPX_TEST_EQ(fib(hpx::execution::parallel_executor{}, 22), 17711);
}

///////////////////////////////////////////////////////////////////////////////
void task_group_test3()
{
    bool throws_exception = true;
    bool caught_exception = false;
    try
    {
        hpx::execution::experimental::task_group g;
        g.run([] { throw std::runtime_error("test1"); });
        g.run([] { throw std::runtime_error("test2"); });
        throws_exception = false;

        g.wait();    // rethrows after waiting for all tasks to finish
        HPX_TEST(false);
    }
    catch (hpx::exception_list const& l)
    {
        caught_exception = true;
        HPX_TEST_EQ(l.size(), std::size_t(2));
    }
    catch (...)
    {
        HPX_TEST(false);
    }
    HPX_TEST(!throws_exception);
    HPX_TEST(caught_exception);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    task_group_test1();
    task_group_test2();
    task_group_test3();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.cfg = cfg;
    HPX_TEST_EQ_MSG(hpx::init(argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
