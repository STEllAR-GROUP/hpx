//  Copyright (C) 2012-2013 Vicente Botet
//  Copyright (c) 2013 Agustin Berge
//  Copyright (c) 2015-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int p1()
{
    hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
    return 1;
}

int p2(hpx::future<int> f)
{
    HPX_TEST(f.valid());
    int i = f.get();
    hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
    return 2 * i;
}

void p3(hpx::future<int> f)
{
    HPX_TEST(f.valid());
    int i = f.get();
    (void) i;
    hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
    return;
}

hpx::future<int> p4(hpx::future<int> f)
{
    return hpx::async(p2, std::move(f));
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_return_int(Executor& exec)
{
    hpx::future<int> f1 = hpx::async(exec, &p1);
    HPX_TEST(f1.valid());
    hpx::future<int> f2 = f1.then(exec, &p2);
    HPX_TEST(f2.valid());
    try
    {
        HPX_TEST_EQ(f2.get(), 2);
    }
    catch (hpx::exception const& /*ex*/)
    {
        HPX_TEST(false);
    }
    catch (...)
    {
        HPX_TEST(false);
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_return_void(Executor& exec)
{
    hpx::future<int> f1 = hpx::async(exec, &p1);
    HPX_TEST(f1.valid());
    hpx::future<void> f2 = f1.then(exec, &p3);
    HPX_TEST(f2.valid());
    try
    {
        f2.wait();
    }
    catch (hpx::exception const& /*ex*/)
    {
        HPX_TEST(false);
    }
    catch (...)
    {
        HPX_TEST(false);
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_implicit_unwrapping(Executor& exec)
{
    hpx::future<int> f1 = hpx::async(exec, &p1);
    HPX_TEST(f1.valid());
    hpx::future<int> f2 = f1.then(exec, &p4);
    HPX_TEST(f2.valid());
    try
    {
        HPX_TEST_EQ(f2.get(), 2);
    }
    catch (hpx::exception const& /*ex*/)
    {
        HPX_TEST(false);
    }
    catch (...)
    {
        HPX_TEST(false);
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_simple_then(Executor& exec)
{
    hpx::future<int> f2 = hpx::async(exec, p1).then(exec, &p2);
    HPX_TEST_EQ(f2.get(), 2);
}

template <typename Executor>
void test_simple_deferred_then(Executor& exec)
{
    hpx::future<int> f2 = hpx::async(exec, p1).then(exec, &p2);
    HPX_TEST_EQ(f2.get(), 2);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_complex_then(Executor& exec)
{
    hpx::future<int> f1 = hpx::async(exec, p1);
    hpx::future<int> f21 = f1.then(exec, &p2);
    hpx::future<int> f2 = f21.then(exec, &p2);
    HPX_TEST_EQ(f2.get(), 4);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_complex_then_chain_one(Executor& exec)
{
    hpx::future<int> f1 = hpx::async(exec, p1);
    hpx::future<int> f2 = f1.then(exec, &p2).then(exec, &p2);
    HPX_TEST_EQ(f2.get(), 4);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_complex_then_chain_two(Executor& exec)
{
    hpx::future<int> f2 = hpx::async(exec, p1).then(exec, &p2).then(exec, &p2);
    HPX_TEST(f2.get() == 4);
}

template <typename Executor>
void test_then(Executor& exec)
{
    test_return_int(exec);
    test_return_void(exec);
    test_implicit_unwrapping(exec);
    test_simple_then(exec);
    test_simple_deferred_then(exec);
    test_complex_then(exec);
    test_complex_then_chain_one(exec);
    test_complex_then_chain_two(exec);
}

///////////////////////////////////////////////////////////////////////////////
using hpx::program_options::options_description;
using hpx::program_options::variables_map;

int hpx_main(variables_map&)
{
    {
        hpx::execution::sequenced_executor exec;
        test_then(exec);
    }

    {
        hpx::execution::parallel_executor exec;
        test_then(exec);
    }

    hpx::local::finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.desc_cmdline = cmdline;
    init_args.cfg = cfg;

    return hpx::local::init(hpx_main, argc, argv, init_args);
}
