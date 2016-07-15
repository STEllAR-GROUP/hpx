//  Copyright (C) 2012-2013 Vicente Botet
//  Copyright (c) 2013 Agustin Berge
//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/include/threadmanager.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int p1()
{
    hpx::this_thread::sleep_for(boost::chrono::milliseconds(500));
    return 1;
}

int p2(hpx::lcos::future<int> f)
{
    HPX_TEST(f.valid());
    int i = f.get();
    hpx::this_thread::sleep_for(boost::chrono::milliseconds(500));
    return 2 * i;
}

void p3(hpx::lcos::future<int> f)
{
    HPX_TEST(f.valid());
    int i = f.get();
    (void)i;
    hpx::this_thread::sleep_for(boost::chrono::milliseconds(500));
    return;
}

hpx::lcos::future<int> p4(hpx::lcos::future<int> f)
{
    return hpx::async(p2, std::move(f));
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_return_int(Executor& exec)
{
    hpx::lcos::future<int> f1 = hpx::async(exec, &p1);
    HPX_TEST(f1.valid());
    hpx::lcos::future<int> f2 = f1.then(exec, &p2);
    HPX_TEST(f2.valid());
    try {
        HPX_TEST(f2.get()==2);
    }
    catch (hpx::exception const& /*ex*/) {
        HPX_TEST(false);
    }
    catch (...) {
        HPX_TEST(false);
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_return_void(Executor& exec)
{
    hpx::lcos::future<int> f1 = hpx::async(exec, &p1);
    HPX_TEST(f1.valid());
    hpx::lcos::future<void> f2 = f1.then(exec, &p3);
    HPX_TEST(f2.valid());
    try {
        f2.wait();
    }
    catch (hpx::exception const& /*ex*/) {
        HPX_TEST(false);
    }
    catch (...) {
        HPX_TEST(false);
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_implicit_unwrapping(Executor& exec)
{
    hpx::lcos::future<int> f1 = hpx::async(exec, &p1);
    HPX_TEST(f1.valid());
    hpx::lcos::future<int> f2 = f1.then(exec, &p4);
    HPX_TEST(f2.valid());
    try {
        HPX_TEST(f2.get()==2);
    }
    catch (hpx::exception const& /*ex*/) {
        HPX_TEST(false);
    }
    catch (...) {
        HPX_TEST(false);
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_simple_then(Executor& exec)
{
    hpx::lcos::future<int> f2 = hpx::async(exec, p1).then(exec, &p2);
    HPX_TEST(f2.get()==2);
}

template <typename Executor>
void test_simple_deferred_then(Executor& exec)
{
    hpx::lcos::future<int> f2 = hpx::async(exec, p1).then(exec, &p2);
    HPX_TEST(f2.get()==2);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_complex_then(Executor& exec)
{
    hpx::lcos::future<int> f1 = hpx::async(exec, p1);
    hpx::lcos::future<int> f21 = f1.then(exec, &p2);
    hpx::lcos::future<int> f2= f21.then(exec, &p2);
    HPX_TEST(f2.get()==4);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_complex_then_chain_one(Executor& exec)
{
    hpx::lcos::future<int> f1 = hpx::async(exec, p1);
    hpx::lcos::future<int> f2= f1.then(exec, &p2).then(exec, &p2);
    HPX_TEST(f2.get()==4);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Executor>
void test_complex_then_chain_two(Executor& exec)
{
    hpx::lcos::future<int> f2 =
        hpx::async(exec, p1).then(exec, &p2).then(exec, &p2);
    HPX_TEST(f2.get()==4);
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
using boost::program_options::variables_map;
using boost::program_options::options_description;

int hpx_main(variables_map&)
{
    {
        hpx::parallel::sequential_executor exec;
        test_then(exec);
    }

    {
        hpx::parallel::parallel_executor exec;
        test_then(exec);
    }

    hpx::finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {
        "hpx.os_threads=" +
            std::to_string(hpx::threads::hardware_concurrency())
    };

    // Initialize and run HPX
    return hpx::init(cmdline, argc, argv, cfg);
}

