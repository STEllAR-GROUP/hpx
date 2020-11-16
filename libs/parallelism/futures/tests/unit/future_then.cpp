//  Copyright (C) 2012-2013 Vicente Botet
//  Copyright (c) 2013 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/threadmanager.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int p1()
{
    hpx::this_thread::sleep_for(std::chrono::milliseconds(500));
    return 1;
}

int p2(hpx::future<int> f)
{
    HPX_TEST(f.valid());
    int i = f.get();
    hpx::this_thread::sleep_for(std::chrono::milliseconds(500));
    return 2 * i;
}

void p3(hpx::future<int> f)
{
    HPX_TEST(f.valid());
    int i = f.get();
    (void) i;
    hpx::this_thread::sleep_for(std::chrono::milliseconds(500));
    return;
}

hpx::future<int> p4(hpx::future<int> f)
{
    return hpx::async(p2, std::move(f));
}

///////////////////////////////////////////////////////////////////////////////
void test_return_int()
{
    hpx::future<int> f1 = hpx::async(hpx::launch::async, &p1);
    HPX_TEST(f1.valid());
    hpx::future<int> f2 = f1.then(&p2);
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

void test_return_int_launch()
{
    hpx::future<int> f1 = hpx::async(hpx::launch::async, &p1);
    HPX_TEST(f1.valid());
    hpx::future<int> f2 = f1.then(hpx::launch::async, &p2);
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
void test_return_void()
{
    hpx::future<int> f1 = hpx::async(hpx::launch::async, &p1);
    HPX_TEST(f1.valid());
    hpx::future<void> f2 = f1.then(&p3);
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

void test_return_void_launch()
{
    hpx::future<int> f1 = hpx::async(hpx::launch::async, &p1);
    HPX_TEST(f1.valid());
    hpx::future<void> f2 = f1.then(hpx::launch::sync, &p3);
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
void test_implicit_unwrapping()
{
    hpx::future<int> f1 = hpx::async(hpx::launch::async, &p1);
    HPX_TEST(f1.valid());
    hpx::future<int> f2 = f1.then(&p4);
    HPX_TEST(f2.valid());
    try
    {
        HPX_TEST(f2.get() == 2);
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
void test_simple_then()
{
    hpx::future<int> f2 = hpx::async(p1).then(&p2);
    HPX_TEST(f2.get() == 2);
}

void test_simple_deferred_then()
{
    hpx::future<int> f2 = hpx::async(hpx::launch::deferred, p1).then(&p2);
    HPX_TEST(f2.get() == 2);
}

///////////////////////////////////////////////////////////////////////////////
void test_complex_then()
{
    hpx::future<int> f1 = hpx::async(p1);
    hpx::future<int> f21 = f1.then(&p2);
    hpx::future<int> f2 = f21.then(&p2);
    HPX_TEST_EQ(f2.get(), 4);
}

void test_complex_then_launch()
{
    auto policy = hpx::launch::select([]() { return hpx::launch::async; });

    hpx::future<int> f1 = hpx::async(p1);
    hpx::future<int> f21 = f1.then(policy, &p2);
    hpx::future<int> f2 = f21.then(policy, &p2);
    HPX_TEST_EQ(f2.get(), 4);
}

///////////////////////////////////////////////////////////////////////////////
void test_complex_then_chain_one()
{
    hpx::future<int> f1 = hpx::async(p1);
    hpx::future<int> f2 = f1.then(&p2).then(&p2);
    HPX_TEST(f2.get() == 4);
}

void test_complex_then_chain_one_launch()
{
    std::atomic<int> count(0);
    auto policy = hpx::launch::select([&count]() -> hpx::launch {
        if (count++ == 0)
            return hpx::launch::async;
        return hpx::launch::sync;
    });

    hpx::future<int> f1 = hpx::async(p1);
    hpx::future<int> f2 = f1.then(policy, &p2).then(policy, &p2);
    HPX_TEST(f2.get() == 4);
}

///////////////////////////////////////////////////////////////////////////////
void test_complex_then_chain_two()
{
    hpx::future<int> f2 = hpx::async(p1).then(&p2).then(&p2);
    HPX_TEST(f2.get() == 4);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    {
        test_return_int();
        test_return_int_launch();
        test_return_void();
        test_return_void_launch();
        test_implicit_unwrapping();
        test_simple_then();
        test_simple_deferred_then();
        test_complex_then();
        test_complex_then_launch();
        test_complex_then_chain_one();
        test_complex_then_chain_one_launch();
        test_complex_then_chain_two();
    }

    hpx::finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.cfg = cfg;

    return hpx::init(argc, argv, init_args);
}
