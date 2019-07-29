// Copyright (C) 2015-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>

#if !defined(HPX_HAVE_AWAIT)
#error "This test requires compiler support for await"
#endif

#include <hpx/hpx_init.hpp>
#include <hpx/include/local_lcos.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/include/async.hpp>

#include <hpx/testing.hpp>

#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
hpx::future<int> fib1(int n)
{
    if (n >= 2)
        n = co_await fib1(n - 1) + co_await fib1(n - 2);
    co_return n;
}

hpx::future<int> fib2(int n)
{
    if (n >= 2)
        n = co_await hpx::async(&fib2, n - 1) + co_await fib2(n - 2);
    co_return n;
}

void simple_await_test()
{
    HPX_TEST_EQ(fib1(10).get(), 55);
    HPX_TEST_EQ(fib2(10).get(), 55);
}

int hpx_main()
{
    simple_await_test();

    HPX_TEST_EQ(hpx::finalize(), 0);
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    return hpx::init(argc, argv, cfg);
}
