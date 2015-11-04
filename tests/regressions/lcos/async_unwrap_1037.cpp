//  Copyright 2013 (c) Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #1037:
// implicit unwrapping of futures in async broken

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/async.hpp>
#include <hpx/util/lightweight_test.hpp>

hpx::future<int> f(int i)
{
    if(i == 0)
    {
        return hpx::async(f, i+1);
    }

    return hpx::make_ready_future(i);
}

int ff()
{
    return 1;
}

hpx::future<int> g()
{
    return f(1);
}

HPX_PLAIN_ACTION(g);

hpx::future<int> h()
{
    return hpx::async(f, 1);
}

HPX_PLAIN_ACTION(h);

hpx::future<int> i()
{
    return hpx::async(ff);
}

HPX_PLAIN_ACTION(i);

int main()
{
    HPX_TEST_EQ(f(0).get(), 1);
    HPX_TEST_EQ(f(1).get(), 1);
    {
        hpx::future<int> fut = hpx::async(f, 0);
        HPX_TEST_EQ(fut.get(), 1);
    }
    {
        hpx::future<int> fut = hpx::async(f, 1);
        HPX_TEST_EQ(fut.get(), 1);
    }
    {
        hpx::future<int> fut = hpx::async(g);
        HPX_TEST_EQ(fut.get(), 1);
    }
    hpx::future<int> f1 = hpx::async(g_action(), hpx::find_here());
    HPX_TEST_EQ(f1.get(), 1);
    {
        hpx::future<int> fut = hpx::async(h);
        HPX_TEST_EQ(fut.get(), 1);
    }
    hpx::future<int> f2 = hpx::async(h_action(), hpx::find_here());
    HPX_TEST_EQ(f2.get(), 1);
    {
        hpx::future<int> fut = hpx::async(i);
        HPX_TEST_EQ(fut.get(), 1);
    }
    hpx::future<int> f3 = hpx::async(i_action(), hpx::find_here());
    HPX_TEST_EQ(f3.get(), 1);
    return hpx::util::report_errors();
}
