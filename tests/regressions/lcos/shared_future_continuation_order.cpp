//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Making sure the continuations of a shared_future are invoked in the same
// order as they have been attached.

#include <hpx/hpx_main.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

boost::atomic<int> invocation_count(0);

///////////////////////////////////////////////////////////////////////////////
int main()
{
    hpx::lcos::local::promise<int> p;
    hpx::shared_future<int> f1 = p.get_future();

    hpx::future<int> f2 =
        f1.then(
            [](hpx::shared_future<int> && f)
            {
                HPX_TEST_EQ(f.get(), 42);
                return ++invocation_count;
            });

    hpx::future<int> f3 =
        f1.then(
            [](hpx::shared_future<int> && f)
            {
                HPX_TEST_EQ(f.get(), 42);
                return ++invocation_count;
            });

    p.set_value(42);

    HPX_TEST_EQ(f1.get(), 1);
    HPX_TEST_EQ(f2.get(), 2);

    return hpx::util::report_errors();
}

