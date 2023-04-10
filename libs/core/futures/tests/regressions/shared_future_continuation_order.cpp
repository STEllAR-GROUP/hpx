//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Making sure the continuations of a shared_future are invoked in the same
// order as they have been attached.

#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>

std::atomic<int> invocation_count(0);

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    hpx::promise<int> p;
    hpx::shared_future<int> f1 = p.get_future();

    hpx::future<int> f2 = f1.then([](hpx::shared_future<int>&& f) {
        HPX_TEST_EQ(f.get(), 42);
        return ++invocation_count;
    });

    hpx::future<int> f3 = f1.then([](hpx::shared_future<int>&& f) {
        HPX_TEST_EQ(f.get(), 42);
        return ++invocation_count;
    });

    p.set_value(42);

    HPX_TEST_EQ(f1.get(), 42);
    HPX_TEST_EQ(f2.get(), 1);
    HPX_TEST_EQ(f3.get(), 2);

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
