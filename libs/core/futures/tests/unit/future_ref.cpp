// Copyright (C) 2014 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <chrono>
#include <functional>

int global;

int& foo()
{
    return global;
}

void test_make_ready_future()
{
    hpx::future<int&> f = hpx::make_ready_future(std::ref(global));
    HPX_TEST_EQ(&f.get(), &global);

    hpx::future<int&> f_at = hpx::make_ready_future_at(
        std::chrono::system_clock::now() + std::chrono::seconds(1),
        std::ref(global));
    HPX_TEST_EQ(&f_at.get(), &global);

    hpx::future<int&> f_after =
        hpx::make_ready_future_after(std::chrono::seconds(1), std::ref(global));
    HPX_TEST_EQ(&f_after.get(), &global);
}

void test_async()
{
    hpx::future<int&> f = hpx::async(&foo);
    HPX_TEST_EQ(&f.get(), &global);

    hpx::future<int&> f_sync = hpx::async(hpx::launch::sync, &foo);
    HPX_TEST_EQ(&f_sync.get(), &global);
}

int hpx_main()
{
    test_make_ready_future();
    test_async();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
