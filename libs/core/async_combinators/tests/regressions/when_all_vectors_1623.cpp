//  Copyright 2015 (c) Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #1623: hpx::wait_all()
// invoked with two vector<future<T>> fails

#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
void test_when_all()
{
    std::vector<hpx::future<void>> v1, v2;
    v1.push_back(hpx::make_ready_future());
    v2.push_back(hpx::make_ready_future());

    hpx::when_all(v1, v2).get();
}

void test_when_any()
{
    std::vector<hpx::future<void>> v1, v2;
    v1.push_back(hpx::make_ready_future());
    v2.push_back(hpx::make_ready_future());

    hpx::when_any(v1, v2).get();
}

void test_when_some()
{
    std::vector<hpx::future<void>> v1, v2;
    v1.push_back(hpx::make_ready_future());
    v2.push_back(hpx::make_ready_future());

    hpx::when_some(1, v1, v2).get();
}

///////////////////////////////////////////////////////////////////////////////
void test_wait_all()
{
    std::vector<hpx::future<void>> v1, v2;
    v1.push_back(hpx::make_ready_future());
    v2.push_back(hpx::make_ready_future());

    hpx::wait_all(v1, v2);
}

void test_wait_any()
{
    std::vector<hpx::future<void>> v1, v2;
    v1.push_back(hpx::make_ready_future());
    v2.push_back(hpx::make_ready_future());

    hpx::wait_any(v1, v2);
}

void test_wait_some()
{
    std::vector<hpx::future<void>> v1, v2;
    v1.push_back(hpx::make_ready_future());
    v2.push_back(hpx::make_ready_future());

    hpx::wait_some(1, v1, v2);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    test_when_all();
    test_when_any();
    test_when_some();

    test_wait_all();
    test_wait_any();
    test_wait_some();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
