//  Copyright 2015 (c) Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #1623: hpx::wait_all()
// invoked with two vector<future<T>> fails

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/util/lightweight_test.hpp>

///////////////////////////////////////////////////////////////////////////////
void test_when_all()
{
    std::vector<hpx::future<void> > v1, v2;
    v1.push_back(hpx::make_ready_future());
    v2.push_back(hpx::make_ready_future());

    hpx::when_all(v1, v2).get();
}

void test_when_any()
{
    std::vector<hpx::future<void> > v1, v2;
    v1.push_back(hpx::make_ready_future());
    v2.push_back(hpx::make_ready_future());

    hpx::when_any(v1, v2).get();
}

void test_when_some()
{
    std::vector<hpx::future<void> > v1, v2;
    v1.push_back(hpx::make_ready_future());
    v2.push_back(hpx::make_ready_future());

    hpx::when_some(1, v1, v2).get();
}

///////////////////////////////////////////////////////////////////////////////
void test_wait_all()
{
    std::vector<hpx::future<void> > v1, v2;
    v1.push_back(hpx::make_ready_future());
    v2.push_back(hpx::make_ready_future());

    hpx::wait_all(v1, v2);
}

void test_wait_any()
{
    std::vector<hpx::future<void> > v1, v2;
    v1.push_back(hpx::make_ready_future());
    v2.push_back(hpx::make_ready_future());

    hpx::wait_any(v1, v2);
}

void test_wait_some()
{
    std::vector<hpx::future<void> > v1, v2;
    v1.push_back(hpx::make_ready_future());
    v2.push_back(hpx::make_ready_future());

    hpx::wait_some(1, v1, v2);
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    test_when_all();
    test_when_any();
    test_when_some();

    test_wait_all();
    test_wait_any();
    test_wait_some();

    return hpx::util::report_errors();
}
