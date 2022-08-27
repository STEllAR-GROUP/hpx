//  Copyright (C) 2012-2022 Hartmut Kaiser
//  (C) Copyright 2008-10 Anthony Williams
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/future.hpp>
#include <hpx/local/init.hpp>
#include <hpx/local/thread.hpp>
#include <hpx/modules/testing.hpp>

#include <chrono>
#include <deque>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int make_int_slowly()
{
    hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
    return 42;
}

template <class Container>
void test_when_all_from_list()
{
    unsigned const count = 10;
    Container futures;
    for (unsigned j = 0; j < count; ++j)
    {
        hpx::lcos::local::futures_factory<int()> task(make_int_slowly);
        futures.push_back(task.get_future());
        task.apply();
    }

    hpx::future<Container> r = hpx::when_all(futures);

    Container result = r.get();

    HPX_TEST_EQ(futures.size(), result.size());
    for (const auto& f : futures)
        HPX_TEST(!f.valid());
    for (const auto& r : result)
        HPX_TEST(r.is_ready());
}

template <class Container>
void test_when_all_from_list_iterators()
{
    unsigned const count = 10;

    Container futures;
    for (unsigned j = 0; j < count; ++j)
    {
        hpx::lcos::local::futures_factory<int()> task(make_int_slowly);
        futures.push_back(task.get_future());
        task.apply();
    }

    hpx::future<Container> r =
        hpx::when_all<typename Container::iterator, Container>(
            futures.begin(), futures.end());

    Container result = r.get();

    HPX_TEST_EQ(futures.size(), result.size());
    for (const auto& f : futures)
        HPX_TEST(!f.valid());
    for (const auto& r : result)
        HPX_TEST(r.is_ready());
}

void test_when_all_one_future()
{
    hpx::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    hpx::future<int> f1 = pt1.get_future();
    pt1.apply();

    typedef hpx::tuple<hpx::future<int>> result_type;
    hpx::future<result_type> r = hpx::when_all(f1);

    result_type result = r.get();

    HPX_TEST(!f1.valid());

    HPX_TEST(hpx::get<0>(result).is_ready());
}

void test_when_all_two_futures()
{
    hpx::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    hpx::future<int> f1 = pt1.get_future();
    pt1.apply();
    hpx::lcos::local::futures_factory<int()> pt2(make_int_slowly);
    hpx::future<int> f2 = pt2.get_future();
    pt2.apply();

    typedef hpx::tuple<hpx::future<int>, hpx::future<int>> result_type;
    hpx::future<result_type> r = hpx::when_all(f1, f2);

    result_type result = r.get();

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());

    HPX_TEST(hpx::get<0>(result).is_ready());
    HPX_TEST(hpx::get<1>(result).is_ready());
}

void test_when_all_three_futures()
{
    hpx::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    hpx::future<int> f1 = pt1.get_future();
    pt1.apply();
    hpx::lcos::local::futures_factory<int()> pt2(make_int_slowly);
    hpx::future<int> f2 = pt2.get_future();
    pt2.apply();
    hpx::lcos::local::futures_factory<int()> pt3(make_int_slowly);
    hpx::future<int> f3 = pt3.get_future();
    pt3.apply();

    typedef hpx::tuple<hpx::future<int>, hpx::future<int>, hpx::future<int>>
        result_type;
    hpx::future<result_type> r = hpx::when_all(f1, f2, f3);

    result_type result = r.get();

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());

    HPX_TEST(hpx::get<0>(result).is_ready());
    HPX_TEST(hpx::get<1>(result).is_ready());
    HPX_TEST(hpx::get<2>(result).is_ready());
}

void test_when_all_four_futures()
{
    hpx::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    hpx::future<int> f1 = pt1.get_future();
    pt1.apply();
    hpx::lcos::local::futures_factory<int()> pt2(make_int_slowly);
    hpx::future<int> f2 = pt2.get_future();
    pt2.apply();
    hpx::lcos::local::futures_factory<int()> pt3(make_int_slowly);
    hpx::future<int> f3 = pt3.get_future();
    pt3.apply();
    hpx::lcos::local::futures_factory<int()> pt4(make_int_slowly);
    hpx::future<int> f4 = pt4.get_future();
    pt4.apply();

    typedef hpx::tuple<hpx::future<int>, hpx::future<int>, hpx::future<int>,
        hpx::future<int>>
        result_type;
    hpx::future<result_type> r = hpx::when_all(f1, f2, f3, f4);

    result_type result = r.get();

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());
    HPX_TEST(!f4.valid());

    HPX_TEST(hpx::get<0>(result).is_ready());
    HPX_TEST(hpx::get<1>(result).is_ready());
    HPX_TEST(hpx::get<2>(result).is_ready());
    HPX_TEST(hpx::get<3>(result).is_ready());
}

void test_when_all_five_futures()
{
    hpx::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    hpx::future<int> f1 = pt1.get_future();
    pt1.apply();
    hpx::lcos::local::futures_factory<int()> pt2(make_int_slowly);
    hpx::future<int> f2 = pt2.get_future();
    pt2.apply();
    hpx::lcos::local::futures_factory<int()> pt3(make_int_slowly);
    hpx::future<int> f3 = pt3.get_future();
    pt3.apply();
    hpx::lcos::local::futures_factory<int()> pt4(make_int_slowly);
    hpx::future<int> f4 = pt4.get_future();
    pt4.apply();
    hpx::lcos::local::futures_factory<int()> pt5(make_int_slowly);
    hpx::future<int> f5 = pt5.get_future();
    pt5.apply();

    typedef hpx::tuple<hpx::future<int>, hpx::future<int>, hpx::future<int>,
        hpx::future<int>, hpx::future<int>>
        result_type;
    hpx::future<result_type> r = hpx::when_all(f1, f2, f3, f4, f5);

    result_type result = r.get();

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());
    HPX_TEST(!f4.valid());
    HPX_TEST(!f5.valid());

    HPX_TEST(hpx::get<0>(result).is_ready());
    HPX_TEST(hpx::get<1>(result).is_ready());
    HPX_TEST(hpx::get<2>(result).is_ready());
    HPX_TEST(hpx::get<3>(result).is_ready());
    HPX_TEST(hpx::get<4>(result).is_ready());
}

void test_when_all_late_futures()
{
    hpx::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    hpx::future<int> f1 = pt1.get_future();
    pt1.apply();
    hpx::lcos::local::futures_factory<int()> pt2(make_int_slowly);
    hpx::future<int> f2 = pt2.get_future();

    typedef hpx::tuple<hpx::future<int>, hpx::future<int>> result_type;
    hpx::future<result_type> r = hpx::when_all(f1, f2);

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());

    pt2.apply();

    result_type result = r.get();

    HPX_TEST(hpx::get<0>(result).is_ready());
    HPX_TEST(hpx::get<1>(result).is_ready());
}

void test_when_all_deferred_futures()
{
    hpx::future<int> f1 = hpx::async(hpx::launch::deferred, &make_int_slowly);
    hpx::future<int> f2 = hpx::async(hpx::launch::deferred, &make_int_slowly);

    typedef hpx::tuple<hpx::future<int>, hpx::future<int>> result_type;
    hpx::future<result_type> r = hpx::when_all(f1, f2);

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());

    result_type result = r.get();

    HPX_TEST(hpx::get<0>(result).is_ready());
    HPX_TEST(hpx::get<1>(result).is_ready());
}

///////////////////////////////////////////////////////////////////////////////
using hpx::program_options::options_description;
using hpx::program_options::variables_map;

using hpx::future;

int hpx_main(variables_map&)
{
    {
        test_when_all_from_list<std::vector<future<int>>>();
        test_when_all_from_list<std::list<future<int>>>();
        test_when_all_from_list<std::deque<future<int>>>();
        test_when_all_from_list_iterators<std::vector<future<int>>>();
        test_when_all_from_list_iterators<std::list<future<int>>>();
        test_when_all_from_list_iterators<std::deque<future<int>>>();
        test_when_all_one_future();
        test_when_all_two_futures();
        test_when_all_three_futures();
        test_when_all_four_futures();
        test_when_all_five_futures();
        test_when_all_late_futures();
        test_when_all_deferred_futures();
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
