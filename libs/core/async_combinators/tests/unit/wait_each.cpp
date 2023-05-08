//  Copyright (c) 2016 Lukas Troska
//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <chrono>
#include <cstddef>
#include <deque>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
template <unsigned id>
unsigned make_unsigned_slowly()
{
    hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
    return id;
}

template <class Container>
void test_wait_each_from_list()
{
    unsigned count = 10;
    unsigned call_count = 0;
    unsigned call_with_index_count = 0;

    auto callback = [count, &call_count](hpx::future<unsigned> fut) {
        ++call_count;

        unsigned id = fut.get();

        HPX_TEST_LT(id, count);
    };

    auto callback_with_index = [count, &call_with_index_count](
                                   std::size_t idx, hpx::future<unsigned> fut) {
        ++call_with_index_count;

        unsigned id = fut.get();

        HPX_TEST_EQ(idx, id);
        HPX_TEST_LT(id, count);
    };

    Container futures1;
    Container futures2;

    for (unsigned j = 0; j < count; ++j)
    {
        futures1.push_back(
            hpx::make_ready_future_after(std::chrono::milliseconds(100), j));

        futures2.push_back(
            hpx::make_ready_future_after(std::chrono::milliseconds(100), j));
    }

    hpx::wait_each(callback, futures1);
    hpx::wait_each(callback_with_index, futures2);

    HPX_TEST_EQ(call_count, count);
    HPX_TEST_EQ(call_with_index_count, count);

    for (const auto& f : futures1)
    {
        HPX_TEST(!f.valid());
    }

    for (const auto& f : futures2)
    {
        HPX_TEST(!f.valid());
    }
}

template <class Container>
void test_wait_each_from_list_iterators()
{
    unsigned count = 10;
    unsigned call_count = 0;
    unsigned call_with_index_count = 0;

    auto callback = [count, &call_count](hpx::future<unsigned> fut) {
        ++call_count;

        unsigned id = fut.get();

        HPX_TEST_LT(id, count);
    };

    auto callback_with_index = [count, &call_with_index_count](
                                   std::size_t idx, hpx::future<unsigned> fut) {
        ++call_with_index_count;

        unsigned id = fut.get();

        HPX_TEST_EQ(idx, id);
        HPX_TEST_LT(id, count);
    };

    Container futures1;
    Container futures2;

    for (unsigned j = 0; j < count; ++j)
    {
        futures1.push_back(
            hpx::make_ready_future_after(std::chrono::milliseconds(100), j));

        futures2.push_back(
            hpx::make_ready_future_after(std::chrono::milliseconds(100), j));
    }

    hpx::wait_each(callback, futures1.begin(), futures1.end());
    hpx::wait_each(callback_with_index, futures2.begin(), futures2.end());

    HPX_TEST_EQ(call_count, count);
    HPX_TEST_EQ(call_with_index_count, count);

    for (const auto& f : futures1)
    {
        HPX_TEST(!f.valid());
    }

    for (const auto& f : futures2)
    {
        HPX_TEST(!f.valid());
    }
}

template <class Container>
void test_wait_each_n_from_list_iterators()
{
    unsigned count = 10;
    unsigned n = 5;

    unsigned call_count = 0;
    unsigned call_with_index_count = 0;

    auto callback_n = [n, &call_count](hpx::future<unsigned> fut) {
        ++call_count;

        unsigned id = fut.get();

        HPX_TEST_LT(id, n);
    };

    auto callback_with_index_n = [n, &call_with_index_count](std::size_t idx,
                                     hpx::future<unsigned> fut) {
        ++call_with_index_count;

        unsigned id = fut.get();

        HPX_TEST_EQ(idx, id);
        HPX_TEST_LT(id, n);
    };

    Container futures1;
    Container futures2;

    for (unsigned j = 0; j < count; ++j)
    {
        futures1.push_back(
            hpx::make_ready_future_after(std::chrono::milliseconds(100), j));

        futures2.push_back(
            hpx::make_ready_future_after(std::chrono::milliseconds(100), j));
    }

    hpx::wait_each_n(callback_n, futures1.begin(), n);
    hpx::wait_each_n(callback_with_index_n, futures2.begin(), n);

    HPX_TEST_EQ(call_count, n);
    HPX_TEST_EQ(call_with_index_count, n);

    unsigned num = 0;
    for (auto it = futures1.begin(); num < n; ++num, ++it)
    {
        HPX_TEST(!it->valid());
    }

    num = 0;
    for (auto it = futures2.begin(); num < n; ++num, ++it)
    {
        HPX_TEST(!it->valid());
    }
}

void test_wait_each_one_future()
{
    unsigned count = 1;
    unsigned call_count = 0;
    unsigned call_with_index_count = 0;

    auto callback = [count, &call_count](hpx::future<unsigned> fut) {
        ++call_count;

        unsigned id = fut.get();

        HPX_TEST_LT(id, count);
    };

    auto callback_with_index = [count, &call_with_index_count](
                                   std::size_t idx, hpx::future<unsigned> fut) {
        ++call_with_index_count;

        unsigned id = fut.get();

        HPX_TEST_EQ(idx, id);
        HPX_TEST_LT(id, count);
    };

    hpx::future<unsigned> f = hpx::make_ready_future(static_cast<unsigned>(0));
    hpx::future<unsigned> g = hpx::make_ready_future(static_cast<unsigned>(0));

    hpx::wait_each(callback, std::move(f));
    hpx::wait_each(callback_with_index, std::move(g));

    HPX_TEST_EQ(call_count, count);
    HPX_TEST_EQ(call_with_index_count, count);

    HPX_TEST(!f.valid());    // NOLINT
    HPX_TEST(!g.valid());    // NOLINT
}

void test_wait_each_two_futures()
{
    unsigned count = 2;
    unsigned call_count = 0;
    unsigned call_with_index_count = 0;

    auto callback = [count, &call_count](hpx::future<unsigned> fut) {
        ++call_count;

        unsigned id = fut.get();

        HPX_TEST_LT(id, count);
    };

    auto callback_with_index = [count, &call_with_index_count](
                                   std::size_t idx, hpx::future<unsigned> fut) {
        ++call_with_index_count;

        unsigned id = fut.get();

        HPX_TEST_EQ(idx, id);
        HPX_TEST_LT(id, count);
    };

    hpx::future<unsigned> f1 = hpx::make_ready_future(static_cast<unsigned>(0));
    hpx::future<unsigned> f2 = hpx::make_ready_future(static_cast<unsigned>(1));
    hpx::future<unsigned> g1 = hpx::make_ready_future(static_cast<unsigned>(0));
    hpx::future<unsigned> g2 = hpx::make_ready_future(static_cast<unsigned>(1));

    hpx::wait_each(callback, std::move(f1), std::move(f2));
    hpx::wait_each(callback_with_index, std::move(g1), std::move(g2));

    HPX_TEST_EQ(call_count, count);
    HPX_TEST_EQ(call_with_index_count, count);

    HPX_TEST(!f1.valid());    // NOLINT
    HPX_TEST(!f2.valid());    // NOLINT
    HPX_TEST(!g1.valid());    // NOLINT
    HPX_TEST(!g2.valid());    // NOLINT
}

void test_wait_each_three_futures()
{
    unsigned count = 3;
    unsigned call_count = 0;
    unsigned call_with_index_count = 0;

    auto callback = [count, &call_count](hpx::future<unsigned> fut) {
        ++call_count;

        unsigned id = fut.get();

        HPX_TEST_LT(id, count);
    };

    auto callback_with_index = [count, &call_with_index_count](
                                   std::size_t idx, hpx::future<unsigned> fut) {
        ++call_with_index_count;

        unsigned id = fut.get();

        HPX_TEST_EQ(idx, id);
        HPX_TEST_LT(id, count);
    };

    hpx::future<unsigned> f1 = hpx::make_ready_future(static_cast<unsigned>(0));
    hpx::future<unsigned> f2 = hpx::make_ready_future(static_cast<unsigned>(1));
    hpx::future<unsigned> f3 = hpx::make_ready_future(static_cast<unsigned>(2));
    hpx::future<unsigned> g1 = hpx::make_ready_future(static_cast<unsigned>(0));
    hpx::future<unsigned> g2 = hpx::make_ready_future(static_cast<unsigned>(1));
    hpx::future<unsigned> g3 = hpx::make_ready_future(static_cast<unsigned>(2));

    hpx::wait_each(callback, std::move(f1), std::move(f2), std::move(f3));
    hpx::wait_each(
        callback_with_index, std::move(g1), std::move(g2), std::move(g3));

    HPX_TEST_EQ(call_count, count);
    HPX_TEST_EQ(call_with_index_count, count);

    HPX_TEST(!f1.valid());    // NOLINT
    HPX_TEST(!f2.valid());    // NOLINT
    HPX_TEST(!f3.valid());    // NOLINT
    HPX_TEST(!g1.valid());    // NOLINT
    HPX_TEST(!g2.valid());    // NOLINT
    HPX_TEST(!g3.valid());    // NOLINT
}

void test_wait_each_four_futures()
{
    unsigned count = 4;
    unsigned call_count = 0;
    unsigned call_with_index_count = 0;

    auto callback = [count, &call_count](hpx::future<unsigned> fut) {
        ++call_count;

        unsigned id = fut.get();

        HPX_TEST_LT(id, count);
    };

    auto callback_with_index = [count, &call_with_index_count](
                                   std::size_t idx, hpx::future<unsigned> fut) {
        ++call_with_index_count;

        unsigned id = fut.get();

        HPX_TEST_EQ(idx, id);
        HPX_TEST_LT(id, count);
    };

    hpx::future<unsigned> f1 = hpx::make_ready_future(static_cast<unsigned>(0));
    hpx::future<unsigned> f2 = hpx::make_ready_future(static_cast<unsigned>(1));
    hpx::future<unsigned> f3 = hpx::make_ready_future(static_cast<unsigned>(2));
    hpx::future<unsigned> f4 = hpx::make_ready_future(static_cast<unsigned>(3));
    hpx::future<unsigned> g1 = hpx::make_ready_future(static_cast<unsigned>(0));
    hpx::future<unsigned> g2 = hpx::make_ready_future(static_cast<unsigned>(1));
    hpx::future<unsigned> g3 = hpx::make_ready_future(static_cast<unsigned>(2));
    hpx::future<unsigned> g4 = hpx::make_ready_future(static_cast<unsigned>(3));

    hpx::wait_each(
        callback, std::move(f1), std::move(f2), std::move(f3), std::move(f4));
    hpx::wait_each(callback_with_index, std::move(g1), std::move(g2),
        std::move(g3), std::move(g4));

    HPX_TEST_EQ(call_count, count);
    HPX_TEST_EQ(call_with_index_count, count);

    HPX_TEST(!f1.valid());    // NOLINT
    HPX_TEST(!f2.valid());    // NOLINT
    HPX_TEST(!f3.valid());    // NOLINT
    HPX_TEST(!f4.valid());    // NOLINT
    HPX_TEST(!g1.valid());    // NOLINT
    HPX_TEST(!g2.valid());    // NOLINT
    HPX_TEST(!g3.valid());    // NOLINT
    HPX_TEST(!g4.valid());    // NOLINT
}

void test_wait_each_five_futures()
{
    unsigned count = 5;
    unsigned call_count = 0;
    unsigned call_with_index_count = 0;

    auto callback = [count, &call_count](hpx::future<unsigned> fut) {
        ++call_count;

        unsigned id = fut.get();

        HPX_TEST_LT(id, count);
    };

    auto callback_with_index = [count, &call_with_index_count](
                                   std::size_t idx, hpx::future<unsigned> fut) {
        ++call_with_index_count;

        unsigned id = fut.get();

        HPX_TEST_EQ(idx, id);
        HPX_TEST_LT(id, count);
    };

    hpx::future<unsigned> f1 = hpx::make_ready_future(static_cast<unsigned>(0));
    hpx::future<unsigned> f2 = hpx::make_ready_future(static_cast<unsigned>(1));
    hpx::future<unsigned> f3 = hpx::make_ready_future(static_cast<unsigned>(2));
    hpx::future<unsigned> f4 = hpx::make_ready_future(static_cast<unsigned>(3));
    hpx::future<unsigned> f5 = hpx::make_ready_future(static_cast<unsigned>(4));
    hpx::future<unsigned> g1 = hpx::make_ready_future(static_cast<unsigned>(0));
    hpx::future<unsigned> g2 = hpx::make_ready_future(static_cast<unsigned>(1));
    hpx::future<unsigned> g3 = hpx::make_ready_future(static_cast<unsigned>(2));
    hpx::future<unsigned> g4 = hpx::make_ready_future(static_cast<unsigned>(3));
    hpx::future<unsigned> g5 = hpx::make_ready_future(static_cast<unsigned>(4));

    hpx::wait_each(callback, std::move(f1), std::move(f2), std::move(f3),
        std::move(f4), std::move(f5));

    hpx::wait_each(callback_with_index, std::move(g1), std::move(g2),
        std::move(g3), std::move(g4), std::move(g5));

    HPX_TEST_EQ(call_count, count);
    HPX_TEST_EQ(call_with_index_count, count);

    HPX_TEST(!f1.valid());    // NOLINT
    HPX_TEST(!f2.valid());    // NOLINT
    HPX_TEST(!f3.valid());    // NOLINT
    HPX_TEST(!f4.valid());    // NOLINT
    HPX_TEST(!f5.valid());    // NOLINT
    HPX_TEST(!g1.valid());    // NOLINT
    HPX_TEST(!g2.valid());    // NOLINT
    HPX_TEST(!g3.valid());    // NOLINT
    HPX_TEST(!g4.valid());    // NOLINT
    HPX_TEST(!g5.valid());    // NOLINT
}

void test_wait_each_late_future()
{
    unsigned count = 2;
    unsigned call_count = 0;
    unsigned call_with_index_count = 0;

    auto callback = [count, &call_count](hpx::future<unsigned> fut) {
        ++call_count;

        unsigned id = fut.get();

        HPX_TEST_LT(id, count);
    };

    auto callback_with_index = [count, &call_with_index_count](
                                   std::size_t idx, hpx::future<unsigned> fut) {
        ++call_with_index_count;

        unsigned id = fut.get();

        HPX_TEST_EQ(idx, id);
        HPX_TEST_LT(id, count);
    };

    hpx::lcos::local::futures_factory<unsigned()> pt0(make_unsigned_slowly<0>);
    hpx::lcos::local::futures_factory<unsigned()> pt1(make_unsigned_slowly<1>);
    hpx::lcos::local::futures_factory<unsigned()> pt2(make_unsigned_slowly<0>);
    hpx::lcos::local::futures_factory<unsigned()> pt3(make_unsigned_slowly<1>);

    hpx::future<unsigned> f1 = pt0.get_future();
    hpx::future<unsigned> f2 = pt1.get_future();

    hpx::async([pt0 = std::move(pt0)]() { pt0.post(); });
    hpx::async([pt1 = std::move(pt1)]() { pt1.post(); });

    hpx::wait_each(callback, std::move(f1), std::move(f2));

    HPX_TEST(!f1.valid());    // NOLINT
    HPX_TEST(!f2.valid());    // NOLINT

    hpx::future<unsigned> g1 = pt2.get_future();
    hpx::future<unsigned> g2 = pt3.get_future();

    hpx::async([pt2 = std::move(pt2)]() { pt2.post(); });
    hpx::async([pt3 = std::move(pt3)]() { pt3.post(); });

    hpx::wait_each(callback_with_index, std::move(g1), std::move(g2));

    HPX_TEST_EQ(call_count, count);
    HPX_TEST_EQ(call_with_index_count, count);

    HPX_TEST(!g1.valid());    // NOLINT
    HPX_TEST(!g2.valid());    // NOLINT
}

void test_wait_each_deferred_futures()
{
    unsigned count = 2;
    unsigned call_count = 0;
    unsigned call_with_index_count = 0;

    auto callback = [count, &call_count](hpx::future<unsigned> fut) {
        ++call_count;

        unsigned id = fut.get();

        HPX_TEST_LT(id, count);
    };

    auto callback_with_index = [count, &call_with_index_count](
                                   std::size_t idx, hpx::future<unsigned> fut) {
        ++call_with_index_count;

        unsigned id = fut.get();

        HPX_TEST_EQ(idx, id);
        HPX_TEST_LT(id, count);
    };

    hpx::future<unsigned> f1 =
        hpx::async(hpx::launch::deferred, &make_unsigned_slowly<0>);
    hpx::future<unsigned> f2 =
        hpx::async(hpx::launch::deferred, &make_unsigned_slowly<1>);

    hpx::future<unsigned> g1 =
        hpx::async(hpx::launch::deferred, &make_unsigned_slowly<0>);
    hpx::future<unsigned> g2 =
        hpx::async(hpx::launch::deferred, &make_unsigned_slowly<1>);

    hpx::wait_each(callback, std::move(f1), std::move(f2));
    hpx::wait_each(callback_with_index, std::move(g1), std::move(g2));

    HPX_TEST_EQ(call_count, count);
    HPX_TEST_EQ(call_with_index_count, count);

    HPX_TEST(!f1.valid());    // NOLINT
    HPX_TEST(!f2.valid());    // NOLINT

    HPX_TEST(!g1.valid());    // NOLINT
    HPX_TEST(!g2.valid());    // NOLINT
}

///////////////////////////////////////////////////////////////////////////////
using hpx::program_options::options_description;
using hpx::program_options::variables_map;

using hpx::future;

int hpx_main(variables_map&)
{
    {
        test_wait_each_from_list<std::vector<future<unsigned>>>();

        test_wait_each_from_list_iterators<std::vector<future<unsigned>>>();
        test_wait_each_from_list_iterators<std::list<future<unsigned>>>();
        test_wait_each_from_list_iterators<std::deque<future<unsigned>>>();

        test_wait_each_n_from_list_iterators<std::vector<future<unsigned>>>();
        test_wait_each_n_from_list_iterators<std::list<future<unsigned>>>();
        test_wait_each_n_from_list_iterators<std::deque<future<unsigned>>>();

        test_wait_each_one_future();
        test_wait_each_two_futures();
        test_wait_each_three_futures();
        test_wait_each_four_futures();
        test_wait_each_five_futures();

        test_wait_each_late_future();

        test_wait_each_deferred_futures();
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
