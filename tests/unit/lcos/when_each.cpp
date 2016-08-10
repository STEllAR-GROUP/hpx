//  Copyright (c) 2016 Lukas Troska
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <chrono>
#include <deque>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
template<unsigned id>
unsigned make_unsigned_slowly()
{
    hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
    return id;
}

template <class Container>
void test_when_each_from_list()
{
    unsigned const count = 10;
    unsigned call_count = 0;
    unsigned call_with_index_count = 0;

    auto callback =
        [count, &call_count](hpx::future<unsigned> fut)
        {
            ++call_count;

            unsigned id = fut.get();

            HPX_TEST(id >= 0);
            HPX_TEST(id < count);
        };

    auto callback_with_index =
        [count, &call_with_index_count](std::size_t idx, hpx::future<unsigned> fut)
        {
            ++call_with_index_count;

            unsigned id = fut.get();

            HPX_TEST_EQ(idx, id);
            HPX_TEST(id >= 0);
            HPX_TEST(id < count);
        };

    Container futures1;
    Container futures2;

    for (unsigned j = 0; j < count; ++j)
    {
        futures1.push_back(
            hpx::make_ready_future_after(std::chrono::milliseconds(100), j)
        );

        futures2.push_back(
            hpx::make_ready_future_after(std::chrono::milliseconds(100), j)
        );
    }

    hpx::future<void> r = hpx::when_each(callback, futures1);

    hpx::future<void> rwi = hpx::when_each(callback_with_index, futures2);

    r.get();
    rwi.get();

    HPX_TEST_EQ(call_count, count);
    HPX_TEST_EQ(call_with_index_count, count);

    for (const auto& f : futures1)
        HPX_TEST(!f.valid());

    for (const auto& f : futures2)
        HPX_TEST(!f.valid());
}

template <class Container>
void test_when_each_from_list_iterators()
{
    unsigned const count = 10;
    unsigned call_count = 0;
    unsigned call_with_index_count = 0;

    auto callback =
        [count, &call_count](hpx::future<unsigned> fut)
        {
            ++call_count;

            unsigned id = fut.get();

            HPX_TEST(id >= 0);
            HPX_TEST(id < count);
        };

    auto callback_with_index =
        [count, &call_with_index_count](std::size_t idx, hpx::future<unsigned> fut)
        {
            ++call_with_index_count;

            unsigned id = fut.get();

            HPX_TEST_EQ(idx, id);
            HPX_TEST(id >= 0);
            HPX_TEST(id < count);
        };

    Container futures1;
    Container futures2;

    for (unsigned j = 0; j < count; ++j)
    {
        futures1.push_back(
            hpx::make_ready_future_after(std::chrono::milliseconds(100), j)
        );

        futures2.push_back(
            hpx::make_ready_future_after(std::chrono::milliseconds(100), j)
        );
    }

    hpx::future<void> r =
        hpx::when_each(callback, futures1.begin(), futures1.end());

    hpx::future<void> rwi =
        hpx::when_each(callback_with_index, futures2.begin(), futures2.end());

    r.get();
    rwi.get();

    HPX_TEST_EQ(call_count, count);
    HPX_TEST_EQ(call_with_index_count, count);

    for (const auto& f : futures1)
        HPX_TEST(!f.valid());

    for (const auto& f : futures2)
        HPX_TEST(!f.valid());
}

template <class Container>
void test_when_each_n_from_list_iterators()
{
    unsigned const count = 10;
    unsigned const n = 5;

    unsigned call_count = 0;
    unsigned call_with_index_count = 0;

    auto callback_n =
        [n, &call_count](hpx::future<unsigned> fut)
        {
            ++call_count;

            unsigned id = fut.get();

            HPX_TEST(id >= 0);
            HPX_TEST(id < n);
        };

    auto callback_with_index_n =
        [n, &call_with_index_count](std::size_t idx, hpx::future<unsigned> fut)
        {
            ++call_with_index_count;

            unsigned id = fut.get();

            HPX_TEST_EQ(idx, id);
            HPX_TEST(id >= 0);
            HPX_TEST(id < n);
        };

    Container futures1;
    Container futures2;

    for (unsigned j = 0; j < count; ++j)
    {
        futures1.push_back(
            hpx::make_ready_future_after(std::chrono::milliseconds(100), j)
        );

        futures2.push_back(
            hpx::make_ready_future_after(std::chrono::milliseconds(100), j)
        );
    }

    hpx::future<void> r =
        hpx::when_each_n(callback_n, futures1.begin(), n);

    hpx::future<void> rwi =
        hpx::when_each_n(callback_with_index_n, futures2.begin(), n);

    r.get();
    rwi.get();

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

void test_when_each_one_future()
{
    unsigned const count = 1;
    unsigned call_count = 0;
    unsigned call_with_index_count = 0;

    auto callback =
        [count, &call_count](hpx::future<unsigned> fut)
        {
            ++call_count;

            unsigned id = fut.get();

            HPX_TEST(id >= 0);
            HPX_TEST(id < count);
        };

    auto callback_with_index =
        [count, &call_with_index_count](std::size_t idx, hpx::future<unsigned> fut)
        {
            ++call_with_index_count;

            unsigned id = fut.get();

            HPX_TEST_EQ(idx, id);
            HPX_TEST(id >= 0);
            HPX_TEST(id < count);
        };

    hpx::future<unsigned> f = hpx::make_ready_future(static_cast<unsigned>(0));
    hpx::future<unsigned> g = hpx::make_ready_future(static_cast<unsigned>(0));

    hpx::future<void> r = hpx::when_each(callback, std::move(f));
    hpx::future<void> rwi = hpx::when_each(callback_with_index, std::move(g));

    r.get();
    rwi.get();

    HPX_TEST_EQ(call_count, count);
    HPX_TEST_EQ(call_with_index_count, count);

    HPX_TEST(!f.valid());
    HPX_TEST(!g.valid());
}

void test_when_each_two_futures()
{
    unsigned const count = 2;
    unsigned call_count = 0;
    unsigned call_with_index_count = 0;

    auto callback =
        [count, &call_count](hpx::future<unsigned> fut)
        {
            ++call_count;

            unsigned id = fut.get();

            HPX_TEST(id >= 0);
            HPX_TEST(id < count);
        };

    auto callback_with_index =
        [count, &call_with_index_count](std::size_t idx, hpx::future<unsigned> fut)
        {
            ++call_with_index_count;

            unsigned id = fut.get();

            HPX_TEST_EQ(idx, id);
            HPX_TEST(id >= 0);
            HPX_TEST(id < count);
        };

    hpx::future<unsigned> f1 = hpx::make_ready_future(static_cast<unsigned>(0));
    hpx::future<unsigned> f2 = hpx::make_ready_future(static_cast<unsigned>(1));
    hpx::future<unsigned> g1 = hpx::make_ready_future(static_cast<unsigned>(0));
    hpx::future<unsigned> g2 = hpx::make_ready_future(static_cast<unsigned>(1));

    hpx::future<void> r = hpx::when_each(callback,
        std::move(f1), std::move(f2));

    hpx::future<void> rwi = hpx::when_each(callback_with_index,
        std::move(g1), std::move(g2));

    r.get();
    rwi.get();

    HPX_TEST_EQ(call_count, count);
    HPX_TEST_EQ(call_with_index_count, count);

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!g1.valid());
    HPX_TEST(!g2.valid());
}

void test_when_each_three_futures()
{
    unsigned const count = 3;
    unsigned call_count = 0;
    unsigned call_with_index_count = 0;

    auto callback =
        [count, &call_count](hpx::future<unsigned> fut)
        {
            ++call_count;

            unsigned id = fut.get();

            HPX_TEST(id >= 0);
            HPX_TEST(id < count);
        };

    auto callback_with_index =
        [count, &call_with_index_count](std::size_t idx, hpx::future<unsigned> fut)
        {
            ++call_with_index_count;

            unsigned id = fut.get();

            HPX_TEST_EQ(idx, id);
            HPX_TEST(id >= 0);
            HPX_TEST(id < count);
        };

    hpx::future<unsigned> f1 = hpx::make_ready_future(static_cast<unsigned>(0));
    hpx::future<unsigned> f2 = hpx::make_ready_future(static_cast<unsigned>(1));
    hpx::future<unsigned> f3 = hpx::make_ready_future(static_cast<unsigned>(2));
    hpx::future<unsigned> g1 = hpx::make_ready_future(static_cast<unsigned>(0));
    hpx::future<unsigned> g2 = hpx::make_ready_future(static_cast<unsigned>(1));
    hpx::future<unsigned> g3 = hpx::make_ready_future(static_cast<unsigned>(2));

    hpx::future<void> r = hpx::when_each(callback,
        std::move(f1), std::move(f2), std::move(f3));

    hpx::future<void> rwi = hpx::when_each(callback_with_index,
        std::move(g1), std::move(g2), std::move(g3));

    r.get();
    rwi.get();

    HPX_TEST_EQ(call_count, count);
    HPX_TEST_EQ(call_with_index_count, count);

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());
    HPX_TEST(!g1.valid());
    HPX_TEST(!g2.valid());
    HPX_TEST(!g3.valid());
}

void test_when_each_four_futures()
{
    unsigned const count = 4;
    unsigned call_count = 0;
    unsigned call_with_index_count = 0;

    auto callback =
        [count, &call_count](hpx::future<unsigned> fut)
        {
            ++call_count;

            unsigned id = fut.get();

            HPX_TEST(id >= 0);
            HPX_TEST(id < count);
        };

    auto callback_with_index =
        [count, &call_with_index_count](std::size_t idx, hpx::future<unsigned> fut)
        {
            ++call_with_index_count;

            unsigned id = fut.get();

            HPX_TEST_EQ(idx, id);
            HPX_TEST(id >= 0);
            HPX_TEST(id < count);
        };

    hpx::future<unsigned> f1 = hpx::make_ready_future(static_cast<unsigned>(0));
    hpx::future<unsigned> f2 = hpx::make_ready_future(static_cast<unsigned>(1));
    hpx::future<unsigned> f3 = hpx::make_ready_future(static_cast<unsigned>(2));
    hpx::future<unsigned> f4 = hpx::make_ready_future(static_cast<unsigned>(3));
    hpx::future<unsigned> g1 = hpx::make_ready_future(static_cast<unsigned>(0));
    hpx::future<unsigned> g2 = hpx::make_ready_future(static_cast<unsigned>(1));
    hpx::future<unsigned> g3 = hpx::make_ready_future(static_cast<unsigned>(2));
    hpx::future<unsigned> g4 = hpx::make_ready_future(static_cast<unsigned>(3));

    hpx::future<void> r = hpx::when_each(callback,
        std::move(f1), std::move(f2), std::move(f3), std::move(f4));

    hpx::future<void> rwi = hpx::when_each(callback_with_index,
        std::move(g1), std::move(g2), std::move(g3), std::move(g4));

    r.get();
    rwi.get();

    HPX_TEST_EQ(call_count, count);
    HPX_TEST_EQ(call_with_index_count, count);

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());
    HPX_TEST(!f4.valid());
    HPX_TEST(!g1.valid());
    HPX_TEST(!g2.valid());
    HPX_TEST(!g3.valid());
    HPX_TEST(!g4.valid());
}

void test_when_each_five_futures()
{
    unsigned const count = 5;
    unsigned call_count = 0;
    unsigned call_with_index_count = 0;

    auto callback =
        [count, &call_count](hpx::future<unsigned> fut)
        {
            ++call_count;

            unsigned id = fut.get();

            HPX_TEST(id >= 0);
            HPX_TEST(id < count);
        };

    auto callback_with_index =
        [count, &call_with_index_count](std::size_t idx, hpx::future<unsigned> fut)
        {
            ++call_with_index_count;

            unsigned id = fut.get();

            HPX_TEST_EQ(idx, id);
            HPX_TEST(id >= 0);
            HPX_TEST(id < count);
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

    hpx::future<void> r = hpx::when_each(callback,
        std::move(f1), std::move(f2), std::move(f3), std::move(f4), std::move(f5));

    hpx::future<void> rwi = hpx::when_each(callback_with_index,
        std::move(g1), std::move(g2), std::move(g3), std::move(g4), std::move(g5));

    r.get();
    rwi.get();

    HPX_TEST_EQ(call_count, count);
    HPX_TEST_EQ(call_with_index_count, count);

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());
    HPX_TEST(!f4.valid());
    HPX_TEST(!f5.valid());
    HPX_TEST(!g1.valid());
    HPX_TEST(!g2.valid());
    HPX_TEST(!g3.valid());
    HPX_TEST(!g4.valid());
    HPX_TEST(!g5.valid());
}

void test_when_each_late_future()
{
    unsigned const count = 2;
    unsigned call_count = 0;
    unsigned call_with_index_count = 0;

    auto callback =
        [count, &call_count](hpx::future<unsigned> fut)
        {
            ++call_count;

            unsigned id = fut.get();

            HPX_TEST(id >= 0);
            HPX_TEST(id < count);
        };

    auto callback_with_index =
        [count, &call_with_index_count](std::size_t idx, hpx::future<unsigned> fut)
        {
            ++call_with_index_count;

            unsigned id = fut.get();

            HPX_TEST_EQ(idx, id);
            HPX_TEST(id >= 0);
            HPX_TEST(id < count);
        };

    hpx::lcos::local::futures_factory<unsigned()> pt0(make_unsigned_slowly<0>);
    hpx::lcos::local::futures_factory<unsigned()> pt1(make_unsigned_slowly<1>);
    hpx::lcos::local::futures_factory<unsigned()> pt2(make_unsigned_slowly<0>);
    hpx::lcos::local::futures_factory<unsigned()> pt3(make_unsigned_slowly<1>);

    hpx::future<unsigned> f1 = pt0.get_future();
    pt0.apply();
    hpx::future<unsigned> f2 = pt1.get_future();



    hpx::future<void> r =
        hpx::when_each(callback, std::move(f1), std::move(f2));

    pt1.apply();

    r.get();

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());

    hpx::future<unsigned> g1 = pt2.get_future();
    pt2.apply();
    hpx::future<unsigned> g2 = pt3.get_future();


    hpx::future<void> rwi =
        hpx::when_each(callback_with_index, std::move(g1), std::move(g2));

    pt3.apply();

    rwi.get();

    HPX_TEST_EQ(call_count, count);
    HPX_TEST_EQ(call_with_index_count, count);

    HPX_TEST(!g1.valid());
    HPX_TEST(!g2.valid());
}

void test_when_each_deferred_futures()
{
    unsigned const count = 2;
    unsigned call_count = 0;
    unsigned call_with_index_count = 0;

    auto callback =
        [count, &call_count](hpx::future<unsigned> fut)
        {
            ++call_count;

            unsigned id = fut.get();

            HPX_TEST(id >= 0);
            HPX_TEST(id < count);
        };

    auto callback_with_index =
        [count, &call_with_index_count](std::size_t idx, hpx::future<unsigned> fut)
        {
            ++call_with_index_count;

            unsigned id = fut.get();

            HPX_TEST_EQ(idx, id);
            HPX_TEST(id >= 0);
            HPX_TEST(id < count);
        };

    hpx::lcos::future<unsigned> f1 =
        hpx::async(hpx::launch::deferred, &make_unsigned_slowly<0>);
    hpx::lcos::future<unsigned> f2 =
        hpx::async(hpx::launch::deferred, &make_unsigned_slowly<1>);

    hpx::lcos::future<unsigned> g1 =
        hpx::async(hpx::launch::deferred, &make_unsigned_slowly<0>);
    hpx::lcos::future<unsigned> g2 =
        hpx::async(hpx::launch::deferred, &make_unsigned_slowly<1>);


    hpx::future<void> r =
        hpx::when_each(callback, std::move(f1), std::move(f2));

    hpx::future<void> rwi =
        hpx::when_each(callback_with_index, std::move(g1), std::move(g2));

    r.get();
    rwi.get();

    HPX_TEST_EQ(call_count, count);
    HPX_TEST_EQ(call_with_index_count, count);

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());

    HPX_TEST(!g1.valid());
    HPX_TEST(!g2.valid());
}

///////////////////////////////////////////////////////////////////////////////
using boost::program_options::variables_map;
using boost::program_options::options_description;

using hpx::lcos::future;

int hpx_main(variables_map&)
{
    {
        test_when_each_from_list<std::vector<future<unsigned> > >();

        test_when_each_from_list_iterators<std::vector<future<unsigned> > >();
        test_when_each_from_list_iterators<std::list<future<unsigned> > >();
        test_when_each_from_list_iterators<std::deque<future<unsigned> > >();

        test_when_each_n_from_list_iterators<std::vector<future<unsigned> > >();
        test_when_each_n_from_list_iterators<std::list<future<unsigned> > >();
        test_when_each_n_from_list_iterators<std::deque<future<unsigned> > >();

        test_when_each_one_future();
        test_when_each_two_futures();
        test_when_each_three_futures();
        test_when_each_four_futures();
        test_when_each_five_futures();

        test_when_each_late_future();

        test_when_each_deferred_futures();
    }

    hpx::finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    // We force this test to use several threads by default.
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    return hpx::init(cmdline, argc, argv, cfg);
}
