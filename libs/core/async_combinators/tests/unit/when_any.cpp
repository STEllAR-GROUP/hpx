//  Copyright (C) 2012 Hartmut Kaiser
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

void test_wait_for_either_of_two_futures_1()
{
    hpx::packaged_task<int()> pt1(make_int_slowly);
    hpx::future<int> f1(pt1.get_future());
    hpx::packaged_task<int()> pt2(make_int_slowly);
    hpx::future<int> f2(pt2.get_future());

    pt1();

    hpx::future<
        hpx::when_any_result<hpx::tuple<hpx::future<int>, hpx::future<int>>>>
        r = hpx::when_any(f1, f2);
    hpx::tuple<hpx::future<int>, hpx::future<int>> t = r.get().futures;

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());

    HPX_TEST(hpx::get<0>(t).is_ready());
    HPX_TEST_EQ(hpx::get<0>(t).get(), 42);
}

void test_wait_for_either_of_two_futures_2()
{
    hpx::packaged_task<int()> pt(make_int_slowly);
    hpx::future<int> f1(pt.get_future());
    hpx::packaged_task<int()> pt2(make_int_slowly);
    hpx::future<int> f2(pt2.get_future());

    pt2();

    hpx::future<
        hpx::when_any_result<hpx::tuple<hpx::future<int>, hpx::future<int>>>>
        r = hpx::when_any(f1, f2);
    hpx::tuple<hpx::future<int>, hpx::future<int>> t = r.get().futures;

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());

    HPX_TEST(hpx::get<1>(t).is_ready());
    HPX_TEST_EQ(hpx::get<1>(t).get(), 42);
}

template <class Container>
void test_wait_for_either_of_two_futures_list_1()
{
    Container futures;
    hpx::packaged_task<int()> pt1(make_int_slowly);
    futures.push_back(pt1.get_future());
    hpx::packaged_task<int()> pt2(make_int_slowly);
    futures.push_back(pt2.get_future());

    pt1();

    hpx::future<hpx::when_any_result<Container>> r = hpx::when_any(futures);
    hpx::when_any_result<Container> raw = r.get();

    HPX_TEST_EQ(raw.index, 0u);

    Container t = std::move(raw.futures);

    HPX_TEST(!futures.front().valid());
    HPX_TEST(!futures.back().valid());

    HPX_TEST(t.front().is_ready());
    HPX_TEST_EQ(t.front().get(), 42);
}

template <class Container>
void test_wait_for_either_of_two_futures_list_2()
{
    Container futures;
    hpx::packaged_task<int()> pt1(make_int_slowly);
    futures.push_back(pt1.get_future());
    hpx::packaged_task<int()> pt2(make_int_slowly);
    futures.push_back(pt2.get_future());

    pt2();

    hpx::future<hpx::when_any_result<Container>> r = hpx::when_any(futures);
    hpx::when_any_result<Container> raw = r.get();

    HPX_TEST_EQ(raw.index, 1u);

    Container t = std::move(raw.futures);

    HPX_TEST(!futures.front().valid());
    HPX_TEST(!futures.back().valid());

    HPX_TEST(t.back().is_ready());
    HPX_TEST_EQ(t.back().get(), 42);
}

void test_wait_for_either_of_three_futures_1()
{
    hpx::packaged_task<int()> pt1(make_int_slowly);
    hpx::future<int> f1(pt1.get_future());
    hpx::packaged_task<int()> pt2(make_int_slowly);
    hpx::future<int> f2(pt2.get_future());
    hpx::packaged_task<int()> pt3(make_int_slowly);
    hpx::future<int> f3(pt3.get_future());

    pt1();

    hpx::future<hpx::when_any_result<
        hpx::tuple<hpx::future<int>, hpx::future<int>, hpx::future<int>>>>
        r = hpx::when_any(f1, f2, f3);
    hpx::tuple<hpx::future<int>, hpx::future<int>, hpx::future<int>> t =
        r.get().futures;

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());

    HPX_TEST(hpx::get<0>(t).is_ready());
    HPX_TEST_EQ(hpx::get<0>(t).get(), 42);
}

void test_wait_for_either_of_three_futures_2()
{
    hpx::packaged_task<int()> pt1(make_int_slowly);
    hpx::future<int> f1(pt1.get_future());
    hpx::packaged_task<int()> pt2(make_int_slowly);
    hpx::future<int> f2(pt2.get_future());
    hpx::packaged_task<int()> pt3(make_int_slowly);
    hpx::future<int> f3(pt3.get_future());

    pt2();

    hpx::future<hpx::when_any_result<
        hpx::tuple<hpx::future<int>, hpx::future<int>, hpx::future<int>>>>
        r = hpx::when_any(f1, f2, f3);
    hpx::tuple<hpx::future<int>, hpx::future<int>, hpx::future<int>> t =
        r.get().futures;

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());

    HPX_TEST(hpx::get<1>(t).is_ready());
    HPX_TEST_EQ(hpx::get<1>(t).get(), 42);
}

void test_wait_for_either_of_three_futures_3()
{
    hpx::packaged_task<int()> pt1(make_int_slowly);
    hpx::future<int> f1(pt1.get_future());
    hpx::packaged_task<int()> pt2(make_int_slowly);
    hpx::future<int> f2(pt2.get_future());
    hpx::packaged_task<int()> pt3(make_int_slowly);
    hpx::future<int> f3(pt3.get_future());

    pt3();

    hpx::future<hpx::when_any_result<
        hpx::tuple<hpx::future<int>, hpx::future<int>, hpx::future<int>>>>
        r = hpx::when_any(f1, f2, f3);
    hpx::tuple<hpx::future<int>, hpx::future<int>, hpx::future<int>> t =
        r.get().futures;

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());

    HPX_TEST(hpx::get<2>(t).is_ready());
    HPX_TEST_EQ(hpx::get<2>(t).get(), 42);
}

void test_wait_for_either_of_four_futures_1()
{
    hpx::packaged_task<int()> pt1(make_int_slowly);
    hpx::future<int> f1(pt1.get_future());
    hpx::packaged_task<int()> pt2(make_int_slowly);
    hpx::future<int> f2(pt2.get_future());
    hpx::packaged_task<int()> pt3(make_int_slowly);
    hpx::future<int> f3(pt3.get_future());
    hpx::packaged_task<int()> pt4(make_int_slowly);
    hpx::future<int> f4(pt4.get_future());

    pt1();

    hpx::future<hpx::when_any_result<hpx::tuple<hpx::future<int>,
        hpx::future<int>, hpx::future<int>, hpx::future<int>>>>
        r = hpx::when_any(f1, f2, f3, f4);
    hpx::tuple<hpx::future<int>, hpx::future<int>, hpx::future<int>,
        hpx::future<int>>
        t = r.get().futures;

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());
    HPX_TEST(!f4.valid());

    HPX_TEST(hpx::get<0>(t).is_ready());
    HPX_TEST_EQ(hpx::get<0>(t).get(), 42);
}

void test_wait_for_either_of_four_futures_2()
{
    hpx::packaged_task<int()> pt1(make_int_slowly);
    hpx::future<int> f1(pt1.get_future());
    hpx::packaged_task<int()> pt2(make_int_slowly);
    hpx::future<int> f2(pt2.get_future());
    hpx::packaged_task<int()> pt3(make_int_slowly);
    hpx::future<int> f3(pt3.get_future());
    hpx::packaged_task<int()> pt4(make_int_slowly);
    hpx::future<int> f4(pt4.get_future());

    pt2();

    hpx::future<hpx::when_any_result<hpx::tuple<hpx::future<int>,
        hpx::future<int>, hpx::future<int>, hpx::future<int>>>>
        r = hpx::when_any(f1, f2, f3, f4);
    hpx::tuple<hpx::future<int>, hpx::future<int>, hpx::future<int>,
        hpx::future<int>>
        t = r.get().futures;

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());
    HPX_TEST(!f4.valid());

    HPX_TEST(hpx::get<1>(t).is_ready());
    HPX_TEST_EQ(hpx::get<1>(t).get(), 42);
}

void test_wait_for_either_of_four_futures_3()
{
    hpx::packaged_task<int()> pt1(make_int_slowly);
    hpx::future<int> f1(pt1.get_future());
    hpx::packaged_task<int()> pt2(make_int_slowly);
    hpx::future<int> f2(pt2.get_future());
    hpx::packaged_task<int()> pt3(make_int_slowly);
    hpx::future<int> f3(pt3.get_future());
    hpx::packaged_task<int()> pt4(make_int_slowly);
    hpx::future<int> f4(pt4.get_future());

    pt3();

    hpx::future<hpx::when_any_result<hpx::tuple<hpx::future<int>,
        hpx::future<int>, hpx::future<int>, hpx::future<int>>>>
        r = hpx::when_any(f1, f2, f3, f4);
    hpx::tuple<hpx::future<int>, hpx::future<int>, hpx::future<int>,
        hpx::future<int>>
        t = r.get().futures;

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());
    HPX_TEST(!f4.valid());

    HPX_TEST(hpx::get<2>(t).is_ready());
    HPX_TEST_EQ(hpx::get<2>(t).get(), 42);
}

void test_wait_for_either_of_four_futures_4()
{
    hpx::packaged_task<int()> pt1(make_int_slowly);
    hpx::future<int> f1(pt1.get_future());
    hpx::packaged_task<int()> pt2(make_int_slowly);
    hpx::future<int> f2(pt2.get_future());
    hpx::packaged_task<int()> pt3(make_int_slowly);
    hpx::future<int> f3(pt3.get_future());
    hpx::packaged_task<int()> pt4(make_int_slowly);
    hpx::future<int> f4(pt4.get_future());

    pt4();

    hpx::future<hpx::when_any_result<hpx::tuple<hpx::future<int>,
        hpx::future<int>, hpx::future<int>, hpx::future<int>>>>
        r = hpx::when_any(f1, f2, f3, f4);
    hpx::tuple<hpx::future<int>, hpx::future<int>, hpx::future<int>,
        hpx::future<int>>
        t = r.get().futures;

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());
    HPX_TEST(!f4.valid());

    HPX_TEST(hpx::get<3>(t).is_ready());
    HPX_TEST_EQ(hpx::get<3>(t).get(), 42);
}

template <class Container>
void test_wait_for_either_of_five_futures_1_from_list()
{
    Container futures;

    hpx::packaged_task<int()> pt1(make_int_slowly);
    hpx::future<int> f1(pt1.get_future());
    futures.push_back(std::move(f1));
    hpx::packaged_task<int()> pt2(make_int_slowly);
    hpx::future<int> f2(pt2.get_future());
    futures.push_back(std::move(f2));
    hpx::packaged_task<int()> pt3(make_int_slowly);
    hpx::future<int> f3(pt3.get_future());
    futures.push_back(std::move(f3));
    hpx::packaged_task<int()> pt4(make_int_slowly);
    hpx::future<int> f4(pt4.get_future());
    futures.push_back(std::move(f4));
    hpx::packaged_task<int()> pt5(make_int_slowly);
    hpx::future<int> f5(pt5.get_future());
    futures.push_back(std::move(f5));

    pt1();

    hpx::future<hpx::when_any_result<Container>> r = hpx::when_any(futures);
    hpx::when_any_result<Container> raw = r.get();

    HPX_TEST_EQ(raw.index, 0u);

    Container t = std::move(raw.futures);

    HPX_TEST(!f1.valid());    // NOLINT
    HPX_TEST(!f2.valid());    // NOLINT
    HPX_TEST(!f3.valid());    // NOLINT
    HPX_TEST(!f4.valid());    // NOLINT
    HPX_TEST(!f5.valid());    // NOLINT

    HPX_TEST(t.front().is_ready());
    HPX_TEST_EQ(t.front().get(), 42);
}

template <class Container>
void test_wait_for_either_of_five_futures_1_from_list_iterators()
{
    typedef typename Container::iterator iterator;

    Container futures;

    hpx::packaged_task<int()> pt1(make_int_slowly);
    hpx::future<int> f1(pt1.get_future());
    futures.push_back(std::move(f1));
    hpx::packaged_task<int()> pt2(make_int_slowly);
    hpx::future<int> f2(pt2.get_future());
    futures.push_back(std::move(f2));
    hpx::packaged_task<int()> pt3(make_int_slowly);
    hpx::future<int> f3(pt3.get_future());
    futures.push_back(std::move(f3));
    hpx::packaged_task<int()> pt4(make_int_slowly);
    hpx::future<int> f4(pt4.get_future());
    futures.push_back(std::move(f4));
    hpx::packaged_task<int()> pt5(make_int_slowly);
    hpx::future<int> f5(pt5.get_future());
    futures.push_back(std::move(f5));

    pt1();

    hpx::future<hpx::when_any_result<Container>> r =
        hpx::when_any<iterator, Container>(futures.begin(), futures.end());
    hpx::when_any_result<Container> raw = r.get();

    HPX_TEST_EQ(raw.index, 0u);

    Container t = std::move(raw.futures);

    HPX_TEST(!f1.valid());    // NOLINT
    HPX_TEST(!f2.valid());    // NOLINT
    HPX_TEST(!f3.valid());    // NOLINT
    HPX_TEST(!f4.valid());    // NOLINT
    HPX_TEST(!f5.valid());    // NOLINT

    HPX_TEST(t.front().is_ready());
    HPX_TEST_EQ(t.front().get(), 42);
}

void test_wait_for_either_of_five_futures_1()
{
    hpx::packaged_task<int()> pt1(make_int_slowly);
    hpx::future<int> f1(pt1.get_future());
    hpx::packaged_task<int()> pt2(make_int_slowly);
    hpx::future<int> f2(pt2.get_future());
    hpx::packaged_task<int()> pt3(make_int_slowly);
    hpx::future<int> f3(pt3.get_future());
    hpx::packaged_task<int()> pt4(make_int_slowly);
    hpx::future<int> f4(pt4.get_future());
    hpx::packaged_task<int()> pt5(make_int_slowly);
    hpx::future<int> f5(pt5.get_future());

    pt1();

    hpx::future<
        hpx::when_any_result<hpx::tuple<hpx::future<int>, hpx::future<int>,
            hpx::future<int>, hpx::future<int>, hpx::future<int>>>>
        r = hpx::when_any(f1, f2, f3, f4, f5);
    hpx::tuple<hpx::future<int>, hpx::future<int>, hpx::future<int>,
        hpx::future<int>, hpx::future<int>>
        t = r.get().futures;

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());
    HPX_TEST(!f4.valid());
    HPX_TEST(!f5.valid());

    HPX_TEST(hpx::get<0>(t).is_ready());
    HPX_TEST_EQ(hpx::get<0>(t).get(), 42);
}

void test_wait_for_either_of_five_futures_2()
{
    hpx::packaged_task<int()> pt1(make_int_slowly);
    hpx::future<int> f1(pt1.get_future());
    hpx::packaged_task<int()> pt2(make_int_slowly);
    hpx::future<int> f2(pt2.get_future());
    hpx::packaged_task<int()> pt3(make_int_slowly);
    hpx::future<int> f3(pt3.get_future());
    hpx::packaged_task<int()> pt4(make_int_slowly);
    hpx::future<int> f4(pt4.get_future());
    hpx::packaged_task<int()> pt5(make_int_slowly);
    hpx::future<int> f5(pt5.get_future());

    pt2();

    hpx::future<
        hpx::when_any_result<hpx::tuple<hpx::future<int>, hpx::future<int>,
            hpx::future<int>, hpx::future<int>, hpx::future<int>>>>
        r = hpx::when_any(f1, f2, f3, f4, f5);
    hpx::tuple<hpx::future<int>, hpx::future<int>, hpx::future<int>,
        hpx::future<int>, hpx::future<int>>
        t = r.get().futures;

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());
    HPX_TEST(!f4.valid());
    HPX_TEST(!f5.valid());

    HPX_TEST(hpx::get<1>(t).is_ready());
    HPX_TEST_EQ(hpx::get<1>(t).get(), 42);
}

void test_wait_for_either_of_five_futures_3()
{
    hpx::packaged_task<int()> pt1(make_int_slowly);
    hpx::future<int> f1(pt1.get_future());
    hpx::packaged_task<int()> pt2(make_int_slowly);
    hpx::future<int> f2(pt2.get_future());
    hpx::packaged_task<int()> pt3(make_int_slowly);
    hpx::future<int> f3(pt3.get_future());
    hpx::packaged_task<int()> pt4(make_int_slowly);
    hpx::future<int> f4(pt4.get_future());
    hpx::packaged_task<int()> pt5(make_int_slowly);
    hpx::future<int> f5(pt5.get_future());

    pt3();

    hpx::future<
        hpx::when_any_result<hpx::tuple<hpx::future<int>, hpx::future<int>,
            hpx::future<int>, hpx::future<int>, hpx::future<int>>>>
        r = hpx::when_any(f1, f2, f3, f4, f5);
    hpx::tuple<hpx::future<int>, hpx::future<int>, hpx::future<int>,
        hpx::future<int>, hpx::future<int>>
        t = r.get().futures;

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());
    HPX_TEST(!f4.valid());
    HPX_TEST(!f5.valid());

    HPX_TEST(hpx::get<2>(t).is_ready());
    HPX_TEST_EQ(hpx::get<2>(t).get(), 42);
}

void test_wait_for_either_of_five_futures_4()
{
    hpx::packaged_task<int()> pt1(make_int_slowly);
    hpx::future<int> f1(pt1.get_future());
    hpx::packaged_task<int()> pt2(make_int_slowly);
    hpx::future<int> f2(pt2.get_future());
    hpx::packaged_task<int()> pt3(make_int_slowly);
    hpx::future<int> f3(pt3.get_future());
    hpx::packaged_task<int()> pt4(make_int_slowly);
    hpx::future<int> f4(pt4.get_future());
    hpx::packaged_task<int()> pt5(make_int_slowly);
    hpx::future<int> f5(pt5.get_future());

    pt4();

    hpx::future<
        hpx::when_any_result<hpx::tuple<hpx::future<int>, hpx::future<int>,
            hpx::future<int>, hpx::future<int>, hpx::future<int>>>>
        r = hpx::when_any(f1, f2, f3, f4, f5);
    hpx::tuple<hpx::future<int>, hpx::future<int>, hpx::future<int>,
        hpx::future<int>, hpx::future<int>>
        t = r.get().futures;

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());
    HPX_TEST(!f4.valid());
    HPX_TEST(!f5.valid());

    HPX_TEST(hpx::get<3>(t).is_ready());
    HPX_TEST_EQ(hpx::get<3>(t).get(), 42);
}

void test_wait_for_either_of_five_futures_5()
{
    hpx::packaged_task<int()> pt1(make_int_slowly);
    hpx::future<int> f1(pt1.get_future());
    hpx::packaged_task<int()> pt2(make_int_slowly);
    hpx::future<int> f2(pt2.get_future());
    hpx::packaged_task<int()> pt3(make_int_slowly);
    hpx::future<int> f3(pt3.get_future());
    hpx::packaged_task<int()> pt4(make_int_slowly);
    hpx::future<int> f4(pt4.get_future());
    hpx::packaged_task<int()> pt5(make_int_slowly);
    hpx::future<int> f5(pt5.get_future());

    pt5();

    hpx::future<
        hpx::when_any_result<hpx::tuple<hpx::future<int>, hpx::future<int>,
            hpx::future<int>, hpx::future<int>, hpx::future<int>>>>
        r = hpx::when_any(f1, f2, f3, f4, f5);
    hpx::tuple<hpx::future<int>, hpx::future<int>, hpx::future<int>,
        hpx::future<int>, hpx::future<int>>
        t = r.get().futures;

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());
    HPX_TEST(!f4.valid());
    HPX_TEST(!f5.valid());

    HPX_TEST(hpx::get<4>(t).is_ready());
    HPX_TEST_EQ(hpx::get<4>(t).get(), 42);
}

///////////////////////////////////////////////////////////////////////////////
// void test_wait_for_either_invokes_callbacks()
// {
//     callback_called = 0;
//     hpx::packaged_task<int()> pt1(make_int_slowly);
//     hpx::future<int> fi = pt1.get_future();
//     hpx::packaged_task<int()> pt2(make_int_slowly);
//     hpx::future<int> fi2 = pt2.get_future();
//     pt1.set_wait_callback(wait_callback_for_task);
//
//     hpx::thread t(std::move(pt));
//
//     boost::wait_for_any(fi, fi2);
//     HPX_TEST_EQ(callback_called, 1U);
//     HPX_TEST_EQ(fi.get(), 42);
// }

// void test_wait_for_any_from_range()
// {
//     unsigned const count = 10;
//     for(unsigned i = 0; i < count; ++i)
//     {
//         hpx::packaged_task<int()> tasks[count];
//         hpx::future<int> futures[count];
//         for(unsigned j = 0; j < count; ++j)
//         {
//             tasks[j] =
//               std::move(hpx::packaged_task<int()>(make_int_slowly));
//             futures[j] = tasks[j].get_future();
//         }
//         hpx::thread t(std::move(tasks[i]));
//
//         hpx::wait_any(futures, futures);
//
//         hpx::future<int>* const future =
//               boost::wait_for_any(futures, futures+count);
//
//         HPX_TEST_EQ(future, (futures + i));
//         for(unsigned j = 0; j < count; ++j)
//         {
//             if (j != i)
//             {
//                 HPX_TEST(!futures[j].is_ready());
//             }
//             else
//             {
//                 HPX_TEST(futures[j].is_ready());
//             }
//         }
//         HPX_TEST_EQ(futures[i].get(), 42);
//     }
// }

void test_wait_for_either_of_two_late_futures()
{
    hpx::packaged_task<int()> pt1(make_int_slowly);
    hpx::future<int> f1(pt1.get_future());
    hpx::packaged_task<int()> pt2(make_int_slowly);
    hpx::future<int> f2(pt2.get_future());

    hpx::future<
        hpx::when_any_result<hpx::tuple<hpx::future<int>, hpx::future<int>>>>
        r = hpx::when_any(f1, f2);

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());

    pt2();
    pt1();

    hpx::tuple<hpx::future<int>, hpx::future<int>> t = r.get().futures;

    HPX_TEST(hpx::get<1>(t).is_ready());
    HPX_TEST_EQ(hpx::get<1>(t).get(), 42);
}

void test_wait_for_either_of_two_deferred_futures()
{
    hpx::future<int> f1 = hpx::async(hpx::launch::deferred, &make_int_slowly);
    hpx::future<int> f2 = hpx::async(hpx::launch::deferred, &make_int_slowly);

    hpx::future<
        hpx::when_any_result<hpx::tuple<hpx::future<int>, hpx::future<int>>>>
        r = hpx::when_any(f1, f2);

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());

    hpx::tuple<hpx::future<int>, hpx::future<int>> t = r.get().futures;

    HPX_TEST(hpx::get<0>(t).is_ready());
    HPX_TEST_EQ(hpx::get<0>(t).get(), 42);
}

///////////////////////////////////////////////////////////////////////////////
using hpx::program_options::options_description;
using hpx::program_options::variables_map;

using hpx::future;

int hpx_main(variables_map&)
{
    {
        test_wait_for_either_of_two_futures_1();
        test_wait_for_either_of_two_futures_2();
        test_wait_for_either_of_two_futures_list_1<std::vector<future<int>>>();
        test_wait_for_either_of_two_futures_list_1<std::list<future<int>>>();
        test_wait_for_either_of_two_futures_list_1<std::deque<future<int>>>();
        test_wait_for_either_of_two_futures_list_2<std::vector<future<int>>>();
        test_wait_for_either_of_two_futures_list_2<std::list<future<int>>>();
        test_wait_for_either_of_two_futures_list_2<std::deque<future<int>>>();
        test_wait_for_either_of_three_futures_1();
        test_wait_for_either_of_three_futures_2();
        test_wait_for_either_of_three_futures_3();
        test_wait_for_either_of_four_futures_1();
        test_wait_for_either_of_four_futures_2();
        test_wait_for_either_of_four_futures_3();
        test_wait_for_either_of_four_futures_4();
        test_wait_for_either_of_five_futures_1_from_list<
            std::vector<future<int>>>();
        test_wait_for_either_of_five_futures_1_from_list<
            std::list<future<int>>>();
        test_wait_for_either_of_five_futures_1_from_list<
            std::deque<future<int>>>();
        test_wait_for_either_of_five_futures_1_from_list_iterators<
            std::vector<future<int>>>();
        test_wait_for_either_of_five_futures_1_from_list_iterators<
            std::list<future<int>>>();
        test_wait_for_either_of_five_futures_1_from_list_iterators<
            std::deque<future<int>>>();
        test_wait_for_either_of_five_futures_1();
        test_wait_for_either_of_five_futures_2();
        test_wait_for_either_of_five_futures_3();
        test_wait_for_either_of_five_futures_4();
        test_wait_for_either_of_five_futures_5();
        //         test_wait_for_either_invokes_callbacks();
        //         test_wait_for_any_from_range();
        test_wait_for_either_of_two_late_futures();
        test_wait_for_either_of_two_deferred_futures();
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
