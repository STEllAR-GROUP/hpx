//  Copyright (C) 2012 Hartmut Kaiser
//  (C) Copyright 2008-10 Anthony Williams
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <utility>
#include <memory>
#include <string>
#include <vector>
#include <deque>
#include <list>

#include <boost/move/move.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/assign/std/vector.hpp>

///////////////////////////////////////////////////////////////////////////////
int make_int_slowly()
{
    hpx::this_thread::sleep_for(boost::chrono::milliseconds(100));
    return 42;
}

template <template <class...> class Container>
void test_wait_for_all_from_list()
{
    unsigned const count = 10;
    Container<hpx::lcos::future<int> > futures;
    for (unsigned j = 0; j < count; ++j)
    {
        hpx::lcos::local::futures_factory<int()> task(make_int_slowly);
        futures.push_back(task.get_future());
        task.apply();
    }

    hpx::lcos::future<Container<hpx::lcos::future<int> > > r =
        hpx::when_all(futures);

    Container<hpx::lcos::future<int> > result = r.get();

    HPX_TEST_EQ(futures.size(), result.size());
    for (const auto& f : futures)
        HPX_TEST(!f.valid());
    for (const auto& r : result)
        HPX_TEST(r.is_ready());
}

template <template <class...> class Container>
void test_wait_for_all_from_list_iterators()
{
    typedef Container<hpx::lcos::future<int> > container;
    unsigned const count = 10;

    container futures;
    for (unsigned j = 0; j < count; ++j)
    {
        hpx::lcos::local::futures_factory<int()> task(make_int_slowly);
        futures.push_back(task.get_future());
        task.apply();
    }

    hpx::lcos::future<container> r =
        hpx::when_all<typename container::iterator,
            container>(futures.begin(), futures.end());

    Container<hpx::lcos::future<int> > result = r.get();

    HPX_TEST_EQ(futures.size(), result.size());
    for (const auto& f : futures)
        HPX_TEST(!f.valid());
    for (const auto& r : result)
        HPX_TEST(r.is_ready());
}

void test_wait_for_all_two_futures()
{
    hpx::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1 = pt1.get_future();
    pt1.apply();
    hpx::lcos::local::futures_factory<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2 = pt2.get_future();
    pt2.apply();

    typedef hpx::util::tuple<
        hpx::lcos::future<int>
      , hpx::lcos::future<int> > result_type;
    hpx::lcos::future<result_type> r =
        hpx::when_all(f1, f2);

    result_type result = r.get();

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());

    HPX_TEST(hpx::util::get<0>(result).is_ready());
    HPX_TEST(hpx::util::get<1>(result).is_ready());
}

void test_wait_for_all_three_futures()
{
    hpx::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1 = pt1.get_future();
    pt1.apply();
    hpx::lcos::local::futures_factory<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2 = pt2.get_future();
    pt2.apply();
    hpx::lcos::local::futures_factory<int()> pt3(make_int_slowly);
    hpx::lcos::future<int> f3 = pt3.get_future();
    pt3.apply();

    typedef hpx::util::tuple<
        hpx::lcos::future<int>
      , hpx::lcos::future<int>
      , hpx::lcos::future<int> > result_type;
    hpx::lcos::future<result_type> r =
        hpx::when_all(f1, f2, f3);

    result_type result = r.get();

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());

    HPX_TEST(hpx::util::get<0>(result).is_ready());
    HPX_TEST(hpx::util::get<1>(result).is_ready());
    HPX_TEST(hpx::util::get<2>(result).is_ready());
}

void test_wait_for_all_four_futures()
{
    hpx::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1 = pt1.get_future();
    pt1.apply();
    hpx::lcos::local::futures_factory<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2 = pt2.get_future();
    pt2.apply();
    hpx::lcos::local::futures_factory<int()> pt3(make_int_slowly);
    hpx::lcos::future<int> f3 = pt3.get_future();
    pt3.apply();
    hpx::lcos::local::futures_factory<int()> pt4(make_int_slowly);
    hpx::lcos::future<int> f4 = pt4.get_future();
    pt4.apply();

    typedef hpx::util::tuple<
        hpx::lcos::future<int>
      , hpx::lcos::future<int>
      , hpx::lcos::future<int>
      , hpx::lcos::future<int> > result_type;
    hpx::lcos::future<result_type> r =
        hpx::when_all(f1, f2, f3, f4);

    result_type result = r.get();

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());
    HPX_TEST(!f4.valid());

    HPX_TEST(hpx::util::get<0>(result).is_ready());
    HPX_TEST(hpx::util::get<1>(result).is_ready());
    HPX_TEST(hpx::util::get<2>(result).is_ready());
    HPX_TEST(hpx::util::get<3>(result).is_ready());
}

void test_wait_for_all_five_futures()
{
    hpx::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1 = pt1.get_future();
    pt1.apply();
    hpx::lcos::local::futures_factory<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2 = pt2.get_future();
    pt2.apply();
    hpx::lcos::local::futures_factory<int()> pt3(make_int_slowly);
    hpx::lcos::future<int> f3 = pt3.get_future();
    pt3.apply();
    hpx::lcos::local::futures_factory<int()> pt4(make_int_slowly);
    hpx::lcos::future<int> f4 = pt4.get_future();
    pt4.apply();
    hpx::lcos::local::futures_factory<int()> pt5(make_int_slowly);
    hpx::lcos::future<int> f5 = pt5.get_future();
    pt5.apply();

    typedef hpx::util::tuple<
        hpx::lcos::future<int>
      , hpx::lcos::future<int>
      , hpx::lcos::future<int>
      , hpx::lcos::future<int>
      , hpx::lcos::future<int> > result_type;
    hpx::lcos::future<result_type> r =
        hpx::when_all(f1, f2, f3, f4, f5);

    result_type result = r.get();

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());
    HPX_TEST(!f3.valid());
    HPX_TEST(!f4.valid());
    HPX_TEST(!f5.valid());

    HPX_TEST(hpx::util::get<0>(result).is_ready());
    HPX_TEST(hpx::util::get<1>(result).is_ready());
    HPX_TEST(hpx::util::get<2>(result).is_ready());
    HPX_TEST(hpx::util::get<3>(result).is_ready());
    HPX_TEST(hpx::util::get<4>(result).is_ready());
}

void test_wait_for_all_late_futures()
{
    hpx::lcos::local::futures_factory<int()> pt1(make_int_slowly);
    hpx::lcos::future<int> f1 = pt1.get_future();
    pt1.apply();
    hpx::lcos::local::futures_factory<int()> pt2(make_int_slowly);
    hpx::lcos::future<int> f2 = pt2.get_future();

    typedef hpx::util::tuple<
        hpx::lcos::future<int>
      , hpx::lcos::future<int> > result_type;
    hpx::lcos::future<result_type> r =
        hpx::when_all(f1, f2);

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());

    pt2.apply();

    result_type result = r.get();

    HPX_TEST(hpx::util::get<0>(result).is_ready());
    HPX_TEST(hpx::util::get<1>(result).is_ready());
}

void test_wait_for_all_deferred_futures()
{
    hpx::lcos::future<int> f1 = hpx::async(hpx::launch::deferred, &make_int_slowly);
    hpx::lcos::future<int> f2 = hpx::async(hpx::launch::deferred, &make_int_slowly);

    typedef hpx::util::tuple<
        hpx::lcos::future<int>
      , hpx::lcos::future<int> > result_type;
    hpx::lcos::future<result_type> r =
        hpx::when_all(f1, f2);

    HPX_TEST(!f1.valid());
    HPX_TEST(!f2.valid());

    result_type result = r.get();

    HPX_TEST(hpx::util::get<0>(result).is_ready());
    HPX_TEST(hpx::util::get<1>(result).is_ready());
}

///////////////////////////////////////////////////////////////////////////////
using boost::program_options::variables_map;
using boost::program_options::options_description;

int hpx_main(variables_map&)
{
    {
        test_wait_for_all_from_list<std::vector>();
        test_wait_for_all_from_list<std::list>();
        test_wait_for_all_from_list<std::deque>();
        test_wait_for_all_from_list_iterators<std::vector>();
        test_wait_for_all_from_list_iterators<std::list>();
        test_wait_for_all_from_list_iterators<std::deque>();
        test_wait_for_all_two_futures();
        test_wait_for_all_three_futures();
        test_wait_for_all_four_futures();
        test_wait_for_all_five_futures();
        test_wait_for_all_late_futures();
        test_wait_for_all_deferred_futures();
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
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency());

    // Initialize and run HPX
    return hpx::init(cmdline, argc, argv, cfg);
}

