//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <string>
#include <vector>

#include <boost/assign/std/vector.hpp>

///////////////////////////////////////////////////////////////////////////////
hpx::util::tuple<> make_tuple0_slowly()
{
    hpx::this_thread::sleep_for(boost::chrono::milliseconds(100));
    return hpx::util::make_tuple();
}

void test_split_all0()
{
    hpx::lcos::local::futures_factory<hpx::util::tuple<>()> pt(
        make_tuple0_slowly);
    pt.apply();

    hpx::util::tuple<> result = hpx::lcos::split_all(pt.get_future());
}

///////////////////////////////////////////////////////////////////////////////
hpx::util::tuple<int> make_tuple1_slowly()
{
    hpx::this_thread::sleep_for(boost::chrono::milliseconds(100));
    return hpx::util::make_tuple(42);
}

void test_split_all1()
{
    hpx::lcos::local::futures_factory<hpx::util::tuple<int>()> pt(
        make_tuple1_slowly);
    pt.apply();

    hpx::util::tuple<hpx::future<int> > result =
        hpx::lcos::split_all(pt.get_future());

    HPX_TEST_EQ(hpx::util::get<0>(result).get(), 42);
}

///////////////////////////////////////////////////////////////////////////////
hpx::util::tuple<int, int> make_tuple2_slowly()
{
    hpx::this_thread::sleep_for(boost::chrono::milliseconds(100));
    return hpx::util::make_tuple(42, 43);
}

void test_split_all2()
{
    hpx::lcos::local::futures_factory<hpx::util::tuple<int, int>()> pt(
        make_tuple2_slowly);
    pt.apply();

    hpx::util::tuple<hpx::future<int>, hpx::future<int> > result =
        hpx::lcos::split_all(pt.get_future());

    HPX_TEST_EQ(hpx::util::get<0>(result).get(), 42);
    HPX_TEST_EQ(hpx::util::get<1>(result).get(), 43);
}

///////////////////////////////////////////////////////////////////////////////
hpx::util::tuple<int, int, int> make_tuple3_slowly()
{
    hpx::this_thread::sleep_for(boost::chrono::milliseconds(100));
    return hpx::util::make_tuple(42, 43, 44);
}

void test_split_all3()
{
    hpx::lcos::local::futures_factory<hpx::util::tuple<int, int, int>()> pt(
        make_tuple3_slowly);
    pt.apply();

    hpx::util::tuple<hpx::future<int>, hpx::future<int>, hpx::future<int> >
        result = hpx::lcos::split_all(pt.get_future());

    HPX_TEST_EQ(hpx::util::get<0>(result).get(), 42);
    HPX_TEST_EQ(hpx::util::get<1>(result).get(), 43);
    HPX_TEST_EQ(hpx::util::get<2>(result).get(), 44);
}

///////////////////////////////////////////////////////////////////////////////
std::pair<int, int> make_pair_slowly()
{
    hpx::this_thread::sleep_for(boost::chrono::milliseconds(100));
    return std::make_pair(42, 43);
}

void test_split_allpair()
{
    hpx::lcos::local::futures_factory<std::pair<int, int>()> pt(
        make_pair_slowly);
    pt.apply();

    std::pair<hpx::future<int>, hpx::future<int> > result =
        hpx::lcos::split_all(pt.get_future());

    HPX_TEST_EQ(result.first.get(), 42);
    HPX_TEST_EQ(result.second.get(), 43);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(int argc, char* argv[])
{
    test_split_all0();
    test_split_all1();
    test_split_all2();
    test_split_all3();

    test_split_allpair();

    hpx::finalize();
    return hpx::util::report_errors();
}

int main(int argc, char* argv[])
{
    // We force this test to use several threads by default.
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.os_threads=" +
        std::to_string(hpx::threads::hardware_concurrency());

    // Initialize and run HPX
    return hpx::init(argc, argv, cfg);
}

