//  Copyright (C) 2012-2017 Hartmut Kaiser
//  (C) Copyright 2008-10 Anthony Williams
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CXX11_STD_ARRAY)
#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/assign/std/vector.hpp>

#include <array>
#include <string>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
int make_int_slowly()
{
    hpx::this_thread::sleep_for(std::chrono::milliseconds(100));
    return 42;
}

void test_wait_for_either_of_two_futures_list()
{
    std::array<hpx::future<int>, 2> futures;
    hpx::lcos::local::packaged_task<int()> pt1(make_int_slowly);
    futures[0] = pt1.get_future();
    hpx::lcos::local::packaged_task<int()> pt2(make_int_slowly);
    futures[1] = pt2.get_future();

    pt1();

    hpx::lcos::future<hpx::when_some_result<
            std::array<hpx::future<int>, 2>
        > > r = hpx::when_some(1u, futures);
    hpx::when_some_result<std::array<hpx::future<int>, 2> > raw = r.get();

    HPX_TEST_EQ(raw.indices.size(), 1u);
    HPX_TEST_EQ(raw.indices[0], 0u);

    std::array<hpx::future<int>, 2> t = std::move(raw.futures);

    HPX_TEST(!futures.front().valid());
    HPX_TEST(!futures.back().valid());

    HPX_TEST(t.front().is_ready());
    HPX_TEST_EQ(t.front().get(), 42);
}

///////////////////////////////////////////////////////////////////////////////
using boost::program_options::variables_map;
using boost::program_options::options_description;

using hpx::lcos::future;

int hpx_main(variables_map&)
{
    test_wait_for_either_of_two_futures_list();

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

#endif
