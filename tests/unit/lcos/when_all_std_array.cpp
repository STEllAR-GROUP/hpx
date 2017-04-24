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

void test_wait_for_all_from_array()
{
    unsigned const count = 10;
    std::array<hpx::future<int>, 10> futures;
    for (unsigned j = 0; j < count; ++j)
    {
        hpx::lcos::local::futures_factory<int()> task(make_int_slowly);
        futures[j] = task.get_future();
        task.apply();
    }

    hpx::lcos::future<std::array<hpx::future<int>, 10> > r =
        hpx::when_all(futures);

    std::array<hpx::future<int>, 10> result = r.get();

    for (const auto& f : futures)
        HPX_TEST(!f.valid());
    for (const auto& r : result)
        HPX_TEST(r.is_ready());
}

///////////////////////////////////////////////////////////////////////////////
using boost::program_options::variables_map;
using boost::program_options::options_description;

using hpx::lcos::future;

int hpx_main(variables_map&)
{
    test_wait_for_all_from_array();

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

