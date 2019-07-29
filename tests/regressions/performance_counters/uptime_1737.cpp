//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// Demonstrating #1737: Uptime problems

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/testing.hpp>

#include <chrono>

int hpx_main(int argc, char ** argv)
{
    using hpx::performance_counters::performance_counter;

    performance_counter uptime("/runtime/uptime");

    // verify that attempts to reset the uptime counter fail
    {
        bool exception_thrown = false;
        try {
            uptime.reset(hpx::launch::sync);
            HPX_TEST(false);
        }
        catch (hpx::exception const&) {
            exception_thrown = true;
        }
        HPX_TEST(exception_thrown);
    }

    {
        bool exception_thrown = false;
        try {
            uptime.get_value<double>(hpx::launch::sync, true);
            HPX_TEST(false);
        }
        catch (hpx::exception const&) {
            exception_thrown = true;
        }
        HPX_TEST(exception_thrown);
    }

    // make sure reported value is in seconds
    double start = uptime.get_value<double>(hpx::launch::sync);
    hpx::this_thread::sleep_for(std::chrono::seconds(1));
    double end = uptime.get_value<double>(hpx::launch::sync);

    HPX_TEST(end - start >= 1.0 && end - start < 1.1);

    // make sure start/stop return false
    HPX_TEST(!uptime.start(hpx::launch::sync));
    HPX_TEST(!uptime.stop(hpx::launch::sync));

    return hpx::finalize();
}

int main(int argc, char **argv)
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}
