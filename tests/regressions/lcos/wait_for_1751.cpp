//  Copyright 2015 (c) Dominic Marcello
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #1751:
// hpx::future::wait_for fails a simple test

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <chrono>
#include <iostream>

int hpx_main()
{
    auto overall_start_time = std::chrono::high_resolution_clock::now();

    while(true)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        // run for 3 seconds max
        std::chrono::duration<double> overall_dif = start_time - overall_start_time;
        if (overall_dif.count() > 3.0)
            break;

        auto f = hpx::async([](){});

        if (f.wait_for(std::chrono::seconds(1)) ==
            hpx::lcos::future_status::timeout)
        {
            auto now = std::chrono::high_resolution_clock::now();
            overall_dif = now - overall_start_time;
            std::chrono::duration<double> dif = now - start_time;

            HPX_TEST_LTE(1.0, dif.count());
            break;
        }
        else
        {
            f.get();
        }

        auto now = std::chrono::high_resolution_clock::now();
        overall_dif = now - overall_start_time;
        std::chrono::duration<double> dif = now - start_time;
        HPX_TEST_LT(dif.count(), 1.0);
    }

    return hpx::finalize();
}

int main(int argc, char *argv[])
{
    hpx::init(argc, argv);
    return hpx::util::report_errors();
}
