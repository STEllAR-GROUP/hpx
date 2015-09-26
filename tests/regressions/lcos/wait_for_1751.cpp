//  Copyright 2015 (c) Dominic Marcello
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #1751:
// hpx::future::wait_for fails a simple test

#include <hpx/hpx.hpp>
#include <boost/chrono.hpp>

int hpx_main()
{
    auto overall_start_time = boost::chrono::high_resolution_clock::now();

    while(true)
    {
        auto start_time = boost::chrono::high_resolution_clock::now();

        // run for 3 seconds max
        boost::chrono::duration<double> overall_dif = start_time - overall_start_time;
        if (overall_dif.count() > 3.0)
            break;

        auto f = hpx::async([](){});

        if (f.wait_for(boost::posix_time::seconds(1)) ==
            hpx::lcos::future_status::timeout)
        {
            auto now = boost::chrono::high_resolution_clock::now();
            overall_dif = now - overall_start_time;
            boost::chrono::duration<double> dif = now - start_time;

            std::cout << "Future timed out after "
                      << dif.count() << " (" << overall_dif << ") seconds.\n";
            break;
        }
        else
        {
            f.get();
        }

        auto now = boost::chrono::high_resolution_clock::now();
        overall_dif = now - overall_start_time;
        boost::chrono::duration<double> dif = now - start_time;
        std::cout << "Future took "
                  << dif.count() << " (" << overall_dif << ") seconds.\n";
    }

    return hpx::finalize();
}

int main(int argc, char *argv[])
{
    return hpx::init(argc, argv);
}
