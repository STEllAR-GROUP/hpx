//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>

// This application will just sit and wait for being terminated from the
// console window for the specified amount of time. This is useful for testing
// the heartbeat tool which connects and disconnects to a running application.
int hpx_main(boost::program_options::variables_map& vm)
{
    double const runfor = vm["runfor"].as<double>();

    hpx::cout << "Heartbeat Console, waiting for";
    if (runfor > 0)
        hpx::cout << " " << runfor << "[s].\n" << hpx::flush;
    else
        hpx::cout << "ever.\n" << hpx::flush;

    hpx::util::high_resolution_timer t;
    while (runfor <= 0 || t.elapsed() < runfor)
    {
        hpx::this_thread::suspend(1000);
        hpx::cout << "." << hpx::flush;
    }

    hpx::cout << "\n" << hpx::flush;
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // Configure application-specific options.
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    using boost::program_options::value;
    desc_commandline.add_options()
        ( "runfor", value<double>()->default_value(600.0),
          "time to wait before this application exits ([s], default: 600)")
        ;

    // we expect other localities to connect
    std::vector<std::string> cfg;
    cfg.push_back("hpx.expect_connecting_localities=1");

    return hpx::init(desc_commandline, argc, argv, cfg);
}
