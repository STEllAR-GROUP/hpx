//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define HPX_NO_VERSION_CHECK        // avoid linker errors

#include "worker_timed.hpp"

#include <hpx/util/high_resolution_timer.hpp>

#include <stdexcept>
#include <iostream>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::store;
using boost::program_options::command_line_parser;
using boost::program_options::notify;

using hpx::util::high_resolution_timer;

///////////////////////////////////////////////////////////////////////////////
boost::uint64_t tasks = 500000;
boost::uint64_t delay = 5;
bool header = true;

///////////////////////////////////////////////////////////////////////////////
void print_results(
    variables_map& vm
  , double sum_
  , double mean_
    )
{
    if (header)
        std::cout << "Tasks,Specified Task Length (microseconds),"
                     "Walltime (seconds),Average Task Length (microseconds)\n";

    std::string const tasks_str = boost::str(boost::format("%lu,") % tasks);
    std::string const delay_str = boost::str(boost::format("%lu,") % delay);

    std::cout << ( boost::format("%-21s %-21s %10.12g %10.12g\n")
                 % tasks_str % delay_str % sum_ % mean_);
}

///////////////////////////////////////////////////////////////////////////////
int app_main(
    variables_map& vm
    )
{
    if (vm.count("no-header"))
        header = false;

    if (0 == tasks)
        throw std::invalid_argument("count of 0 tasks specified\n");

    // Start the clock.
    high_resolution_timer t;

    for (boost::uint64_t i = 0; i < tasks; ++i)
    {
        worker_timed(delay);
    }

    double elapsed = t.elapsed();

    // Print out the results.
    print_results(vm, elapsed, (elapsed * 1e6) / double(tasks));

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(
    int argc
  , char* argv[]
    )
{
    ///////////////////////////////////////////////////////////////////////////
    // Parse command line.
    variables_map vm;

    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ( "help,h"
        , "print out program usage (this message)")

        ( "tasks"
        , value<boost::uint64_t>(&tasks)->default_value(100000)
        , "number of tasks to invoke")

        ( "delay"
        , value<boost::uint64_t>(&delay)->default_value(5)
        , "duration of delay in microseconds")
        ;

    store(command_line_parser(argc, argv).options(cmdline).run(), vm);

    notify(vm);

    // Print help screen.
    if (vm.count("help"))
    {
        std::cout << cmdline;
        return 0;
    }

    return app_main(vm);
}

