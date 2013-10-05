//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "worker.hpp"

#include <hpx/util/high_resolution_timer.hpp>

#include <stdexcept>
#include <iostream>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/sum.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::store;
using boost::program_options::command_line_parser;
using boost::program_options::notify;

using boost::accumulators::accumulator_set;
using boost::accumulators::stats;
using boost::accumulators::median;
using boost::accumulators::mean;
using boost::accumulators::sum;

typedef boost::accumulators::tag::median median_tag;
typedef boost::accumulators::tag::mean mean_tag;
typedef boost::accumulators::tag::sum sum_tag;

using hpx::util::high_resolution_timer;

///////////////////////////////////////////////////////////////////////////////
boost::uint64_t tasks = 64;
boost::uint64_t delay = 0;
bool header = true;

///////////////////////////////////////////////////////////////////////////////
void print_results(
    variables_map& vm
  , double median_
  , double mean_
  , double sum_
    )
{
    if (header)
        std::cout << "Tasks,Delay (iterations),Total Walltime (seconds),"
                     "Median Walltime (seconds),Average Walltime (seconds),\n";

    std::string const tasks_str = boost::str(boost::format("%lu,") % tasks);
    std::string const delay_str = boost::str(boost::format("%lu,") % delay);

    std::cout << ( boost::format("%-21s %-21s %10.12s, %10.12s, %10.12s\n")
                 % tasks_str % delay_str % sum_ % median_ % mean_);
}

///////////////////////////////////////////////////////////////////////////////
int app_main(
    variables_map& vm
    )
{
    if (0 == tasks)
        throw std::invalid_argument("error: count of 0 tasks specified\n");

    accumulator_set<double, stats<median_tag, mean_tag, sum_tag> > results;

    volatile double d = 0.0;

    for (boost::uint64_t i = 0; i < tasks; ++i)
    {
        // Start the clock.
        high_resolution_timer t;

        worker(delay, &d);

        results(t.elapsed());
    }

    // Print out the results.
    print_results(vm, median(results), mean(results), sum(results));

    return static_cast<int>(d);
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
        , value<boost::uint64_t>(&tasks)->default_value(64)
        , "number of tasks to invoke")

        ( "delay"
        , value<boost::uint64_t>(&delay)->default_value(0)
        , "number of iterations in the delay loop")

        ( "no-header"
        , "do not print out the csv header row")
        ;

    store(command_line_parser(argc, argv).options(cmdline).run(), vm);

    notify(vm);

    // Print help screen.
    if (vm.count("help"))
    {
        std::cout << cmdline;
        return 0;
    }

    if (vm.count("no-header"))
        header = false;

    return app_main(vm);
}

