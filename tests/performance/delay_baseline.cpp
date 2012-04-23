//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

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

///////////////////////////////////////////////////////////////////////////////
void worker()
{
    double volatile d = 0.;
    for (boost::uint64_t i = 0; i < delay; ++i)
        d += 1. / (2. * i + 1.);
}

///////////////////////////////////////////////////////////////////////////////
void print_results(
    double median_
  , double mean_
  , double sum_
    )
{
    std::string const delay_str = boost::str(boost::format("%lu,") % delay);

    std::cout << ( boost::format("%-21s %10.10s, %10.10s, %10.10s\n")
                 % delay_str % median_ % mean_ % sum_);
}

///////////////////////////////////////////////////////////////////////////////
int app_main(
    variables_map&
    )
{
    if (0 == tasks)
        throw std::invalid_argument("error: count of 0 tasks specified\n");

    accumulator_set<double, stats<median_tag, mean_tag, sum_tag> > results;
  
    for (boost::uint64_t i = 0; i < tasks; ++i)
    {
        // Start the clock.
        high_resolution_timer t;

        worker(); 

        results(t.elapsed());
    }

    // Print out the results.
    print_results(median(results), mean(results), sum(results));

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
        , value<boost::uint64_t>(&tasks)->default_value(64)
        , "number of tasks to invoke")

        ( "delay"
        , value<boost::uint64_t>(&delay)->default_value(0)
        , "number of iterations in the delay loop")
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

