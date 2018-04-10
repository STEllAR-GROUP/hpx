//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define HPX_NO_VERSION_CHECK        // avoid linker errors

#include <hpx/config.hpp>

#include "worker_timed.hpp"

#include <hpx/util/format.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <chrono>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <string>

#include <boost/program_options.hpp>

char const* benchmark_name = "Delay Baseline";

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::store;
using boost::program_options::command_line_parser;
using boost::program_options::notify;

using hpx::util::high_resolution_timer;

using std::cout;

///////////////////////////////////////////////////////////////////////////////
std::uint64_t tasks = 500000;
std::uint64_t delay = 5;
bool header = true;

///////////////////////////////////////////////////////////////////////////////
std::string format_build_date(std::string timestamp)
{
    std::chrono::time_point<std::chrono::system_clock> now =
        std::chrono::system_clock::now();

    std::time_t current_time = std::chrono::system_clock::to_time_t(now);

    std::string ts = std::ctime(&current_time);
    ts.resize(ts.size()-1);     // remove trailing '\n'
    return ts;
}

///////////////////////////////////////////////////////////////////////////////
void print_results(
    variables_map& vm
  , double sum_
  , double mean_
    )
{
    if (header)
    {
        cout << "# BENCHMARK: " << benchmark_name << "\n";

        cout << "# VERSION: " << HPX_HAVE_GIT_COMMIT << " "
                 << format_build_date(__DATE__) << "\n"
             << "#\n";

        // Note that if we change the number of fields above, we have to
        // change the constant that we add when printing out the field # for
        // performance counters below (e.g. the last_index part).
        cout <<
                "## 0:DELAY:Delay [micro-seconds] - Independent Variable\n"
                "## 1:TASKS:# of Tasks - Independent Variable\n"
                "## 2:WTIME_THR:Total Walltime/Thread [micro-seconds]\n"
                ;
    }

    std::string const tasks_str = hpx::util::format("{},", tasks);
    std::string const delay_str = hpx::util::format("{},", delay);

    hpx::util::format_to(cout, "{} {} {:.14g}\n",
        delay, tasks, mean_);
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

    for (std::uint64_t i = 0; i < tasks; ++i)
    {
        worker_timed(delay * 1000);
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
        , value<std::uint64_t>(&tasks)->default_value(100000)
        , "number of tasks to invoke")

        ( "delay"
        , value<std::uint64_t>(&delay)->default_value(5)
        , "duration of delay in microseconds")

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

    return app_main(vm);
}

