//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2012      Dylan Stark
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/util/high_resolution_timer.hpp>

#include <stdexcept>
#include <iostream>

#include <qthread/qthread.h>

#include <boost/atomic.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::store;
using boost::program_options::command_line_parser;
using boost::program_options::notify;

using hpx::util::high_resolution_timer;

///////////////////////////////////////////////////////////////////////////////
boost::atomic<boost::uint64_t> donecount(0);
boost::uint64_t delay = 0;

///////////////////////////////////////////////////////////////////////////////
extern "C" aligned_t null_thread(
    void* args
    )
{
    double volatile d = 0.;
    for (boost::uint64_t i = 0; i < delay; ++i)
        d += 1 / (2. * i + 1);

    ++donecount;

    return aligned_t();
}

///////////////////////////////////////////////////////////////////////////////
void print_results(
    boost::uint64_t cores
  , boost::uint64_t tasks
  , boost::uint64_t delay
  , double walltime
    )
{
    std::string const cores_str = boost::str(boost::format("%lu,") % cores);
    std::string const tasks_str = boost::str(boost::format("%lu,") % tasks);
    std::string const delay_str = boost::str(boost::format("%lu,") % delay);

    std::cout << ( boost::format("%-21s %-21s %-21s %-08.8g\n")
                 % cores_str 
                 % tasks_str
                 % delay_str 
                 % walltime);
}

///////////////////////////////////////////////////////////////////////////////
int qthreads_main(
    variables_map& vm
    )
{
    delay = vm["delay"].as<boost::uint64_t>();
    boost::uint64_t const tasks = vm["tasks"].as<boost::uint64_t>();

    if (0 == tasks)
        throw std::invalid_argument("count of 0 tasks specified\n");

    if (qthread_initialize() != 0)
        throw std::runtime_error("qthreads failed to initialize\n");

    // Start the clock.
    high_resolution_timer t;

	for (boost::uint64_t i = 0; i < tasks; ++i)
        qthread_fork(&null_thread, NULL, NULL);

    // Yield until all our null qthreads are done.
	do {
	    qthread_yield();
	} while (donecount != tasks);

    print_results(qthread_num_workers(), tasks, delay, t.elapsed());

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(
    int argc
  , char** argv
    )
{
    // Parse command line.
    variables_map vm;

    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ( "help,h"
        , "print out program usage (this message)")
        
        ( "shepherds,s"
        , value<boost::uint64_t>()->default_value(1),
         "number of shepherds to use")

        ( "workers-per-shepherd,w"
        , value<boost::uint64_t>()->default_value(1),
         "number of worker OS-threads per shepherd")

        ( "tasks"
        , value<boost::uint64_t>()->default_value(500000)
        , "number of tasks (e.g. qthreads) to invoke")

        ( "delay"
        , value<boost::uint64_t>()->default_value(0)
        , "number of iterations in the delay loop")
        ;
    ;

    store(command_line_parser(argc, argv).options(cmdline).run(), vm);

    notify(vm);

    // Print help screen.
    if (vm.count("help"))
    {
        std::cout << cmdline;
        return 0;
    }

    // Set qthreads environment variables.
    std::string const shepherds = boost::lexical_cast<std::string>
        (vm["shepherds"].as<boost::uint64_t>());
    std::string const workers = boost::lexical_cast<std::string>
        (vm["workers-per-shepherd"].as<boost::uint64_t>());

    setenv("QT_NUM_SHEPHERDS", shepherds.c_str(), 1);
    setenv("QT_NUM_WORKERS_PER_SHEPHERD", workers.c_str(), 1);

    return qthreads_main(vm);
}

