//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/util/high_resolution_timer.hpp>

#include <swarm/Runtime.h>
#include <swarm/Scheduler.h>

#include <stdexcept>
#include <iostream>

#include <boost/assert.hpp>
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
boost::uint64_t delay = 0;
boost::uint64_t tasks = 0;
swarm_dependency_t flag;
swarm_Runtime_params params = swarm_Runtime_params_INITIALIZER;

///////////////////////////////////////////////////////////////////////////////
extern "C" void worker(
    void*
    )
{
    double volatile d = 0.;
    for (boost::uint64_t i = 0; i < delay; ++i)
        d += 1 / (2. * i + 1);

    swarm_satisfy(&flag, 1);
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
extern "C" void finish(
    void* p 
    )
{
    BOOST_ASSERT(p);

    high_resolution_timer* t = static_cast<high_resolution_timer*>(p);

    print_results(params.maxThreadCount, tasks, delay, t->elapsed());

    delete t;

    swarm_shutdownRuntime(0);
}

///////////////////////////////////////////////////////////////////////////////
extern "C" void spawner(
    void* p 
    )
{
    BOOST_ASSERT(p);

    swarm_dependency_init(&flag, tasks, finish, p);

    for (boost::uint64_t i = 0; i < tasks; ++i)
        swarm_scheduleGeneral(worker, 0);
}

///////////////////////////////////////////////////////////////////////////////
int swarm_main(
    variables_map& vm
    )
{
    delay = vm["delay"].as<boost::uint64_t>();
    tasks = vm["tasks"].as<boost::uint64_t>();
    params.maxThreadCount = vm["threads"].as<boost::uint32_t>();

    if (0 == tasks)
        throw std::invalid_argument("count of 0 tasks specified\n");

    if (0 == params.maxThreadCount)
        throw std::invalid_argument("count of 0 OS-threads specified\n");

    // Start the clock.
    high_resolution_timer* t = new high_resolution_timer;

    return !swarm_enterRuntime(&params, spawner, t);
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
        
        ( "threads,t"
        , value<boost::uint32_t>()->default_value(1),
         "number of OS-threads to use")

        ( "tasks"
        , value<boost::uint64_t>()->default_value(500000)
        , "number of tasks to invoke")

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
 
    return swarm_main(vm);
}

