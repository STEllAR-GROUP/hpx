//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Copyright (c) 2007, Sandia Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the Sandia Corporation nor the names of its
//       contributors may be used to endorse or promote products derived from
//       this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
// EVENT SHALL SANDIA CORPORATION BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
// OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#define HPX_NO_VERSION_CHECK

#include <hpx/config.hpp>

#include "worker_timed.hpp"

#include <hpx/util/assert.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <tbb/task.h>
#include <tbb/task_scheduler_init.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

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
// Command-line variables.
std::uint64_t tasks = 500000;
std::uint64_t delay = 0;
bool header = true;

///////////////////////////////////////////////////////////////////////////////
void print_results(
    std::uint64_t cores
  , double walltime
    )
{
    if (header)
        std::cout << "OS-threads,Tasks,Delay (iterations),"
                     "Total Walltime (seconds),Walltime per Task (seconds)\n";

    std::string const cores_str = boost::str(boost::format("%lu,") % cores);
    std::string const tasks_str = boost::str(boost::format("%lu,") % tasks);
    std::string const delay_str = boost::str(boost::format("%lu,") % delay);

    std::cout << ( boost::format("%-21s %-21s %-21s %10.12s, %10.12s\n")
                 % cores_str % tasks_str % delay_str
                 % walltime % (walltime / tasks));
}

///////////////////////////////////////////////////////////////////////////////
struct worker_func : tbb::task
{
    tbb::task* execute()
    {
        worker_timed(delay * 1000);
        return 0;
    }
};

///////////////////////////////////////////////////////////////////////////////
struct spawner : tbb::task
{
    tbb::task* execute()
    {
        set_ref_count(static_cast<int>(tasks + 1));

        for (std::uint64_t i = 0; i < tasks; ++i)
        {
            worker_func& a = *new (tbb::task::allocate_child()) worker_func();

            if (i == (tasks - 1))
                spawn_and_wait_for_all(a);
            else
                spawn(a);
        }

        return 0;
    }
};

///////////////////////////////////////////////////////////////////////////////
int tbb_main(
    variables_map& vm
    )
{
    // Validate command line.
    if (0 == tasks)
        throw std::invalid_argument("count of 0 tasks specified\n");

    // Start the clock.
    high_resolution_timer t;

    spawner& a = *new (tbb::task::allocate_root()) spawner();

    tbb::task::spawn_root_and_wait(a);

    print_results(vm["threads"].as<std::uint64_t>(), t.elapsed());

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

        ( "threads,t"
        , value<std::uint64_t>()->default_value(1),
         "number of OS-threads to use")

        ( "tasks"
        , value<std::uint64_t>(&tasks)->default_value(500000)
        , "number of tasks (e.g. px-threads) to invoke")

        ( "delay"
        , value<std::uint64_t>(&delay)->default_value(0)
        , "number of iterations in the delay loop")

        ( "no-header"
        , "do not print out the csv header row")
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

    if (vm.count("no-header"))
        header = false;

    // Setup the TBB environment.
    tbb::task_scheduler_init init(vm["threads"].as<std::uint64_t>());

    return tbb_main(vm);
}

