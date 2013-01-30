//  Copyright (c) 2011-2012 Patricia Grubel
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
// Applications globals.
boost::atomic<boost::uint64_t> donecount(0);

// Command-line variables.
boost::uint64_t tasks = 500000;
boost::uint64_t delay = 0;
bool header = true;

///////////////////////////////////////////////////////////////////////////////
void print_results(
    boost::uint64_t cores
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
extern "C" aligned_t worker(
    void* 
    )
{
    double volatile d = 0.;
    for (boost::uint64_t i = 0; i < delay; ++i)
        d += 1. / (2. * i + 1.);

    ++donecount;

    return aligned_t();
}

///////////////////////////////////////////////////////////////////////////////
int qthreads_main(
    variables_map& vm
    )
{
    if (vm.count("no-header"))
        header = false;
    {
        // Validate command line.
        if (0 == tasks)
            throw std::invalid_argument("count of 0 tasks specified\n");

        // Start the clock.
        high_resolution_timer t;

        for (boost::uint64_t i = 0; i < tasks; ++i)
            qthread_fork(&worker, NULL, NULL);

        // Yield until all our null qthreads are done.
        do {
            qthread_yield();
        } while (donecount != tasks);

        print_results(qthread_num_workers(), t.elapsed());
    }

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
        , value<boost::uint64_t>(&tasks)->default_value(500000)
        , "number of tasks (e.g. qthreads) to invoke")

        ( "delay"
        , value<boost::uint64_t>(&delay)->default_value(0)
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

    if (vm.count("no-header"))
        header = false;

    // Set qthreads environment variables.
    std::string const shepherds = boost::lexical_cast<std::string>
        (vm["shepherds"].as<boost::uint64_t>());
    std::string const workers = boost::lexical_cast<std::string>
        (vm["workers-per-shepherd"].as<boost::uint64_t>());

    setenv("QT_NUM_SHEPHERDS", shepherds.c_str(), 1);
    setenv("QT_NUM_WORKERS_PER_SHEPHERD", workers.c_str(), 1);

    // Setup the qthreads environment.
    if (qthread_initialize() != 0)
        throw std::runtime_error("qthreads failed to initialize\n");

    return qthreads_main(vm);
}

