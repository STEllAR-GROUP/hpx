//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#include <hpx/compat/thread.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include <boost/config.hpp>
#include <boost/format.hpp>
#include <boost/thread/tss.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/program_options.hpp>

#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::store;
using boost::program_options::command_line_parser;
using boost::program_options::notify;

namespace compat = hpx::compat;
using hpx::util::high_resolution_timer;

///////////////////////////////////////////////////////////////////////////////
// thread local globals
static boost::thread_specific_ptr<double> global_scratch;

///////////////////////////////////////////////////////////////////////////////
inline void worker(
    boost::barrier& b
  , std::uint64_t updates
    )
{
    b.wait();

    for (double i = 0.; i < updates; ++i)
    {
        global_scratch.reset(new double);

        *global_scratch += 1. / (2. * i * (*global_scratch) + 1.);

        global_scratch.reset();
    }
}

///////////////////////////////////////////////////////////////////////////////
int main(
    int argc
  , char** argv
    )
{
    ///////////////////////////////////////////////////////////////////////////
    // parse command line
    variables_map vm;

    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    unsigned threads;
    std::uint64_t updates;

    cmdline.add_options()
        ( "help,h"
        , "print out program usage (this message)")

        ( "threads,t"
        , value<unsigned>(&threads)->default_value(1),
         "number of OS-threads")

        ( "updates,u"
        , value<std::uint64_t>(&updates)->default_value(1 << 22)
        , "updates made to the TLS variable per OS-thread")

        ( "csv"
        , "output results as csv (format: updates,OS-threads,duration)")
        ;
    ;

    store(command_line_parser(argc, argv).options(cmdline).run(), vm);

    notify(vm);

    ///////////////////////////////////////////////////////////////////////////
    // print help screen
    if (vm.count("help"))
    {
        std::cout << cmdline;
        return 0;
    }

    ///////////////////////////////////////////////////////////////////////////
    // run the test
    std::vector<compat::thread> workers;

    boost::barrier b(threads);

    high_resolution_timer t;

    for (unsigned i = 0; i != threads; ++i)
        workers.push_back(compat::thread(worker, std::ref(b), updates));

    for (compat::thread& thread : workers)
    {
        if (thread.joinable())
            thread.join();
    }

    const double duration = t.elapsed();

    ///////////////////////////////////////////////////////////////////////////
    // output results
    if (vm.count("csv"))
        std::cout << ( boost::format("%1%,%2%,%3%\n")
                     % updates
                     % threads
                     % duration);
    else
        std::cout << ( boost::format("ran %1% updates per OS-thread on %2% "
                                     "OS-threads in %3% seconds\n")
                     % updates
                     % threads
                     % duration);
}

