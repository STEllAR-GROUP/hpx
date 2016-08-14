//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>

#include <hpx/config.hpp>

#include <boost/config.hpp>
#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/tss.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/program_options.hpp>

#include <hpx/util/high_resolution_timer.hpp>

#include <functional>

#if defined(__has_feature)
#  if __has_feature(cxx_thread_local)
#    define HPX_NATIVE_TLS thread_local
#  endif
#endif

#if !defined(HPX_NATIVE_TLS)
#  if defined(_GLIBCXX_HAVE_TLS)
#    define HPX_NATIVE_TLS __thread
#  elif defined(HPX_WINDOWS)
#    define HPX_NATIVE_TLS __declspec(thread)
#  elif defined(__FreeBSD__) || (defined(__APPLE__) && defined(__MACH__))
#    define HPX_NATIVE_TLS __thread
#  else
#    error Unsupported platform.
#  endif
#endif

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::store;
using boost::program_options::command_line_parser;
using boost::program_options::notify;

using hpx::util::high_resolution_timer;

///////////////////////////////////////////////////////////////////////////////
// thread local globals
static HPX_NATIVE_TLS double* global_scratch;

///////////////////////////////////////////////////////////////////////////////
inline void worker(
    boost::barrier& b
  , boost::uint64_t updates
    )
{
    b.wait();

    for (double i = 0.; i < updates; ++i)
    {
        global_scratch = new double;

        *global_scratch += 1. / (2. * i * (*global_scratch) + 1.);

        delete global_scratch;
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

    boost::uint32_t threads, updates;

    cmdline.add_options()
        ( "help,h"
        , "print out program usage (this message)")

        ( "threads,t"
        , value<boost::uint32_t>(&threads)->default_value(1),
         "number of OS-threads")

        ( "updates,u"
        , value<boost::uint32_t>(&updates)->default_value(1 << 22)
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
    boost::thread_group workers;

    boost::barrier b(threads);

    high_resolution_timer t;

    for (boost::uint32_t i = 0; i != threads; ++i)
        workers.add_thread(new boost::thread(worker, std::ref(b), updates));

    workers.join_all();

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

