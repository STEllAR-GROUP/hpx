//  Copyright (c) 2011-2014 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2014 Hartmut Kaiser
//  Copyright (c) 2013-2014 Thomas Heller
//  Copyright (c) 2013-2014 Patricia Grubel
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#define HPX_NO_VERSION_CHECK
#include "htts2.hpp"

#include <boost/program_options.hpp>

#include <cstdint>
#include <string>

namespace htts2
{

driver::driver(int argc, char** argv, bool allow_unregistered)
  : osthreads_(1)
  , tasks_(500000)
  , payload_duration_(5000)
  , io_(csv_with_headers)
  , argc_(argc)
  , argv_(argv)
  , allow_unregistered_(allow_unregistered)
{
    boost::program_options::variables_map vm;

    boost::program_options::options_description cmdline
        (std::string("Usage: ") + argv[0] + " [options]");

    cmdline.add_options()
        ( "help,h"
        , "print out program usage (this message)")

        ( "osthreads,t"
        , boost::program_options::value<std::uint64_t>
                (&osthreads_)->default_value(1)
        , "number of OS-threads to use")

        ( "tasks"
        , boost::program_options::value<std::uint64_t>
                (&tasks_)->default_value(500000)
        , "number of tasks per OS-thread to invoke")

        ( "payload"
        , boost::program_options::value<std::uint64_t>
                (&payload_duration_)->default_value(5000)
        , "duration of payload in nanoseconds")

        ( "no-header"
        , "don't print out column headers")
        ;

    if (allow_unregistered_)
    {
        boost::program_options::store(
            boost::program_options::command_line_parser(argc, argv)
                .options(cmdline).allow_unregistered().run(), vm);
    }
    else
    {
        boost::program_options::store(
            boost::program_options::command_line_parser(argc, argv)
                .options(cmdline).run(), vm);
    }

    boost::program_options::notify(vm);

    // Print help screen.
    if (vm.count("help"))
    {
        std::cout << cmdline;
        std::exit(0);
    }

    if (vm.count("no-header"))
        io_ = csv_without_headers;
}

}

