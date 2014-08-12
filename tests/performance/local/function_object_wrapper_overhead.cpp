//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2014 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/function.hpp>
#include <hpx/util/high_resolution_timer.hpp>

#include "worker_timed.hpp"

#include <functional>

#include <boost/function.hpp>
#include <boost/program_options.hpp>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;
using boost::program_options::store;
using boost::program_options::command_line_parser;
using boost::program_options::notify;

boost::uint64_t iterations = 500000;
boost::uint64_t delay = 5;

struct foo
{
    void operator()() const
    {
        worker_timed(delay * 1000);
    }

    template <typename Archive> void serialize(Archive&, unsigned int) {}
};

template <typename F>
void run(F const & f, boost::uint64_t local_iterations)
{
    boost::uint64_t i = 0;
    hpx::util::high_resolution_timer t;

    for (; i < local_iterations; ++i)
        f();

    double elapsed = t.elapsed();
    std::cout << " walltime/iteration: "
              << ((elapsed/i)*1e9) << " ns\n";
}

int app_main(
    variables_map& vm
    )
{
    {
        foo f;
        std::cout << "baseline";
        run(f, iterations);
    }
    {
        hpx::util::function<void(), void, void> f = foo();
        std::cout << "hpx::util::function (non-serializable)";
        run(f, iterations);
    }
    {
        hpx::util::function<void()> f = foo();
        std::cout << "hpx::util::function (serializable)";
        run(f, iterations);
    }
    {
        boost::function<void()> f = foo();
        std::cout << "boost::function";
        run(f, iterations);
    }
    {
        std::function<void()> f = foo();
        std::cout << "std::function";
        run(f, iterations);
    }

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

        ( "iterations"
        , value<boost::uint64_t>(&iterations)->default_value(500000)
        , "number of iterations to invoke for each test")

        ( "delay"
        , value<boost::uint64_t>(&delay)->default_value(5)
        , "duration of delay in microseconds")
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

