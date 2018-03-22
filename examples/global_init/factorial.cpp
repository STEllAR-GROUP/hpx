////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2018 Nikunj Gupta
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// Naive SMP version implemented with futures.

#include <hpx/hpx.hpp>
#include "hpx_initialize.hpp"
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/util.hpp>
#include <hpx/exception.hpp>
#include <hpx/include/iostreams.hpp>

#include <cstdint>
#include <typeinfo>
#include <iostream>

///////////////////////////////////////////////////////////////////////////////
std::uint64_t factorial(std::uint64_t m);

HPX_PLAIN_ACTION(factorial, factorial_action);

///////////////////////////////////////////////////////////////////////////////
std::uint64_t factorial(std::uint64_t n)
{
    if (0 >= n)
        return 1;

    hpx::lcos::future<std::uint64_t> n1 =
        hpx::async<factorial_action>(hpx::find_here(), n - 1);
    return n * n1.get();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    using boost::program_options::options_description;
    using boost::program_options::value;
    using boost::program_options::variables_map;

    // Configure application-specific options
    options_description
        desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "n-value"
        , value<std::uint64_t>()->default_value(10)
        , "n value for the factorial function")
        ;

    // Add application-specific options into variables_map
    variables_map vm;
    store(parse_command_line(argc, argv, desc_commandline), vm);
    notify(vm);

    std::uint64_t n;

    if (vm.count("n-value"))
    n = vm["n-value"].as<std::uint64_t>();

    {
        hpx::util::high_resolution_timer t;

        hpx::future<std::uint64_t> f = hpx::async(factorial, n);
        std::uint64_t r = f.get();

        double elapsed = t.elapsed();
        hpx::util::format_to(std::cout,
            "factorial(%1%) == %2%\n"
            "elapsed time == %3% [s]\n",
            n,r, elapsed);

    }

    return hpx::finalize();
}
