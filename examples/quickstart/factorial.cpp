////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// Naive SMP version implemented with futures.

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/util.hpp>

#include <cstdint>
#include <iostream>

#include <boost/format.hpp>

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
int hpx_main(boost::program_options::variables_map& vm)
{
    std::uint64_t n = vm["n-value"].as<std::uint64_t>();

    {
        hpx::util::high_resolution_timer t;

        std::uint64_t r =
            hpx::async<factorial_action>(hpx::find_here(), n).get();

        double elapsed = t.elapsed();
        std::cout
            << ( boost::format("factorial(%1%) == %2%\n"
                               "elapsed time == %3% [s]\n")
               % n % r % elapsed);
    }

    return hpx::finalize();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    using boost::program_options::options_description;
    using boost::program_options::value;

    // Configure application-specific options
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "n-value"
        , value<std::uint64_t>()->default_value(10)
        , "n value for the factorial function")
        ;

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}

