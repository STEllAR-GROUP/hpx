////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// Naive SMP version implemented with futures.

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>

#include <iostream>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>

///////////////////////////////////////////////////////////////////////////////
//[fib_action
// forward declaration of the Fibonacci function
boost::uint64_t fibonacci(boost::uint64_t n);

// This is to generate the required boilerplate we need for the remote
// invocation to work.
HPX_PLAIN_ACTION(fibonacci, fibonacci_action);
//]

///////////////////////////////////////////////////////////////////////////////
//[fib_func
boost::uint64_t fibonacci(boost::uint64_t n)
{
    if (n < 2)
        return n;

    // We restrict ourselves to execute the Fibonacci function locally.
    hpx::naming::id_type const locality_id = hpx::find_here();

    // Invoking the Fibonacci algorithm twice is inefficient.
    // However, we intentionally demonstrate it this way to create some
    // heavy workload.
    using hpx::lcos::future;
    using hpx::async;
    future<boost::uint64_t> n1 = async<fibonacci_action>(locality_id, n - 1);
    future<boost::uint64_t> n2 = async<fibonacci_action>(locality_id, n - 2);

    return n1.get() + n2.get();   // wait for the Futures to return their values
}
//]

///////////////////////////////////////////////////////////////////////////////
//[fib_hpx_main
int hpx_main(boost::program_options::variables_map& vm)
{
    // extract command line argument, i.e. fib(N)
    boost::uint64_t n = vm["n-value"].as<boost::uint64_t>();

    {
        // Keep track of the time required to execute.
        hpx::util::high_resolution_timer t;

        // Create a Future for the whole calculation, execute it locally, and
        // wait for it.
        hpx::lcos::future<boost::uint64_t> f =
            hpx::async<fibonacci_action>(hpx::find_here(), n);

        // wait for future f to return value
        boost::uint64_t r = f.get();

        char const* fmt = "fibonacci(%1%) == %2%\nelapsed time: %3% [s]\n";
        std::cout << (boost::format(fmt) % n % r % t.elapsed());
    }

    return hpx::finalize(); // Handles HPX shutdown
}
//]

///////////////////////////////////////////////////////////////////////////////
//[fib_main
int main(int argc, char* argv[])
{
    // Configure application-specific options
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "n-value",
          boost::program_options::value<boost::uint64_t>()->default_value(10),
          "n value for the Fibonacci function")
        ;

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}
//]
