//  Copyright (c)      2013 Thomas Heller
//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This is a purely local version demonstrating the proposed extension to
// C++ implementing resumable functions (see N3564,
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3564.pdf). The
// necessary transformations are performed by hand.

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/lcos.hpp>

#include <hpx/util/unwrapped.hpp>

#include <iostream>
#include <utility>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>

///////////////////////////////////////////////////////////////////////////////
boost::uint64_t threshold = 2;

///////////////////////////////////////////////////////////////////////////////
BOOST_NOINLINE boost::uint64_t fibonacci_serial(boost::uint64_t n)
{
    if (n < 2)
        return n;
    return fibonacci_serial(n-1) + fibonacci_serial(n-2);
}

///////////////////////////////////////////////////////////////////////////////
hpx::future<boost::uint64_t> fibonacci(boost::uint64_t n)
{
    if (n < 2) return hpx::make_ready_future(n);
    if (n < threshold) return hpx::make_ready_future(fibonacci_serial(n));

    hpx::future<boost::uint64_t> lhs_future = hpx::async(&fibonacci, n-1);
    hpx::future<boost::uint64_t> rhs_future = fibonacci(n-2);

    return
        hpx::lcos::local::dataflow(
            hpx::util::unwrapped(
            [](boost::uint64_t lhs, boost::uint64_t rhs)
            {
                return lhs + rhs;
            })
          , std::move(lhs_future), std::move(rhs_future)
        );
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    // extract command line argument, i.e. fib(N)
    boost::uint64_t n = vm["n-value"].as<boost::uint64_t>();
    std::string test = vm["test"].as<std::string>();
    boost::uint64_t max_runs = vm["n-runs"].as<boost::uint64_t>();

    if (max_runs == 0) {
        std::cerr << "fibonacci_dataflow: wrong command line argument value for "
            "option 'n-runs', should not be zero" << std::endl;
        return hpx::finalize(); // Handles HPX shutdown
    }

    threshold = vm["threshold"].as<unsigned int>();
    if (threshold < 2 || threshold > n) {
        std::cerr << "fibonacci_dataflow: wrong command line argument value for "
            "option 'threshold', should be in between 2 and n-value"
            ", value specified: " << threshold << std::endl;
        return hpx::finalize(); // Handles HPX shutdown
    }

    bool executed_one = false;
    boost::uint64_t r = 0;

    if (test == "all" || test == "0")
    {
        // Keep track of the time required to execute.
        boost::uint64_t start = hpx::util::high_resolution_clock::now();

        for (boost::uint64_t i = 0; i != max_runs; ++i)
        {
            // serial execution
            r = fibonacci_serial(n);
        }

//      double d = double(hpx::util::high_resolution_clock::now() - start) / 1.e9;
        boost::uint64_t d = hpx::util::high_resolution_clock::now() - start;
        char const* fmt = "fibonacci_serial(%1%) == %2%,"
            "elapsed time:,%3%,[s]\n";
        std::cout << (boost::format(fmt) % n % r % (d / max_runs));

        executed_one = true;
    }

    if (test == "all" || test == "1")
    {
        // Keep track of the time required to execute.
        boost::uint64_t start = hpx::util::high_resolution_clock::now();

        for (boost::uint64_t i = 0; i != max_runs; ++i)
        {
            // Create a future for the whole calculation, execute it locally,
            // and wait for it.
            r = fibonacci(n).get();
        }

//      double d = double(hpx::util::high_resolution_clock::now() - start) / 1.e9;
        boost::uint64_t d = hpx::util::high_resolution_clock::now() - start;
        char const* fmt = "fibonacci_await(%1%) == %2%,"
            "elapsed time:,%3%,[s]\n";
        std::cout << (boost::format(fmt) % n % r % (d / max_runs));

        executed_one = true;
    }

    if (!executed_one)
    {
        std::cerr << "fibonacci_dataflow: wrong command line argument value for "
            "option 'tests', should be either 'all' or a number between zero "
            "and 1, value specified: " << test << std::endl;
    }

    return hpx::finalize(); // Handles HPX shutdown
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    using boost::program_options::value;
    desc_commandline.add_options()
        ( "n-value", value<boost::uint64_t>()->default_value(10),
          "n value for the Fibonacci function")
        ( "n-runs", value<boost::uint64_t>()->default_value(1),
          "number of runs to perform")
        ( "threshold", value<unsigned int>()->default_value(2),
          "threshold for switching to serial code")
        ( "test", value<std::string>()->default_value("all"),
          "select tests to execute (0-1, default: all)")
    ;

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}
