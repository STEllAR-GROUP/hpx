//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This is a purely local version demonstrating different versions of making
// the calculation of a fibonacci asynchronous.

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/lcos.hpp>

#include <iostream>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>

#if defined(BOOST_MSVC)
#define HPX_NO_INLINE __declspec(noinline)
#else
#define HPX_NO_INLINE
#endif

///////////////////////////////////////////////////////////////////////////////
boost::uint64_t threshold = 2;
boost::uint64_t distribute_at = 2;

boost::atomic<std::size_t> next_locality(0);
std::vector<hpx::id_type> localities;

struct when_all_wrapper
{
    typedef boost::uint64_t result_type;

    boost::uint64_t operator()(
        hpx::lcos::future<std::vector<hpx::lcos::future<uint64_t> > > data
    ) const
    {
        std::vector<hpx::lcos::future<uint64_t> > v = data.move();
        return v[0].get() + v[1].get();
    }
};

///////////////////////////////////////////////////////////////////////////////
HPX_NO_INLINE boost::uint64_t fibonacci_serial(boost::uint64_t n)
{
    if (n < 2)
        return n;
    return fibonacci_serial(n-1) + fibonacci_serial(n-2);
}

hpx::future<boost::uint64_t> fibonacci_future(boost::uint64_t n);
HPX_PLAIN_ACTION(fibonacci_future);

hpx::future<boost::uint64_t> fibonacci_future(boost::uint64_t n)
{
    // if we know the answer, we return a future encapsulating the final value
    if (n < 2)
        return hpx::make_ready_future(n);
    if (n < threshold)
        return hpx::make_ready_future(fibonacci_serial(n));

    fibonacci_future_action fib;
    hpx::id_type loc = hpx::find_here();

    if (n == distribute_at)
        loc = localities[++next_locality % localities.size()];

    hpx::future<boost::uint64_t> f = hpx::async(hpx::launch::async, fib, loc, n-1);
    hpx::future<boost::uint64_t> r = hpx::async(hpx::launch::sync, fib, loc, n-2);

    return hpx::when_all(f, r).then(when_all_wrapper());
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    // extract command line argument, i.e. fib(N)
    boost::uint64_t n = vm["n-value"].as<boost::uint64_t>();
    std::string test = vm["test"].as<std::string>();
    boost::uint64_t max_runs = vm["n-runs"].as<boost::uint64_t>();

    if (max_runs == 0) {
        std::cerr << "fibonacci_futures: wrong command line argument value for "
            "option 'n-runs', should not be zero" << std::endl;
        return hpx::finalize(); // Handles HPX shutdown
    }

    threshold = vm["threshold"].as<unsigned int>();
    if (threshold < 2 || threshold > n) {
        std::cerr << "fibonacci_futures: wrong command line argument value for "
            "option 'threshold', should be in between 2 and n-value"
            ", value specified: " << threshold << std::endl;
        return hpx::finalize(); // Handles HPX shutdown
    }

    distribute_at = vm["distribute-at"].as<unsigned int>();
    if (distribute_at < 2 || distribute_at > n) {
        std::cerr << "fibonacci_futures: wrong command line argument value for "
            "option 'distribute-at', should be in between 2 and n-value"
            ", value specified: " << threshold << std::endl;
        return hpx::finalize(); // Handles HPX shutdown
    }

    bool executed_one = false;
    boost::uint64_t r = 0;

    if (test == "all" || test == "0")
    {
        // Keep track of the time required to execute.
        boost::uint64_t start = hpx::util::high_resolution_clock::now();

        for (std::size_t i = 0; i != max_runs; ++i)
        {
            // Create a Future for the whole calculation, execute it locally,
            // and wait for it.
            r = fibonacci_serial(n);
        }

//        double d = double(hpx::util::high_resolution_clock::now() - start) / 1.e9;
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

        for (std::size_t i = 0; i != max_runs; ++i)
        {
            // Create a Future for the whole calculation, execute it locally, and
            // wait for it.
            r = fibonacci_future(n).get();
        }

//        double d = double(hpx::util::high_resolution_clock::now() - start) / 1.e9;
        boost::uint64_t d = hpx::util::high_resolution_clock::now() - start;
        char const* fmt = "fibonacci_future_unwrapped_when_all(%1%) == %2%,elapsed time:,%3%,[s]\n";
        std::cout << (boost::format(fmt) % n % r % (d / max_runs));

        executed_one = true;
    }

    if (!executed_one)
    {
        std::cerr << "fibonacci_futures_distributed: wrong command line argument "
            "value for option 'tests', should be either 'all' or a number between "
            "zero and 1, value specified: " << test << std::endl;
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
        ( "distribute-at", value<unsigned int>()->default_value(2),
          "threshold for distribution to other nodes")
        ( "test", value<std::string>()->default_value("all"),
          "select tests to execute (0-7, default: all)")
    ;

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}
