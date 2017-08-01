//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/unwrap.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <boost/atomic.hpp>
#include <boost/format.hpp>

///////////////////////////////////////////////////////////////////////////////
std::uint64_t threshold = 2;
std::uint64_t distribute_at = 2;
int num_repeats = 1;

///////////////////////////////////////////////////////////////////////////////
boost::atomic<std::size_t> serial_execution_count(0);

std::size_t get_serial_execution_count()
{
    return serial_execution_count.load();
}
HPX_PLAIN_ACTION(get_serial_execution_count);

///////////////////////////////////////////////////////////////////////////////
boost::atomic<std::size_t> next_locality(0);
std::vector<hpx::id_type> localities;
hpx::id_type here;

struct when_all_wrapper
{
    typedef hpx::util::tuple<
            hpx::lcos::future<std::uint64_t>
          , hpx::lcos::future<std::uint64_t> > data_type;

    std::uint64_t operator()(
        hpx::lcos::future<data_type> data
    ) const
    {
        data_type v = data.get();
        return hpx::util::get<0>(v).get() + hpx::util::get<1>(v).get();
    }
};

///////////////////////////////////////////////////////////////////////////////
HPX_NOINLINE std::uint64_t fibonacci_serial_sub(std::uint64_t n)
{
    if (n < 2)
        return n;
    return fibonacci_serial_sub(n-1) + fibonacci_serial_sub(n-2);
}

std::uint64_t fibonacci_serial(std::uint64_t n)
{
    ++serial_execution_count;
    return fibonacci_serial_sub(n);
}

///////////////////////////////////////////////////////////////////////////////
hpx::id_type const& get_next_locality(std::uint64_t next)
{
    return localities[next % localities.size()];
}

///////////////////////////////////////////////////////////////////////////////
hpx::future<std::uint64_t> fibonacci_future(std::uint64_t n);
HPX_PLAIN_ACTION(fibonacci_future);

hpx::future<std::uint64_t> fibonacci_future(std::uint64_t n)
{
    // if we know the answer, we return a future encapsulating the final value
    if (n < 2)
        return hpx::make_ready_future(n);
    if (n < threshold)
        return hpx::make_ready_future(fibonacci_serial(n));

    fibonacci_future_action fib;
    hpx::id_type loc1 = here;
    hpx::id_type loc2 = here;

    if (n == distribute_at) {
        loc2 = get_next_locality(++next_locality);
    }
    else if (n-1 == distribute_at) {
        std::uint64_t next = next_locality += 2;
        loc1 = get_next_locality(next-1);
        loc2 = get_next_locality(next);
    }

    hpx::future<std::uint64_t> f = hpx::async(fib, loc1, n-1);
    hpx::future<std::uint64_t> r = fib(loc2, n-2);

    return hpx::when_all(f, r).then(when_all_wrapper());
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    // extract command line argument, i.e. fib(N)
    std::uint64_t n = vm["n-value"].as<std::uint64_t>();
    std::string test = vm["test"].as<std::string>();
    std::uint64_t max_runs = vm["n-runs"].as<std::uint64_t>();

    if (max_runs == 0) {
        std::cerr << "fibonacci_futures_distributed: wrong command "
            "line argument value for "
            "option 'n-runs', should not be zero" << std::endl;
        return hpx::finalize(); // Handles HPX shutdown
    }

    bool executed_one = false;
    std::uint64_t r = 0;

    if (test == "all" || test == "0")
    {
        // Keep track of the time required to execute.
        std::uint64_t start = hpx::util::high_resolution_clock::now();

        // Synchronous execution, use as reference only.
        r = fibonacci_serial(n);

//        double d = double(hpx::util::high_resolution_clock::now() - start) / 1.e9;
        std::uint64_t d = hpx::util::high_resolution_clock::now() - start;
        char const* fmt = "fibonacci_serial(%1%) == %2%,"
            "elapsed time:,%3%,[s]\n";
        std::cout << (boost::format(fmt) % n % r % d);

        executed_one = true;
    }

    if (test == "all" || test == "1")
    {
        // Keep track of the time required to execute.
        std::uint64_t start = hpx::util::high_resolution_clock::now();

        for (std::size_t i = 0; i != max_runs; ++i)
        {
            // Create a Future for the whole calculation and wait for it.
            next_locality.store(0);
            r = fibonacci_future(n).get();
        }

//        double d = double(hpx::util::high_resolution_clock::now() - start) / 1.e9;
        std::uint64_t d = hpx::util::high_resolution_clock::now() - start;
        char const* fmt = "fibonacci_future(%1%) == %2%,elapsed time:,%3%,[s],%4%\n";
        std::cout << (boost::format(fmt) % n % r % (d / max_runs)
            % next_locality.load());

        get_serial_execution_count_action serial_count;
        for (hpx::id_type const& loc : hpx::find_all_localities())
        {
            std::size_t count = serial_count(loc);
            std::cout << (boost::format("  serial-count,%1%,%2%\n") %
                loc % (count / max_runs));
        }

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
boost::program_options::options_description get_commandline_options()
{
    // Configure application-specific options
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    using boost::program_options::value;
    desc_commandline.add_options()
        ( "n-value", value<std::uint64_t>()->default_value(10),
          "n value for the Fibonacci function")
        ( "n-runs", value<std::uint64_t>()->default_value(1),
          "number of runs to perform")
        ( "threshold", value<unsigned int>()->default_value(2),
          "threshold for switching to serial code")
        ( "distribute-at", value<unsigned int>()->default_value(2),
          "threshold for distribution to other nodes")
        ( "test", value<std::string>()->default_value("all"),
          "select tests to execute (0-7, default: all)")
        ( "loc-repeat", value<int>()->default_value(1),
          "how often should a locality > 0 be used")
    ;
    return desc_commandline;
}

///////////////////////////////////////////////////////////////////////////////
void init_globals()
{
    // Retrieve command line using the Boost.ProgramOptions library.
    boost::program_options::variables_map vm;
    if (!hpx::util::retrieve_commandline_arguments(get_commandline_options(), vm))
    {
        HPX_THROW_EXCEPTION(hpx::commandline_option_error,
            "fibonacci_futures_distributed",
            "failed to handle command line options");
        return;
    }

    std::uint64_t n = vm["n-value"].as<std::uint64_t>();

    threshold = vm["threshold"].as<unsigned int>();
    if (threshold < 2 || threshold > n) {
        HPX_THROW_EXCEPTION(hpx::commandline_option_error,
            "fibonacci_futures_distributed",
            "wrong command line argument value for option 'threshold', "
            "should be in between 2 and n-value, value specified: " +
                std::to_string(threshold));
        return;
    }

    distribute_at = vm["distribute-at"].as<unsigned int>();
    if (distribute_at < 2 || distribute_at > n) {
        HPX_THROW_EXCEPTION(hpx::commandline_option_error,
            "fibonacci_futures_distributed",
            "wrong command line argument value for option 'distribute-at', "
            "should be in between 2 and n-value, value specified: " +
                std::to_string(distribute_at));
        return;
    }

    here = hpx::find_here();
    next_locality.store(0);
    serial_execution_count.store(0);

    // try to more evenly distribute the work over the participating localities
    std::vector<hpx::id_type> locs = hpx::find_all_localities();
    std::size_t num_repeats = vm["loc-repeat"].as<int>();

    localities.push_back(here);      // add ourselves
    for (std::size_t j = 0; j != num_repeats; ++j)
    {
        for (std::size_t i = 0; i != locs.size(); ++i)
        {
            if (here == locs[i])
                continue;
            localities.push_back(locs[i]);
        }
    }
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    hpx::register_startup_function(&init_globals);
    return hpx::init(get_commandline_options(), argc, argv);
}
