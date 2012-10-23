//  Copyright (c) 2007-2012 Hartmut Kaiser
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

///////////////////////////////////////////////////////////////////////////////
boost::uint64_t add(
    hpx::lcos::future<boost::uint64_t> const& f1,
    hpx::lcos::future<boost::uint64_t> const& f2)
{
    return f1.get() + f2.get();
}

///////////////////////////////////////////////////////////////////////////////
hpx::lcos::future<boost::uint64_t> fibonacci_future_one(boost::uint64_t n);

struct fibonacci_future_one_continuation
{
    typedef boost::uint64_t result_type;

    fibonacci_future_one_continuation(boost::uint64_t n)
      : n_(n)
    {}

    result_type operator()(hpx::lcos::future<boost::uint64_t> res) const
    {
        return add(fibonacci_future_one(n_ - 2), res);
    }

    boost::uint64_t n_;
};

boost::uint64_t fib(boost::uint64_t n)
{
    return fibonacci_future_one(n).get();
}

hpx::lcos::future<boost::uint64_t> fibonacci_future_one(boost::uint64_t n)
{
    // if we know the answer, we return a future encapsulating the final value
    if (n < 2)
        return hpx::lcos::make_future(n);

    // asynchronously launch the calculation of one of the sub-terms
    // attach a continuation to this future which is called asynchronously on
    // its completion and which calculates the other sub-term
    return hpx::async(&fib, n-1).then(fibonacci_future_one_continuation(n));
}

///////////////////////////////////////////////////////////////////////////////
boost::uint64_t fibonacci(boost::uint64_t n)
{
    // if we know the answer, we return a future encapsulating the final value
    if (n < 2)
        return n;

    // asynchronously launch the creation of one of the sub-terms of the
    // execution graph
    hpx::lcos::future<boost::uint64_t> f = hpx::async(&fibonacci, n-1);
    boost::uint64_t r = fibonacci(n-2);

    // attach a continuation to this future which is called asynchronously on
    // its completion and which calculates the other sub-term
    return f.get() + r;
}

///////////////////////////////////////////////////////////////////////////////
hpx::lcos::future<boost::uint64_t> fibonacci_future(boost::uint64_t n)
{
    // if we know the answer, we return a future encapsulating the final value
    if (n < 2)
        return hpx::lcos::make_future(n);

    // asynchronously launch the creation of one of the sub-terms of the
    // execution graph
    hpx::lcos::future<hpx::lcos::future<boost::uint64_t> > f =
        hpx::async(hpx::launch::deferred, &fibonacci_future, n-1);
    hpx::lcos::future<boost::uint64_t> r = fibonacci_future(n-2);

    return hpx::async(&add, f.get(), r);
}

/////////////////////////////////////////////////////////////////////////////
hpx::lcos::future<boost::uint64_t> fibonacci_future_all(boost::uint64_t n)
{
    // if we know the answer, we return a future encapsulating the final value
    if (n < 2)
        return hpx::lcos::make_future(n);

    using hpx::lcos::future;

    // asynchronously launch the calculation of both of the sub-terms
    future<boost::uint64_t> f1 = fibonacci_future_all(n - 1);
    future<boost::uint64_t> f2 = fibonacci_future_all(n - 2);

    // create a future representing the successful calculation of both sub-terms
    // attach a continuation to this future which is called asynchronously on
    // its completion and which calculates the the final result
    return hpx::async(&add, f1, f2);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    // extract command line argument, i.e. fib(N)
    boost::uint64_t n = vm["n-value"].as<boost::uint64_t>();
    std::string test = vm["test"].as<std::string>();
    bool executed_one = false;

    if (test == "all" || test == "0")
    {
        // Keep track of the time required to execute.
        hpx::util::high_resolution_timer t;

        // Create a Future for the whole calculation, execute it locally, and
        // wait for it.
        boost::uint64_t r = fibonacci_future_one(n).get();

        double d = t.elapsed();
        char const* fmt = "fibonacci_future_one(%1%) == %2%\n"
            "elapsed time: %3% [s]\n";
        std::cout << (boost::format(fmt) % n % r % d);

        executed_one = true;
    }

    if (test == "all" || test == "1")
    {
        // Keep track of the time required to execute.
        hpx::util::high_resolution_timer t;

        // Create a Future for the whole calculation, execute it locally, and
        // wait for it.
        boost::uint64_t r = fibonacci(n);

        double d = t.elapsed();
        char const* fmt = "fibonacci(%1%) == %2%\nelapsed time: %3% [s]\n";
        std::cout << (boost::format(fmt) % n % r % d);

        executed_one = true;
    }

    if (test == "all" || test == "2")
    {
        // Keep track of the time required to execute.
        hpx::util::high_resolution_timer t;

        // Create a Future for the whole calculation, execute it locally, and
        // wait for it.
        boost::uint64_t r = fibonacci_future(n).get();

        double d = t.elapsed();
        char const* fmt = "fibonacci_future(%1%) == %2%\nelapsed time: %3% [s]\n";
        std::cout << (boost::format(fmt) % n % r % d);

        executed_one = true;
    }

    if (test == "all" || test == "3")
    {
        // Keep track of the time required to execute.
        hpx::util::high_resolution_timer t;

        // Create a future for the whole calculation, execute it locally, and
        // wait for it.
        boost::uint64_t r = fibonacci_future_all(n).get();

        char const* fmt = "fibonacci_future_all(%1%) == %2%\n"
            "elapsed time: %3% [s]\n";
        std::cout << (boost::format(fmt) % n % r % t.elapsed());

        executed_one = true;
    }

    if (!executed_one)
    {
        std::cerr << "fibonacci_futures: wrong command line argument value for "
            "option 'tests', should be either 'all' or a number between zero "
            "and 3, value specified: " << test << std::endl;
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
        ( "test", value<std::string>()->default_value("all"),
          "select tests to execute (0-3, default: all)")
    ;

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}
