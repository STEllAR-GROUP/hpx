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
boost::uint64_t add(
    hpx::future<boost::uint64_t> f1,
    hpx::future<boost::uint64_t> f2)
{
    return f1.get() + f2.get();
}

///////////////////////////////////////////////////////////////////////////////
struct when_all_wrapper
{
    typedef boost::uint64_t result_type;
    typedef hpx::util::tuple<
            hpx::future<boost::uint64_t>
          , hpx::future<boost::uint64_t> > data_type;

    boost::uint64_t operator()(
        hpx::future<data_type> data
    ) const
    {
        data_type v = data.get();
        return hpx::util::get<0>(v).get() + hpx::util::get<1>(v).get();
    }
};

///////////////////////////////////////////////////////////////////////////////
hpx::future<boost::uint64_t> fibonacci_future_one(boost::uint64_t n);

struct fibonacci_future_one_continuation
{
    typedef boost::uint64_t result_type;

    fibonacci_future_one_continuation(boost::uint64_t n)
      : n_(n)
    {}

    result_type operator()(hpx::future<boost::uint64_t> res) const
    {
        return add(fibonacci_future_one(n_ - 2), std::move(res));
    }

    boost::uint64_t n_;
};

boost::uint64_t fib(boost::uint64_t n)
{
    return fibonacci_future_one(n).get();
}

hpx::future<boost::uint64_t> fibonacci_future_one(boost::uint64_t n)
{
    // if we know the answer, we return a future encapsulating the final value
    if (n < 2)
        return hpx::make_ready_future(n);
    if (n < threshold)
        return hpx::make_ready_future(fibonacci_serial(n));

    // asynchronously launch the calculation of one of the sub-terms
    // attach a continuation to this future which is called asynchronously on
    // its completion and which calculates the other sub-term
    return hpx::async(&fib, n-1).then(fibonacci_future_one_continuation(n));
}

///////////////////////////////////////////////////////////////////////////////
boost::uint64_t fibonacci(boost::uint64_t n)
{
    // if we know the answer, we return the final value
    if (n < 2)
        return n;
    if (n < threshold)
        return fibonacci_serial(n);

    // asynchronously launch the creation of one of the sub-terms of the
    // execution graph
    hpx::future<boost::uint64_t> f = hpx::async(&fibonacci, n-1);
    boost::uint64_t r = fibonacci(n-2);

    return f.get() + r;
}

///////////////////////////////////////////////////////////////////////////////
boost::uint64_t fibonacci_fork(boost::uint64_t n)
{
    // if we know the answer, we return the final value
    if (n < 2)
        return n;
    if (n < threshold)
        return fibonacci_serial(n);

    // asynchronously launch the creation of one of the sub-terms of the
    // execution graph
    hpx::future<boost::uint64_t> f =
        hpx::async(hpx::launch::fork, &fibonacci_fork, n-1);
    boost::uint64_t r = fibonacci_fork(n-2);

    return f.get() + r;
}

///////////////////////////////////////////////////////////////////////////////
hpx::future<boost::uint64_t> fibonacci_future(boost::uint64_t n)
{
    // if we know the answer, we return a future encapsulating the final value
    if (n < 2)
        return hpx::make_ready_future(n);
    if (n < threshold)
        return hpx::make_ready_future(fibonacci_serial(n));

    // asynchronously launch the creation of one of the sub-terms of the
    // execution graph
    hpx::future<boost::uint64_t> f =
        hpx::async(&fibonacci_future, n-1);
    hpx::future<boost::uint64_t> r = fibonacci_future(n-2);

    return hpx::async(&add, std::move(f), std::move(r));
}

///////////////////////////////////////////////////////////////////////////////
hpx::future<boost::uint64_t> fibonacci_future_fork(boost::uint64_t n)
{
    // if we know the answer, we return a future encapsulating the final value
    if (n < 2)
        return hpx::make_ready_future(n);
    if (n < threshold)
        return hpx::make_ready_future(fibonacci_serial(n));

    // asynchronously launch the creation of one of the sub-terms of the
    // execution graph
    hpx::future<boost::uint64_t> f =
        hpx::async(hpx::launch::fork, &fibonacci_future_fork, n-1);
    hpx::future<boost::uint64_t> r = fibonacci_future_fork(n-2);

    return hpx::async(&add, std::move(f), std::move(r));
}

///////////////////////////////////////////////////////////////////////////////
hpx::future<boost::uint64_t> fibonacci_future_when_all(boost::uint64_t n)
{
    // if we know the answer, we return a future encapsulating the final value
    if (n < 2)
        return hpx::make_ready_future(n);
    if (n < threshold)
        return hpx::make_ready_future(fibonacci_serial(n));

    // asynchronously launch the creation of one of the sub-terms of the
    // execution graph
    hpx::future<hpx::future<boost::uint64_t> > f =
        hpx::async(&fibonacci_future, n-1);
    hpx::future<boost::uint64_t> r = fibonacci_future(n-2);

    return hpx::when_all(f.get(), r).then(when_all_wrapper());
}

hpx::future<boost::uint64_t> fibonacci_future_unwrapped_when_all(boost::uint64_t n)
{
    // if we know the answer, we return a future encapsulating the final value
    if (n < 2)
        return hpx::make_ready_future(n);
    if (n < threshold)
        return hpx::make_ready_future(fibonacci_serial(n));

    // asynchronously launch the creation of one of the sub-terms of the
    // execution graph
    hpx::future<boost::uint64_t> f = hpx::async(&fibonacci_future, n-1);
    hpx::future<boost::uint64_t> r = fibonacci_future(n-2);

    return hpx::when_all(f, r).then(when_all_wrapper());
}

/////////////////////////////////////////////////////////////////////////////
hpx::future<boost::uint64_t> fibonacci_future_all(boost::uint64_t n)
{
    // if we know the answer, we return a future encapsulating the final value
    if (n < 2)
        return hpx::make_ready_future(n);
    if (n < threshold)
        return hpx::make_ready_future(fibonacci_serial(n));

    // asynchronously launch the calculation of both of the sub-terms
    hpx::future<boost::uint64_t> f1 = fibonacci_future_all(n - 1);
    hpx::future<boost::uint64_t> f2 = fibonacci_future_all(n - 2);

    // create a future representing the successful calculation of both sub-terms
    return hpx::async(&add, std::move(f1), std::move(f2));
}

/////////////////////////////////////////////////////////////////////////////
hpx::future<boost::uint64_t> fibonacci_future_all_when_all(boost::uint64_t n)
{
    // if we know the answer, we return a future encapsulating the final value
    if (n < 2)
        return hpx::make_ready_future(n);
    if (n < threshold)
        return hpx::make_ready_future(fibonacci_serial(n));

    // asynchronously launch the calculation of both of the sub-terms
    hpx::future<boost::uint64_t> f1 = fibonacci_future_all(n - 1);
    hpx::future<boost::uint64_t> f2 = fibonacci_future_all(n - 2);

    // create a future representing the successful calculation of both sub-terms
    // attach a continuation to this future which is called asynchronously on
    // its completion and which calculates the the final result
    return hpx::when_all(f1, f2).then(when_all_wrapper());
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
            // Create a Future for the whole calculation, execute it locally,
            // and wait for it.
            r = fibonacci_future_one(n).get();
        }

        boost::uint64_t d = hpx::util::high_resolution_clock::now() - start;
        char const* fmt = "fibonacci_future_one(%1%) == %2%,"
            "elapsed time:,%3%,[s]\n";
        std::cout << (boost::format(fmt) % n % r % (d / max_runs));

        executed_one = true;
    }

    if (test == "all" || test == "2")
    {
        // Keep track of the time required to execute.
        boost::uint64_t start = hpx::util::high_resolution_clock::now();

        for (std::size_t i = 0; i != max_runs; ++i)
        {
            // Create a Future for the whole calculation, execute it locally, and
            // wait for it.
            r = fibonacci(n);
        }

//        double d = double(hpx::util::high_resolution_clock::now() - start) / 1.e9;
        boost::uint64_t d = hpx::util::high_resolution_clock::now() - start;
        char const* fmt = "fibonacci(%1%) == %2%,elapsed time:,%3%,[s]\n";
        std::cout << (boost::format(fmt) % n % r % (d / max_runs));

        executed_one = true;
    }

    if (test == "all" || test == "9")
    {
        // Keep track of the time required to execute.
        boost::uint64_t start = hpx::util::high_resolution_clock::now();

        for (std::size_t i = 0; i != max_runs; ++i)
        {
            // Create a Future for the whole calculation, execute it locally, and
            // wait for it. Use continuation stealing
            r = fibonacci_fork(n);
        }

        boost::uint64_t d = hpx::util::high_resolution_clock::now() - start;
        char const* fmt = "fibonacci_fork(%1%) == %2%,elapsed time:,%3%,[s]\n";
        std::cout << (boost::format(fmt) % n % r % (d / max_runs));

        executed_one = true;
    }

    if (test == "all" || test == "3")
    {
        // Keep track of the time required to execute.
        boost::uint64_t start = hpx::util::high_resolution_clock::now();

        for (std::size_t i = 0; i != max_runs; ++i)
        {
            // Create a Future for the whole calculation, execute it locally, and
            // wait for it.
            r = fibonacci_future(n).get();
        }

        boost::uint64_t d = hpx::util::high_resolution_clock::now() - start;
        char const* fmt = "fibonacci_future(%1%) == %2%,elapsed time:,%3%,[s]\n";
        std::cout << (boost::format(fmt) % n % r % (d / max_runs));

        executed_one = true;
    }

    if (test == "all" || test == "8")
    {
        // Keep track of the time required to execute.
        boost::uint64_t start = hpx::util::high_resolution_clock::now();

        for (std::size_t i = 0; i != max_runs; ++i)
        {
            // Create a Future for the whole calculation, execute it locally, and
            // wait for it. Use continuation stealing.
            r = fibonacci_future_fork(n).get();
        }

        boost::uint64_t d = hpx::util::high_resolution_clock::now() - start;
        char const* fmt = "fibonacci_future_fork(%1%) == %2%,elapsed time:,%3%,[s]\n";
        std::cout << (boost::format(fmt) % n % r % (d / max_runs));

        executed_one = true;
    }

    if (test == "all" || test == "6")
    {
        // Keep track of the time required to execute.
        boost::uint64_t start = hpx::util::high_resolution_clock::now();

        for (std::size_t i = 0; i != max_runs; ++i)
        {
            // Create a Future for the whole calculation, execute it locally, and
            // wait for it.
            r = fibonacci_future_when_all(n).get();
        }

        boost::uint64_t d = hpx::util::high_resolution_clock::now() - start;
        char const* fmt =
            "fibonacci_future_when_all(%1%) == %2%,elapsed time:,%3%,[s]\n";
        std::cout << (boost::format(fmt) % n % r % (d / max_runs));

        executed_one = true;
    }

    if (test == "all" || test == "7")
    {
        // Keep track of the time required to execute.
        boost::uint64_t start = hpx::util::high_resolution_clock::now();

        for (std::size_t i = 0; i != max_runs; ++i)
        {
            // Create a Future for the whole calculation, execute it locally, and
            // wait for it.
            r = fibonacci_future_unwrapped_when_all(n).get();
        }

        boost::uint64_t d = hpx::util::high_resolution_clock::now() - start;
        char const* fmt =
            "fibonacci_future_unwrapped_when_all(%1%) == %2%,elapsed time:,%3%,[s]\n";
        std::cout << (boost::format(fmt) % n % r % (d / max_runs));

        executed_one = true;
    }

    if (test == "all" || test == "4")
    {
        // Keep track of the time required to execute.
        boost::uint64_t start = hpx::util::high_resolution_clock::now();

        for (std::size_t i = 0; i != max_runs; ++i)
        {
            // Create a future for the whole calculation, execute it locally, and
            // wait for it.
            r = fibonacci_future_all(n).get();
        }

        boost::uint64_t d = hpx::util::high_resolution_clock::now() - start;
        char const* fmt =
            "fibonacci_future_all(%1%) == %2%,elapsed time,%3%,[s]\n";
        std::cout << (boost::format(fmt) % n % r % (d / max_runs));

        executed_one = true;
    }

    if (test == "all" || test == "5")
    {
        // Keep track of the time required to execute.
        boost::uint64_t start = hpx::util::high_resolution_clock::now();

        for (std::size_t i = 0; i != max_runs; ++i)
        {
            // Create a Future for the whole calculation, execute it locally, and
            // wait for it.
            r = fibonacci_future_all_when_all(n).get();
        }

        boost::uint64_t d = hpx::util::high_resolution_clock::now() - start;
        char const* fmt =
            "fibonacci_future_all_when_all(%1%) == %2%,elapsed time:,%3%,[s]\n";
        std::cout << (boost::format(fmt) % n % r % (d / max_runs));

        executed_one = true;
    }

    if (!executed_one)
    {
        std::cerr << "fibonacci_futures: wrong command line argument value for "
            "option 'tests', should be either 'all' or a number between zero "
            "and 7, value specified: " << test << std::endl;
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
          "select tests to execute (0-9, default: all)")
    ;

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}
