//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This is a purely local version demonstrating how to use continuations for
// futures.

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

hpx::lcos::future<boost::uint64_t> fibonacci_future_one(boost::uint64_t n)
{
    // if we know the answer, we return a future encapsulating the final value
    if (n < 2)
        return hpx::lcos::create_value(n);

    using hpx::lcos::future;
    using hpx::async;

    // asynchronously launch the calculation of one of the sub-terms
    future<boost::uint64_t> f = fibonacci_future_one(n-1);

    // attach a continuation to this future which is called asynchronously on
    // its completion and which calculates the other sub-term
    return f.when(fibonacci_future_one_continuation(n));
}

///////////////////////////////////////////////////////////////////////////////
struct fibonacci_future_all_continuation
{
    typedef hpx::lcos::future<
        std::vector<hpx::lcos::future<boost::uint64_t> >
    > argument_type;

    // we return the result of adding the values calculated by the two sub-terms
    typedef boost::uint64_t result_type;

    result_type operator()(argument_type res) const
    {
        std::vector<hpx::lcos::future<boost::uint64_t> > v = res.get();
        return v[0].get() + v[1].get();
    }
};

hpx::lcos::future<boost::uint64_t> fibonacci_future_all(boost::uint64_t n)
{
    // if we know the answer, we return a future encapsulating the final value
    if (n < 2)
        return hpx::lcos::create_value(n);

    using hpx::lcos::future;
    using hpx::async;

    // asynchronously launch the calculation of both of the sub-terms
    future<boost::uint64_t> f1 = fibonacci_future_all(n - 1);
    future<boost::uint64_t> f2 = fibonacci_future_all(n - 2);

    // create a future representing the successful calculation of both sub-terms
    // attach a continuation to this future which is called asynchronously on
    // its completion and which calculates the the final result
    return hpx::wait_all(f1, f2).when(fibonacci_future_all_continuation());
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    // extract command line argument, i.e. fib(N)
    boost::uint64_t n = vm["n-value"].as<boost::uint64_t>();

    {
        // Keep track of the time required to execute.
        hpx::util::high_resolution_timer t;

        // Create a Future for the whole calculation, execute it locally, and
        // wait for it.
        boost::uint64_t r = fibonacci_future_one(n).get();

        char const* fmt =
            "fibonacci_future_one(%1%) == %2%\nelapsed time: %3% [s]\n";
        std::cout << (boost::format(fmt) % n % r % t.elapsed());
    }

    {
        // Keep track of the time required to execute.
        hpx::util::high_resolution_timer t;

        // Create a Future for the whole calculation, execute it locally, and
        // wait for it.
        boost::uint64_t r = fibonacci_future_all(n).get();

        char const* fmt =
            "fibonacci_future_all(%1%) == %2%\nelapsed time: %3% [s]\n";
        std::cout << (boost::format(fmt) % n % r % t.elapsed());
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
    ;

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}
