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

#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include <iostream>
#include <string>

///////////////////////////////////////////////////////////////////////////////
boost::uint64_t threshold = 2;

///////////////////////////////////////////////////////////////////////////////
HPX_NOINLINE boost::uint64_t fibonacci_serial(boost::uint64_t n)
{
    if (n < 2)
        return n;
    return fibonacci_serial(n-1) + fibonacci_serial(n-2);
}

///////////////////////////////////////////////////////////////////////////////
//
// hpx::future<boost::uint64_t> fibonacci(boost::uint64_t) resumable
// {
//     if (n < 2) return hpx::make_ready_future(n);
//     if (n < threshold) return hpx::make_ready_future(fibonacci_serial(n));
//
//     hpx::future<boost::uint64_t> lhs = hpx::async(&fibonacci, n-1);
//     hpx::future<boost::uint64_t> rhs = fibonacci(n-2);
//
//     return await lhs + await rhs;
// }
//

hpx::future<boost::uint64_t> fibonacci(boost::uint64_t n);

struct _fibonacci_frame
{
    int state_;
    hpx::future<boost::uint64_t> result_;
    hpx::lcos::local::promise<boost::uint64_t> result_promise_;

    _fibonacci_frame(boost::uint64_t n)
      : state_(0),
        n_(n), lhs_result_(0), rhs_result_(0)
    {}

    // local variables
    boost::uint64_t n_;
    hpx::future<boost::uint64_t> lhs_;
    hpx::future<boost::uint64_t> rhs_;
    boost::uint64_t lhs_result_;
    boost::uint64_t rhs_result_;
};

void _fibonacci(boost::shared_ptr<_fibonacci_frame> const& frame_)
{
    _fibonacci_frame* frame = frame_.get();
    int state = frame->state_;

    switch (state)
    {
    case 1:
        goto L1;
    case 2:
        goto L2;
    }

    // if (n < 2) return hpx::make_ready_future(n);
    if (frame->n_ < 2)
    {
        if (state == 0)
            // never paused
            frame->result_ = hpx::make_ready_future(frame->n_);
        else
            frame->result_promise_.set_value(frame->n_);
        return;
    }

    // if (n < threshold) return hpx::make_ready_future(fibonacci_serial(n));
    if (frame->n_ < threshold)
    {
        if (state == 0)
            // never paused
            frame->result_ = hpx::make_ready_future(fibonacci_serial(frame->n_));
        else
            frame->result_promise_.set_value(fibonacci_serial(frame->n_));
        return;
    }

    // hpx::future<boost::uint64_t> lhs = hpx::async(&fibonacci, n-1);
    frame->lhs_ = hpx::async(&fibonacci, frame->n_-1);

    // hpx::future<boost::uint64_t> rhs = fibonacci(n-2);
    frame->rhs_ = fibonacci(frame->n_-2);

    if (!frame->lhs_.is_ready())
    {
        frame->state_ = 1;
        if (!frame->result_.valid())
            frame->result_ = frame->result_promise_.get_future();
        frame->lhs_.then(hpx::util::bind(&_fibonacci, frame_));
        return;
    }

L1:
    frame->lhs_result_ = frame->lhs_.get();

    if ( !frame->rhs_.is_ready())
    {
        frame->state_ = 2;
        if (!frame->result_.valid())
            frame->result_ = frame->result_promise_.get_future();
        frame->rhs_.then(hpx::util::bind(&_fibonacci, frame_));
        return;
    }

L2:
    frame->rhs_result_ = frame->rhs_.get();

    if (state == 0)
        // never paused
        frame->result_ = hpx::make_ready_future(frame->lhs_result_ + frame->rhs_result_);
    else
        frame->result_promise_.set_value(frame->lhs_result_ + frame->rhs_result_);
    return;
}

hpx::future<boost::uint64_t> fibonacci(boost::uint64_t n)
{
    boost::shared_ptr<_fibonacci_frame> frame =
        boost::make_shared<_fibonacci_frame>(n);

    _fibonacci(frame);

    return std::move(frame->result_);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    // extract command line argument, i.e. fib(N)
    boost::uint64_t n = vm["n-value"].as<boost::uint64_t>();
    std::string test = vm["test"].as<std::string>();
    boost::uint64_t max_runs = vm["n-runs"].as<boost::uint64_t>();

    if (max_runs == 0) {
        std::cerr << "fibonacci_await: wrong command line argument value for "
            "option 'n-runs', should not be zero" << std::endl;
        return hpx::finalize(); // Handles HPX shutdown
    }

    threshold = vm["threshold"].as<unsigned int>();
    if (threshold < 2 || threshold > n) {
        std::cerr << "fibonacci_await: wrong command line argument value for "
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

        for (std::size_t i = 0; i != max_runs; ++i)
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
        std::cerr << "fibonacci_await: wrong command line argument value for "
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
