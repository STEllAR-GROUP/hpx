//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <boost/program_options.hpp>
#include <boost/format.hpp>
#include <boost/date_time.hpp>

///////////////////////////////////////////////////////////////////////////////
// This example demonstrates the creation and use of different types of
// performance counters. It utilizes the sine component, which implements two
// of the demonstrated counters.
//
// The example prints the values of the counters after every fixed amount of
// time. This time is configurable by a command line option --pause=N, where
// N is the amount of milliseconds to wait in between the sample points. The
// default value is 500ms.
//
// Here is a short description of each of the counters:
//
// /sine{locality#0/instance#0}/immediate/explicit
//    This is a custom performance counter fully implemented in the sine
//    component. It evaluates a new value every 1000ms (1s) and delivers this
//    value whenever it is queried.
//
// /sine{locality#0/total}/immediate/implicit
//    This is an immediate counter implemented by registering a function with
//    the counter framework. This function will be called whenever the counter
//    value is queried. It calculates the current value of a sine based on the
//    uptime of the counter.
//
// /statistics{/sine{locality#0/instance#1}/immediate/explicit}/average#100
//    This is an aggregating counter calculating the average value of a given
//    base counter (in this case /sine{locality#0/instance#1}/immediate/explicit,
//    i.e. a second instance of the explicit counter). The base counter is
//    evaluated every 100ms (as specified by the trailing parameter in the
//    counter name). No special code in the sine example is needed for this
//    counter as it reuses the explicit counter and the predefined averaging
//    performance counter implemented in HPX.
//
// Additionally, this example demonstrates starting and stopping performance
// counters. It stops the evaluation of the first explicit counter instance
// every 5 seconds, restarting it after a while.

///////////////////////////////////////////////////////////////////////////////
int monitor(boost::uint64_t pause, boost::uint64_t values)
{
    // Resolve the GID of the performances counter using it's symbolic name.
    boost::uint32_t const prefix = hpx::get_locality_id();
    boost::format sine_explicit("/sine{locality#%d/instance#%d}/immediate/explicit");
    boost::format sine_implicit("/sine{locality#%d/total}/immediate/implicit");
    boost::format sine_average("/statistics{/sine{locality#%d/instance#%d}/immediate/explicit}/average@100");

    using hpx::naming::id_type;
    using hpx::performance_counters::get_counter;

    id_type id1 = get_counter(boost::str(sine_explicit % prefix % 0));
    id_type id2 = get_counter(boost::str(sine_implicit % prefix));
    id_type id3 = get_counter(boost::str(sine_average % prefix % 1));

    using hpx::performance_counters::stubs::performance_counter;

    // We need to explicitly start all counters before we can use them. For
    // certain counters this could be a no-op, in which case start will return
    // 'false'.
    performance_counter::start(id1);
    performance_counter::start(id2);
    performance_counter::start(id3);

    // retrieve the counter values
    boost::uint64_t start_time = 0;
    bool started = true;
    while (values-- > 0)
    {
        // Query the performance counter.
        using hpx::performance_counters::counter_value;
        using hpx::performance_counters::status_is_valid;

        counter_value value1 = performance_counter::get_value(id1);
        counter_value value2 = performance_counter::get_value(id2);
        counter_value value3 = performance_counter::get_value(id3);
        if (status_is_valid(value1.status_))
        {
            if (!start_time)
                start_time = value2.time_;

            std::cout << (boost::format("%.3f: %.4f, %.4f, %.4f\n") %
                ((value2.time_ - start_time) * 1e-9) %
                value1.get_value<double>() %
                (value2.get_value<double>() / 100000.) %
                value3.get_value<double>());
        }

        // stop/restart the counter referenced by id1 after 5 seconds of
        // evaluation
        bool should_run =
            (int((value2.time_ - start_time) * 1e-9) / 5) % 2 != 0;
        if (should_run == started) {
            if (started) {
                performance_counter::stop(id1);
                started = false;
            }
            else {
                performance_counter::start(id1);
                started = true;
            }
        }

        // give up control to the thread manager, we will be resumed after
        // 'pause' ms
        hpx::this_thread::suspend(boost::chrono::milliseconds(pause));
    }
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    // retrieve the command line arguments
    boost::uint64_t const pause = vm["pause"].as<boost::uint64_t>();
    boost::uint64_t const values = vm["values"].as<boost::uint64_t>();

    // do main work, i.e. query the performance counters
    std::cout << "starting sine monitoring..." << std::endl;

    int result = 0;
    try {
        result = monitor(pause, values);
    }
    catch(hpx::exception const& e) {
        std::cerr << "sine_client: caught exception: " << e.what() << std::endl;
        std::cerr << "Have you specified the command line option "
                     "--sine to enable the sine component?"
                  << std::endl;
    }

    // Initiate shutdown of the runtime system.
    hpx::finalize();
    return result;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    using boost::program_options::options_description;
    using boost::program_options::value;

    // Configure application-specific options.
    options_description desc_commandline("usage: sine_client [options]");
    desc_commandline.add_options()
            ("pause", value<boost::uint64_t>()->default_value(500),
             "milliseconds between each performance counter query")
            ("values", value<boost::uint64_t>()->default_value(100),
             "number of performance counter queries to perform")
        ;

    // Initialize and run HPX.
    return hpx::init(desc_commandline, argc, argv);
}

