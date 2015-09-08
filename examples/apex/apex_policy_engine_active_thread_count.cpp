////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2014-2015 Oregon University
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// Naive SMP version implemented with futures.

#include <hpx/hpx_init.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/util.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/lcos.hpp>

#include <apex.hpp>
#include <apex_api.hpp>

#include <iostream>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>

///////////////////////////////////////////////////////////////////////////////
// forward declaration of the Fibonacci function
boost::uint64_t fibonacci(boost::uint64_t n);

// This is to generate the required boilerplate we need for the remote
// invocation to work.
HPX_PLAIN_ACTION(fibonacci, fibonacci_action);

hpx::naming::id_type counter_id;

///////////////////////////////////////////////////////////////////////////////
boost::uint64_t fibonacci(boost::uint64_t n)
{
    if (n < 2)
        return n;

    // We restrict ourselves to execute the Fibonacci function locally.
    hpx::naming::id_type const locality_id = hpx::find_here();

    // Invoking the Fibonacci algorithm twice is inefficient.
    // However, we intentionally demonstrate it this way to create some
    // heavy workload.

    fibonacci_action fib;
    hpx::future<boost::uint64_t> n1 =
        hpx::async(fib, locality_id, n - 1);
    hpx::future<boost::uint64_t> n2 =
        hpx::async(fib, locality_id, n - 2);

    return n1.get() + n2.get();   // wait for the Futures to return their values
}

using hpx::naming::id_type;
using hpx::performance_counters::get_counter;
using hpx::performance_counters::stubs::performance_counter;
using hpx::performance_counters::counter_value;
using hpx::performance_counters::status_is_valid;
static bool counters_initialized = false;
static const char * counter_name = "/threadqueue{locality#%d/total}/length";

id_type get_counter_id() {
    // Resolve the GID of the performances counter using it's symbolic name.
    boost::uint32_t const prefix = hpx::get_locality_id();
    /*
    boost::format 
      active_threads("/threads{locality#%d/total}/count/instantaneous/active");
    */
    boost::format active_threads(counter_name);
    id_type id = get_counter(boost::str(active_threads % prefix));
    return id;
}

void setup_counters() {
    try {
        id_type id = get_counter_id();
        // We need to explicitly start all counters before we can use them. For
        // certain counters this could be a no-op, in which case start will
        // return 'false'.
        performance_counter::start(id);
        std::cout << "Counters initialized! " << id << std::endl;
        counter_value value = performance_counter::get_value(id);
        std::cout << "Active threads " << value.get_value<int>() << std::endl;
        counter_id = id;
    }
    catch(hpx::exception const& e) {
        std::cerr << "apex_policy_engine_active_thread_count: caught exception: "
            << e.what() << std::endl;
    }
    counters_initialized = true;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    // extract command line argument, i.e. fib(N)
    boost::uint64_t n = vm["n-value"].as<boost::uint64_t>();

    {
        // Keep track of the time required to execute.
        hpx::util::high_resolution_timer t;

        // Wait for fib() to return the value
        fibonacci_action fib;
        boost::uint64_t r = fib(hpx::find_here(), n);

        char const* fmt = "fibonacci(%1%) == %2%\nelapsed time: %3% [s]\n";
        std::cout << (boost::format(fmt) % n % r % t.elapsed());
    }

    return hpx::finalize(); // Handles HPX shutdown
}

bool test_function(apex_context const& context) {
    if (!counters_initialized) return false;
    try {
        //id_type id = get_counter_id();
        counter_value value1 = performance_counter::get_value(counter_id);
        if (value1.get_value<int>() % 2 == 1) {
          return APEX_NOERROR;
        } else {
          std::cerr << "Expecting an error message..." << std::endl;
          return APEX_ERROR;
        }
    }
    catch(hpx::exception const& e) {
        std::cerr << "apex_policy_engine_active_thread_count: caught exception: "
            << e.what() << std::endl;
        return APEX_ERROR;
    }
}

void register_policies() {
    //std::set<apex::event_type> when = {apex::START_EVENT};
    //apex::register_policy(START_EVENT, test_function);
    apex::register_periodic_policy(1000, test_function);
}

///////////////////////////////////////////////////////////////////////////////
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

    hpx::register_startup_function(&setup_counters);
    hpx::register_startup_function(&register_policies);

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}
