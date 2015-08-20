//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// The purpose of this example is to demonstrate the simplest way to create and
// use a performance counter for HPX.

#include <hpx/hpx_init.hpp>
#include <hpx/include/performance_counters.hpp>

#include <boost/atomic.hpp>

///////////////////////////////////////////////////////////////////////////////
// The atomic variable 'counter' ensures the thread safety of the counter.
boost::atomic<boost::int64_t> counter(0);

boost::int64_t some_performance_data(bool reset)
{
    return hpx::util::get_and_reset_value(counter, reset);
}

void register_counter_type()
{
    // Call the HPX API function to register the counter type.
    hpx::performance_counters::install_counter_type(
        "/test/data",
        // counter type name
        &some_performance_data,
        // function providing counter data
        "returns a linearly increasing counter value"
        // description text
    );
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    {
        // Now it is possible to instantiate a new counter instance based on
        // the naming scheme "/test{locality#0/total}/data". Try invoking this
        // example using the command line option:
        //
        //    --hpx:print-counter=/test{locality#0/total}/data
        //
        // This will print something like:
        //
        //    test{locality#0/total}/data,1,1.005240[s],1
        //
        // where the first value is a sequence counter of the invocation of the
        // counter. The next is the timestamp marking the point in time since
        // application startup at which this counter has been queried. The last
        // value printed is the counter value retrieved at the listed point in
        // time.

        // By invoking this example with the command line options:
        //
        //    --hpx:print-counter=/test{locality#0/total}/data
        //    --hpx:print-counter-interval=100
        //
        // the counter will be queried periodically and the output will look
        // like:
        //
        //    test{locality#0/total}/data,1,0.001937[s],1
        //    test{locality#0/total}/data,2,0.109625[s],2
        //    test{locality#0/total}/data,3,0.217192[s],3
        //    test{locality#0/total}/data,4,0.323497[s],4
        //    test{locality#0/total}/data,5,0.430867[s],5
        //    test{locality#0/total}/data,6,0.536965[s],6
        //    test{locality#0/total}/data,7,0.643422[s],7
        //    test{locality#0/total}/data,8,0.750788[s],8
        //    test{locality#0/total}/data,9,0.857031[s],9
        //    test{locality#0/total}/data,10,0.963330[s],10
        //    test{locality#0/total}/data,11,1.015063[s],11
        //
        // which shows that the counter has been queried roughly every 100
        // milliseconds, as specified. Additionally it will be queried during
        // application shutdown, as before.

        hpx::this_thread::suspend(1000);   // wait for one second
    }
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // By registering the counter type we make it available to any consumer
    // creating and querying an instance of the type "/test/data".
    //
    // This registration should be performed during startup. We register the
    // function 'register_counter_type' to be executed as an HPX thread right
    // before hpx_main will be executed.
    hpx::register_startup_function(&register_counter_type);

    // Initialize and run HPX.
    return hpx::init(argc, argv);
}

