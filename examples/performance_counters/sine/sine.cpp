//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>

#include <boost/format.hpp>
#include <boost/foreach.hpp>
#include <boost/chrono/chrono.hpp>

#include "server/sine.hpp"

///////////////////////////////////////////////////////////////////////////////
// Add factory registration functionality
HPX_REGISTER_COMPONENT_MODULE();    // create entry point for component factory

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    ::performance_counters::sine::server::sine_counter
> sine_counter_type;

///////////////////////////////////////////////////////////////////////////////
namespace performance_counters { namespace sine
{
    // This function will be invoked whenever the second counter is queried.
    boost::int64_t immediate_sine()
    {
        using boost::chrono::high_resolution_clock;
        using boost::chrono::duration;

        static high_resolution_clock::time_point started_at =
            high_resolution_clock::now();

        duration<double> up_time = high_resolution_clock::now() - started_at;
        return std::sin(up_time.count() / 10.) * 100000.;
    }

    // This function will be registered as a startup function for HPX below. 
    // 
    // That means it will be executed in a px-thread before hpx_main, but after 
    // the runtime has been initialized and started.
    void startup()
    {
        using namespace hpx::performance_counters;

        // define the counter types
        counter_type_data const counter_types[] = 
        {
            { "/sine/immediate", counter_raw,
              "returns the current value of an sine wave calculated over "
              "an arbitrary timeline", 
              HPX_PERFORMANCE_COUNTER_V1 }
        };

        // install the counter types, un-installation is handled automatically
        install_counter_types(counter_types, 
            sizeof(counter_types)/sizeof(counter_types[0]));

        // create the counter instances

        // The first counter uses our own full counter implementation, we create 
        // the sine_type counter locally and install it to the local counter
        // registry.
        boost::uint32_t const prefix = hpx::applier::get_applier().get_prefix_id();        
        boost::format sine_instance("/sine(locality#%d/instance#%d)/immediate");

        // full info of the counter to create, help text and version will be
        // complemented from counter type info as specified above
        counter_info info(counter_raw, boost::str(sine_instance % prefix % 0));

        // create the 'sine' performance counter component locally
        hpx::naming::id_type id(
            hpx::components::server::create_one<sine_counter_type>(info),
            hpx::naming::id_type::managed);

        // install the created counter, un-installation is automatic
        install_counter(id, info);

        // The second counter is based on the built-in counter type allowing
        // to use a plain function to return the counter values. We do not need
        // to explicitly create the counter instance in this case.
        install_counter(boost::str(sine_instance % prefix % 1), immediate_sine);
    }
}}

///////////////////////////////////////////////////////////////////////////////
// Register a startup function which will be called as a px-thread during 
// runtime startup. We use this function to register our performance counter 
// type and performance counter instances.
// 
// Note that this macro can be used not more than once in one module.
HPX_REGISTER_STARTUP_SHUTDOWN_MODULE(::performance_counters::sine::startup, 0);

