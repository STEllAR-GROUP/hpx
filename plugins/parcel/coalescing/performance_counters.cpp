//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>

namespace hpx { namespace plugins { namespace parcel
{
    ///////////////////////////////////////////////////////////////////////////
    // Discoverer for the explicit (hand-rolled performance counter. The
    // purpose of this function is to invoke the supplied function f for all
    // allowed counter instance names supported by the counter type this
    // function has been registered with.
    bool counter_discoverer(
        hpx::performance_counters::counter_info const& info,
        hpx::performance_counters::discover_counter_func const& f,
        hpx::performance_counters::discover_counters_mode mode, hpx::error_code& ec)
    {
        return true;    // everything is ok
    }

    ///////////////////////////////////////////////////////////////////////////
    // Creation function for explicit sine performance counter. It's purpose is
    // to create and register a new instance of the given name (or reuse an
    // existing instance).
    hpx::naming::gid_type counter_creator(
        hpx::performance_counters::counter_info const& info, hpx::error_code& ec)
    {
        return hpx::naming::invalid_gid;
    }

    ///////////////////////////////////////////////////////////////////////////
    // This function will be registered as a startup function for HPX below.
    //
    // That means it will be executed in a HPX-thread before hpx_main, but after
    // the runtime has been initialized and started.
    void startup()
    {
        using namespace hpx::performance_counters;

        // define the counter types
        generic_counter_type_data const counter_types[] =
        {
            { "/coalescing/count/parcels", counter_raw,
              "returns the number of parcels handled by the message handler "
              "associated with the action which is given by the counter "
              "parameter",
              HPX_PERFORMANCE_COUNTER_V1,
              // We assume that valid counter names have the following scheme:
              //
              //  /coalescing(locality#<locality_id>/total)/count/parcels@action-name
              //
              // where '<locality_id>' is the number of the locality.
              &counter_creator,
              &counter_discoverer,
              ""
            },
            { "/coalescing/count/messages", counter_raw,
              "returns the number of messages creates as the result of "
              "coalescing parcels of the action which is given by the counter "
              "parameter",
              HPX_PERFORMANCE_COUNTER_V1,
              // We assume that valid counter names have the following scheme:
              //
              //  /coalescing(locality#<locality_id>/total)/count/parcels@action-name
              //
              // where '<locality_id>' is the number of the locality.
              &counter_creator,
              &counter_discoverer,
              ""
            }
        };

        // Install the counter types, un-installation of the types is handled
        // automatically.
        install_counter_types(counter_types,
            sizeof(counter_types)/sizeof(counter_types[0]));
    }

    ///////////////////////////////////////////////////////////////////////////
    bool get_startup(hpx::util::function_nonser<void()>& startup_func,
        bool& pre_startup)
    {
        // return our startup-function if performance counters are required
        startup_func = startup;   // function to run during startup
        pre_startup = true;       // run 'startup' as pre-startup function
        return true;
    }
}}}

///////////////////////////////////////////////////////////////////////////////
// Register a startup function which will be called as a HPX-thread during
// runtime startup. We use this function to register our performance counter
// type and performance counter instances.
//
// Note that this macro can be used not more than once in one module.
HPX_REGISTER_STARTUP_MODULE_DYNAMIC(hpx::plugins::parcel::get_startup);
