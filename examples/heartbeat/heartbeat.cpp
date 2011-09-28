//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/components/plain_component_factory.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/lcos/promise.hpp>
#include <hpx/state.hpp>

#include <boost/bind.hpp>
#include <boost/format.hpp>
#include <boost/cstdint.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

// include Windows specific performance counter binding
#include "win_perf_counters.hpp"

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using boost::posix_time::milliseconds;

using boost::format;
using boost::str;

using hpx::init;
using hpx::disconnect;
using hpx::get_runtime;
using hpx::register_shutdown_function;
using hpx::running;
using hpx::runtime_mode_connect;
using hpx::network_error;

using hpx::applier::get_applier;

using hpx::threads::threadmanager_is;
using hpx::threads::thread_priority_critical;
using hpx::threads::wait_timeout;
using hpx::threads::pending;
using hpx::threads::suspended;
using hpx::threads::get_self_id;
using hpx::threads::get_self;
using hpx::threads::set_thread_state;

using hpx::performance_counters::stubs::performance_counter;
using hpx::performance_counters::counter_value;
using hpx::performance_counters::status_valid_data;

using hpx::naming::resolver_client;
using hpx::naming::gid_type;
using hpx::naming::get_prefix_from_gid;
using hpx::naming::get_agas_client;

using hpx::lcos::promise;
using hpx::lcos::eager_future;
using hpx::lcos::base_lco;

///////////////////////////////////////////////////////////////////////////////
void stop_monitor(
    std::string const& name
) {
    // Kill the monitor.
    resolver_client& agas_client = get_agas_client();
    gid_type gid;
    
    if (!agas_client.queryid(name, gid))
    {
        HPX_THROW_EXCEPTION(network_error, "stop_monitor",
            "couldn't find stop flag");
    } 
    
    BOOST_ASSERT(gid);
    
    eager_future<base_lco::set_event_action> stop_future(gid);
    
    stop_future.get();
}

///////////////////////////////////////////////////////////////////////////////
int monitor(
    std::string const& name
  , boost::uint64_t pause
) {
    // Resolve the GID of the performance counter using it's symbolic name.
    gid_type gid;
    get_agas_client().queryid(name, gid);

    if (!gid)
    {
        std::cout << (format(
            "error: performance counter not found (%s)")
            % name) << std::endl;
        return 1; 
    }

    const boost::uint32_t prefix = get_applier().get_prefix_id();

    if (prefix == get_prefix_from_gid(gid))
    {
        std::cout << (format(
            "error: cannot query performance counters on its own locality (%s)")
            % name) << std::endl;
        return 1;
    } 

    promise<void> stop_flag;
    const std::string stop_flag_name
        = str(format("/stop_flag(locality#%d]/heartbeat)") % prefix);

    // Associate the stop flag with a symbolic name.
    get_agas_client().registerid(stop_flag_name, stop_flag.get_gid().get_gid());

    register_shutdown_function(boost::bind(&stop_monitor, stop_flag_name)); 

    boost::int64_t zero_time = 0;

    while (true) 
    {
        if (!threadmanager_is(running) || stop_flag.ready())
            return 0;

        // Query the performance counter.
        counter_value value = performance_counter::get_value(gid); 

        if (HPX_LIKELY(status_valid_data == value.status_))
        {
            if (!zero_time)
                zero_time = value.time_;

            std::cout << ( format("  %s,%d,%d\n")
                         % name
                         % (value.time_ - zero_time)
                         % value.value_);

#if defined(BOOST_WINDOWS)
            update_windows_counters(value.value_);
#endif
        }

        // Schedule a wakeup.
        set_thread_state(get_self_id(), milliseconds(pause)
                       , pending, wait_timeout, thread_priority_critical);
        
        get_self().yield(suspended);
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    int r = 0;

    {
        std::cout << "starting monitor" << std::endl;

        const std::string name = vm["name"].as<std::string>();
        const boost::uint64_t pause = vm["pause"].as<boost::uint64_t>();

        r = monitor(name, pause);
    }

    disconnect();
    return r;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "name"
        , value<std::string>()->default_value
            ("/counter/queue(locality#0/total)/length")
        , "symbolic name of the performance counter")

        ( "pause"
        , value<boost::uint64_t>()->default_value(500) 
        , "milliseconds between each performance counter query")
        ;

    // Initialize and run HPX, enforce connect mode as we connect to an existing 
    // application.
#if defined(BOOST_WINDOWS)
    return init(desc_commandline, argc, argv, install_windows_counters, 
        uninstall_windows_counters, runtime_mode_connect);
#else
    return init(desc_commandline, argc, argv, runtime_mode_connect);
#endif
}

