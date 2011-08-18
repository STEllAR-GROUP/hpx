//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2009-2010 Dylan Stark
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/state.hpp>

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

using hpx::init;
using hpx::finalize;
using hpx::running;
using hpx::runtime_mode_connect;

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

using hpx::naming::gid_type;

///////////////////////////////////////////////////////////////////////////////
void monitor(
    std::string const& name
  , boost::uint64_t pause
) {
    // Resolve the GID of the performance counter using it's symbolic name.
    gid_type gid;
    get_applier().get_agas_client().queryid(name, gid);

    BOOST_ASSERT(gid); 

    boost::int64_t zero_time = 0;

    while (true) 
    {
        if (!threadmanager_is(running))
            return;

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
    {
        std::cout << "starting monitor" << std::endl;

        const std::string name = vm["name"].as<std::string>();
        const boost::uint64_t pause = vm["pause"].as<boost::uint64_t>();

        monitor(name, pause);
    }

    finalize();
    return 0;
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
            ("/pcs/queue([L1]/threadmanager)/length")
        , "symbolic name of the performance counter")

        ( "pause"
        , value<boost::uint64_t>()->default_value(100) 
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

