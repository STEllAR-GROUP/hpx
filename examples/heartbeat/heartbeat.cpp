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
#include <hpx/lcos/future.hpp>
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

using hpx::running;

using hpx::threads::threadmanager_is;

using hpx::performance_counters::stubs::performance_counter;
using hpx::performance_counters::counter_value;
using hpx::performance_counters::status_is_valid;

///////////////////////////////////////////////////////////////////////////////
void stop_monitor(hpx::promise<void> p)
{
    p.set_value();      // Kill the monitor.
}

///////////////////////////////////////////////////////////////////////////////
int monitor(std::string const& name, boost::uint64_t pause)
{
#if defined(BOOST_WINDOWS)
    hpx::register_shutdown_function(&uninstall_windows_counters);
#endif

    // Resolve the GID of the performance counter using it's symbolic name.
    hpx::naming::id_type id = hpx::performance_counters::get_counter(name);
    if (!id)
    {
        std::cout << (format(
            "error: performance counter not found (%s)")
            % name) << std::endl;
        return 1;
    }

    boost::uint32_t const locality_id = hpx::get_locality_id();
    if (locality_id == hpx::naming::get_locality_id_from_gid(id.get_gid()))
    {
        std::cout << (format(
            "error: cannot query performance counters on its own locality (%s)")
            % name) << std::endl;
        return 1;
    }

    hpx::promise<void> stop_flag;
    hpx::register_shutdown_function(boost::bind(&stop_monitor, stop_flag));

    boost::int64_t zero_time = 0;
    hpx::future<void> f = stop_flag.get_future();

    while (true)
    {
        // stop collecting data when the runtime is exiting
        if (!hpx::is_running() || f.is_ready())
            return 0;

        // Query the performance counter.
        counter_value value = performance_counter::get_value(id);

        if (HPX_LIKELY(status_is_valid(value.status_)))
        {
            if (!zero_time)
                zero_time = value.time_;

            std::cout << ( format("  %s,%d[s],%d\n")
                         % name
                         % double((value.time_ - zero_time) * 1e-9)
                         % value.value_);

#if defined(BOOST_WINDOWS)
            update_windows_counters(value.value_);
#endif
        }

        // Schedule a wakeup.
        hpx::this_thread::suspend(milliseconds(pause));
    }

    return hpx::disconnect();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    std::cout << "starting monitor" << std::endl;

    const std::string name = vm["name"].as<std::string>();
    const boost::uint64_t pause = vm["pause"].as<boost::uint64_t>();

    return monitor(name, pause);
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ( "name", value<std::string>()->default_value(
              "/threadqueue{locality#0/total}/length")
        , "symbolic name of the performance counter")

        ( "pause", value<boost::uint64_t>()->default_value(500)
        , "milliseconds between each performance counter query")
        ;

#if defined(BOOST_WINDOWS)
    hpx::register_startup_function(&install_windows_counters);
#endif

    // Initialize and run HPX, enforce connect mode as we connect to an existing
    // application.
    return hpx::init(desc_commandline, argc, argv, hpx::runtime_mode_connect);
}

