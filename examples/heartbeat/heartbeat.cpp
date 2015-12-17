//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2009-2010 Dylan Stark
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// #define HPX_USE_WINDOWS_PERFORMANCE_COUNTERS 1

#include <hpx/hpx_init.hpp>
#include <hpx/exception.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/state.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/format.hpp>
#include <boost/cstdint.hpp>

// include Windows specific performance counter binding
#if defined(HPX_WINDOWS) && HPX_USE_WINDOWS_PERFORMANCE_COUNTERS != 0
#include "win_perf_counters.hpp"
#endif

///////////////////////////////////////////////////////////////////////////////
void stop_monitor(boost::shared_ptr<hpx::promise<void> > p)
{
    p->set_value();      // Kill the monitor.
}

///////////////////////////////////////////////////////////////////////////////
int monitor(double runfor, std::string const& name, boost::uint64_t pause)
{
#if defined(HPX_WINDOWS) && HPX_USE_WINDOWS_PERFORMANCE_COUNTERS != 0
    hpx::register_shutdown_function(&uninstall_windows_counters);
#endif

    // Resolve the GID of the performance counter using it's symbolic name.
    hpx::naming::id_type id = hpx::performance_counters::get_counter(name);
    if (!id)
    {
        std::cout << (boost::format(
            "error: performance counter not found (%s)")
            % name) << std::endl;
        return 1;
    }

    boost::uint32_t const locality_id = hpx::get_locality_id();
    if (locality_id == hpx::naming::get_locality_id_from_gid(id.get_gid()))
    {
        std::cout << (boost::format(
            "error: cannot query performance counters on its own locality (%s)")
            % name) << std::endl;
        return 1;
    }

    boost::shared_ptr<hpx::promise<void> > stop_flag =
        boost::make_shared<hpx::promise<void> >();
    hpx::future<void> f = stop_flag->get_future();

    hpx::register_shutdown_function(
        hpx::util::bind(&stop_monitor, stop_flag));

    boost::int64_t zero_time = 0;

    hpx::util::high_resolution_timer t;
    while (runfor < 0 || t.elapsed() < runfor)
    {
        // stop collecting data when the runtime is exiting
        if (!hpx::is_running() || f.is_ready())
            return 0;

        // Query the performance counter.
        using namespace hpx::performance_counters;
        counter_value value = stubs::performance_counter::get_value(id);

        if (status_is_valid(value.status_))
        {
            if (!zero_time)
                zero_time = value.time_;

            std::cout << (boost::format("  %s,%d,%d[s],%d\n")
                         % name
                         % value.count_
                         % double((value.time_ - zero_time) * 1e-9)
                         % value.value_);

#if defined(HPX_WINDOWS) && HPX_USE_WINDOWS_PERFORMANCE_COUNTERS != 0
            update_windows_counters(value.value_);
#endif
        }

        // Schedule a wakeup.
        hpx::this_thread::suspend(pause);
    }

    return hpx::disconnect();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    std::cout << "starting monitor" << std::endl;

    std::string const name = vm["name"].as<std::string>();
    boost::uint64_t const pause = vm["pause"].as<boost::uint64_t>();
    double const runfor = vm["runfor"].as<double>();

    return monitor(runfor, name, pause);
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    boost::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    using boost::program_options::value;
    desc_commandline.add_options()
        ( "name", value<std::string>()->default_value(
              "/threadqueue{locality#0/total}/length")
        , "symbolic name of the performance counter")

        ( "pause", value<boost::uint64_t>()->default_value(500)
        , "milliseconds between each performance counter query")

        ( "runfor", value<double>()->default_value(-1)
        , "time to wait before this application exits ([s], default: run forever)")
        ;

#if defined(HPX_WINDOWS) && HPX_USE_WINDOWS_PERFORMANCE_COUNTERS != 0
    hpx::register_startup_function(&install_windows_counters);
#endif

    // Initialize and run HPX, enforce connect mode as we connect to an existing
    // application.
    std::vector<std::string> cfg;
    cfg.push_back("hpx.run_hpx_main!=1");

    hpx::util::function_nonser<void()> const empty;
    return hpx::init(desc_commandline, argc, argv, cfg, empty,
        empty, hpx::runtime_mode_connect);
}

