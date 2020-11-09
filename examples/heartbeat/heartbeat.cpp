//  Copyright (c)      2011 Bryce Lelbach
//  Copyright (c) 2009-2010 Dylan Stark
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// #define HPX_USE_WINDOWS_PERFORMANCE_COUNTERS 1

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx_init.hpp>
#include <hpx/chrono.hpp>
#include <hpx/exception.hpp>
#include <hpx/future.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/include/threadmanager.hpp>
#include <hpx/modules/actions_base.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/state.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// include Windows specific performance counter binding
#if defined(HPX_WINDOWS) && HPX_USE_WINDOWS_PERFORMANCE_COUNTERS != 0
#include "win_perf_counters.hpp"
#endif

///////////////////////////////////////////////////////////////////////////////
void stop_monitor(std::shared_ptr<hpx::lcos::promise<void> > p)
{
    p->set_value();      // Kill the monitor.
}

///////////////////////////////////////////////////////////////////////////////
int monitor(double runfor, std::string const& name, std::uint64_t pause)
{
#if defined(HPX_WINDOWS) && HPX_USE_WINDOWS_PERFORMANCE_COUNTERS != 0
    hpx::register_shutdown_function(&uninstall_windows_counters);
#endif

    // Resolve the GID of the performance counter using it's symbolic name.
    hpx::performance_counters::performance_counter c(name);
    if (!c.get_id())
    {
        hpx::util::format_to(std::cout,
            "error: performance counter not found ({})",
            name) << std::endl;
        return 1;
    }

    std::uint32_t const locality_id = hpx::get_locality_id();
    if (locality_id == hpx::naming::get_locality_id_from_id(c.get_id()))
    {
        hpx::util::format_to(std::cout,
            "error: cannot query performance counters on its own locality ({})",
            name) << std::endl;
        return 1;
    }

    std::shared_ptr<hpx::lcos::promise<void> > stop_flag =
        std::make_shared<hpx::lcos::promise<void> >();
    hpx::future<void> f = stop_flag->get_future();

    hpx::register_shutdown_function(
        hpx::util::bind(&stop_monitor, stop_flag));

    std::int64_t zero_time = 0;

    hpx::chrono::high_resolution_timer t;
    while (runfor < 0 || t.elapsed() < runfor)
    {
        // stop collecting data when the runtime is exiting
        if (!hpx::is_running() || f.is_ready())
            return 0;

        // Query the performance counter.
        using namespace hpx::performance_counters;
        counter_value value = c.get_counter_value(hpx::launch::sync);

        if (status_is_valid(value.status_))
        {
            if (!zero_time)
                zero_time = value.time_;

            hpx::util::format_to(std::cout, "  {},{},{}[s],{}\n", name,
                value.count_,
                double(static_cast<double>(value.time_ - zero_time) * 1e-9),
                value.value_);

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
int hpx_main(hpx::program_options::variables_map& vm)
{
    std::cout << "starting monitor" << std::endl;

    std::string const name = vm["name"].as<std::string>();
    std::uint64_t const pause = vm["pause"].as<std::uint64_t>();
    double const runfor = vm["runfor"].as<double>();

    return monitor(runfor, name, pause);
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    hpx::program_options::options_description
       desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

    using hpx::program_options::value;
    desc_commandline.add_options()
        ( "name", value<std::string>()->default_value(
              "/threadqueue{locality#0/total}/length")
        , "symbolic name of the performance counter")

        ( "pause", value<std::uint64_t>()->default_value(500)
        , "milliseconds between each performance counter query")

        ( "runfor", value<double>()->default_value(-1)
        , "time to wait before this application exits ([s], default: run forever)")
        ;

#if defined(HPX_WINDOWS) && HPX_USE_WINDOWS_PERFORMANCE_COUNTERS != 0
    hpx::register_startup_function(&install_windows_counters);
#endif

    // Initialize and run HPX, enforce connect mode as we connect to an existing
    // application.
    std::vector<std::string> const cfg = {
        "hpx.run_hpx_main!=1"
    };

    hpx::init_params init_args;
    init_args.mode = hpx::runtime_mode::connect;
    init_args.cfg = cfg;
    init_args.desc_cmdline = desc_commandline;

    return hpx::init(argc, argv, init_args);
}

#endif
