//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/util/parse_command_line.hpp>

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
        return boost::int64_t(std::sin(up_time.count() / 10.) * 100000.);
    }

    // create an averaging performance counter based on the immediate sine
    // counter
    void create_averaging_sine()
    {
        // First, register the counter type
        hpx::performance_counters::install_counter_type(
            "/sine/average", hpx::performance_counters::counter_average_count,
            "returns the averaged value of a sine wave calculated over "
            "an arbitrary time line");

        // Second, create and register the counter instance
        boost::uint32_t const prefix = hpx::applier::get_applier().get_prefix_id()-1;
        boost::format sine_instance("/sine(locality#%d/instance#0)/average");

        // full info of the counter to create, help text and version will be
        // complemented from counter type info as specified above
        hpx::performance_counters::counter_info info(
            hpx::performance_counters::counter_average_count,
            boost::str(sine_instance % prefix));

        // create the 'sine' performance counter component locally
        boost::format base_instance("/sine(locality#%d/instance#0)/immediate");
        hpx::naming::id_type id =
            hpx::performance_counters::create_average_count_counter(info,
                boost::str(base_instance % prefix), 100);

        // install the counter instance
        hpx::performance_counters::install_counter(id, info);
    }


    ///////////////////////////////////////////////////////////////////////////
    // This will be called to return special command line options supported by
    // this component.
    boost::program_options::options_description command_line_options()
    {
        boost::program_options::options_description opts(
            "Additional command line options for the sine component");
        opts.add_options()
            ("sine", "enables the performance counters implemented by the "
                "sine component")
            ;
        return opts;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Parse the command line to figure out whether the sine performance
    // counters need to be created.
    bool need_perf_counters()
    {
        using boost::program_options::options_description;
        using boost::program_options::variables_map;
        using hpx::util::retrieve_commandline_arguments;

        // Retrieve command line using the Boost.ProgramOptions library.
        variables_map vm;
        if (!retrieve_commandline_arguments(command_line_options(), vm))
        {
            HPX_THROW_EXCEPTION(hpx::not_implemented,
                "sine::startup", "Failed to handle command line options");
            return false;
        }

        // We enable the performance counters if --sine is specified on the
        // command line.
        return (vm.count("sine") != 0) ? true : false;
    }

    ///////////////////////////////////////////////////////////////////////////
    // This function will be registered as a startup function for HPX below.
    //
    // That means it will be executed in a px-thread before hpx_main, but after
    // the runtime has been initialized and started.
    void startup()
    {
        using namespace hpx::performance_counters;

        // define the counter types
        raw_counter_type_data const counter_types[] =
        {
            { "/sine/immediate", counter_raw,
              "returns the current value of a sine wave calculated over "
              "an arbitrary time line",
              HPX_PERFORMANCE_COUNTER_V1 }
        };

        // install the counter types, un-installation is handled automatically
        install_counter_types(counter_types,
            sizeof(counter_types)/sizeof(counter_types[0]));

        // create the counter instances

        // The first counter uses our own full counter implementation, we create
        // the sine_type counter locally and install it to the local counter
        // registry.
        boost::uint32_t const prefix = hpx::applier::get_applier().get_prefix_id()-1;
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

        // The third counter is an averaging performance counter based on the
        // first counter above
        create_averaging_sine();
    }

    ///////////////////////////////////////////////////////////////////////////
    bool get_startup(HPX_STD_FUNCTION<void()>& startup_func)
    {
        // check whether the performance counters need to be enabled
        if (!need_perf_counters()) {
            HPX_THROW_EXCEPTION(hpx::not_implemented, "sine::startup",
                "The Sine component is not enabled on the commandline "
                "(--sine), bailing out");
            return false;
        }

        // return our startup-function if performance counters are required
        startup_func = startup;
        return true;
    }
}}

///////////////////////////////////////////////////////////////////////////////
// Register a startup function which will be called as a px-thread during
// runtime startup. We use this function to register our performance counter
// type and performance counter instances.
//
// Note that this macro can be used not more than once in one module.
HPX_REGISTER_STARTUP_MODULE(::performance_counters::sine::get_startup);

///////////////////////////////////////////////////////////////////////////////
// Register a function to be called to populate the special command line
// options supported by this component.
//
// Note that this macro can be used not more than once in one module.
HPX_REGISTER_COMMANDLINE_MODULE(
    ::performance_counters::sine::command_line_options);

