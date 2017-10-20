////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/runtime.hpp>

#include <hpx/error_code.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/lcos/barrier.hpp>
#include <hpx/lcos/detail/barrier_node.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/runtime_support.hpp>
#include <hpx/runtime/find_localities.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/shutdown_function.hpp>
#include <hpx/runtime/startup_function.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/tuple.hpp>

#include <cstddef>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
static void garbage_collect_non_blocking()
{
    hpx::agas::garbage_collect_non_blocking();
}
static void garbage_collect()
{
    hpx::agas::garbage_collect();
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{

///////////////////////////////////////////////////////////////////////////////
// Install performance counter startup functions for core subsystems.
static void register_counter_types()
{
    naming::get_agas_client().register_counter_types();
    lbt_ << "(2nd stage) pre_main: registered AGAS client-side "
            "performance counter types";

    get_runtime().register_counter_types();
    lbt_ << "(2nd stage) pre_main: registered runtime performance "
            "counter types";

    threads::get_thread_manager().register_counter_types();
    lbt_ << "(2nd stage) pre_main: registered thread-manager performance "
            "counter types";

    applier::get_applier().get_parcel_handler().register_counter_types();
    lbt_ << "(2nd stage) pre_main: registered parcelset performance "
            "counter types";
}

///////////////////////////////////////////////////////////////////////////////
extern std::vector<util::tuple<char const*, char const*> >
    message_handler_registrations;

static void register_message_handlers()
{
    runtime& rt = get_runtime();
    for (auto const& t : message_handler_registrations)
    {
        error_code ec(lightweight);
        rt.register_message_handler(util::get<0>(t), util::get<1>(t), ec);
    }
    lbt_ << "(3rd stage) pre_main: registered message handlers";
}

///////////////////////////////////////////////////////////////////////////////
// Implements second and third stage bootstrapping.
int pre_main(runtime_mode mode);
int pre_main(runtime_mode mode)
{
    // Register pre-shutdown and shutdown functions to flush pending
    // reference counting operations.
    register_pre_shutdown_function(&::garbage_collect_non_blocking);
    register_shutdown_function(&::garbage_collect);

    using components::stubs::runtime_support;

    naming::resolver_client& agas_client = naming::get_agas_client();
    runtime& rt = get_runtime();

    int exit_code = 0;
    if (runtime_mode_connect == mode)
    {
        lbt_ << "(2nd stage) pre_main: locality is in connect mode, "
                "skipping 2nd and 3rd stage startup synchronization";
        lbt_ << "(2nd stage) pre_main: addressing services enabled";

        // Load components, so that we can use the barrier LCO.
        exit_code = runtime_support::load_components(find_here());
        lbt_ << "(2nd stage) pre_main: loaded components"
            << (exit_code ? ", application exit has been requested" : "");

        // Work on registration requests for message handler plugins
        register_message_handlers();

        // Register all counter types before the startup functions are being
        // executed.
        register_counter_types();

        rt.set_state(state_pre_startup);
        runtime_support::call_startup_functions(find_here(), true);
        lbt_ << "(3rd stage) pre_main: ran pre-startup functions";

        rt.set_state(state_startup);
        runtime_support::call_startup_functions(find_here(), false);
        lbt_ << "(4th stage) pre_main: ran startup functions";
    }
    else
    {
        lbt_ << "(2nd stage) pre_main: addressing services enabled";

        // Load components, so that we can use the barrier LCO.
        exit_code = runtime_support::load_components(find_here());
        lbt_ << "(2nd stage) pre_main: loaded components"
            << (exit_code ? ", application exit has been requested" : "");

        // {{{ Second and third stage barrier creation.
        if (agas_client.is_bootstrap())
        {
            naming::gid_type console_;
            if (HPX_UNLIKELY(!agas_client.get_console_locality(console_)))
            {
                HPX_THROW_EXCEPTION(network_error
                    , "pre_main"
                    , "no console locality registered");
            }

            lbt_ << "(2nd stage) pre_main: creating 2nd and 3rd stage boot barriers";
        }
        else // Hosted.
        {
            lbt_ << "(2nd stage) pre_main: finding 2nd and 3rd stage boot barriers";
        }
        // }}}

        // create our global barrier...
        hpx::lcos::barrier::get_global_barrier() =
            hpx::lcos::barrier::create_global_barrier();

        // Second stage bootstrap synchronizes component loading across all
        // localities, ensuring that the component namespace tables are fully
        // populated before user code is executed.
        lcos::barrier::synchronize();
        lbt_ << "(2nd stage) pre_main: passed 2nd stage boot barrier";

        // Work on registration requests for message handler plugins
        register_message_handlers();

        // Register all counter types before the startup functions are being
        // executed.
        register_counter_types();

        // Second stage bootstrap synchronizes performance counter loading
        // across all localities.
        lcos::barrier::synchronize();
        lbt_ << "(3rd stage) pre_main: passed 3rd stage boot barrier";

        runtime_support::call_startup_functions(find_here(), true);
        lbt_ << "(3rd stage) pre_main: ran pre-startup functions";

        // Third stage separates pre-startup and startup function phase.
        lcos::barrier::synchronize();
        lbt_ << "(4th stage) pre_main: passed 4th stage boot barrier";

        runtime_support::call_startup_functions(find_here(), false);
        lbt_ << "(4th stage) pre_main: ran startup functions";

        // Forth stage bootstrap synchronizes startup functions across all
        // localities. This is done after component loading to guarantee that
        // all user code, including startup functions, are only run after the
        // component tables are populated.
        lcos::barrier::synchronize();
        lbt_ << "(5th stage) pre_main: passed 4th stage boot barrier";
    }

    // Enable logging. Even if we terminate at this point we will see all
    // pending log messages so far.
    components::activate_logging();
    lbt_ << "(last stage) pre_main: activated logging";

    // Any error in post-command line handling or any explicit --exit command
    // line option will cause the application to terminate at this point.
    if (exit_code)
    {
        // If load_components returns false, shutdown the system. This
        // essentially only happens if the command line contained --exit.
        runtime_support::shutdown_all(
            naming::get_id_from_locality_id(HPX_AGAS_BOOTSTRAP_PREFIX), -1.0);
        return exit_code;
    }

    return 0;
}

}
