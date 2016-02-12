////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#include <hpx/version.hpp>
#include <hpx/hpx.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/lcos/barrier.hpp>
#include <hpx/runtime/agas/interface.hpp>

#define HPX_USE_FAST_BOOTSTRAP_SYNCHRONIZATION

#if defined(HPX_USE_FAST_BOOTSTRAP_SYNCHRONIZATION)
#include <hpx/lcos/broadcast.hpp>
#endif

///////////////////////////////////////////////////////////////////////////////

#if defined(HPX_USE_FAST_BOOTSTRAP_SYNCHRONIZATION)

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::server::runtime_support::call_startup_functions_action
    call_startup_functions_action;

HPX_REGISTER_BROADCAST_ACTION_DECLARATION(call_startup_functions_action,
        call_startup_functions_action)
HPX_REGISTER_BROADCAST_ACTION_ID(call_startup_functions_action,
        call_startup_functions_action,
        hpx::actions::broadcast_call_startup_functions_action_id)

#endif

///////////////////////////////////////////////////////////////////////////////
namespace
{
    void garbage_collect_non_blocking()
    {
        hpx::agas::garbage_collect_non_blocking();
    }
    void garbage_collect()
    {
        hpx::agas::garbage_collect();
    }
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{

///////////////////////////////////////////////////////////////////////////////
// Create a new barrier and register its id with the given symbolic name.
inline lcos::barrier
create_barrier(std::size_t num_localities, char const* symname)
{
    lcos::barrier b = lcos::barrier::create(find_here(), num_localities);

    // register an unmanaged gid to avoid id-splitting during startup
    agas::register_name_sync(symname, b.get_id().get_gid());
    return b;
}

inline void delete_barrier(lcos::barrier& b, char const* symname)
{
    agas::unregister_name_sync(symname);
    b.free();
}

///////////////////////////////////////////////////////////////////////////////
// Find a registered barrier object from its symbolic name.
inline lcos::barrier
find_barrier(char const* symname)
{
    naming::id_type barrier_id;
    for (std::size_t i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i)
    {
        if (agas::resolve_name_sync(symname, barrier_id))
            break;

        boost::this_thread::sleep(boost::get_system_time() +
            boost::posix_time::milliseconds(HPX_NETWORK_RETRIES_SLEEP));
    }
    if (HPX_UNLIKELY(!barrier_id))
    {
        HPX_THROW_EXCEPTION(network_error, "pre_main::find_barrier",
            std::string("couldn't find boot barrier ") + symname);
    }
    return lcos::barrier(barrier_id);
}

///////////////////////////////////////////////////////////////////////////////
// Symbolic names of global boot barrier objects
static const char* const startup_barrier_name = "/0/agas/startup_barrier";

///////////////////////////////////////////////////////////////////////////////
// Install performance counter startup functions for core subsystems.
inline void register_counter_types()
{
     naming::get_agas_client().register_counter_types();
     LBT_(info) << "(2nd stage) pre_main: registered AGAS client-side "
                   "performance counter types";

     get_runtime().register_counter_types();
     LBT_(info) << "(2nd stage) pre_main: registered runtime performance "
                   "counter types";

     threads::get_thread_manager().register_counter_types();
     LBT_(info) << "(2nd stage) pre_main: registered thread-manager performance "
                   "counter types";

     applier::get_applier().get_parcel_handler().register_counter_types();
     LBT_(info) << "(2nd stage) pre_main: registered parcelset performance "
                   "counter types";
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
    util::runtime_configuration const& cfg = rt.get_config();

    int exit_code = 0;
    if (runtime_mode_connect == mode)
    {
        LBT_(info) << "(2nd stage) pre_main: locality is in connect mode, "
                      "skipping 2nd and 3rd stage startup synchronization";
        LBT_(info) << "(2nd stage) pre_main: addressing services enabled";

        // Load components, so that we can use the barrier LCO.
        exit_code = runtime_support::load_components(find_here());
        LBT_(info) << "(2nd stage) pre_main: loaded components"
            << (exit_code ? ", application exit has been requested" : "");

        register_counter_types();

        rt.set_state(state_pre_startup);
        runtime_support::call_startup_functions(find_here(), true);
        LBT_(info) << "(3rd stage) pre_main: ran pre-startup functions";

        rt.set_state(state_startup);
        runtime_support::call_startup_functions(find_here(), false);
        LBT_(info) << "(3rd stage) pre_main: ran startup functions";
    }
    else
    {
        LBT_(info) << "(2nd stage) pre_main: addressing services enabled";

        // Load components, so that we can use the barrier LCO.
        exit_code = runtime_support::load_components(find_here());
        LBT_(info) << "(2nd stage) pre_main: loaded components"
            << (exit_code ? ", application exit has been requested" : "");

        lcos::barrier startup_barrier;

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

            std::size_t const num_localities =
                static_cast<std::size_t>(cfg.get_num_localities());

            if (num_localities > 1)
            {
                startup_barrier = create_barrier(num_localities, startup_barrier_name);
            }

            LBT_(info) << "(2nd stage) pre_main: created \
                           2nd and 3rd stage boot barriers";
        }
        else // Hosted.
        {
            // Initialize the barrier clients (find them in AGAS)
            startup_barrier = find_barrier(startup_barrier_name);

            LBT_(info) << "(2nd stage) pre_main: found 2nd and 3rd stage boot barriers";
        }
        // }}}

        // Register all counter types before the startup functions are being
        // executed.
        register_counter_types();

        // Second stage bootstrap synchronizes component loading across all
        // localities, ensuring that the component namespace tables are fully
        // populated before user code is executed.
        if (startup_barrier)
        {
            startup_barrier.wait();
            LBT_(info) << "(2nd stage) pre_main: passed 2nd stage boot barrier";
        }

#if defined(HPX_USE_FAST_BOOTSTRAP_SYNCHRONIZATION)
        if (agas_client.is_bootstrap())
        {
            std::vector<naming::id_type> localities = hpx::find_all_localities();

            call_startup_functions_action act;
            lcos::broadcast(act, localities, true).get();
            LBT_(info) << "(3rd stage) pre_main: ran pre-startup functions";

            lcos::broadcast(act, localities, false).get();
            LBT_(info) << "(4th stage) pre_main: ran startup functions";
        }
#else
        runtime_support::call_startup_functions(find_here(), true);
        LBT_(info) << "(3rd stage) pre_main: ran pre-startup functions";

        // Third stage separates pre-startup and startup function phase.
        if (startup_barrier)
        {
            startup_barrier.wait();
            LBT_(info) << "(3rd stage) pre_main: passed 3rd stage boot barrier";
        }

        runtime_support::call_startup_functions(find_here(), false);
        LBT_(info) << "(4th stage) pre_main: ran startup functions";
#endif

        if (startup_barrier)
        {
            // Forth stage bootstrap synchronizes startup functions across all
            // localities. This is done after component loading to guarantee that
            // all user code, including startup functions, are only run after the
            // component tables are populated.
            startup_barrier.wait();
            LBT_(info) << "(4th stage) pre_main: passed 4th stage boot barrier";

            // Tear down the startup barrier.
            if (agas_client.is_bootstrap())
                delete_barrier(startup_barrier, startup_barrier_name);
        }
    }

    // Enable logging. Even if we terminate at this point we will see all
    // pending log messages so far.
    components::activate_logging();
    LBT_(info) << "(4th stage) pre_main: activated logging";

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

