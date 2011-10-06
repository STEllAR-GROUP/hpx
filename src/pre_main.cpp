////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/version.hpp>
#include <hpx/hpx.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/lcos/barrier.hpp>

namespace hpx
{

///////////////////////////////////////////////////////////////////////////////
// Create a new barrier and register its gid with the given symbolic name.
inline lcos::barrier 
create_barrier(naming::resolver_client& agas_client, 
    std::size_t num_localities, char const* symname)
{
    lcos::barrier barrier;
    barrier.create_one(agas_client.local_prefix(), num_localities);

    naming::gid_type gid = barrier.get_gid().get_gid();
    naming::strip_credit_from_gid(gid);

    agas_client.registerid(symname, gid); 
    return barrier;
}

///////////////////////////////////////////////////////////////////////////////
// Find a registered barrier object from its symbolic name.
inline lcos::barrier 
find_barrier(naming::resolver_client& agas_client, char const* symname)
{
    naming::gid_type barrier_id;
    for (int i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i)
    {
        if (agas_client.queryid(symname, barrier_id))
            break; 

        boost::this_thread::sleep(boost::get_system_time() + 
            boost::posix_time::milliseconds(HPX_NETWORK_RETRIES_SLEEP));
    }
    if (HPX_UNLIKELY(!barrier_id))
    {
        HPX_THROW_EXCEPTION(network_error, "pre_main::find_barrier", 
            std::string("couldn't find stage boot barrier ") + symname);
    }
    return lcos::barrier(naming::id_type(barrier_id, naming::id_type::unmanaged));
}

///////////////////////////////////////////////////////////////////////////////
// Symbolic names of global boot barrier objects
const char* second_barrier = "/barrier(agas#0)/second_stage";
const char* third_barrier = "/barrier(agas#0)/third_stage";

///////////////////////////////////////////////////////////////////////////////
void pre_main(runtime_mode mode)
{
    naming::resolver_client& agas_client = naming::get_agas_client();
    util::runtime_configuration const& cfg = get_runtime().get_config();

    if (runtime_mode_connect == mode)
    {
        const std::size_t allocate_size = cfg.get_agas_allocate_response_pool_size();
        const std::size_t bind_size = cfg.get_agas_bind_response_pool_size();

        // Unblock the AGAS router by adding the initial pool size to the
        // future pool semaphores. 
        agas_client.hosted->allocate_response_sema_.signal(allocate_size);
        agas_client.hosted->bind_response_sema_.signal(bind_size);

        // Load components, so that we can use the barrier LCO.
        components::stubs::runtime_support::load_components
            (find_here());

        // Install performance counter startup functions for core subsystems.
        get_runtime().install_counters();
        threads::get_thread_manager().install_counters();
        applier::get_applier().get_parcel_handler().install_counters();

        components::stubs::runtime_support::call_startup_functions(find_here());
    }

    else
    {
        // {{{ unblock router 
        if (!agas_client.is_bootstrap())
        {
            const std::size_t allocate_size = cfg.get_agas_allocate_response_pool_size();
            const std::size_t bind_size = cfg.get_agas_bind_response_pool_size();

            // Unblock the AGAS router by adding the initial pool size to the
            // future pool semaphores. This ensures that no AGAS requests are 
            // sent after first-stage AGAS bootstrap and before second-stage
            // bootstrap.
            agas_client.hosted->allocate_response_sema_.signal(allocate_size);
            agas_client.hosted->bind_response_sema_.signal(bind_size);
        }
        // }}}

        // Load components, so that we can use the barrier LCO.
        components::stubs::runtime_support::load_components
            (find_here());

        lcos::barrier second_stage, third_stage;

        // {{{ Second and third stage barrier creation.
        if (agas_client.is_bootstrap())
        {
            naming::gid_type console_;
            
            if (HPX_UNLIKELY(!agas_client.get_console_prefix(console_, false)))
            {
                HPX_THROW_EXCEPTION(network_error
                    , "pre_main"
                    , "no console locality registered"); 
            }

            std::size_t const num_localities = cfg.get_num_localities();
            second_stage = create_barrier(agas_client, num_localities, second_barrier);
            third_stage = create_barrier(agas_client, num_localities, third_barrier);
        }

        else // Hosted. 
        {
            // Initialize the barrier clients (find them in AGAS)
            second_stage = find_barrier(agas_client, second_barrier);
            third_stage = find_barrier(agas_client, third_barrier);
        }
        // }}}

        // Second stage bootstrap synchronizes component loading across all
        // localities, ensuring that the component namespace tables are fully
        // populated before user code is executed.
        second_stage.wait();

        // Install performance counter startup functions for core subsystems.
        get_runtime().install_counters();
        threads::get_thread_manager().install_counters();
        applier::get_applier().get_parcel_handler().install_counters();

        components::stubs::runtime_support::call_startup_functions(find_here());

        // Third stage bootstrap synchronizes startup functions across all
        // localities. This is done after component loading to guarantee that
        // all user code, including startup functions, are only run after the
        // component tables are populated.
        third_stage.wait();
    }

    // Enable logging.
    components::activate_logging();
}

}

