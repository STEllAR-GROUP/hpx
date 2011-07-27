////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/version.hpp>

#if HPX_AGAS_VERSION > 0x10

#include <hpx/hpx.hpp>
#include <hpx/runtime/components/stubs/runtime_support.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/lcos/barrier.hpp>

namespace hpx
{

void pre_main(runtime_mode mode)
{
    if (runtime_mode_probe == mode)
    {
        naming::resolver_client& agas_client = get_runtime().get_agas_client();

        components::stubs::runtime_support::load_components
            (find_here());

        if (!agas_client.is_bootstrap())
        {
            const std::size_t allocate_size = get_runtime().get_config().
                get_agas_allocate_response_pool_size();
            const std::size_t bind_size = get_runtime().get_config().
                get_agas_bind_response_pool_size();
    
            // Unblock the AGAS router by adding the initial pool size to the
            // future pool semaphores. 
            agas_client.hosted->allocate_response_sema_.signal(allocate_size);
            agas_client.hosted->bind_response_sema_.signal(bind_size);
        }

        components::stubs::runtime_support::call_startup_functions
            (find_here());
    }

    else
    {
        naming::resolver_client& agas_client = get_runtime().get_agas_client();
    
        // Load components, so that we can use the barrier LCO.
        components::stubs::runtime_support::load_components
            (find_here());
    
        lcos::barrier second_stage, third_stage;
    
        // {{{ Second and third stage barrier creation.
        if (agas_client.is_bootstrap())
        {
            second_stage.create_one(agas_client.local_prefix(),
                get_runtime().get_config().get_num_localities());
    
            third_stage.create_one(agas_client.local_prefix(),
                get_runtime().get_config().get_num_localities());
    
            naming::gid_type second = second_stage.get_gid().get_gid();
            naming::strip_credit_from_gid(second);
    
            naming::gid_type third = third_stage.get_gid().get_gid();
            naming::strip_credit_from_gid(third);
    
            agas_client.registerid("/barrier(agas#0)/second_stage", second); 
            agas_client.registerid("/barrier(agas#0)/third_stage", third); 
        }
    
        else // Hosted. 
        {
            const std::size_t allocate_size =
                get_runtime().get_config().get_agas_allocate_response_pool_size();
            const std::size_t bind_size =
                get_runtime().get_config().get_agas_bind_response_pool_size();
    
            // Unblock the AGAS router by adding the initial pool size to the
            // future pool semaphores. This ensures that no AGAS requests are 
            // sent after first-stage AGAS bootstrap and before second-stage
            // bootstrap.
            agas_client.hosted->allocate_response_sema_.signal(allocate_size);
            agas_client.hosted->bind_response_sema_.signal(bind_size);

            naming::gid_type second, third; 
    
            // {{{ Find the global address of the second stage barrier.
            for (int i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i)
            {
                if (agas_client.queryid("/barrier(agas#0)/second_stage", second))
                    break; 
    
                boost::this_thread::sleep(boost::get_system_time() + 
                    boost::posix_time::milliseconds(HPX_NETWORK_RETRIES_SLEEP));
            }
    
            if (HPX_UNLIKELY(second == naming::invalid_gid))
            {
                HPX_THROW_EXCEPTION(network_error, 
                    "pre_main", "couldn't find second stage boot barrier");
            }
            // }}}
    
            // {{{ Find the global address of the third stage barrier.
            for (int i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i)
            {
                if (agas_client.queryid("/barrier(agas#0)/third_stage", third))
                    break; 
    
                boost::this_thread::sleep(boost::get_system_time() + 
                    boost::posix_time::milliseconds(HPX_NETWORK_RETRIES_SLEEP));
            }
    
            if (HPX_UNLIKELY(third == naming::invalid_gid))
            {
                HPX_THROW_EXCEPTION(network_error, 
                    "pre_main", "couldn't find third stage boot barrier");
            }
            // }}}
    
            naming::id_type second_id(second, naming::id_type::unmanaged)
                          , third_id(third, naming::id_type::unmanaged);
    
            // Initialize the barrier clients
            second_stage = lcos::barrier(second_id);
            third_stage = lcos::barrier(third_id);
        }
        // }}}
    
        // Second stage bootstrap synchronizes component loading across all
        // localities, ensuring that the component namespace tables are fully
        // populated before user code is executed.
        second_stage.wait();
    
        // Third stage bootstrap synchronizes startup functions across all
        // localities. This is done after component loading to guarantee that
        // all user code, including startup functions, are only run after the
        // component tables are populated.
        components::stubs::runtime_support::call_startup_functions
            (find_here());
    
        third_stage.wait();
    }
}

}

#endif

