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
#include <hpx/lcos/barrier.hpp>

namespace hpx
{

void pre_main()
{
    hpx::naming::resolver_client& agas_client =
        hpx::get_runtime().get_agas_client();

    hpx::components::stubs::runtime_support::load_components
        (hpx::applier::get_applier().get_runtime_support_gid());

    if (agas_client.is_bootstrap())
    {
        hpx::lcos::barrier b;

        b.create_one(agas_client.local_prefix()
                   , hpx::get_runtime().get_config().get_num_localities());

        hpx::naming::gid_type gid = b.get_gid().get_gid();
        hpx::naming::strip_credit_from_gid(gid);

        agas_client.registerid("/barrier(agas#0)", gid); 

        b.wait();
    }

    else // hosted
    {
        const std::size_t allocate_size =
            get_runtime().get_config().get_agas_allocate_response_pool_size();
        const std::size_t bind_size =
            get_runtime().get_config().get_agas_bind_response_pool_size();

        agas_client.hosted->allocate_response_sema_.signal(allocate_size);
        agas_client.hosted->bind_response_sema_.signal(bind_size);

        hpx::naming::gid_type gid; 

        for (int i = 0; i < HPX_MAX_NETWORK_RETRIES; ++i)
        {
            if (agas_client.queryid("/barrier(agas#0)", gid))
                break; 

            boost::this_thread::sleep(boost::get_system_time() + 
                boost::posix_time::milliseconds(HPX_NETWORK_RETRIES_SLEEP));
        }

        if (HPX_UNLIKELY(gid == hpx::naming::invalid_gid))
        {
            HPX_THROW_EXCEPTION(hpx::network_error, 
                "pre_main", "couldn't find second-stage boot barrier");
        }


        hpx::naming::id_type id(gid, hpx::naming::id_type::unmanaged);

        hpx::lcos::stubs::barrier::wait(id);
    }

    hpx::components::stubs::runtime_support::call_startup_functions
        (hpx::find_here());
}

}

#endif

