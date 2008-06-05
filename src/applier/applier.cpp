//  Copyright (c) 2008 Anshul Tandon
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/applier/applier.hpp>
#include <hpx/threadmanager/threadmanager.hpp>
#include <hpx/parcelset/parcelhandler.hpp>

///////////////////////////////////////////////////////////////////////////////

namespace hpx { namespace applier
{
    // Call-back function for parcelHandler to call when new parcels are received
    template <typename Action>
    void applier::apply (naming::id_type gid)
    {
        // A call to applier's apply function would look like:
        // apl_.apply<add_action>(gid, value)
        // Decode the action that is to be performed
        // Determine whether the gid is local or remote
        // If remote, create a new parcel to be sent to the destination
        // If local, register the function with the thread-manager

        // Resolve the address of the gid
        naming::address addr;
        dgas_client_.resolve(gid, addr);
        
        // Check for local/remote
        if (addr.locality_ == dgas_client_.here_)
        {   // If resource is local, call TM
            // Get the local-virtual address of the resource
            naming::address::address_type lva = addr.address_;
            // register the action and the local-virtual address with the TM
            thread_manager_.register_work(Action->get_thread_function(*this, lva));
        }
        else
        {   // If resource is global, create parcel
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p(gid, new Action());
            // Send the parcel through the parcel handler
            parcelset::parcel_id p_id = parcel_handler_.sync_put_parcel(p);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////
}}
