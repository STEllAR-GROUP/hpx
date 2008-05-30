//  Copyright (c) 2008 Anshul Tandon
// 
//  Parts of this code were taken from the Boost.Asio library
//  Copyright (c) 2003-2007 Christopher M. Kohlhoff (chris at kohlhoff dot com)
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>

#include <hpx/action_manager/action_manager.hpp>
#include <hpx/components/action.hpp>
#include <hpx/parcelset/parcelhandler.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace action_manager
{
    void action_manager::fetchNewParcel (parcelset::parcelhandler& pHandler, naming::address const&)
    {
        parcelset::parcel p;
        if (pHandler.get_parcel(p))            // if new parcel is found
        {
            // decode the local virtual address of the parcel
            naming::address addr = p.get_destination_addr();
            naming::address::address_type lva = addr.address_;
            
            // decode the action-type in the parcel
            components::action_type act = p.get_action();

            // register the action and the local-virtual address with the TM
            tManager.register_work(act->get_thread_function(lva));
        }
    }
    
///////////////////////////////////////////////////////////////////////////////
}}
