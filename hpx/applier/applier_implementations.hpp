//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_APPLIER_APPLIER_IMPLEMENTATIONS_JUN_09_2008_0434PM)
#define HPX_APPLIER_APPLIER_IMPLEMENTATIONS_JUN_09_2008_0434PM

#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (3, HPX_COMPONENT_ARGUMENT_LIMIT,                                     \
    "hpx/applier/applier_implementations.hpp"))                               \
    /**/
    
#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    void apply (naming::id_type gid, 
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        // Determine whether the gid is local or remote
        naming::address addr;
        if (address_is_local(gid, addr))
        {
            // If local, register the function with the thread-manager
            // Get the local-virtual address of the resource and register 
            // the action with the TM
            thread_manager_.register_work(
                Action(BOOST_PP_ENUM_PARAMS(N, arg)).get_thread_function(*this, addr.address_));
            return parcelset::no_parcel_id;     // no parcel has been sent
        }

        // If remote, create a new parcel to be sent to the destination
        // Create a new parcel with the gid, action, and arguments
        parcelset::parcel p (gid, new Action(BOOST_PP_ENUM_PARAMS(N, arg)));
        p.set_destination_addr(addr);   // avoid to resolve address again

        // Send the parcel through the parcel handler
        return parcel_handler_.put_parcel(p);
    }
    
#undef N

#endif
