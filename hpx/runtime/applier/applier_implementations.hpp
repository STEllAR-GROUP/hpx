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
    (3, (2, HPX_COMPONENT_ARGUMENT_LIMIT,                                     \
    "hpx/runtime/applier/applier_implementations.hpp"))                       \
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
    bool apply (naming::id_type const& gid, 
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        // Determine whether the gid is local or remote
        naming::address addr;
        if (address_is_local(gid, addr))
        {
            detail::BOOST_PP_CAT(apply_helper, N)<
                    Action, BOOST_PP_ENUM_PARAMS(N, Arg)
            >::call(thread_manager_, *this, addr.address_,
                BOOST_PP_ENUM_PARAMS(N, arg));
            return true;     // no parcel has been sent (dest is local)
        }

        // If remote, create a new parcel to be sent to the destination
        // Create a new parcel with the gid, action, and arguments
        parcelset::parcel p (gid, new Action(BOOST_PP_ENUM_PARAMS(N, arg)));
        p.set_destination_addr(addr);   // avoid to resolve address again

        // Send the parcel through the parcel handler
        parcel_handler_.put_parcel(p);
        return false;     // destination is remote
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    bool apply (actions::continuation* c, naming::id_type const& gid, 
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        actions::continuation_type cont(c);

        // Determine whether the gid is local or remote
        naming::address addr;
        if (address_is_local(gid, addr))
        {
            detail::BOOST_PP_CAT(apply_helper, N)<
                    Action, BOOST_PP_ENUM_PARAMS(N, Arg)
            >::call(cont, thread_manager_, *this, addr.address_,
                BOOST_PP_ENUM_PARAMS(N, arg));
            return true;     // no parcel has been sent (dest is local)
        }

        // If remote, create a new parcel to be sent to the destination
        // Create a new parcel with the gid, action, and arguments
        parcelset::parcel p (gid, new Action(BOOST_PP_ENUM_PARAMS(N, arg)), cont);
        p.set_destination_addr(addr);   // avoid to resolve address again

        // Send the parcel through the parcel handler
        parcel_handler_.put_parcel(p);
        return false;     // destination is remote
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    bool apply_c (naming::id_type const& targetgid, naming::id_type const& gid,
         BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        return apply<Action>(new actions::continuation(targetgid), 
            gid, BOOST_PP_ENUM_PARAMS(N, arg));
    }

#undef N

#endif
