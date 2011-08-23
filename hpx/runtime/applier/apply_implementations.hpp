//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_APPLIER_APPLY_IMPLEMENTATIONS_JUN_09_2008_0434PM)
#define HPX_APPLIER_APPLY_IMPLEMENTATIONS_JUN_09_2008_0434PM

#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (2, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/applier/apply_implementations.hpp"))                         \
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
    inline bool 
    apply_r_p(naming::address& addr, naming::id_type const& gid, 
        threads::thread_priority priority,
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        // If remote, create a new parcel to be sent to the destination
        // Create a new parcel with the gid, action, and arguments
        parcelset::parcel p (gid.get_gid(), 
            new Action(priority, BOOST_PP_ENUM_PARAMS(N, arg)));
        if (components::component_invalid == addr.type_)
            addr.type_ = components::get_component_type<typename Action::component_type>();
        p.set_destination_addr(addr);   // avoid to resolve address again

        // Send the parcel through the parcel handler
        hpx::applier::get_applier().get_parcel_handler().put_parcel(p);
        return false;     // destination is remote
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool 
    apply_r (naming::address& addr, naming::id_type const& gid, 
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        return apply_r_p<Action>(addr, gid, action_priority<Action>(),
            BOOST_PP_ENUM_PARAMS(N, arg));
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool 
    apply_l_p(naming::address const& addr, threads::thread_priority priority,
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        BOOST_ASSERT(components::types_are_compatible(addr.type_, 
            components::get_component_type<typename Action::component_type>()));
        detail::BOOST_PP_CAT(apply_helper, N)<
                Action, BOOST_PP_ENUM_PARAMS(N, Arg)
        >::call(addr.address_, priority, BOOST_PP_ENUM_PARAMS(N, arg));
        return true;     // no parcel has been sent (dest is local)
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool 
    apply_l (naming::address const& addr, 
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        return apply_l_p<Action>(addr, action_priority<Action>(), 
            BOOST_PP_ENUM_PARAMS(N, arg));
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool 
    apply_p(naming::id_type const& gid, threads::thread_priority priority,
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        // Determine whether the gid is local or remote
        naming::address addr;
        if (hpx::applier::get_applier().address_is_local(gid, addr))
            return apply_l_p<Action>(addr, priority, BOOST_PP_ENUM_PARAMS(N, arg));

        // apply remotely
        return apply_r_p<Action>(addr, gid, priority, BOOST_PP_ENUM_PARAMS(N, arg));
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool 
    apply (naming::id_type const& gid, 
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        return apply_p<Action>(gid, action_priority<Action>(), 
            BOOST_PP_ENUM_PARAMS(N, arg));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool 
    apply_r_p(naming::address& addr, actions::continuation* c, 
        naming::id_type const& gid, threads::thread_priority priority,
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        actions::continuation_type cont(c);

        // If remote, create a new parcel to be sent to the destination
        // Create a new parcel with the gid, action, and arguments
        parcelset::parcel p (gid.get_gid(), 
            new Action(priority, BOOST_PP_ENUM_PARAMS(N, arg)), cont);
        if (components::component_invalid == addr.type_)
            addr.type_ = components::get_component_type<typename Action::component_type>();
        p.set_destination_addr(addr);   // avoid to resolve address again

        // Send the parcel through the parcel handler
        hpx::applier::get_applier().get_parcel_handler().put_parcel(p);
        return false;     // destination is remote
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool 
    apply_r (naming::address& addr, actions::continuation* c, 
        naming::id_type const& gid, 
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        return apply_r_p<Action>(addr, c, gid, action_priority<Action>(),
            BOOST_PP_ENUM_PARAMS(N, arg));
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool 
    apply_l_p(actions::continuation* c, naming::address const& addr, 
        threads::thread_priority priority,
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        BOOST_ASSERT(components::types_are_compatible(addr.type_, 
            components::get_component_type<typename Action::component_type>()));
        actions::continuation_type cont(c);
        detail::BOOST_PP_CAT(apply_helper, N)<
                Action, BOOST_PP_ENUM_PARAMS(N, Arg)
        >::call(cont, addr.address_, priority, BOOST_PP_ENUM_PARAMS(N, arg));
        return true;     // no parcel has been sent (dest is local)
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool 
    apply_l (actions::continuation* c, naming::address const& addr, 
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        return apply_l_p<Action>(c, addr, action_priority<Action>(),
            BOOST_PP_ENUM_PARAMS(N, arg));
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool 
    apply_p(actions::continuation* c, naming::id_type const& gid, 
        threads::thread_priority priority,
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        // Determine whether the gid is local or remote
        naming::address addr;
        if (hpx::applier::get_applier().address_is_local(gid, addr))
            return apply_l_p<Action>(c, addr, priority, BOOST_PP_ENUM_PARAMS(N, arg));

        // apply remotely
        return apply_r_p<Action>(addr, c, gid, priority, BOOST_PP_ENUM_PARAMS(N, arg));
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool 
    apply (actions::continuation* c, naming::id_type const& gid, 
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        return apply_p<Action>(c, gid, action_priority<Action>(),
            BOOST_PP_ENUM_PARAMS(N, arg));
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool 
    apply_c_p(naming::address& addr, naming::id_type const& contgid, 
        naming::id_type const& gid, threads::thread_priority priority,
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        return apply_r_p<Action>(addr, new actions::continuation(contgid), 
            gid, priority, BOOST_PP_ENUM_PARAMS(N, arg));
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool 
    apply_c (naming::address& addr, naming::id_type const& contgid, 
        naming::id_type const& gid,
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        return apply_r_p<Action>(addr, new actions::continuation(contgid), 
            gid, action_priority<Action>(), BOOST_PP_ENUM_PARAMS(N, arg));
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool 
    apply_c_p(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority,
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        return apply<Action>(new actions::continuation(contgid), 
            gid, priority, BOOST_PP_ENUM_PARAMS(N, arg));
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool 
    apply_c (naming::id_type const& contgid, naming::id_type const& gid,
        BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, const& arg))
    {
        return apply_p<Action>(new actions::continuation(contgid), 
            gid, action_priority<Action>(), BOOST_PP_ENUM_PARAMS(N, arg));
    }

#undef N

#endif
