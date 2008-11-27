//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLIER_APPLY_NOV_27_2008_0957AM)
#define HPX_APPLIER_APPLY_NOV_27_2008_0957AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/applier/apply_helper.hpp>
#include <hpx/runtime/actions/component_action.hpp>

namespace hpx { namespace applier 
{
    ///////////////////////////////////////////////////////////////////////////
    // zero parameter version of apply()
    // Invoked by a running PX-thread to apply an action to any resource

    /// \note A call to applier's apply function would look like:
    /// \code
    ///    appl_.apply<add_action>(gid, ...);
    /// \endcode
    template <typename Action>
    bool apply (naming::address& addr, naming::id_type const& gid)
    {
        // If remote, create a new parcel to be sent to the destination
        // Create a new parcel with the gid, action, and arguments
        parcelset::parcel p (gid, new Action());
        if (components::component_invalid == addr.type_)
            addr.type_ = components::get_component_type<typename Action::component_type>();
        p.set_destination_addr(addr);   // avoid to resolve address again

        // Send the parcel through the parcel handler
        hpx::applier::get_applier().get_parcel_handler().put_parcel(p);
        return false;     // destination is remote
    }

    template <typename Action>
    bool apply (naming::id_type const& gid)
    {
        // Determine whether the gid is local or remote
        naming::address addr;
        if (hpx::applier::get_applier().address_is_local(gid, addr)) {
            BOOST_ASSERT(components::types_are_compatible(addr.type_, 
                components::get_component_type<typename Action::component_type>()));
            detail::apply_helper0<Action>::call(addr.address_);
            return true;     // no parcel has been sent (dest is local)
        }

        // apply remotely
        return apply<Action>(addr, gid);
    }

    /// \note A call to applier's apply function would look like:
    /// \code
    ///    appl_.apply<add_action>(cont, gid, ...);
    /// \endcode
    template <typename Action>
    bool apply (naming::address& addr, actions::continuation* c, 
        naming::id_type const& gid)
    {
        actions::continuation_type cont(c);

        // If remote, create a new parcel to be sent to the destination
        // Create a new parcel with the gid, action, and arguments
        parcelset::parcel p (gid, new Action(), cont);
        if (components::component_invalid == addr.type_)
            addr.type_ = components::get_component_type<typename Action::component_type>();
        p.set_destination_addr(addr);   // avoid to resolve address again

        // Send the parcel through the parcel handler
        hpx::applier::get_applier().get_parcel_handler().put_parcel(p);
        return false;     // destination is remote
    }

    template <typename Action>
    bool apply (actions::continuation* c, naming::id_type const& gid)
    {
        // Determine whether the gid is local or remote
        naming::address addr;
        if (hpx::applier::get_applier().address_is_local(gid, addr)) {
            BOOST_ASSERT(components::types_are_compatible(addr.type_, 
                components::get_component_type<typename Action::component_type>()));
            actions::continuation_type cont(c);
            detail::apply_helper0<Action>::call(cont, addr.address_);
            return true;     // no parcel has been sent (dest is local)
        }

        // apply remotely
        return apply<Action>(addr, c, gid);
    }

    template <typename Action>
    bool apply_c (naming::address& addr, naming::id_type const& targetgid, 
        naming::id_type const& gid)
    {
        return apply<Action>(addr, new actions::continuation(targetgid), gid);
    }

    template <typename Action>
    bool apply_c (naming::id_type const& targetgid, 
        naming::id_type const& gid)
    {
        return apply<Action>(new actions::continuation(targetgid), gid);
    }

    ///////////////////////////////////////////////////////////////////////////
    // one parameter version
    template <typename Action, typename Arg0>
    bool apply (naming::address& addr, naming::id_type const& gid, 
        Arg0 const& arg0)
    {
        // If remote, create a new parcel to be sent to the destination
        // Create a new parcel with the gid, action, and arguments
        parcelset::parcel p (gid, new Action(arg0));
        if (components::component_invalid == addr.type_)
            addr.type_ = components::get_component_type<typename Action::component_type>();
        p.set_destination_addr(addr);   // avoid to resolve address again

        // Send the parcel through the parcel handler
        hpx::applier::get_applier().get_parcel_handler().put_parcel(p);
        return false;     // destination is remote
    }

    template <typename Action, typename Arg0>
    bool apply (naming::id_type const& gid, Arg0 const& arg0)
    {
        // Determine whether the gid is local or remote
        naming::address addr;
        if (hpx::applier::get_applier().address_is_local(gid, addr)) {
            BOOST_ASSERT(components::types_are_compatible(addr.type_, 
                components::get_component_type<typename Action::component_type>()));
            detail::apply_helper1<Action, Arg0>::call(addr.address_, arg0);
            return true;     // no parcel has been sent (dest is local)
        }

        // apply remotely
        return apply<Action>(addr, gid, arg0);
    }

    template <typename Action, typename Arg0>
    bool apply (naming::address& addr, actions::continuation* c, 
        naming::id_type const& gid, Arg0 const& arg0)
    {
        actions::continuation_type cont(c);

        // If remote, create a new parcel to be sent to the destination
        // Create a new parcel with the gid, action, and arguments
        parcelset::parcel p (gid, new Action(arg0), cont);
        if (components::component_invalid == addr.type_)
            addr.type_ = components::get_component_type<typename Action::component_type>();
        p.set_destination_addr(addr);   // avoid to resolve address again

        // Send the parcel through the parcel handler
        hpx::applier::get_applier().get_parcel_handler().put_parcel(p);
        return false;     // destination is remote
    }

    template <typename Action, typename Arg0>
    bool apply (actions::continuation* c, naming::id_type const& gid, 
        Arg0 const& arg0)
    {
        // Determine whether the gid is local or remote
        naming::address addr;
        if (hpx::applier::get_applier().address_is_local(gid, addr)) {
            BOOST_ASSERT(components::types_are_compatible(addr.type_, 
                components::get_component_type<typename Action::component_type>()));
            actions::continuation_type cont(c);
            detail::apply_helper1<Action, Arg0>::call(cont, addr.address_, arg0);
            return true;     // no parcel has been sent (dest is local)
        }

        // apply remotely
        return apply<Action>(addr, c, gid, arg0);
    }

    template <typename Action, typename Arg0>
    bool apply_c (naming::address& addr, naming::id_type const& targetgid, 
        naming::id_type const& gid, Arg0 const& arg0)
    {
        return apply<Action>(addr, new actions::continuation(targetgid), gid, arg0);
    }

    template <typename Action, typename Arg0>
    bool apply_c (naming::id_type const& targetgid, naming::id_type const& gid, 
        Arg0 const& arg0)
    {
        return apply<Action>(new actions::continuation(targetgid), gid, arg0);
    }

    // bring in the rest of the apply<> overloads
    #include <hpx/runtime/applier/apply_implementations.hpp>

}}

#endif
