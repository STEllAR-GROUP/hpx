//  Copyright (c) 2007-2011 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLIER_APPLY_NOV_27_2008_0957AM)
#define HPX_APPLIER_APPLY_NOV_27_2008_0957AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/applier/apply_helper.hpp>
#include <hpx/runtime/actions/component_action.hpp>

// FIXME: Error codes?

namespace hpx { namespace applier
{
    template <typename Action>
    threads::thread_priority action_priority()
    {
        return static_cast<threads::thread_priority>(Action::priority_value);
    }

    ///////////////////////////////////////////////////////////////////////////
    // zero parameter version of apply()
    // Invoked by a running PX-thread to apply an action to any resource

    /// \note A call to applier's apply function would look like:
    /// \code
    ///    appl_.apply<add_action>(gid, ...);
    /// \endcode

    // we know, it's remote
    template <typename Action>
    inline bool
    apply_r_p(naming::address& addr, naming::id_type const& gid,
        threads::thread_priority priority)
    {
        // If remote, create a new parcel to be sent to the destination
        // Create a new parcel with the gid, action, and arguments
        parcelset::parcel p (gid.get_gid(), new Action(priority));
        if (components::component_invalid == addr.type_)
            addr.type_ = components::get_component_type<typename Action::component_type>();
        p.set_destination_addr(addr);   // avoid to resolve address again

        // Send the parcel through the parcel handler
        hpx::applier::get_applier().get_parcel_handler().put_parcel(p);
        return false;     // destination is remote
    }

    template <typename Action>
    inline bool
    apply_r (naming::address& addr, naming::id_type const& gid)
    {
        return apply_r_p<Action>(addr, gid, action_priority<Action>());
    }

    // we  know, it's local and has to be directly executed
    template <typename Action>
    inline bool
    apply_l_p(naming::address const& addr, threads::thread_priority priority)
    {
        BOOST_ASSERT(components::types_are_compatible(addr.type_,
            components::get_component_type<typename Action::component_type>()));
        detail::apply_helper0<Action>::call(addr.address_, priority);
        return true;     // no parcel has been sent (dest is local)
    }

    template <typename Action>
    inline bool
    apply_l (naming::address const& addr)
    {
        return apply_l_p<Action>(addr, action_priority<Action>());
    }

    template <typename Action>
    inline bool
    apply_p (naming::id_type const& gid, threads::thread_priority priority)
    {
        // Determine whether the gid is local or remote
        naming::address addr;
        if (hpx::applier::get_applier().address_is_local(gid, addr))
            return apply_l_p<Action>(addr, priority);   // apply locally

        // apply remotely
        return apply_r_p<Action>(addr, gid, priority);
    }

    template <typename Action>
    inline bool apply (naming::id_type const& gid)
    {
        return apply_p<Action>(gid, action_priority<Action>());
    }

    /// \note A call to applier's apply function would look like:
    /// \code
    ///    appl_.apply<add_action>(cont, gid, ...);
    /// \endcode
    template <typename Action>
    inline bool
    apply_r_p(naming::address& addr, actions::continuation* c,
        naming::id_type const& gid, threads::thread_priority priority)
    {
        actions::continuation_type cont(c);

        // If remote, create a new parcel to be sent to the destination
        // Create a new parcel with the gid, action, and arguments
        parcelset::parcel p (gid.get_gid(), new Action(priority), cont);
        if (components::component_invalid == addr.type_)
            addr.type_ = components::get_component_type<typename Action::component_type>();
        p.set_destination_addr(addr);   // avoid to resolve address again

        // Send the parcel through the parcel handler
        hpx::applier::get_applier().get_parcel_handler().put_parcel(p);
        return false;     // destination is remote
    }

    template <typename Action>
    inline bool
    apply_r (naming::address& addr, actions::continuation* c,
        naming::id_type const& gid)
    {
        return apply_r_p<Action>(addr, c, gid, action_priority<Action>());
    }

    template <typename Action>
    inline bool
    apply_r_sync_p(naming::address& addr, naming::id_type const& gid,
        threads::thread_priority priority)
    {
        // If remote, create a new parcel to be sent to the destination
        // Create a new parcel with the gid, action, and arguments
        parcelset::parcel p (gid.get_gid(), new Action(priority));
        if (components::component_invalid == addr.type_)
            addr.type_ = components::get_component_type<typename Action::component_type>();
        p.set_destination_addr(addr);   // avoid to resolve address again

        // Send the parcel through the parcel handler
        hpx::applier::get_applier().get_parcel_handler().sync_put_parcel(p);
        return false;     // destination is remote
    }

    template <typename Action>
    inline bool
    apply_r_sync (naming::address& addr, naming::id_type const& gid)
    {
        return apply_r_sync_p<Action>(addr, gid, action_priority<Action>());
    }

    // we know, it's local and has to be directly executed
    template <typename Action>
    inline bool apply_l_p(actions::continuation* c,
        naming::address const& addr, threads::thread_priority priority)
    {
        BOOST_ASSERT(components::types_are_compatible(addr.type_,
            components::get_component_type<typename Action::component_type>()));
        actions::continuation_type cont(c);
        detail::apply_helper0<Action>::call(cont, addr.address_, priority);
        return true;     // no parcel has been sent (dest is local)
    }

    template <typename Action>
    inline bool apply_l (actions::continuation* c, naming::address const& addr)
    {
        return apply_l_p<Action>(c, addr, action_priority<Action>());
    }

    template <typename Action>
    inline bool apply_p(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority)
    {
        // Determine whether the gid is local or remote
        naming::address addr;
        if (hpx::applier::get_applier().address_is_local(gid, addr))
            return apply_l_p<Action>(c, addr, priority);

        // apply remotely
        return apply_r_p<Action>(addr, c, gid, priority);
    }

    template <typename Action>
    inline bool apply (actions::continuation* c, naming::id_type const& gid)
    {
        return apply_p<Action>(c, gid, action_priority<Action>());
    }

    template <typename Action>
    inline bool
    apply_c_p(naming::address& addr, naming::id_type const& contgid,
        naming::id_type const& gid, threads::thread_priority priority)
    {
        return apply_r_p<Action>(addr, new actions::continuation(contgid),
            gid, priority);
    }

    template <typename Action>
    inline bool
    apply_c (naming::address& addr, naming::id_type const& contgid,
        naming::id_type const& gid)
    {
        return apply_r<Action>(addr, new actions::continuation(contgid), gid);
    }

    template <typename Action>
    inline bool
    apply_c_p(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority)
    {
        return apply_p<Action>(new actions::continuation(contgid), gid, priority);
    }

    template <typename Action>
    inline bool
    apply_c (naming::id_type const& contgid, naming::id_type const& gid)
    {
        return apply<Action>(new actions::continuation(contgid), gid);
    }

    ///////////////////////////////////////////////////////////////////////////
    // one parameter version
    template <typename Action, typename Arg0>
    inline bool
    apply_r_p(naming::address& addr, naming::id_type const& gid,
        threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0)
    {
        // If remote, create a new parcel to be sent to the destination
        // Create a new parcel with the gid, action, and arguments
        parcelset::parcel p (gid.get_gid(),
            new Action(priority, boost::forward<Arg0>(arg0)));
        if (components::component_invalid == addr.type_)
            addr.type_ = components::get_component_type<typename Action::component_type>();
        p.set_destination_addr(addr);   // avoid to resolve address again

        // Send the parcel through the parcel handler
        hpx::applier::get_applier().get_parcel_handler().put_parcel(p);
        return false;     // destination is remote
    }

    template <typename Action, typename Arg0>
    inline bool
    apply_r (naming::address& addr, naming::id_type const& gid,
        BOOST_FWD_REF(Arg0) arg0)
    {
        return apply_r_p<Action>(addr, gid, action_priority<Action>(),
            boost::forward<Arg0>(arg0));
    }

    template <typename Action, typename Arg0>
    inline bool
    apply_r_sync_p(naming::address& addr, naming::id_type const& gid,
        threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0)
    {
        // If remote, create a new parcel to be sent to the destination
        // Create a new parcel with the gid, action, and arguments
        parcelset::parcel p (gid.get_gid(),
            new Action(priority, boost::forward<Arg0>(arg0)));
        if (components::component_invalid == addr.type_)
            addr.type_ = components::get_component_type<typename Action::component_type>();
        p.set_destination_addr(addr);   // avoid to resolve address again

        // Send the parcel through the parcel handler
        hpx::applier::get_applier().get_parcel_handler().sync_put_parcel(p);
        return false;     // destination is remote
    }

    template <typename Action, typename Arg0>
    inline bool
    apply_r_sync (naming::address& addr, naming::id_type const& gid,
        BOOST_FWD_REF(Arg0) arg0)
    {
        return apply_r_sync_p<Action>(addr, gid, action_priority<Action>(),
            boost::forward<Arg0>(arg0));
    }

    template <typename Action, typename Arg0>
    inline bool
    apply_l_p(naming::address const& addr, threads::thread_priority priority,
        BOOST_FWD_REF(Arg0) arg0)
    {
        BOOST_ASSERT(components::types_are_compatible(addr.type_,
            components::get_component_type<typename Action::component_type>()));
        detail::apply_helper1<Action>::call(addr.address_, priority,
            boost::forward<Arg0>(arg0));
        return true;     // no parcel has been sent (dest is local)
    }

    template <typename Action, typename Arg0>
    inline bool
    apply_l (naming::address const& addr, BOOST_FWD_REF(Arg0) arg0)
    {
        return apply_l_p<Action>(addr, action_priority<Action>(),
            boost::forward<Arg0>(arg0));
    }

    template <typename Action, typename Arg0>
    inline bool
    apply_p(naming::id_type const& gid, threads::thread_priority priority,
        BOOST_FWD_REF(Arg0) arg0)
    {
        // Determine whether the gid is local or remote
        naming::address addr;
        if (hpx::applier::get_applier().address_is_local(gid, addr))
            return apply_l_p<Action>(addr, priority, boost::forward<Arg0>(arg0));   // apply locally

        // apply remotely
        return apply_r_p<Action>(addr, gid, priority, boost::forward<Arg0>(arg0));
    }

    template <typename Action, typename Arg0>
    inline bool
    apply (naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0)
    {
        return apply_p<Action>(gid, action_priority<Action>(),
            boost::forward<Arg0>(arg0));
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Arg0>
    inline bool
    apply_r_p(naming::address& addr, actions::continuation* c,
        naming::id_type const& gid, threads::thread_priority priority,
        BOOST_FWD_REF(Arg0) arg0)
    {
        actions::continuation_type cont(c);

        // If remote, create a new parcel to be sent to the destination
        // Create a new parcel with the gid, action, and arguments
        parcelset::parcel p (gid.get_gid(),
            new Action(priority, boost::forward<Arg0>(arg0)), cont);
        if (components::component_invalid == addr.type_)
            addr.type_ = components::get_component_type<typename Action::component_type>();
        p.set_destination_addr(addr);   // avoid to resolve address again

        // Send the parcel through the parcel handler
        hpx::applier::get_applier().get_parcel_handler().put_parcel(p);
        return false;     // destination is remote
    }

    template <typename Action, typename Arg0>
    inline bool
    apply_r (naming::address& addr, actions::continuation* c,
        naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0)
    {
        return apply_r_p<Action>(addr, c, gid, action_priority<Action>(),
            boost::forward<Arg0>(arg0));
    }

    template <typename Action, typename Arg0>
    inline bool
    apply_l_p(actions::continuation* c, naming::address const& addr,
        threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0)
    {
        BOOST_ASSERT(components::types_are_compatible(addr.type_,
            components::get_component_type<typename Action::component_type>()));
        actions::continuation_type cont(c);
        detail::apply_helper1<Action>::call(cont, addr.address_, priority,
            boost::forward<Arg0>(arg0));
        return true;     // no parcel has been sent (dest is local)
    }

    template <typename Action, typename Arg0>
    inline bool
    apply_l (actions::continuation* c, naming::address const& addr,
        BOOST_FWD_REF(Arg0) arg0)
    {
        return apply_l_p<Action>(c, addr, action_priority<Action>(),
            boost::forward<Arg0>(arg0));
    }

    template <typename Action, typename Arg0>
    inline bool
    apply_p(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0)
    {
        // Determine whether the gid is local or remote
        naming::address addr;
        if (hpx::applier::get_applier().address_is_local(gid, addr))
            return apply_l_p<Action>(c, addr, priority, boost::forward<Arg0>(arg0));    // apply locally

        // apply remotely
        return apply_r_p<Action>(addr, c, gid, priority, boost::forward<Arg0>(arg0));
    }

    template <typename Action, typename Arg0>
    inline bool
    apply (actions::continuation* c, naming::id_type const& gid,
        BOOST_FWD_REF(Arg0) arg0)
    {
        return apply_p<Action>(c, gid, action_priority<Action>(),
            boost::forward<Arg0>(arg0));
    }

    /// \brief Invoke an unary action with a \b actions::continuation
    ///        at a specific priority using a pre-resolved remote
    ///        \b naming::address. Takes a continuation by GID.
    template <typename Action, typename Arg0>
    inline bool
    apply_c_p(naming::address& addr, naming::id_type const& contgid,
        naming::id_type const& gid, threads::thread_priority priority,
        BOOST_FWD_REF(Arg0) arg0)
    {
        return apply_r_p<Action>(addr, new actions::continuation(contgid), gid,
            action_priority<Action>(), boost::forward<Arg0>(arg0));
    }

    /// \brief Invoke an unary action with a \b actions::continuation
    ///        using a pre-resolved remote \b naming::address. Takes a
    ///        continuation by GID.
    template <typename Action, typename Arg0>
    inline bool
    apply_c (naming::address& addr, naming::id_type const& contgid,
        naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0)
    {
        return apply_r<Action>(addr, new actions::continuation(contgid), gid,
            boost::forward<Arg0>(arg0));
    }

    /// \brief Invoke an unary action with a \b actions::continuation at a
    ///        specific priority on a local or remote GID. Takes a continuation
    ///        by GID.
    template <typename Action, typename Arg0>
    inline bool
    apply_c_p(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0)
    {
        return apply_p<Action>(new actions::continuation(contgid), gid,
            priority, boost::forward<Arg0>(arg0));
    }

    /// \brief Invoke an unary action with a \b actions::continuation on a
    ///        local or remote GID. Takes a continuation by GID.
    ///
    /// \param contgid      [in] The GID of the continuation.
    /// \param gid          [in] The target of the action.
    /// \param arg0         [in] Value to bind to the first argument of the
    ///                     action.
    template <typename Action, typename Arg0>
    inline bool
    apply_c (naming::id_type const& contgid, naming::id_type const& gid,
        BOOST_FWD_REF(Arg0) arg0)
    {
        return apply<Action>(new actions::continuation(contgid), gid,
            boost::forward<Arg0>(arg0));
    }
}}

// bring in the rest of the apply<> overloads (arity 2+)
#include <hpx/runtime/applier/apply_implementations.hpp>

#endif
