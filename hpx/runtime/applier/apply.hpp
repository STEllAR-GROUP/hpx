//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLIER_APPLY_NOV_27_2008_0957AM)
#define HPX_APPLIER_APPLY_NOV_27_2008_0957AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>

#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/parcelset/parcelhandler.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/applier/apply_helper.hpp>
#include <hpx/runtime/actions/component_action.hpp>
#include <hpx/runtime/actions/base_lco_continuation.hpp>

// FIXME: Error codes?

namespace hpx { namespace actions
{
    template <typename Action>
    threads::thread_priority action_priority()
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        return static_cast<threads::thread_priority>(action_type::priority_value);
    }
}}

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    // zero parameter version of apply()
    // Invoked by a running HPX-thread to apply an action to any resource

    namespace applier { namespace detail
    {
        // We know it is remote.
        template <typename Action>
        inline bool
        apply_r_p(naming::address& addr, naming::id_type const& gid,
            threads::thread_priority priority)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid.get_gid(), new action_type(priority));
            if (components::component_invalid == addr.type_)
                addr.type_ = components::get_component_type<
                    typename action_type::component_type>();
            p.set_destination_addr(addr);   // avoid to resolve address again

            // Send the parcel through the parcel handler
            hpx::applier::get_applier().get_parcel_handler().put_parcel(p);
            return false;     // destination is remote
        }

        template <typename Action>
        inline bool
        apply_r_p_route(naming::address& addr, naming::id_type const& gid,
            threads::thread_priority priority)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            // Create a parcel and send it to the AGAS server
            // New parcel with the gid, action, and arguments
            parcelset::parcel p (gid.get_gid(), new action_type(priority));
            if (components::component_invalid == addr.type_)
                addr.type_ = components::get_component_type<
                    typename action_type::component_type>();
            p.set_destination_addr(addr);   // redundant

            // Send the parcel to applier to be sent to the AGAS server
            return hpx::applier::get_applier().route(p);
        }

        template <typename Action>
        inline bool
        apply_r (naming::address& addr, naming::id_type const& gid)
        {
            return apply_r_p<Action>(addr, gid,
                actions::action_priority<Action>());
        }

        template <typename Action>
        inline bool
        apply_r_route(naming::address& addr, naming::id_type const& gid)
        {
            return apply_r_p_route<Action>(addr, gid,
                actions::action_priority<Action>());
        }

        // We know it is local and has to be directly executed.
        template <typename Action>
        inline bool
        apply_l_p(naming::address const& addr, threads::thread_priority priority)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            BOOST_ASSERT(components::types_are_compatible(addr.type_,
                components::get_component_type<
                    typename action_type::component_type>()));
            apply_helper0<action_type>::call(addr.address_, priority);
            return true;     // no parcel has been sent (dest is local)
        }

        template <typename Action>
        inline bool
        apply_l (naming::address const& addr)
        {
            return apply_l_p<Action>(addr, actions::action_priority<Action>());
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    /// \note A call to applier's apply function would look like:
    /// \code
    ///    appl_.apply<add_action>(gid, ...);
    /// \endcode
    template <typename Action>
    inline bool
    apply_p (naming::id_type const& gid, threads::thread_priority priority)
    {
        // Determine whether the gid is local or remote
        naming::address addr;
        if (agas::is_local_address(gid, addr))
            return applier::detail::apply_l_p<Action>(addr, priority);   // apply locally

        // apply remotely
        return applier::detail::apply_r_p<Action>(addr, gid, priority);
    }

    template <typename Action>
    inline bool apply (naming::id_type const& gid)
    {
        return apply_p<Action>(gid, actions::action_priority<Action>());
    }

    namespace applier
    {
        /// routed version
        template <typename Action>
        inline bool
        apply_p_route (naming::id_type const& gid, threads::thread_priority priority)
        {
            // check if the address is in the local cache
            // send a parcel to agas if address is not found in cache
            naming::address addr;
            if (agas::is_local_address_cached(gid, addr))
                return detail::apply_l_p<Action>(addr, priority);   // apply locally

            // since we already know the address is local and its value
            // we can apply the function locally

            // parameter addr is redundant here
            return detail::apply_r_p_route<Action>(addr, gid, priority);
        }

        // routed version
        template <typename Action>
        inline bool apply_route (naming::id_type const& gid)
        {
            return apply_p_route<Action>(gid, actions::action_priority<Action>());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action>
        inline bool
        apply_r_p(naming::address& addr, actions::continuation* c,
            naming::id_type const& gid, threads::thread_priority priority)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            actions::continuation_type cont(c);

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid.get_gid(), new action_type(priority), cont);
            if (components::component_invalid == addr.type_)
                addr.type_ = components::get_component_type<
                    typename action_type::component_type>();
            p.set_destination_addr(addr);   // avoid to resolve address again

            // Send the parcel through the parcel handler
            hpx::applier::get_applier().get_parcel_handler().put_parcel(p);
            return false;     // destination is remote
        }

        template <typename Action>
        inline bool
        apply_r_p_route(naming::address& addr, actions::continuation* c,
            naming::id_type const& gid, threads::thread_priority priority)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            actions::continuation_type cont(c);

            parcelset::parcel p (gid.get_gid(), new action_type(priority), cont);
            if (components::component_invalid == addr.type_)
                addr.type_ = components::get_component_type<
                    typename action_type::component_type>();
            p.set_destination_addr(addr); // redundant

            // send parcel to agas
            return hpx::applier::get_applier().route(p);
        }

        template <typename Action>
        inline bool
        apply_r (naming::address& addr, actions::continuation* c,
            naming::id_type const& gid)
        {
            return apply_r_p<Action>(addr, c, gid,
                actions::action_priority<Action>());
        }

        template <typename Action>
        inline bool
        apply_r_route (naming::address& addr, actions::continuation* c,
            naming::id_type const& gid)
        {
            return apply_r_p_route<Action>(addr, c, gid,
                actions::action_priority<Action>());
        }

        template <typename Action>
        inline bool
        apply_r_sync_p(naming::address& addr, naming::id_type const& gid,
            threads::thread_priority priority)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid.get_gid(), new action_type(priority));
            if (components::component_invalid == addr.type_)
                addr.type_ = components::get_component_type<
                    typename action_type::component_type>();
            p.set_destination_addr(addr);   // avoid to resolve address again

            // Send the parcel through the parcel handler
            hpx::applier::get_applier().get_parcel_handler().sync_put_parcel(p);
            return false;     // destination is remote
        }

        template <typename Action>
        inline bool
        apply_r_sync (naming::address& addr, naming::id_type const& gid)
        {
            return apply_r_sync_p<Action>(addr, gid,
                actions::action_priority<Action>());
        }

        // We know it is local and has to be directly executed.
        template <typename Action>
        inline bool apply_l_p(actions::continuation* c,
            naming::address const& addr, threads::thread_priority priority)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            BOOST_ASSERT(components::types_are_compatible(addr.type_,
                components::get_component_type<
                    typename action_type::component_type>()));
            actions::continuation_type cont(c);
            apply_helper0<action_type>::call(cont, addr.address_, priority);
            return true;     // no parcel has been sent (dest is local)
        }

        template <typename Action>
        inline bool apply_l (actions::continuation* c, naming::address const& addr)
        {
            return apply_l_p<Action>(c, addr, actions::action_priority<Action>());
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    inline bool apply_p(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority)
    {
        // Determine whether the gid is local or remote
        naming::address addr;
        if (agas::is_local_address(gid, addr))
            return applier::detail::apply_l_p<Action>(c, addr, priority);

        // apply remotely
        return applier::detail::apply_r_p<Action>(addr, c, gid, priority);
    }

    template <typename Action>
    inline bool apply (actions::continuation* c, naming::id_type const& gid)
    {
        return apply_p<Action>(c, gid, actions::action_priority<Action>());
    }

    namespace applier
    {
        template <typename Action>
        inline bool apply_p_route(actions::continuation* c, naming::id_type const& gid,
            threads::thread_priority priority)
        {
            // Determine whether gid is local or remote
            naming::address addr;
            if (agas::is_local_address_cached(gid, addr))
                return detail::apply_l_p<Action>(c, addr, priority); // apply locally

            // apply remotely
            return detail::apply_r_p_route<Action>(addr, c, gid, priority);
        }

        template <typename Action>
        inline bool apply_route (actions::continuation* c, naming::id_type const& gid)
        {
            return apply_p_route<Action>(c, gid, actions::action_priority<Action>());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action>
        inline bool
        apply_c_p(naming::address& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r_p<Action>(addr,
                new actions::base_lco_continuation<result_type>(contgid),
                gid, priority);
        }

        template <typename Action>
        inline bool
        apply_c_p_route(naming::address& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r_p_route<Action>(addr,
                new actions::base_lco_continuation<result_type>(contgid),
                gid, priority);
        }

        template <typename Action>
        inline bool
        apply_c (naming::address& addr, naming::id_type const& contgid,
            naming::id_type const& gid)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r<Action>(addr,
                new actions::base_lco_continuation<result_type>(contgid), gid);
        }

        template <typename Action>
        inline bool
        apply_c_route (naming::address& addr, naming::id_type const& contgid,
            naming::id_type const& gid)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r_route<Action>(addr,
                new actions::base_lco_continuation<result_type>(contgid), gid);
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action>
    inline bool
    apply_c_p(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p<Action>(
            new actions::base_lco_continuation<result_type>(contgid),
            gid, priority);
    }

    template <typename Action>
    inline bool
    apply_c (naming::id_type const& contgid, naming::id_type const& gid)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply<Action>(
            new actions::base_lco_continuation<result_type>(contgid), gid);
    }

    namespace applier
    {
        template <typename Action>
        inline bool
        apply_c_p_route(naming::id_type const& contgid, naming::id_type const& gid,
            threads::thread_priority priority)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_p_route<Action>(
                new actions::base_lco_continuation<result_type>(contgid),
                gid, priority);
        }

        template <typename Action>
        inline bool
        apply_c_route (naming::id_type const& contgid, naming::id_type const& gid)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_route<Action>(
                new actions::base_lco_continuation<result_type>(contgid), gid);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // one parameter version
    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action, typename Arg0>
        inline bool
        apply_r_p(naming::address& addr, naming::id_type const& gid,
            threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid.get_gid(),
                new action_type(priority, boost::forward<Arg0>(arg0)));
            if (components::component_invalid == addr.type_)
                addr.type_ = components::get_component_type<
                    typename action_type::component_type>();
            p.set_destination_addr(addr);   // avoid to resolve address again

            // Send the parcel through the parcel handler
            hpx::applier::get_applier().get_parcel_handler().put_parcel(p);
            return false;     // destination is remote
        }

        template <typename Action, typename Arg0>
        inline bool
        apply_r_p_route(naming::address& addr, naming::id_type const& gid,
            threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            // create parcel
            // addr is probably redundant here!!
            parcelset::parcel p (gid.get_gid(), new action_type(priority,
                boost::forward<Arg0>(arg0)));
            if (components::component_invalid == addr.type_)
                addr.type_ = components::get_component_type<
                    typename action_type::component_type>();
            p.set_destination_addr(addr);   // probably redundant

            //send parcel to agas
            return hpx::applier::get_applier().route(p);
        }

        template <typename Action, typename Arg0>
        inline bool
        apply_r (naming::address& addr, naming::id_type const& gid,
            BOOST_FWD_REF(Arg0) arg0)
        {
            return apply_r_p<Action>(addr, gid,
                actions::action_priority<Action>(), boost::forward<Arg0>(arg0));
        }

        template <typename Action, typename Arg0>
        inline bool
        apply_r_route (naming::address& addr, naming::id_type const& gid,
            BOOST_FWD_REF(Arg0) arg0)
        {
            return apply_r_p_route<Action>(addr, gid,
                actions::action_priority<Action>(), boost::forward<Arg0>(arg0));
        }

        template <typename Action, typename Arg0>
        inline bool
        apply_r_sync_p(naming::address& addr, naming::id_type const& gid,
            threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid.get_gid(),
                new action_type(priority, boost::forward<Arg0>(arg0)));
            if (components::component_invalid == addr.type_)
                addr.type_ = components::get_component_type<
                    typename action_type::component_type>();
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
            return apply_r_sync_p<Action>(addr, gid,
                actions::action_priority<Action>(), boost::forward<Arg0>(arg0));
        }

        template <typename Action, typename Arg0>
        inline bool
        apply_l_p(naming::address const& addr, threads::thread_priority priority,
            BOOST_FWD_REF(Arg0) arg0)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            BOOST_ASSERT(components::types_are_compatible(addr.type_,
                components::get_component_type<
                    typename action_type::component_type>()));
            apply_helper1<action_type>::call(addr.address_, priority,
                boost::forward<Arg0>(arg0));
            return true;     // no parcel has been sent (dest is local)
        }

        template <typename Action, typename Arg0>
        inline bool
        apply_l (naming::address const& addr, BOOST_FWD_REF(Arg0) arg0)
        {
            return apply_l_p<Action>(addr, actions::action_priority<Action>(),
                boost::forward<Arg0>(arg0));
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Arg0>
    inline bool
    apply_p(naming::id_type const& gid, threads::thread_priority priority,
        BOOST_FWD_REF(Arg0) arg0)
    {
        // Determine whether the gid is local or remote
        naming::address addr;
        if (agas::is_local_address(gid, addr))    // apply locally
            return applier::detail::apply_l_p<Action>(addr, priority,
                boost::forward<Arg0>(arg0));

        // apply remotely
        return applier::detail::apply_r_p<Action>(addr, gid, priority,
            boost::forward<Arg0>(arg0));
    }

    template <typename Action, typename Arg0>
    inline bool
    apply (naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0)
    {
        return apply_p<Action>(gid, actions::action_priority<Action>(),
            boost::forward<Arg0>(arg0));
    }

    namespace applier
    {
        template <typename Action, typename Arg0>
        inline bool
        apply_p_route(naming::id_type const& gid, threads::thread_priority priority,
            BOOST_FWD_REF(Arg0) arg0)
        {
            // Determine if the gid is local, if it is, resolve it from the cache.
            naming::address addr;
            if (agas::is_local_address_cached(gid, addr))
            {
                // addr. is in cache, apply locally
                return detail::apply_l_p<Action>(addr, priority,
                    boost::forward<Arg0>(arg0));
            }

            // apply remote - route
            return detail::apply_r_p_route<Action>(addr, gid, priority,
                boost::forward<Arg0>(arg0));
        }

        template <typename Action, typename Arg0>
        inline bool
        apply_route (naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0)
        {
            return apply_p_route<Action>(gid, actions::action_priority<Action>(),
                boost::forward<Arg0>(arg0));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action, typename Arg0>
        inline bool
        apply_r_p(naming::address& addr, actions::continuation* c,
            naming::id_type const& gid, threads::thread_priority priority,
            BOOST_FWD_REF(Arg0) arg0)
        {
            typedef typename hpx::actions::extract_action<Action>::type
                action_type;

            actions::continuation_type cont(c);

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid.get_gid(),
                new action_type(priority, boost::forward<Arg0>(arg0)), cont);
            if (components::component_invalid == addr.type_)
                addr.type_ = components::get_component_type<
                    typename action_type::component_type>();
            p.set_destination_addr(addr);   // avoid to resolve address again

            // Send the parcel through the parcel handler
            hpx::applier::get_applier().get_parcel_handler().put_parcel(p);
            return false;     // destination is remote
        }

        template <typename Action, typename Arg0>
        inline bool
        apply_r_p_route(naming::address& addr, actions::continuation* c,
            naming::id_type const& gid, threads::thread_priority priority,
            BOOST_FWD_REF(Arg0) arg0)
        {
            typedef typename hpx::actions::extract_action<Action>::type
                action_type;

            actions::continuation_type cont(c);

            // create a parcel and forward it to agas server
            parcelset::parcel p (gid.get_gid(),
                new action_type(priority, boost::forward<Arg0>(arg0)), cont);
            if (components::component_invalid == addr.type_)
                addr.type_ = components::get_component_type<
                    typename action_type::component_type>();
            p.set_destination_addr(addr);

            // send the parcel to the applier to send to agas server
            return hpx::applier::get_applier().route(p);
        }

        template <typename Action, typename Arg0>
        inline bool
        apply_r (naming::address& addr, actions::continuation* c,
            naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0)
        {
            return apply_r_p<Action>(addr, c, gid,
                actions::action_priority<Action>(), boost::forward<Arg0>(arg0));
        }

        template <typename Action, typename Arg0>
        inline bool
        apply_r_route (naming::address& addr, actions::continuation* c,
            naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0)
        {
            return apply_r_p_route<Action>(addr, c, gid,
                actions::action_priority<Action>(), boost::forward<Arg0>(arg0));
        }

        template <typename Action, typename Arg0>
        inline bool
        apply_l_p(actions::continuation* c, naming::address const& addr,
            threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            BOOST_ASSERT(components::types_are_compatible(addr.type_,
                components::get_component_type<
                    typename action_type::component_type>()));
            actions::continuation_type cont(c);
            apply_helper1<action_type>::call(cont, addr.address_, priority,
                boost::forward<Arg0>(arg0));
            return true;     // no parcel has been sent (dest is local)
        }

        template <typename Action, typename Arg0>
        inline bool
        apply_l (actions::continuation* c, naming::address const& addr,
            BOOST_FWD_REF(Arg0) arg0)
        {
            return apply_l_p<Action>(c, addr,
                actions::action_priority<Action>(), boost::forward<Arg0>(arg0));
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Arg0>
    inline bool
    apply_p(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0)
    {
        // Determine whether the gid is local or remote
        naming::address addr;
        if (agas::is_local_address(gid, addr))    // apply locally
            return applier::detail::apply_l_p<Action>(c, addr, priority,
                boost::forward<Arg0>(arg0));

        // apply remotely
        return applier::detail::apply_r_p<Action>(addr, c, gid, priority,
            boost::forward<Arg0>(arg0));
    }

    template <typename Action, typename Arg0>
    inline bool
    apply (actions::continuation* c, naming::id_type const& gid,
        BOOST_FWD_REF(Arg0) arg0)
    {
        return apply_p<Action>(c, gid, actions::action_priority<Action>(),
            boost::forward<Arg0>(arg0));
    }

    namespace applier
    {
        template <typename Action, typename Arg0>
        inline bool
        apply_p_route(actions::continuation* c, naming::id_type const& gid,
            threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0)
        {
            naming::address addr;
            if (agas::is_local_address_cached(gid, addr))
                return detail::apply_l_p<Action>(c, addr, priority,
                    boost::forward<Arg0>(arg0));  // apply locally

            // send parcel to agas
            return detail::apply_r_p_route<Action>(addr, c, gid, priority,
                boost::forward<Arg0>(arg0));
        }

        template <typename Action, typename Arg0>
        inline bool
        apply_route (actions::continuation* c, naming::id_type const& gid,
            BOOST_FWD_REF(Arg0) arg0)
        {
            return apply_p_route<Action>(c, gid,
                actions::action_priority<Action>(), boost::forward<Arg0>(arg0));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        // \brief Invoke an unary action with a \b actions::continuation
        //        at a specific priority using a pre-resolved remote
        //        \b naming::address. Takes a continuation by GID.
        template <typename Action, typename Arg0>
        inline bool
        apply_c_p(naming::address& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            BOOST_FWD_REF(Arg0) arg0)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return detail::apply_r_p<Action>(addr,
                new actions::base_lco_continuation<result_type>(contgid), gid,
                actions::action_priority<Action>(), boost::forward<Arg0>(arg0));
        }

        template <typename Action, typename Arg0>
        inline bool
        apply_c_p_route(naming::address& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            BOOST_FWD_REF(Arg0) arg0)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return detail::apply_r_p_route<Action>(addr,
                new actions::base_lco_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                boost::forward<Arg0>(arg0));
        }

        // \brief Invoke an unary action with a \b actions::continuation
        //        using a pre-resolved remote \b naming::address. Takes a
        //        continuation by GID.
        template <typename Action, typename Arg0>
        inline bool
        apply_c (naming::address& addr, naming::id_type const& contgid,
            naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r<Action>(addr,
                new actions::base_lco_continuation<result_type>(contgid), gid,
                boost::forward<Arg0>(arg0));
        }

        template <typename Action, typename Arg0>
        inline bool
        apply_c_route (naming::address& addr, naming::id_type const& contgid,
            naming::id_type const& gid, BOOST_FWD_REF(Arg0) arg0)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r_route<Action>(addr,
                new actions::base_lco_continuation<result_type>(contgid),
                gid, boost::forward<Arg0>(arg0));
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    /// \brief Invoke an unary action with a \b actions::continuation at a
    ///        specific priority on a local or remote GID. Takes a continuation
    ///        by GID.
    ///
    /// \param contgid      [in] The GID of the continuation.
    /// \param gid          [in] The target of the action.
    /// \param arg0         [in] Value to bind to the first argument of the
    ///                     action.
    template <typename Action, typename Arg0>
    inline bool
    apply_c_p(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p<Action>(
            new actions::base_lco_continuation<result_type>(contgid),
            gid, priority, boost::forward<Arg0>(arg0));
    }

    template <typename Action, typename Arg0>
    inline bool
    apply_c (naming::id_type const& contgid, naming::id_type const& gid,
        BOOST_FWD_REF(Arg0) arg0)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply<Action>(
            new actions::base_lco_continuation<result_type>(contgid),
            gid, boost::forward<Arg0>(arg0));
    }

    namespace applier
    {
        template <typename Action, typename Arg0>
        inline bool
        apply_c_p_route(naming::id_type const& contgid, naming::id_type const& gid,
            threads::thread_priority priority, BOOST_FWD_REF(Arg0) arg0)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_p_route<Action>(
                new actions::base_lco_continuation<result_type>(contgid),
                gid, priority, boost::forward<Arg0>(arg0));
        }

        template <typename Action, typename Arg0>
        inline bool
        apply_c_route (naming::id_type const& contgid, naming::id_type const& gid,
            BOOST_FWD_REF(Arg0) arg0)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_route<Action>(
                new actions::base_lco_continuation<result_type>(contgid),
                gid, boost::forward<Arg0>(arg0));
        }
    }
}

// bring in the rest of the apply<> overloads (arity 2+)
#include <hpx/runtime/applier/apply_implementations.hpp>

#endif
