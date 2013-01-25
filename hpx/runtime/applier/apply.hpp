//  Copyright (c) 2007-2013 Hartmut Kaiser
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
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/util/remove_local_destinations.hpp>

#include <boost/dynamic_bitset.hpp>

#include <vector>
#include <map>
#include <algorithm>

// FIXME: Error codes?

namespace hpx { namespace actions
{
    template <typename Action>
    threads::thread_priority action_priority()
    {
        typedef typename hpx::actions::extract_action<Action>::type action_type;
        return static_cast<threads::thread_priority>(
            traits::action_priority<action_type>::value);
    }
}}

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    // zero parameter version of apply()
    // Invoked by a running HPX-thread to apply an action to any resource
    namespace applier { namespace detail
    {
        template <typename Action>
        inline naming::address& complement_addr(naming::address& addr)
        {
            if (components::component_invalid == addr.type_)
            {
                addr.type_ = components::get_component_type<
                    typename Action::component_type>();
            }
            return addr;
        }

        inline naming::gid_type const& convert_to_gid(naming::id_type const& id)
        {
            return id.get_gid();
        }

        // We know it is remote.
        template <typename Action>
        inline bool
        apply_r_p(naming::address& addr, naming::id_type const& gid,
            threads::thread_priority priority)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid.get_gid(), complement_addr<action_type>(addr),
                new hpx::actions::transfer_action<action_type>(priority));

            // Send the parcel through the parcel handler
            hpx::applier::get_applier().get_parcel_handler().put_parcel(p);
            return false;     // destination is remote
        }

        struct destinations
        {
            std::vector<naming::gid_type> gids_;
            std::vector<naming::address> addrs_;
        };

        struct send_parcel
        {
            send_parcel(parcelset::parcelhandler& ph, actions::action_type act)
              : ph_(ph), act_(act)
            {}

            void operator()(
                std::pair<naming::locality const, destinations>& e) const
            {
                // Create a new parcel to be sent to the destination with the
                // gid, action, and arguments
                parcelset::parcel p(e.second.gids_, e.second.addrs_, act_);

                // Send the parcel through the parcel handler
                ph_.put_parcel(p);
            }

            parcelset::parcelhandler& ph_;
            actions::action_type act_;
        };

        template <typename Action>
        inline bool
        apply_r_p(std::vector<naming::address>& addrs,
            std::vector<naming::gid_type> const& gids,
            threads::thread_priority priority)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            // sort destinations
            std::map<naming::locality, destinations> dests;

            std::size_t count = gids.size();
            for (std::size_t i = 0; i < count; ++i) {
                complement_addr<action_type>(addrs[i]);

                destinations& dest = dests[addrs[i].locality_];
                dest.gids_.push_back(gids[i]);
                dest.addrs_.push_back(addrs[i]);
            }

            // send one parcel to each of the destination localities
            parcelset::parcelhandler& ph =
                hpx::applier::get_applier().get_parcel_handler();
            actions::action_type act(
                new hpx::actions::transfer_action<action_type>(priority));

            std::for_each(dests.begin(), dests.end(), send_parcel(ph, act));

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
            parcelset::parcel p (gid.get_gid(), complement_addr<action_type>(addr),
                new hpx::actions::transfer_action<action_type>(priority));

            // Send the parcel to applier to be sent to the AGAS server
            return hpx::applier::get_applier().route(p);
        }

        template <typename Action>
        inline bool
        apply_r (naming::address& addr, naming::id_type const& gid)
        {
            return apply_r_p<Action>(addr, gid, actions::action_priority<Action>());
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
            util::tuple0<> env;
            apply_helper<action_type>::call(addr.address_, priority, env);
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

    template <typename Component, typename Result,
        typename Arguments, typename Derived, threads::thread_priority Priority>
    inline bool apply (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /*act*/, naming::id_type const& gid)
    {
        return apply_p<Derived>(gid, actions::action_priority<Derived>());
    }

    // same for multiple destinations
    template <typename Action>
    inline bool
    apply_p (std::vector<naming::id_type> const& ids, threads::thread_priority priority)
    {
        // Determine whether the gids are local or remote
        std::vector<naming::gid_type> gids;
        std::vector<naming::address> addrs;
        boost::dynamic_bitset<> locals;

        std::size_t count = ids.size();
        gids.reserve(count);
        if (agas::is_local_address(ids, addrs, locals)) {
            // at least one destination is local
            for (std::size_t i = 0; i < count; ++i) {
                if (locals.test(i))
                    applier::detail::apply_l_p<Action>(addrs[i], priority);
                gids.push_back(applier::detail::convert_to_gid(ids[i]));
            }

            // remove local destinations
            std::vector<naming::gid_type>::iterator it =
                util::remove_local_destinations(gids, addrs, locals);
            if (it == gids.begin())
                return true;        // all destinations are local

            gids.erase(it, gids.end());
            addrs.resize(gids.size());
        }
        else {
            std::transform(ids.begin(), ids.end(), std::back_inserter(gids),
                applier::detail::convert_to_gid);
        }

        // apply remotely
        return applier::detail::apply_r_p<Action>(addrs, gids, priority);
    }

    template <typename Action>
    inline bool apply (std::vector<naming::id_type> const& gids)
    {
        return apply_p<Action>(gids, actions::action_priority<Action>());
    }

    template <typename Component, typename Result,
        typename Arguments, typename Derived>
    inline bool apply (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /*act*/, std::vector<naming::id_type> const& gids)
    {
        return apply_p<Derived>(gids, actions::action_priority<Derived>());
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
            parcelset::parcel p (gid.get_gid(), complement_addr<action_type>(addr),
                new hpx::actions::transfer_action<action_type>(priority), cont);

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

            parcelset::parcel p (gid.get_gid(), complement_addr<action_type>(addr),
                new hpx::actions::transfer_action<action_type>(priority), cont);

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
            parcelset::parcel p (gid.get_gid(), complement_addr<action_type>(addr),
                new hpx::actions::transfer_action<action_type>(priority));

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
            util::tuple0<> env;
            apply_helper<action_type>::call(cont, addr.address_, priority, env);
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

    template <typename Component, typename Result,
        typename Arguments, typename Derived>
    inline bool apply (actions::continuation* c,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /*act*/, naming::id_type const& gid)
    {
        return apply_p<Derived>(c, gid, actions::action_priority<Derived>());
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
                new actions::typed_continuation<result_type>(contgid),
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
                new actions::typed_continuation<result_type>(contgid),
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
                new actions::typed_continuation<result_type>(contgid), gid);
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
                new actions::typed_continuation<result_type>(contgid), gid);
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
            new actions::typed_continuation<result_type>(contgid),
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
            new actions::typed_continuation<result_type>(contgid), gid);
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
                new actions::typed_continuation<result_type>(contgid),
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
                new actions::typed_continuation<result_type>(contgid), gid);
        }
    }
}

// bring in the rest of the apply<> overloads (arity 1+)
#include <hpx/runtime/applier/apply_implementations.hpp>

#endif
