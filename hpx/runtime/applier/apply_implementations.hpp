//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_APPLIER_APPLY_IMPLEMENTATIONS_JUN_09_2008_0434PM)
#define HPX_APPLIER_APPLY_IMPLEMENTATIONS_JUN_09_2008_0434PM

#include <hpx/util/move.hpp>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#if !defined(HPX_DONT_USE_PREPROCESSED_FILES)
#  include <hpx/runtime/applier/preprocessed/apply_implementations.hpp>
#else

#if defined(__WAVE__) && defined(HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(preserve: 1, line: 0, output: "preprocessed/apply_implementations_" HPX_LIMIT_STR ".hpp")
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/applier/apply_implementations.hpp"))                         \
    /**/

#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined (HPX_CREATE_PREPROCESSED_FILES)
#  pragma wave option(output: null)
#endif

#endif // !defined(HPX_DONT_USE_PREPROCESSED_FILES)

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_r_p(naming::address& addr, naming::id_type const& gid,
            threads::thread_priority priority, HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid.get_gid(), complement_addr<action_type>(addr),
                new hpx::actions::transfer_action<action_type>(
                    priority, HPX_ENUM_FORWARD_ARGS(N, Arg, arg)));

            // Send the parcel through the parcel handler
            hpx::applier::get_applier().get_parcel_handler().put_parcel(p);
            return false;     // destinations are remote
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_r_p(std::vector<naming::address>& addrs,
            std::vector<naming::gid_type> const& gids,
            threads::thread_priority priority, HPX_ENUM_FWD_ARGS(N, Arg, arg))
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
                new hpx::actions::transfer_action<action_type>(
                    priority, HPX_ENUM_FORWARD_ARGS(N, Arg, arg)));

            std::for_each(dests.begin(), dests.end(), send_parcel(ph, act));

            return false;     // destinations are remote
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_r_p_route(naming::address& addr, naming::id_type const& gid,
            threads::thread_priority priority, HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            // create parcel
            parcelset::parcel p(gid.get_gid(), complement_addr<action_type>(addr),
                new hpx::actions::transfer_action<action_type>(
                    priority, HPX_ENUM_FORWARD_ARGS(N, Arg, arg)));

            // send parcel to agas
            return hpx::applier::get_applier().route(p);
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_r (naming::address& addr, naming::id_type const& gid,
            HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            return apply_r_p<Action>(addr, gid,
                actions::action_priority<Action>(),
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_r_route (naming::address& addr, naming::id_type const& gid,
            HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            return apply_r_p_route<Action>(addr, gid,
                actions::action_priority<Action>(),
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_l_p(naming::address const& addr, threads::thread_priority priority,
            HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            BOOST_ASSERT(components::types_are_compatible(addr.type_,
                components::get_component_type<
                    typename action_type::component_type>()));

            apply_helper<action_type>::call(addr.address_, priority,
                util::forward_as_tuple(HPX_ENUM_FORWARD_ARGS(N, Arg, arg)));
            return true;     // no parcel has been sent (dest is local)
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_l (naming::address const& addr, HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            return apply_l_p<Action>(addr,
                actions::action_priority<Action>(),
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply_p(naming::id_type const& gid, threads::thread_priority priority,
        HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        // Determine whether the gid is local or remote
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            return applier::detail::apply_l_p<Action>(addr, priority,
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }

        // apply remotely
        return applier::detail::apply_r_p<Action>(addr, gid, priority,
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply (naming::id_type const& gid, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return apply_p<Action>(gid, actions::action_priority<Action>(),
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    template <typename Component, int Action, typename Result,
        typename Arguments, typename Derived, threads::thread_priority Priority,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply (
        hpx::actions::action<
            Component, Action, Result, Arguments, Derived, Priority
        > /*act*/,
        naming::id_type const& gid,
        HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return apply_p<Derived>(gid, actions::action_priority<Derived>(),
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    // same for multiple destinations
    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply_p(std::vector<naming::id_type> const& ids,
        threads::thread_priority priority, HPX_ENUM_FWD_ARGS(N, Arg, arg))
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
                if (locals.test(i)) {
                    applier::detail::apply_l_p<Action>(addrs[i], priority,
                        HPX_ENUM_FORWARD_ARGS(N, Arg, arg));   // apply locally
                }
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
        return applier::detail::apply_r_p<Action>(addrs, gids, priority,
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply (std::vector<naming::id_type> const& gids,
        HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return apply_p<Action>(gids, actions::action_priority<Action>(),
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    template <typename Component, int Action, typename Result,
        typename Arguments, typename Derived, threads::thread_priority Priority,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply (
        hpx::actions::action<
            Component, Action, Result, Arguments, Derived, Priority
        > /*act*/,
        std::vector<naming::id_type> const& gids,
        HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return apply_p<Derived>(gids, actions::action_priority<Derived>(),
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    namespace applier
    {
        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_p_route(naming::id_type const& gid,
            threads::thread_priority priority,
            HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            // Determine whether the gid is local or remote
            naming::address addr;
            if (agas::is_local_address_cached(gid, addr)) {
                return detail::apply_l_p<Action>(addr, priority,
                    HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
            }

            // apply remotely
            return detail::apply_r_p_route<Action>(addr, gid, priority,
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_route (naming::id_type const& gid, HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            return apply_p_route<Action>(gid,
                actions::action_priority<Action>(),
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_r_p(naming::address& addr, actions::continuation* c,
            naming::id_type const& gid, threads::thread_priority priority,
            HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            actions::continuation_type cont(c);

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid.get_gid(), complement_addr<action_type>(addr),
                new hpx::actions::transfer_action<action_type>(
                    priority, HPX_ENUM_FORWARD_ARGS(N, Arg, arg)), cont);

            // Send the parcel through the parcel handler
            hpx::applier::get_applier().get_parcel_handler().put_parcel(p);
            return false;     // destination is remote
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_r_p_route(naming::address& addr, actions::continuation* c,
            naming::id_type const& gid, threads::thread_priority priority,
            HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            actions::continuation_type cont(c);

            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid.get_gid(), complement_addr<action_type>(addr),
                new hpx::actions::transfer_action<action_type>(
                    priority, HPX_ENUM_FORWARD_ARGS(N, Arg, arg)), cont);

            // Send the parcel to agas
            return hpx::applier::get_applier().route(p);
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_r (naming::address& addr, actions::continuation* c,
            naming::id_type const& gid, HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            return apply_r_p<Action>(addr, c, gid,
                actions::action_priority<Action>(),
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_r_route (naming::address& addr, actions::continuation* c,
            naming::id_type const& gid, HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            return apply_r_p_route<Action>(addr, c, gid,
                actions::action_priority<Action>(),
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_l_p(actions::continuation* c, naming::address const& addr,
            threads::thread_priority priority, HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            BOOST_ASSERT(components::types_are_compatible(addr.type_,
                components::get_component_type<
                    typename action_type::component_type>()));
            actions::continuation_type cont(c);

            apply_helper<action_type>::call(
                cont, addr.address_, priority,
                util::forward_as_tuple(HPX_ENUM_FORWARD_ARGS(N, Arg, arg)));
            return true;     // no parcel has been sent (dest is local)
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_l (actions::continuation* c, naming::address const& addr,
            HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            return apply_l_p<Action>(c, addr,
                actions::action_priority<Action>(),
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply_p(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        // Determine whether the gid is local or remote
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            return applier::detail::apply_l_p<Action>(c, addr, priority,
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }

        // apply remotely
        return applier::detail::apply_r_p<Action>(addr, c, gid, priority,
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply (actions::continuation* c, naming::id_type const& gid,
        HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return apply_p<Action>(c, gid, actions::action_priority<Action>(),
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    template <typename Component, int Action, typename Result,
        typename Arguments, typename Derived, threads::thread_priority Priority,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply (actions::continuation* c,
        hpx::actions::action<
            Component, Action, Result, Arguments, Derived, Priority
        > /*act*/,
        naming::id_type const& gid,
        HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return apply_p<Derived>(c, gid, actions::action_priority<Derived>(),
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    namespace applier
    {
        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_p_route(actions::continuation* c, naming::id_type const& gid,
            threads::thread_priority priority, HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            // Determine whether the gid is local or remote
            naming::address addr;
            if (agas::is_local_address_cached(gid, addr)) {
                return detail::apply_l_p<Action>(c, addr, priority,
                    HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
            }

            // apply remotely
            return detail::apply_r_p_route<Action>(addr, c, gid, priority,
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_route (actions::continuation* c, naming::id_type const& gid,
            HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            return apply_p_route<Action>(c, gid,
                actions::action_priority<Action>(),
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_c_p(naming::address& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r_p<Action>(addr,
                new actions::base_lco_continuation<result_type>(contgid),
                gid, priority, HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_c_p_route(naming::address& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r_p_route<Action>(addr,
                new actions::base_lco_continuation<result_type>(contgid),
                gid, priority, HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_c (naming::address& addr, naming::id_type const& contgid,
            naming::id_type const& gid, HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r_p<Action>(addr,
                new actions::base_lco_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_c_route (naming::address& addr, naming::id_type const& contgid,
            naming::id_type const& gid, HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r_p_route<Action>(addr,
                new actions::base_lco_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }
    }}

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply_c_p(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p<Action>(
            new actions::base_lco_continuation<result_type>(contgid),
            gid, priority, HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply_c (naming::id_type const& contgid, naming::id_type const& gid,
        HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p<Action>(
            new actions::base_lco_continuation<result_type>(contgid),
            gid, actions::action_priority<Action>(),
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    namespace applier
    {
        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_c_p_route(naming::id_type const& contgid, naming::id_type const& gid,
            threads::thread_priority priority, HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_route<Action>(
                new actions::base_lco_continuation<result_type>(contgid),
                gid, priority, HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_c_route (naming::id_type const& contgid, naming::id_type const& gid,
            HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_p_route<Action>(
                new actions::base_lco_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
#undef N

#endif
