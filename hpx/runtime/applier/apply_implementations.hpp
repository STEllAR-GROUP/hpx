//  Copyright (c) 2007-2013 Hartmut Kaiser
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

#if !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)
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

#endif // !defined(HPX_USE_PREPROCESSOR_LIMIT_EXPANSION)

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
        apply_r (naming::address& addr, naming::id_type const& gid,
            HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            return apply_r_p<Action>(addr, gid,
                actions::action_priority<Action>(),
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_l_p(naming::id_type const& target, naming::address const& addr,
            threads::thread_priority priority, HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            BOOST_ASSERT(components::types_are_compatible(addr.type_,
                components::get_component_type<
                    typename action_type::component_type>()));

            apply_helper<action_type>::call(target, addr.address_, priority,
                util::forward_as_tuple(HPX_ENUM_FORWARD_ARGS(N, Arg, arg)));
            return true;     // no parcel has been sent (dest is local)
        }

        // same as above, but taking all arguments by value
        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_l_p_val(naming::id_type const& target, naming::address const& addr,
            threads::thread_priority priority, BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, arg))
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            BOOST_ASSERT(components::types_are_compatible(addr.type_,
                components::get_component_type<
                    typename action_type::component_type>()));

            apply_helper<action_type>::call(target, addr.address_, priority,
                util::forward_as_tuple(HPX_ENUM_MOVE_ARGS(N, arg)));
            return true;     // no parcel has been sent (dest is local)
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_l (naming::id_type const& target, naming::address const& addr,
            HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            return apply_l_p<Action>(target, addr,
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
        if (!Action::is_target_valid(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p", 
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }

        // Determine whether the gid is local or remote
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            return applier::detail::apply_l_p<Action>(gid, addr, priority,
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

    template <typename Component, typename Result, typename Arguments,
        typename Derived, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /*act*/, naming::id_type const& gid, HPX_ENUM_FWD_ARGS(N, Arg, arg))
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
                    // apply locally, do not move arguments
                    applier::detail::apply_l_p_val<Action>(ids[i], addrs[i], priority,
                        BOOST_PP_ENUM_PARAMS(N, arg));
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

    template <typename Component, typename Result, typename Arguments, 
        typename Derived, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /*act*/,
        std::vector<naming::id_type> const& gids,
        HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return apply_p<Derived>(gids, actions::action_priority<Derived>(),
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
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
        apply_r (naming::address& addr, actions::continuation* c,
            naming::id_type const& gid, HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            return apply_r_p<Action>(addr, c, gid,
                actions::action_priority<Action>(),
                HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_l_p(actions::continuation* c, naming::id_type const& target,
            naming::address const& addr, threads::thread_priority priority,
            HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            BOOST_ASSERT(components::types_are_compatible(addr.type_,
                components::get_component_type<
                    typename action_type::component_type>()));
            actions::continuation_type cont(c);

            apply_helper<action_type>::call(
                cont, target, addr.address_, priority,
                util::forward_as_tuple(HPX_ENUM_FORWARD_ARGS(N, Arg, arg)));
            return true;     // no parcel has been sent (dest is local)
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_l (actions::continuation* c, naming::id_type const& target,
            naming::address const& addr, HPX_ENUM_FWD_ARGS(N, Arg, arg))
        {
            return apply_l_p<Action>(c, target, addr,
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
        if (!Action::is_target_valid(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p", 
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }

        // Determine whether the gid is local or remote
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            return applier::detail::apply_l_p<Action>(c, gid, addr, priority,
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

    template <typename Component, typename Result,
        typename Arguments, typename Derived,
        BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply (actions::continuation* c,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /*act*/,
        naming::id_type const& gid,
        HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        return apply_p<Derived>(c, gid, actions::action_priority<Derived>(),
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
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
                new actions::typed_continuation<result_type>(contgid),
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
                new actions::typed_continuation<result_type>(contgid),
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
            new actions::typed_continuation<result_type>(contgid),
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
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Action>(),
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }

    template <typename Component, typename Result, typename Arguments,
        typename Derived, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply_c (
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /*act*/, naming::id_type const& contgid, naming::id_type const& gid, 
        HPX_ENUM_FWD_ARGS(N, Arg, arg))
    {
        typedef
            typename hpx::actions::extract_action<Derived>::result_type
            result_type;

        return apply_p<Derived>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Derived>(),
            HPX_ENUM_FORWARD_ARGS(N, Arg, arg));
    }
}

///////////////////////////////////////////////////////////////////////////////
#undef N

#endif
