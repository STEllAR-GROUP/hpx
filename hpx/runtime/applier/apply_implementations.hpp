//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_APPLIER_APPLY_IMPLEMENTATIONS_JUN_09_2008_0434PM)
#define HPX_APPLIER_APPLY_IMPLEMENTATIONS_JUN_09_2008_0434PM

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (1, HPX_ACTION_ARGUMENT_LIMIT,                                        \
    "hpx/runtime/applier/apply_implementations.hpp"))                         \
    /**/

#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

#define HPX_FWD_ARGS(z, n, _)                                                 \
        BOOST_PP_COMMA_IF(n)                                                  \
            BOOST_FWD_REF(BOOST_PP_CAT(Arg, n)) BOOST_PP_CAT(arg, n)          \
    /**/
#define HPX_FORWARD_ARGS(z, n, _)                                             \
        BOOST_PP_COMMA_IF(n)                                                  \
            boost::forward<BOOST_PP_CAT(Arg, n)>(BOOST_PP_CAT(arg, n))        \
    /**/

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_r_p(naming::address& addr, naming::id_type const& gid,
            threads::thread_priority priority, BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid.get_gid(),
                new hpx::actions::transfer_action<action_type>(
                    priority, BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _)));
            if (components::component_invalid == addr.type_)
                addr.type_ = components::get_component_type<
                    typename action_type::component_type>();
            p.set_destination_addr(addr);   // avoid to resolve address again

            // Send the parcel through the parcel handler
            hpx::applier::get_applier().get_parcel_handler().put_parcel(p);
            return false;     // destination is remote
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_r_p_route(naming::address& addr, naming::id_type const& gid,
            threads::thread_priority priority, BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            // create parcel
            parcelset::parcel p(gid.get_gid(),
                new hpx::actions::transfer_action<action_type>(
                    priority, BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _)));
            if (components::component_invalid == addr.type_)
                addr.type_ = components::get_component_type<
                    typename action_type::component_type>();
            p.set_destination_addr(addr); // redundant

            // send parcel to agas
            return hpx::applier::get_applier().route(p);
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_r (naming::address& addr, naming::id_type const& gid,
            BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            return apply_r_p<Action>(addr, gid,
                actions::action_priority<Action>(),
                BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_r_route (naming::address& addr, naming::id_type const& gid,
            BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            return apply_r_p_route<Action>(addr, gid,
                actions::action_priority<Action>(),
                BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_l_p(naming::address const& addr, threads::thread_priority priority,
            BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            BOOST_ASSERT(components::types_are_compatible(addr.type_,
                components::get_component_type<
                    typename action_type::component_type>()));

            apply_helper<action_type>::call(addr.address_, priority,
                util::make_argument_pack(BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _)));
            return true;     // no parcel has been sent (dest is local)
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_l (naming::address const& addr, BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            return apply_l_p<Action>(addr,
                actions::action_priority<Action>(),
                BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply_p(naming::id_type const& gid, threads::thread_priority priority,
        BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        // Determine whether the gid is local or remote
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            return applier::detail::apply_l_p<Action>(addr, priority,
                BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        }

        // apply remotely
        return applier::detail::apply_r_p<Action>(addr, gid, priority,
            BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply (naming::id_type const& gid, BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        return apply_p<Action>(gid, actions::action_priority<Action>(),
            BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
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
        BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        return apply_p<Derived>(gid, actions::action_priority<Derived>(),
            BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
    }

    namespace applier
    {
        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_p_route(naming::id_type const& gid,
            threads::thread_priority priority,
            BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            // Determine whether the gid is local or remote
            naming::address addr;
            if (agas::is_local_address_cached(gid, addr)) {
                return detail::apply_l_p<Action>(addr, priority,
                    BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
            }

            // apply remotely
            return detail::apply_r_p_route<Action>(addr, gid, priority,
                BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_route (naming::id_type const& gid, BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            return apply_p_route<Action>(gid,
                actions::action_priority<Action>(),
                BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_r_p(naming::address& addr, actions::continuation* c,
            naming::id_type const& gid, threads::thread_priority priority,
            BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            actions::continuation_type cont(c);

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid.get_gid(),
                new hpx::actions::transfer_action<action_type>(
                    priority, BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _)), cont);
            if (components::component_invalid == addr.type_)
                addr.type_ = components::get_component_type<
                    typename action_type::component_type>();
            p.set_destination_addr(addr);   // avoid to resolve address again

            // Send the parcel through the parcel handler
            hpx::applier::get_applier().get_parcel_handler().put_parcel(p);
            return false;     // destination is remote
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_r_p_route(naming::address& addr, actions::continuation* c,
            naming::id_type const& gid, threads::thread_priority priority,
            BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            actions::continuation_type cont(c);

            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid.get_gid(),
                new hpx::actions::transfer_action<action_type>(
                    priority, BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _)), cont);
            if (components::component_invalid == addr.type_)
                addr.type_ = components::get_component_type<
                    typename action_type::component_type>();
            p.set_destination_addr(addr);   // redundant

            // Send the parcel to agas
            return hpx::applier::get_applier().route(p);
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_r (naming::address& addr, actions::continuation* c,
            naming::id_type const& gid, BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            return apply_r_p<Action>(addr, c, gid,
                actions::action_priority<Action>(),
                BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_r_route (naming::address& addr, actions::continuation* c,
            naming::id_type const& gid, BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            return apply_r_p_route<Action>(addr, c, gid,
                actions::action_priority<Action>(),
                BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_l_p(actions::continuation* c, naming::address const& addr,
            threads::thread_priority priority, BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            BOOST_ASSERT(components::types_are_compatible(addr.type_,
                components::get_component_type<
                    typename action_type::component_type>()));
            actions::continuation_type cont(c);

            apply_helper<action_type>::call(
                cont, addr.address_, priority,
                util::make_argument_pack(BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _)));
            return true;     // no parcel has been sent (dest is local)
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_l (actions::continuation* c, naming::address const& addr,
            BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            return apply_l_p<Action>(c, addr,
                actions::action_priority<Action>(),
                BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply_p(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority, BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        // Determine whether the gid is local or remote
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            return applier::detail::apply_l_p<Action>(c, addr, priority,
                BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        }

        // apply remotely
        return applier::detail::apply_r_p<Action>(addr, c, gid, priority,
            BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply (actions::continuation* c, naming::id_type const& gid,
        BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        return apply_p<Action>(c, gid, actions::action_priority<Action>(),
            BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
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
        BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        return apply_p<Derived>(c, gid, actions::action_priority<Derived>(),
            BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
    }

    namespace applier
    {
        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_p_route(actions::continuation* c, naming::id_type const& gid,
            threads::thread_priority priority, BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            // Determine whether the gid is local or remote
            naming::address addr;
            if (agas::is_local_address_cached(gid, addr)) {
                return detail::apply_l_p<Action>(c, addr, priority,
                    BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
            }

            // apply remotely
            return detail::apply_r_p_route<Action>(addr, c, gid, priority,
                BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_route (actions::continuation* c, naming::id_type const& gid,
            BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            return apply_p_route<Action>(c, gid,
                actions::action_priority<Action>(),
                BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_c_p(naming::address& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r_p<Action>(addr,
                new actions::base_lco_continuation<result_type>(contgid),
                gid, priority, BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_c_p_route(naming::address& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r_p_route<Action>(addr,
                new actions::base_lco_continuation<result_type>(contgid),
                gid, priority, BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_c (naming::address& addr, naming::id_type const& contgid,
            naming::id_type const& gid, BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r_p<Action>(addr,
                new actions::base_lco_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_c_route (naming::address& addr, naming::id_type const& contgid,
            naming::id_type const& gid, BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r_p_route<Action>(addr,
                new actions::base_lco_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        }
    }}

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply_c_p(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p<Action>(
            new actions::base_lco_continuation<result_type>(contgid),
            gid, priority, BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
    }

    template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
    inline bool
    apply_c (naming::id_type const& contgid, naming::id_type const& gid,
        BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p<Action>(
            new actions::base_lco_continuation<result_type>(contgid),
            gid, actions::action_priority<Action>(),
            BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
    }

    namespace applier
    {
        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_c_p_route(naming::id_type const& contgid, naming::id_type const& gid,
            threads::thread_priority priority, BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_route<Action>(
                new actions::base_lco_continuation<result_type>(contgid),
                gid, priority, BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        }

        template <typename Action, BOOST_PP_ENUM_PARAMS(N, typename Arg)>
        inline bool
        apply_c_route (naming::id_type const& contgid, naming::id_type const& gid,
            BOOST_PP_REPEAT(N, HPX_FWD_ARGS, _))
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_p_route<Action>(
                new actions::base_lco_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                BOOST_PP_REPEAT(N, HPX_FORWARD_ARGS, _));
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
#undef HPX_APPLY_BIND_REFERENCE
#undef HPX_FORWARD_ARGS
#undef HPX_FWD_ARGS
#undef N

#endif
