//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLIER_APPLY_CALLBACK_DEC_16_2012_1228PM)
#define HPX_APPLIER_APPLY_CALLBACK_DEC_16_2012_1228PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>

#include <hpx/runtime/applier/apply.hpp>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    // zero parameter version of apply()
    // Invoked by a running HPX-thread to apply an action to any resource
    namespace applier { namespace detail
    {
        // We know it is remote.
        template <typename Action, typename Callback>
        inline bool
        apply_r_p_cb(naming::address& addr, naming::id_type const& gid,
            threads::thread_priority priority, BOOST_FWD_REF(Callback) cb)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid.get_gid(), complement_addr<action_type>(addr),
                new hpx::actions::transfer_action<action_type>(priority));

            // Send the parcel through the parcel handler
            hpx::applier::get_applier().get_parcel_handler()
                .put_parcel(p, boost::forward<Callback>(cb));
            return false;     // destination is remote
        }

        template <typename Action, typename Callback>
        inline bool
        apply_r_cb(naming::address& addr, naming::id_type const& gid,
            BOOST_FWD_REF(Callback) cb)
        {
            return apply_r_p_cb<Action>(addr, gid, actions::action_priority<Action>(),
                boost::forward<Callback>(cb));
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback>
    inline bool
    apply_p_cb(naming::id_type const& gid, threads::thread_priority priority,
        BOOST_FWD_REF(Callback) cb)
    {
        if (!Action::is_target_valid(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb", 
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }

        // Determine whether the gid is local or remote
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            bool result = applier::detail::apply_l_p<Action>(addr, priority);   // apply locally
            cb(boost::system::error_code(), 0);     // invoke callback
            return result;
        }

        // apply remotely
        return applier::detail::apply_r_p_cb<Action>(addr, gid,
            priority, boost::forward<Callback>(cb));
    }

    template <typename Action, typename Callback>
    inline bool apply_cb(naming::id_type const& gid, BOOST_FWD_REF(Callback) cb)
    {
        return apply_p_cb<Action>(gid, actions::action_priority<Action>(),
            boost::forward<Callback>(cb));
    }

    template <typename Component, typename Result,
        typename Arguments, typename Derived, threads::thread_priority Priority,
        typename Callback>
    inline bool apply_cb(
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /*act*/, naming::id_type const& gid, BOOST_FWD_REF(Callback) cb)
    {
        return apply_p_cb<Derived>(gid, actions::action_priority<Derived>(),
            boost::forward<Callback>(cb));
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback>
        inline bool
        apply_r_p_cb(naming::address& addr, actions::continuation* c,
            naming::id_type const& gid, threads::thread_priority priority,
            BOOST_FWD_REF(Callback) cb)
        {
            typedef typename hpx::actions::extract_action<Action>::type action_type;

            actions::continuation_type cont(c);

            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            parcelset::parcel p (gid.get_gid(), complement_addr<action_type>(addr),
                new hpx::actions::transfer_action<action_type>(priority), cont);

            // Send the parcel through the parcel handler
            hpx::applier::get_applier().get_parcel_handler()
                .put_parcel(p, boost::forward<Callback>(cb));
            return false;     // destination is remote
        }

        template <typename Action, typename Callback>
        inline bool
        apply_r_cb(naming::address& addr, actions::continuation* c,
            naming::id_type const& gid, BOOST_FWD_REF(Callback) cb)
        {
            return apply_r_p_cb<Action>(addr, c, gid,
                actions::action_priority<Action>(), boost::forward<Callback>(cb));
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback>
    inline bool apply_p_cb(actions::continuation* c, naming::address& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        BOOST_FWD_REF(Callback) cb)
    {
        if (!Action::is_target_valid(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb", 
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }

        // Determine whether the gid is local or remote
        if (addr.locality_ == hpx::get_locality()) {
            bool result = applier::detail::apply_l_p<Action>(c, addr, priority);
            cb(boost::system::error_code(), 0);     // invoke callback
            return result;
        }

        // apply remotely
        return applier::detail::apply_r_p_cb<Action>(addr, c, gid, priority,
            boost::forward<Callback>(cb));
    }

    template <typename Action, typename Callback>
    inline bool apply_p_cb(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority, BOOST_FWD_REF(Callback) cb)
    {
        if (!Action::is_target_valid(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb", 
                "the target (destination) does not match the action type");
            return false;
        }

        // Determine whether the gid is local or remote
        naming::address addr;
        if (agas::is_local_address(gid, addr)) {
            bool result = applier::detail::apply_l_p<Action>(c, addr, priority);
            cb(boost::system::error_code(), 0);     // invoke callback
            return result;
        }

        // apply remotely
        return applier::detail::apply_r_p_cb<Action>(addr, c, gid, priority,
            boost::forward<Callback>(cb));
    }

    template <typename Action, typename Callback>
    inline bool apply_cb(actions::continuation* c, naming::id_type const& gid,
        BOOST_FWD_REF(Callback) cb)
    {
        return apply_p_cb<Action>(c, gid, actions::action_priority<Action>(),
            boost::forward<Callback>(cb));
    }

    template <typename Component, typename Result, typename Arguments,
        typename Derived, typename Callback>
    inline bool apply_cb(actions::continuation* c,
        hpx::actions::action<
            Component, Result, Arguments, Derived
        > /*act*/, naming::id_type const& gid, BOOST_FWD_REF(Callback) cb)
    {
        return apply_p_cb<Derived>(c, gid, actions::action_priority<Derived>(),
            boost::forward<Callback>(cb));
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback>
        inline bool
        apply_c_p_cb(naming::address& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            BOOST_FWD_REF(Callback) cb)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r_p_cb<Action>(addr,
                new actions::typed_continuation<result_type>(contgid),
                gid, priority, boost::forward<Callback>(cb));
        }

        template <typename Action, typename Callback>
        inline bool
        apply_c_cb(naming::address& addr, naming::id_type const& contgid,
            naming::id_type const& gid, BOOST_FWD_REF(Callback) cb)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r_cb<Action>(addr,
                new actions::typed_continuation<result_type>(contgid),
                gid, boost::forward<Callback>(cb));
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, BOOST_FWD_REF(Callback) cb)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, priority, boost::forward<Callback>(cb));
    }

    template <typename Action, typename Callback>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::id_type const& gid,
        BOOST_FWD_REF(Callback) cb)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Action>(),
            boost::forward<Callback>(cb));
    }

    template <typename Action, typename Callback>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::address& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        BOOST_FWD_REF(Callback) cb)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            addr, gid, priority, boost::forward<Callback>(cb));
    }

    template <typename Action, typename Callback>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::address& addr,
        naming::id_type const& gid, BOOST_FWD_REF(Callback) cb)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            addr, gid, actions::action_priority<Action>(),
            boost::forward<Callback>(cb));
    }
}

// bring in the rest of the apply_cb<> overloads (arity 1+)
#include <hpx/runtime/applier/apply_implementations_callback.hpp>

#endif
