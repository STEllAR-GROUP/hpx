//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_APPLIER_APPLY_IMPLEMENTATIONS_CALLBACK_DEC_17_2012_0240PM)
#define HPX_APPLIER_APPLY_IMPLEMENTATIONS_CALLBACK_DEC_17_2012_0240PM

#include <hpx/util/move.hpp>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback, typename ...Ts>
        inline bool
        apply_r_p_cb(naming::address&& addr, naming::id_type const& id,
            threads::thread_priority priority, Callback && cb, Ts&&... vs)
        {
            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(), std::forward<Callback>(cb)),
                std::forward<Ts>(vs)...);
            return false;     // destinations are remote
        }

        template <typename Action, typename Callback, typename ...Ts>
        inline bool
        apply_r_cb(naming::address&& addr, naming::id_type const& gid,
            Callback && cb, Ts&&... vs)
        {
            return apply_r_p_cb<Action>(std::move(addr), gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Ts>(vs)...);
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback, typename ...Ts>
    inline bool
    apply_p_cb(naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Ts&&... vs)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }

        // Determine whether the gid is local or remote
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            // apply locally
            bool result = applier::detail::apply_l_p<Action>(gid,
                std::move(addr), priority,
                std::forward<Ts>(vs)...);
            cb(boost::system::error_code(), parcelset::parcel());     // invoke callback
            return result;
        }

        // apply remotely
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), gid,
            priority, std::forward<Callback>(cb),
            std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Callback, typename ...Ts>
    inline bool
    apply_cb(naming::id_type const& gid, Callback && cb, Ts&&... vs)
    {
        return apply_p_cb<Action>(gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename Callback, typename ...Ts>
    inline bool
    apply_cb(
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        naming::id_type const& gid, Callback && cb, Ts&&... vs)
    {
        return apply_p_cb<Derived>(gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback, typename ...Ts>
        inline bool
        apply_r_p_cb(naming::address&& addr, actions::continuation* c,
            naming::id_type const& id, threads::thread_priority priority,
            Callback && cb, Ts&&... vs)
        {
            // If remote, create a new parcel to be sent to the destination
            // Create a new parcel with the gid, action, and arguments
            lcos::local::detail::invoke_when_ready(
                detail::put_parcel<Action>(id, std::move(addr), priority,
                    actions::continuation_type(c), std::forward<Callback>(cb)),
                std::forward<Ts>(vs)...);
            return false;     // destination is remote
        }

        template <typename Action, typename Callback, typename ...Ts>
        inline bool
        apply_r_cb(naming::address&& addr, actions::continuation* c,
            naming::id_type const& gid, Callback && cb, Ts&&... vs)
        {
            return apply_r_p_cb<Action>(std::move(addr), c, gid,
                actions::action_priority<Action>(), std::forward<Callback>(cb),
                std::forward<Ts>(vs)...);
        }
    }}

    ///////////////////////////////////////////////////////////////////////////
    template <typename Action, typename Callback, typename ...Ts>
    inline bool
    apply_p_cb(actions::continuation* c, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Ts&&... vs)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                boost::str(boost::format(
                    "the target (destination) does not match the action type (%s)"
                ) % hpx::actions::detail::get_action_name<Action>()));
            return false;
        }

        // Determine whether the gid is local or remote
        if (addr.locality_ == hpx::get_locality()) {
            // apply locally
            bool result = applier::detail::apply_l_p<Action>(c, gid,
                std::move(addr), priority,
                std::forward<Ts>(vs)...);
            cb(boost::system::error_code(), parcelset::parcel());     // invoke callback
            return result;
        }

        // apply remotely
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, gid,
            priority, std::forward<Callback>(cb),
            std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Callback, typename ...Ts>
    inline bool
    apply_p_cb(actions::continuation* c, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb, Ts&&... vs)
    {
        if (!traits::action_is_target_valid<Action>::call(gid)) {
            HPX_THROW_EXCEPTION(bad_parameter, "apply_p_cb",
                "the target (destination) does not match the action type");
            return false;
        }

        // Determine whether the gid is local or remote
        naming::address addr;
        if (agas::is_local_address_cached(gid, addr)) {
            // apply locally
            bool result = applier::detail::apply_l_p<Action>(c, gid,
                std::move(addr), priority,
                std::forward<Ts>(vs)...);
            cb(boost::system::error_code(), parcelset::parcel());     // invoke callback
            return result;
        }

        // apply remotely
        return applier::detail::apply_r_p_cb<Action>(std::move(addr), c, gid,
            priority, std::forward<Callback>(cb),
            std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Callback, typename ...Ts>
    inline bool
    apply_cb(actions::continuation* c, naming::id_type const& gid,
        Callback && cb, Ts&&... vs)
    {
        return apply_p_cb<Action>(c, gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Component, typename Signature, typename Derived,
        typename Callback, typename ...Ts>
    inline bool
    apply_cb(actions::continuation* c,
        hpx::actions::basic_action<Component, Signature, Derived> /*act*/,
        naming::id_type const& gid, Callback && cb, Ts&&... vs)
    {
        return apply_p<Derived>(c, gid, actions::action_priority<Derived>(),
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    namespace applier { namespace detail
    {
        template <typename Action, typename Callback, typename ...Ts>
        inline bool
        apply_c_p_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, threads::thread_priority priority,
            Callback && cb, Ts&&... vs)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r_p_cb<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, priority, std::forward<Callback>(cb),
                std::forward<Ts>(vs)...);
        }

        template <typename Action, typename Callback, typename ...Ts>
        inline bool
        apply_c_cb(naming::address&& addr, naming::id_type const& contgid,
            naming::id_type const& gid, Callback && cb, Ts&&... vs)
        {
            typedef
                typename hpx::actions::extract_action<Action>::result_type
                result_type;

            return apply_r_p_cb<Action>(std::move(addr),
                new actions::typed_continuation<result_type>(contgid),
                gid, actions::action_priority<Action>(),
                std::forward<Callback>(cb),
                std::forward<Ts>(vs)...);
        }
    }}

    template <typename Action, typename Callback, typename ...Ts>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::id_type const& gid,
        threads::thread_priority priority, Callback && cb, Ts&&... vs)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, priority, std::forward<Callback>(cb),
            std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Callback, typename ...Ts>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::id_type const& gid,
        Callback && cb, Ts&&... vs)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Callback, typename ...Ts>
    inline bool
    apply_c_p_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, threads::thread_priority priority,
        Callback && cb, Ts&&... vs)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            std::move(addr), gid, priority, std::forward<Callback>(cb),
            std::forward<Ts>(vs)...);
    }

    template <typename Action, typename Callback, typename ...Ts>
    inline bool
    apply_c_cb(naming::id_type const& contgid, naming::address&& addr,
        naming::id_type const& gid, Callback && cb, Ts&&... vs)
    {
        typedef
            typename hpx::actions::extract_action<Action>::result_type
            result_type;

        return apply_p_cb<Action>(
            new actions::typed_continuation<result_type>(contgid),
            std::move(addr), gid, actions::action_priority<Action>(),
            std::forward<Callback>(cb), std::forward<Ts>(vs)...);
    }
}

#endif
